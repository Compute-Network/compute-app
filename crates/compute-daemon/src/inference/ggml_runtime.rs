use std::path::PathBuf;
use std::process::Command;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::inference::engine::{InferenceBackend, ShardConfig};
use crate::inference::stage_acceleration::StageAccelerationTarget;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GgmlRuntimeMode {
    NativeLlamaServer { path: PathBuf },
    DockerCuda,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgmlRuntimePlan {
    pub target: StageAccelerationTarget,
    pub mode: GgmlRuntimeMode,
    pub detail: String,
}

impl GgmlRuntimePlan {
    pub fn summary_label(&self) -> String {
        match &self.mode {
            GgmlRuntimeMode::NativeLlamaServer { path } => {
                format!("native-llama-server:{} ({})", path.display(), self.detail)
            }
            GgmlRuntimeMode::DockerCuda => format!("docker-cuda ({})", self.detail),
            GgmlRuntimeMode::Unavailable => format!("unavailable ({})", self.detail),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgmlLaunchSpec {
    pub runtime: GgmlRuntimePlan,
    pub program: String,
    pub args: Vec<String>,
}

impl GgmlLaunchSpec {
    pub fn summary_label(&self) -> String {
        format!("{} {}", self.program, self.args.join(" "))
    }

    pub fn into_command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command.args(&self.args);
        command
    }
}

pub fn detect_ggml_runtime_plan(target: StageAccelerationTarget) -> GgmlRuntimePlan {
    let native_path = find_llama_server().ok();
    let docker_available = docker_cli_available();
    let os_name = std::env::consts::OS;
    ggml_runtime_plan_from_probe(target, native_path, docker_available, os_name)
}

pub fn ggml_runtime_plan_from_probe(
    target: StageAccelerationTarget,
    native_path: Option<PathBuf>,
    docker_available: bool,
    os_name: &str,
) -> GgmlRuntimePlan {
    match target {
        StageAccelerationTarget::Cpu => match native_path {
            Some(path) => GgmlRuntimePlan {
                target,
                mode: GgmlRuntimeMode::NativeLlamaServer { path },
                detail: "cpu target via native llama-server".into(),
            },
            None => GgmlRuntimePlan {
                target,
                mode: GgmlRuntimeMode::Unavailable,
                detail: "native llama-server not found for cpu target".into(),
            },
        },
        StageAccelerationTarget::Metal => match native_path {
            Some(path) => GgmlRuntimePlan {
                target,
                mode: GgmlRuntimeMode::NativeLlamaServer { path },
                detail: format!("{os_name} target via native llama-server"),
            },
            None => GgmlRuntimePlan {
                target,
                mode: GgmlRuntimeMode::Unavailable,
                detail: "native llama-server not found for metal target".into(),
            },
        },
        StageAccelerationTarget::Cuda => {
            if docker_available {
                GgmlRuntimePlan {
                    target,
                    mode: GgmlRuntimeMode::DockerCuda,
                    detail: "cuda target via docker llama.cpp server image".into(),
                }
            } else if let Some(path) = native_path {
                GgmlRuntimePlan {
                    target,
                    mode: GgmlRuntimeMode::NativeLlamaServer { path },
                    detail: "cuda target falling back to native llama-server discovery".into(),
                }
            } else {
                GgmlRuntimePlan {
                    target,
                    mode: GgmlRuntimeMode::Unavailable,
                    detail: "no docker cuda path or native llama-server found".into(),
                }
            }
        }
        StageAccelerationTarget::Vulkan => GgmlRuntimePlan {
            target,
            mode: GgmlRuntimeMode::Unavailable,
            detail: "vulkan ggml runtime path is not wired in this repo yet".into(),
        },
        StageAccelerationTarget::DirectMl => GgmlRuntimePlan {
            target,
            mode: GgmlRuntimeMode::Unavailable,
            detail: "directml ggml runtime path is not wired in this repo yet".into(),
        },
    }
}

pub fn docker_cli_available() -> bool {
    Command::new("docker")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

pub fn build_ggml_launch_spec_for_backend(
    backend: InferenceBackend,
    config: &ShardConfig,
    server_port: u16,
    thread_count: usize,
) -> Result<GgmlLaunchSpec> {
    match backend {
        InferenceBackend::NativeMetal => {
            let plan = GgmlRuntimePlan {
                target: StageAccelerationTarget::Metal,
                mode: GgmlRuntimeMode::NativeLlamaServer { path: find_llama_server()? },
                detail: "native metal llama-server launch".into(),
            };
            build_ggml_launch_spec_for_plan(&plan, config, server_port, thread_count)
        }
        InferenceBackend::DockerCuda => {
            let plan = GgmlRuntimePlan {
                target: StageAccelerationTarget::Cuda,
                mode: GgmlRuntimeMode::DockerCuda,
                detail: "docker cuda llama.cpp launch".into(),
            };
            build_ggml_launch_spec_for_plan(&plan, config, server_port, thread_count)
        }
        InferenceBackend::Cpu => {
            let plan = GgmlRuntimePlan {
                target: StageAccelerationTarget::Cpu,
                mode: GgmlRuntimeMode::NativeLlamaServer { path: find_llama_server()? },
                detail: "cpu llama-server launch".into(),
            };
            build_ggml_launch_spec_for_plan(&plan, config, server_port, thread_count)
        }
    }
}

pub fn build_ggml_launch_spec_for_plan(
    plan: &GgmlRuntimePlan,
    config: &ShardConfig,
    server_port: u16,
    thread_count: usize,
) -> Result<GgmlLaunchSpec> {
    match &plan.mode {
        GgmlRuntimeMode::NativeLlamaServer { path } => {
            let gpu_layers = match plan.target {
                StageAccelerationTarget::Cpu => "0".to_string(),
                StageAccelerationTarget::Metal
                | StageAccelerationTarget::Cuda
                | StageAccelerationTarget::Vulkan
                | StageAccelerationTarget::DirectMl => "999".to_string(),
            };
            Ok(GgmlLaunchSpec {
                runtime: plan.clone(),
                program: path.display().to_string(),
                args: vec![
                    "--model".into(),
                    config.shard_path.to_string_lossy().into_owned(),
                    "--port".into(),
                    server_port.to_string(),
                    "--ctx-size".into(),
                    config.context_length.to_string(),
                    "--batch-size".into(),
                    config.max_batch_size.to_string(),
                    "--n-gpu-layers".into(),
                    gpu_layers,
                    "--threads".into(),
                    thread_count.to_string(),
                ],
            })
        }
        GgmlRuntimeMode::DockerCuda => {
            let shard_dir = config
                .shard_path
                .parent()
                .unwrap_or(&PathBuf::from("/tmp"))
                .to_string_lossy()
                .to_string();
            let shard_filename =
                config.shard_path.file_name().unwrap_or_default().to_string_lossy().to_string();
            Ok(GgmlLaunchSpec {
                runtime: plan.clone(),
                program: "docker".into(),
                args: vec![
                    "run".into(),
                    "--rm".into(),
                    "--gpus".into(),
                    "all".into(),
                    "-p".into(),
                    format!("{server_port}:8080"),
                    "-v".into(),
                    format!("{shard_dir}:/models"),
                    "--name".into(),
                    format!("compute-inference-{server_port}"),
                    "ghcr.io/ggerganov/llama.cpp:server-cuda".into(),
                    "--model".into(),
                    format!("/models/{shard_filename}"),
                    "--port".into(),
                    "8080".into(),
                    "--ctx-size".into(),
                    config.context_length.to_string(),
                    "--batch-size".into(),
                    config.max_batch_size.to_string(),
                    "--n-gpu-layers".into(),
                    "999".into(),
                ],
            })
        }
        GgmlRuntimeMode::Unavailable => {
            anyhow::bail!("Cannot build ggml launch spec: {}", plan.detail)
        }
    }
}

pub fn find_llama_server() -> Result<PathBuf> {
    if let Ok(output) = Command::new("which").arg("llama-server").output()
        && output.status.success()
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() {
            return Ok(PathBuf::from(path));
        }
    }

    let candidates = [
        "/usr/local/bin/llama-server",
        "/opt/homebrew/bin/llama-server",
        "~/.local/bin/llama-server",
    ];

    for candidate in &candidates {
        let expanded = shellexpand::tilde(candidate).to_string();
        let path = PathBuf::from(&expanded);
        if path.exists() {
            return Ok(path);
        }
    }

    anyhow::bail!(
        "llama-server not found. Install it:\n\
         macOS:  brew install llama.cpp\n\
         Linux:  See https://github.com/ggerganov/llama.cpp#build\n\
         Or use Docker mode with an NVIDIA GPU."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_prefers_native_llama_server_when_available() {
        let plan = ggml_runtime_plan_from_probe(
            StageAccelerationTarget::Metal,
            Some(PathBuf::from("/opt/homebrew/bin/llama-server")),
            false,
            "macos",
        );
        assert!(matches!(plan.mode, GgmlRuntimeMode::NativeLlamaServer { .. }));
        assert!(plan.summary_label().contains("native-llama-server"));
    }

    #[test]
    fn cuda_prefers_docker_when_available() {
        let plan = ggml_runtime_plan_from_probe(
            StageAccelerationTarget::Cuda,
            Some(PathBuf::from("/usr/local/bin/llama-server")),
            true,
            "linux",
        );
        assert_eq!(plan.mode, GgmlRuntimeMode::DockerCuda);
    }

    #[test]
    fn vulkan_remains_explicitly_unavailable() {
        let plan =
            ggml_runtime_plan_from_probe(StageAccelerationTarget::Vulkan, None, false, "linux");
        assert_eq!(plan.mode, GgmlRuntimeMode::Unavailable);
        assert!(plan.detail.contains("vulkan"));
    }

    #[test]
    fn native_launch_spec_matches_expected_llama_server_shape() {
        let config = ShardConfig {
            model_id: "gemma-4-e4b-q4".into(),
            shard_path: PathBuf::from("/tmp/stage.index.json"),
            start_layer: 0,
            end_layer: 20,
            total_layers: 42,
            is_first_stage: true,
            is_last_stage: false,
            max_batch_size: 16,
            context_length: 8192,
        };
        let plan = ggml_runtime_plan_from_probe(
            StageAccelerationTarget::Metal,
            Some(PathBuf::from("/opt/homebrew/bin/llama-server")),
            false,
            "macos",
        );
        let launch = build_ggml_launch_spec_for_plan(&plan, &config, 8090, 12).unwrap();
        assert_eq!(launch.program, "/opt/homebrew/bin/llama-server");
        assert!(launch.args.contains(&"--model".into()));
        assert!(launch.args.contains(&"999".into()));
        assert!(launch.summary_label().contains("llama-server"));
    }

    #[test]
    fn docker_launch_spec_matches_expected_shape() {
        let config = ShardConfig {
            model_id: "gemma-4-e4b-q4".into(),
            shard_path: PathBuf::from("/tmp/models/stage.gguf"),
            start_layer: 21,
            end_layer: 41,
            total_layers: 42,
            is_first_stage: false,
            is_last_stage: true,
            max_batch_size: 32,
            context_length: 4096,
        };
        let plan = ggml_runtime_plan_from_probe(StageAccelerationTarget::Cuda, None, true, "linux");
        let launch = build_ggml_launch_spec_for_plan(&plan, &config, 8091, 8).unwrap();
        assert_eq!(launch.program, "docker");
        assert!(launch.args.contains(&"ghcr.io/ggerganov/llama.cpp:server-cuda".into()));
        assert!(launch.args.contains(&"8091:8080".into()));
    }
}
