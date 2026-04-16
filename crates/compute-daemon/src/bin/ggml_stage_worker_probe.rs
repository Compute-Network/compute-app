use anyhow::{Context, Result};
use compute_daemon::inference::engine::ShardConfig;
use compute_daemon::inference::ggml_runtime::detect_ggml_runtime_plan;
use compute_daemon::inference::ggml_stage_executor::GgmlStageExecutorKind;
use compute_daemon::inference::ggml_stage_worker::{
    GgmlStageWorkerHostLaunchSpec, GgmlStageWorkerInitSpec, GgmlStageWorkerRequest,
    run_stage_worker_request,
};
use compute_daemon::inference::real_forward_artifact::RealForwardStageLoadSpec;
use compute_daemon::inference::stage_acceleration::StageAccelerationTarget;
use stage_forward_lab::real_forward::RealGemmaBackend;
use stage_forward_lab::{StageForwardBackend, StageLayout};
use std::env;
use std::path::PathBuf;

fn parse_executor(value: &str) -> GgmlStageExecutorKind {
    match value.trim().to_ascii_lowercase().as_str() {
        "ggml" | "ggml-worker" | "ggml_worker" => GgmlStageExecutorKind::Ggml,
        _ => GgmlStageExecutorKind::ReferenceCpu,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let shard_path = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "/Users/macintosh/Documents/projects/Compute/compute-backend/out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json",
            )
        });
    let model_id = args.get(2).cloned().unwrap_or_else(|| "gemma-4-e4b-q4".to_string());
    let start_layer = args.get(3).and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
    let end_layer = args.get(4).and_then(|v| v.parse::<u32>().ok()).unwrap_or(20);
    let stage_role = args.get(5).map(|v| v.as_str()).unwrap_or("head");
    let requested = args.get(6).map(|v| v.as_str()).unwrap_or("metal");
    let operation = args.get(7).map(|v| v.as_str()).unwrap_or("tokenize_generation_prompt");
    let payload = args.get(8).cloned().unwrap_or_else(|| "Hello".to_string());
    let request_id = args.get(9).cloned().unwrap_or_else(|| "ggml-stage-worker-probe".to_string());
    let default_executor = GgmlStageExecutorKind::ReferenceCpu;
    let executor = if operation == "continue_forward_summary" {
        args.get(11).map(|value| parse_executor(value)).unwrap_or(default_executor)
    } else {
        args.get(10).map(|value| parse_executor(value)).unwrap_or(default_executor)
    };

    let target = match requested {
        "cpu" => StageAccelerationTarget::Cpu,
        "cuda" => StageAccelerationTarget::Cuda,
        "vulkan" => StageAccelerationTarget::Vulkan,
        "directml" => StageAccelerationTarget::DirectMl,
        _ => StageAccelerationTarget::Metal,
    };

    let is_head = matches!(stage_role, "head" | "first");
    let is_tail = matches!(stage_role, "tail" | "last");
    let load_spec = RealForwardStageLoadSpec::from_shard_config(&ShardConfig {
        model_id: model_id.clone(),
        shard_path: shard_path.clone(),
        start_layer,
        end_layer,
        total_layers: end_layer + 1,
        is_first_stage: is_head,
        is_last_stage: is_tail,
        max_batch_size: 16,
        context_length: 8192,
    })?;
    let runtime = detect_ggml_runtime_plan(target);
    let init = GgmlStageWorkerInitSpec::from_load_spec(&load_spec, &runtime, executor);
    let launch = GgmlStageWorkerHostLaunchSpec::from_init_spec(&init)?;
    let request = match operation {
        "tokenize_text" => GgmlStageWorkerRequest::TokenizeText { text: payload.clone() },
        "decode_token_ids" => {
            let token_ids = payload
                .split(',')
                .filter(|part| !part.is_empty())
                .map(|part| part.trim().parse::<u32>())
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("parse comma-separated token ids")?;
            GgmlStageWorkerRequest::DecodeTokenIds { token_ids }
        }
        "begin_token_ids_summary" => {
            let token_ids = payload
                .split(',')
                .filter(|part| !part.is_empty())
                .map(|part| part.trim().parse::<u32>())
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("parse comma-separated token ids")?;
            GgmlStageWorkerRequest::BeginTokenIdsSummary {
                request_id,
                token_ids,
                max_tokens: Some(1),
            }
        }
        "continue_forward_summary" => {
            let upstream_path = args
                .get(10)
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    PathBuf::from(
                        "/Users/macintosh/Documents/projects/Compute/compute-backend/out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json",
                    )
                });
            let upstream_spec = RealForwardStageLoadSpec::from_shard_config(&ShardConfig {
                model_id: model_id.clone(),
                shard_path: upstream_path,
                start_layer: 0,
                end_layer: start_layer.saturating_sub(1),
                total_layers: end_layer + 1,
                is_first_stage: true,
                is_last_stage: false,
                max_batch_size: 16,
                context_length: 8192,
            })?;
            let mut upstream_backend = RealGemmaBackend::new(&upstream_spec.index_path);
            if let Some(vocab_path) = upstream_spec.vocab_path.as_deref() {
                upstream_backend
                    .load_tokenizer(vocab_path, upstream_spec.vocab_scores_path.as_deref())?;
            }
            upstream_backend.load_layout(StageLayout {
                model_id: model_id.clone(),
                stage_id: upstream_spec.layout.stage_id.clone(),
                start_layer: upstream_spec.layout.start_layer,
                end_layer: upstream_spec.layout.end_layer,
                is_head: true,
                is_tail: false,
            })?;
            let prompt_tokens = upstream_backend.tokenize_generation_prompt(&payload);
            let input =
                upstream_backend.begin_token_ids(&request_id, &prompt_tokens, Some(1), 0)?;
            GgmlStageWorkerRequest::ContinueForwardSummary { input }
        }
        "eos_token_id" => GgmlStageWorkerRequest::EosTokenId,
        _ => GgmlStageWorkerRequest::TokenizeGenerationPrompt { text: payload.clone() },
    };
    let response = run_stage_worker_request(&launch, &request)?;

    println!("=== GGML Stage Worker Probe ===");
    println!("model        : {model_id}");
    println!("shard        : {}", shard_path.display());
    println!("layers       : {start_layer}-{end_layer}");
    println!("stage role   : {stage_role}");
    println!("target       : {}", target.as_str());
    println!("executor     : {}", executor.as_str());
    println!("runtime      : {}", runtime.summary_label());
    println!("launch       : {}", launch.summary_label());
    println!("request      : {}", serde_json::to_string(&request)?);
    println!("response     : {}", serde_json::to_string(&response)?);

    Ok(())
}
