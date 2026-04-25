use anyhow::Result;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use crate::hardware::{GpuBackend, HardwareInfo};

/// Results from a full node benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub cpu_single_thread: f64,
    pub cpu_multi_thread: f64,
    pub memory_bandwidth_gbs: f64,
    pub download_speed_mbps: Option<f64>,
    pub upload_speed_mbps: Option<f64>,
    pub disk_read_mbps: f64,
    pub estimated_tflops_fp16: f64,
}

/// Fast startup assessment used by the native app splash screen and scheduler
/// metadata. This deliberately avoids heavyweight timed benchmarks so startup
/// stays instant; the score is derived from detected hardware plus known GPU
/// estimates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartupAssessment {
    pub score: u8,
    pub tier: String,
    pub assigned_model_id: String,
    pub assigned_model_label: String,
    pub split_capable: bool,
    pub split_role: String,
    pub split_model_id: Option<String>,
    pub split_model_label: Option<String>,
    pub split_reason: String,
    pub estimated_tflops_fp16: f64,
    pub estimated_memory_bandwidth_gbps: f64,
    pub usable_accelerator_memory_mb: u64,
    pub gpu_backend: String,
    pub gpu_label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaBenchmarkCase {
    pub name: String,
    pub ctx_size: u32,
    pub parallel: u32,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub threads: u32,
    pub flash_attn: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaBenchmarkResult {
    pub case: LlamaBenchmarkCase,
    pub predicted_per_second: Option<f64>,
    pub prompt_per_second: Option<f64>,
    pub total_ms: Option<f64>,
    pub success: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathBenchmarkResult {
    pub name: String,
    pub first_token_ms: Option<f64>,
    pub total_ms: Option<f64>,
    pub completion_tokens: Option<u64>,
    pub effective_tok_per_sec: Option<f64>,
    pub profile: Option<BTreeMap<String, f64>>,
    pub success: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathBenchmarkSuite {
    pub direct_temp: PathBenchmarkResult,
    pub daemon_local: PathBenchmarkResult,
    pub orchestrator: PathBenchmarkResult,
}

pub fn assess_node_startup(hw: &HardwareInfo) -> StartupAssessment {
    let gpu = hw.gpus.first();
    let backend = gpu.map(|g| &g.backend).unwrap_or(&GpuBackend::Cpu);
    let gpu_label = gpu.map(|g| g.name.clone()).unwrap_or_else(|| "CPU".into());
    let gpu_backend = match backend {
        GpuBackend::Cuda => "cuda",
        GpuBackend::Metal => "metal",
        GpuBackend::Cpu => "cpu",
    }
    .to_string();

    let raw_vram_mb = gpu.map(|g| g.vram_mb).unwrap_or(0);
    let usable_accelerator_memory_mb = usable_accelerator_memory_mb(hw, backend, raw_vram_mb);
    let estimated_tflops_fp16 =
        gpu.map(|g| estimate_tflops(&g.name, g.vram_mb)).unwrap_or_default();
    let estimated_memory_bandwidth_gbps =
        estimate_memory_bandwidth_gbps(hw, backend, estimated_tflops_fp16);

    let score = startup_score(
        backend,
        usable_accelerator_memory_mb,
        estimated_tflops_fp16,
        estimated_memory_bandwidth_gbps,
        hw.cpu.threads,
        hw.disk.available_gb,
    );
    let tier = capability_tier(score).to_string();
    let (assigned_model_id, assigned_model_label) =
        assigned_model_for_node(backend, usable_accelerator_memory_mb, estimated_tflops_fp16);
    let (split_capable, split_role, split_reason) = split_assessment(
        backend,
        usable_accelerator_memory_mb,
        estimated_tflops_fp16,
        estimated_memory_bandwidth_gbps,
    );
    let (split_model_id, split_model_label) = if split_capable {
        (
            Some("gemma-4-e4b-q4".to_string()),
            catalog_model_label("gemma-4-e4b-q4").map(str::to_string),
        )
    } else {
        (None, None)
    };

    StartupAssessment {
        score,
        tier,
        assigned_model_id,
        assigned_model_label,
        split_capable,
        split_role,
        split_model_id,
        split_model_label,
        split_reason,
        estimated_tflops_fp16,
        estimated_memory_bandwidth_gbps,
        usable_accelerator_memory_mb,
        gpu_backend,
        gpu_label,
    }
}

fn usable_accelerator_memory_mb(hw: &HardwareInfo, backend: &GpuBackend, raw_vram_mb: u64) -> u64 {
    match backend {
        GpuBackend::Cuda => raw_vram_mb,
        GpuBackend::Metal => {
            let total_memory_mb = (hw.memory.total_gb.max(0.0) * 1024.0) as u64;
            let unified_mb = raw_vram_mb.min(total_memory_mb);
            // Keep enough unified memory free for the OS, browser, and daemon.
            ((unified_mb as f64) * 0.72) as u64
        }
        GpuBackend::Cpu => {
            let available_mb = (hw.memory.available_gb.max(0.0) * 1024.0) as u64;
            ((available_mb as f64) * 0.55) as u64
        }
    }
}

fn startup_score(
    backend: &GpuBackend,
    usable_mem_mb: u64,
    tflops: f64,
    memory_bandwidth_gbps: f64,
    threads: usize,
    disk_available_gb: f64,
) -> u8 {
    let mut score: f64 = match backend {
        GpuBackend::Cuda | GpuBackend::Metal => 22.0,
        GpuBackend::Cpu => 4.0,
    };

    score += match usable_mem_mb {
        32_000.. => 26.0,
        22_000.. => 23.0,
        16_000.. => 19.0,
        8_000.. => 13.0,
        5_000.. => 8.0,
        2_000.. => 4.0,
        _ => 0.0,
    };

    score += if tflops >= 40.0 {
        22.0
    } else if tflops >= 14.0 {
        18.0
    } else if tflops >= 7.0 {
        14.0
    } else if tflops >= 3.0 {
        9.0
    } else if tflops > 0.0 {
        4.0
    } else {
        0.0
    };

    score += if memory_bandwidth_gbps >= 500.0 {
        15.0
    } else if memory_bandwidth_gbps >= 200.0 {
        12.0
    } else if memory_bandwidth_gbps >= 100.0 {
        8.0
    } else if memory_bandwidth_gbps >= 40.0 {
        4.0
    } else {
        0.0
    };

    score += match threads {
        16.. => 8.0,
        12.. => 7.0,
        8.. => 5.0,
        4.. => 3.0,
        _ => 1.0,
    };

    if disk_available_gb >= 80.0 {
        score += 7.0;
    } else if disk_available_gb >= 40.0 {
        score += 4.0;
    } else if disk_available_gb >= 20.0 {
        score += 2.0;
    }

    score.round().clamp(0.0, 100.0) as u8
}

fn capability_tier(score: u8) -> &'static str {
    match score {
        85..=100 => "ultra",
        70..=84 => "plus",
        50..=69 => "standard",
        30..=49 => "lite",
        _ => "cpu-only",
    }
}

fn assigned_model_for_node(
    backend: &GpuBackend,
    usable_mem_mb: u64,
    tflops: f64,
) -> (String, String) {
    let model_id = match backend {
        GpuBackend::Metal | GpuBackend::Cuda if usable_mem_mb >= 22_000 && tflops >= 4.0 => {
            "qwen-3.6"
        }
        GpuBackend::Metal | GpuBackend::Cuda if usable_mem_mb >= 8_000 && tflops >= 3.0 => {
            "gemma-4-e4b-q4"
        }
        GpuBackend::Metal | GpuBackend::Cuda if usable_mem_mb >= 5_000 => "mistral-7b-q4",
        _ => "gemma-3-270m-q4-draft",
    };

    let label = catalog_model_label(model_id).unwrap_or(model_id).to_string();
    (model_id.to_string(), label)
}

fn split_assessment(
    backend: &GpuBackend,
    usable_mem_mb: u64,
    tflops: f64,
    memory_bandwidth_gbps: f64,
) -> (bool, String, String) {
    let has_accelerator = matches!(backend, GpuBackend::Cuda | GpuBackend::Metal);
    if !has_accelerator {
        return (false, "single-node only".into(), "no GPU acceleration detected".into());
    }
    if usable_mem_mb < 6_000 {
        return (
            false,
            "single-node only".into(),
            format!("{} usable accelerator memory", format_mb(usable_mem_mb)),
        );
    }
    if tflops < 3.0 {
        return (
            false,
            "single-node only".into(),
            format!("{tflops:.1} estimated FP16 TFLOPS"),
        );
    }

    let role = if usable_mem_mb >= 20_000 && memory_bandwidth_gbps >= 120.0 && tflops >= 6.0 {
        "head/tail candidate"
    } else if tflops >= 7.0 && memory_bandwidth_gbps >= 90.0 {
        "head candidate"
    } else {
        "tail candidate"
    };

    (
        true,
        role.into(),
        format!(
            "{} usable · {tflops:.1} TFLOPS · {:.0} GB/s memory",
            format_mb(usable_mem_mb),
            memory_bandwidth_gbps
        ),
    )
}

fn estimate_memory_bandwidth_gbps(hw: &HardwareInfo, backend: &GpuBackend, tflops: f64) -> f64 {
    let gpu_name = hw.gpus.first().map(|g| g.name.to_lowercase()).unwrap_or_default();

    if matches!(backend, GpuBackend::Metal) {
        if gpu_name.contains("ultra") {
            return 800.0;
        }
        if gpu_name.contains("max") {
            return 400.0;
        }
        if gpu_name.contains("pro") {
            return 150.0;
        }
        return 100.0;
    }

    if matches!(backend, GpuBackend::Cuda) {
        if gpu_name.contains("h100") {
            return 3350.0;
        }
        if gpu_name.contains("a100") {
            return 1555.0;
        }
        if gpu_name.contains("4090") {
            return 1008.0;
        }
        if gpu_name.contains("3090") {
            return 936.0;
        }
        if gpu_name.contains("4080") {
            return 716.0;
        }
        if gpu_name.contains("3080") {
            return 760.0;
        }
        if gpu_name.contains("4070") {
            return 504.0;
        }
        return (tflops * 18.0).clamp(120.0, 700.0);
    }

    let thread_factor = hw.cpu.threads.max(1) as f64;
    (18.0 + thread_factor * 2.0).clamp(20.0, 80.0)
}

fn catalog_model_label(model_id: &str) -> Option<&'static str> {
    match model_id {
        "qwen-3.6" => Some("Qwen 3.6 35B-A3B"),
        "gemma-4-e4b-q4" => Some("Gemma 4 E4B Q4"),
        "mistral-7b-q4" => Some("Mistral 7B Q4"),
        "gemma-3-270m-q4-draft" => Some("Gemma 3 270M utility"),
        _ => None,
    }
}

fn format_mb(mb: u64) -> String {
    if mb >= 1024 { format!("{:.1}GB", mb as f64 / 1024.0) } else { format!("{mb}MB") }
}

/// Run a CPU benchmark (simple computation throughput).
pub fn bench_cpu() -> (f64, f64) {
    // Single-thread: count iterations of a compute-heavy loop in 1 second
    let single = {
        let start = Instant::now();
        let mut count: u64 = 0;
        let mut x: f64 = 1.0;
        while start.elapsed() < Duration::from_secs(1) {
            for _ in 0..1000 {
                x = (x * 1.0000001).sin().abs() + 0.5;
                count += 1;
            }
        }
        let _ = x; // prevent optimization
        count as f64 / 1_000_000.0 // Mops
    };

    // Multi-thread: same but across all cores
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let multi = {
        let _start = Instant::now();
        let handles: Vec<_> = (0..cores)
            .map(|_| {
                std::thread::spawn(move || {
                    let mut count: u64 = 0;
                    let mut x: f64 = 1.0;
                    let thread_start = Instant::now();
                    while thread_start.elapsed() < Duration::from_secs(1) {
                        for _ in 0..1000 {
                            x = (x * 1.0000001).sin().abs() + 0.5;
                            count += 1;
                        }
                    }
                    let _ = x;
                    count
                })
            })
            .collect();

        let total: u64 = handles.into_iter().map(|h| h.join().unwrap_or(0)).sum();
        total as f64 / 1_000_000.0 // Mops
    };

    (single, multi)
}

/// Estimate memory bandwidth (simplified: measure sequential write throughput).
pub fn bench_memory() -> f64 {
    let size = 256 * 1024 * 1024; // 256 MB
    let mut buf = vec![0u8; size];
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(8);

    let start = Instant::now();
    let iterations = 6;
    for iter in 0..iterations {
        std::thread::scope(|scope| {
            let chunk_len = buf.len().div_ceil(threads);
            for (idx, chunk) in buf.chunks_mut(chunk_len).enumerate() {
                scope.spawn(move || {
                    let value = ((iter + idx) & 0xFF) as u8;
                    chunk.fill(value);
                });
            }
        });
    }
    let elapsed = start.elapsed();
    let _ = buf[0];

    let bytes_written = size as f64 * iterations as f64;
    bytes_written / elapsed.as_secs_f64() / 1_073_741_824.0
}

/// Estimate disk sequential read speed.
pub fn bench_disk() -> f64 {
    let temp_path = std::env::temp_dir().join("compute-bench-temp");

    // Write a temp file
    let size = 32 * 1024 * 1024; // 32 MB
    let data = vec![42u8; size];
    if std::fs::write(&temp_path, &data).is_err() {
        return 0.0;
    }

    // Read it back
    let start = Instant::now();
    let _ = std::fs::read(&temp_path);
    let elapsed = start.elapsed();

    // Cleanup
    let _ = std::fs::remove_file(&temp_path);

    size as f64 / elapsed.as_secs_f64() / 1_048_576.0
}

/// Run a download speed test using a public endpoint.
pub async fn bench_network_download() -> Option<f64> {
    let client = reqwest::Client::builder().timeout(Duration::from_secs(15)).build().ok()?;

    // Use Cloudflare's speed test endpoint (1MB file)
    let url = "https://speed.cloudflare.com/__down?bytes=1000000";

    let start = Instant::now();
    let resp = client.get(url).send().await.ok()?;
    let bytes = resp.bytes().await.ok()?;
    let elapsed = start.elapsed();

    let mbps = bytes.len() as f64 * 8.0 / elapsed.as_secs_f64() / 1_000_000.0;
    Some(mbps)
}

/// Estimate FP16 TFLOPS based on hardware.
pub fn estimate_tflops(gpu_name: &str, vram_mb: u64) -> f64 {
    let name_lower = gpu_name.to_lowercase();

    // Known GPU estimates (approximate FP16 TFLOPS)
    if name_lower.contains("m3 ultra") {
        return 27.0;
    }
    if name_lower.contains("m3 max") {
        return 14.2;
    }
    if name_lower.contains("m3 pro") {
        return 7.4;
    }
    if name_lower.contains("m3") {
        return 4.1;
    }
    if name_lower.contains("m2 ultra") {
        return 27.2;
    }
    if name_lower.contains("m2 max") {
        return 13.5;
    }
    if name_lower.contains("m2 pro") {
        return 6.8;
    }
    if name_lower.contains("m2") {
        return 3.6;
    }
    if name_lower.contains("m1 ultra") {
        return 21.0;
    }
    if name_lower.contains("m1 max") {
        return 10.4;
    }
    if name_lower.contains("m1 pro") {
        return 5.2;
    }
    if name_lower.contains("m1") {
        return 2.6;
    }
    if name_lower.contains("m4") {
        return 8.0;
    } // Rough estimate
    if name_lower.contains("4090") {
        return 82.6;
    }
    if name_lower.contains("4080") {
        return 48.7;
    }
    if name_lower.contains("4070 ti") {
        return 40.1;
    }
    if name_lower.contains("4070") {
        return 29.1;
    }
    if name_lower.contains("3090") {
        return 35.6;
    }
    if name_lower.contains("3080") {
        return 29.8;
    }
    if name_lower.contains("3070") {
        return 20.3;
    }
    if name_lower.contains("5090") {
        return 104.8;
    }
    if name_lower.contains("a100") {
        return 77.6;
    }
    if name_lower.contains("h100") {
        return 267.6;
    }

    // Fallback: rough estimate from VRAM (lower bound)
    vram_mb as f64 / 4000.0
}

/// Run the complete benchmark suite.
pub async fn run_full_benchmark(gpu_name: &str, vram_mb: u64) -> BenchmarkResults {
    let (cpu_single, cpu_multi) = bench_cpu();
    let memory_bw = bench_memory();
    let disk_read = bench_disk();
    let download = bench_network_download().await;
    let tflops = estimate_tflops(gpu_name, vram_mb);

    BenchmarkResults {
        cpu_single_thread: cpu_single,
        cpu_multi_thread: cpu_multi,
        memory_bandwidth_gbs: memory_bw,
        download_speed_mbps: download,
        upload_speed_mbps: None, // TODO
        disk_read_mbps: disk_read,
        estimated_tflops_fp16: tflops,
    }
}

pub fn default_llama_sweep_cases() -> Vec<LlamaBenchmarkCase> {
    vec![
        LlamaBenchmarkCase {
            name: "current".into(),
            ctx_size: 32_768,
            parallel: 2,
            batch_size: 2048,
            ubatch_size: 512,
            threads: 6,
            flash_attn: true,
        },
        LlamaBenchmarkCase {
            name: "low-ctx".into(),
            ctx_size: 8192,
            parallel: 2,
            batch_size: 2048,
            ubatch_size: 512,
            threads: 6,
            flash_attn: true,
        },
        LlamaBenchmarkCase {
            name: "small-batch".into(),
            ctx_size: 8192,
            parallel: 2,
            batch_size: 1024,
            ubatch_size: 256,
            threads: 6,
            flash_attn: true,
        },
        LlamaBenchmarkCase {
            name: "more-threads".into(),
            ctx_size: 8192,
            parallel: 2,
            batch_size: 1024,
            ubatch_size: 256,
            threads: 8,
            flash_attn: true,
        },
        LlamaBenchmarkCase {
            name: "flash-off".into(),
            ctx_size: 8192,
            parallel: 2,
            batch_size: 2048,
            ubatch_size: 512,
            threads: 6,
            flash_attn: false,
        },
        LlamaBenchmarkCase {
            name: "parallel-1".into(),
            ctx_size: 8192,
            parallel: 1,
            batch_size: 2048,
            ubatch_size: 512,
            threads: 6,
            flash_attn: true,
        },
        LlamaBenchmarkCase {
            name: "ubatch-1024".into(),
            ctx_size: 8192,
            parallel: 2,
            batch_size: 2048,
            ubatch_size: 1024,
            threads: 6,
            flash_attn: true,
        },
    ]
}

pub async fn run_llama_benchmark_sweep(model_name: &str) -> Result<Vec<LlamaBenchmarkResult>> {
    let model_path = find_model_path(model_name)
        .ok_or_else(|| anyhow::anyhow!("Model not found locally: {model_name}"))?;
    let llama_server = find_llama_server()?;
    let prompt = "Write one short paragraph about efficient local inference on Apple Silicon.";

    let mut results = Vec::new();
    for (idx, case) in default_llama_sweep_cases().into_iter().enumerate() {
        let port = 18090 + idx as u16;
        let result = run_single_llama_case(&llama_server, &model_path, port, prompt, case).await;
        results.push(result);
    }

    Ok(results)
}

async fn run_single_llama_case(
    llama_server: &Path,
    model_path: &Path,
    port: u16,
    prompt: &str,
    case: LlamaBenchmarkCase,
) -> LlamaBenchmarkResult {
    let mut command = Command::new(llama_server);
    command
        .arg("--model")
        .arg(model_path)
        .arg("--port")
        .arg(port.to_string())
        .arg("--ctx-size")
        .arg(case.ctx_size.to_string())
        .arg("--n-gpu-layers")
        .arg("999")
        .arg("--parallel")
        .arg(case.parallel.to_string())
        .arg("--cache-type-k")
        .arg("q8_0")
        .arg("--cache-type-v")
        .arg("q8_0")
        .arg("--batch-size")
        .arg(case.batch_size.to_string())
        .arg("--ubatch-size")
        .arg(case.ubatch_size.to_string())
        .arg("--threads")
        .arg(case.threads.to_string())
        .arg("--jinja")
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    if case.flash_attn {
        command.arg("--flash-attn").arg("on");
    }

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(e) => {
            return LlamaBenchmarkResult {
                case,
                predicted_per_second: None,
                prompt_per_second: None,
                total_ms: None,
                success: false,
                error: Some(format!("Failed to start llama-server: {e}")),
            };
        }
    };

    let result = async {
        let case_for_success = case.clone();
        wait_for_health(port, 60).await?;

        let client = reqwest::Client::builder().timeout(Duration::from_secs(120)).build()?;

        let started = Instant::now();
        let resp = client
            .post(format!("http://127.0.0.1:{port}/completion"))
            .json(&serde_json::json!({
                "prompt": prompt,
                "n_predict": 512,
                "temperature": 0.2,
                "stream": false,
            }))
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("benchmark request failed ({}): {}", status, body);
        }

        let total_ms = started.elapsed().as_secs_f64() * 1000.0;
        let body = resp.json::<serde_json::Value>().await?;
        let timings = body.get("timings").cloned().unwrap_or_default();

        Ok::<LlamaBenchmarkResult, anyhow::Error>(LlamaBenchmarkResult {
            case: case_for_success,
            predicted_per_second: timings.get("predicted_per_second").and_then(|v| v.as_f64()),
            prompt_per_second: timings.get("prompt_per_second").and_then(|v| v.as_f64()),
            total_ms: Some(total_ms),
            success: true,
            error: None,
        })
    }
    .await;

    let _ = stop_child(&mut child);

    match result {
        Ok(result) => result,
        Err(e) => LlamaBenchmarkResult {
            case,
            predicted_per_second: None,
            prompt_per_second: None,
            total_ms: None,
            success: false,
            error: Some(e.to_string()),
        },
    }
}

pub async fn run_path_comparison(
    model_name: &str,
    api_key: Option<&str>,
) -> Result<PathBenchmarkSuite> {
    let direct_temp = run_direct_temp_path_case(model_name).await?;
    let daemon_local = run_daemon_local_path_case().await;
    let orchestrator = run_orchestrator_path_case(model_name, api_key).await;

    Ok(PathBenchmarkSuite { direct_temp, daemon_local, orchestrator })
}

pub async fn run_direct_temp_path_case(model_name: &str) -> Result<PathBenchmarkResult> {
    let model_path = find_model_path(model_name)
        .ok_or_else(|| anyhow::anyhow!("Model not found locally: {model_name}"))?;
    let llama_server = find_llama_server()?;

    let direct_case = LlamaBenchmarkCase {
        name: "direct-temp".into(),
        ctx_size: 32_768,
        parallel: 2,
        batch_size: 2048,
        ubatch_size: 512,
        threads: 6,
        flash_attn: true,
    };

    Ok(run_single_path_case_temp(&llama_server, &model_path, 18100, model_name, direct_case).await)
}

pub async fn run_daemon_local_path_case() -> PathBenchmarkResult {
    run_single_path_case_daemon_local().await
}

pub async fn run_orchestrator_path_case(
    model_name: &str,
    api_key: Option<&str>,
) -> PathBenchmarkResult {
    run_single_path_case_orchestrator(model_name, api_key).await
}

async fn run_single_path_case_temp(
    llama_server: &Path,
    model_path: &Path,
    port: u16,
    model_name: &str,
    case: LlamaBenchmarkCase,
) -> PathBenchmarkResult {
    let mut command = Command::new(llama_server);
    command
        .arg("--model")
        .arg(model_path)
        .arg("--port")
        .arg(port.to_string())
        .arg("--ctx-size")
        .arg(case.ctx_size.to_string())
        .arg("--n-gpu-layers")
        .arg("999")
        .arg("--parallel")
        .arg(case.parallel.to_string())
        .arg("--cache-type-k")
        .arg("q8_0")
        .arg("--cache-type-v")
        .arg("q8_0")
        .arg("--batch-size")
        .arg(case.batch_size.to_string())
        .arg("--ubatch-size")
        .arg(case.ubatch_size.to_string())
        .arg("--threads")
        .arg(case.threads.to_string())
        .arg("--jinja")
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    if case.flash_attn {
        command.arg("--flash-attn").arg("on");
    }

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(e) => {
            return PathBenchmarkResult {
                name: "direct-temp".into(),
                first_token_ms: None,
                total_ms: None,
                completion_tokens: None,
                effective_tok_per_sec: None,
                profile: None,
                success: false,
                error: Some(format!("Failed to start llama-server: {e}")),
            };
        }
    };

    let result = match wait_for_health(port, 60).await {
        Ok(()) => {
            run_chat_stream_path_case(
                "direct-temp",
                &format!("http://127.0.0.1:{port}/v1/chat/completions"),
                Some(model_name),
                None,
                false,
            )
            .await
        }
        Err(e) => PathBenchmarkResult {
            name: "direct-temp".into(),
            first_token_ms: None,
            total_ms: None,
            completion_tokens: None,
            effective_tok_per_sec: None,
            profile: None,
            success: false,
            error: Some(e.to_string()),
        },
    };

    let _ = stop_child(&mut child);
    result
}

async fn run_single_path_case_daemon_local() -> PathBenchmarkResult {
    run_chat_stream_path_case(
        "daemon-local",
        "http://127.0.0.1:8090/v1/chat/completions",
        None,
        None,
        false,
    )
    .await
}

async fn run_single_path_case_orchestrator(
    model_name: &str,
    api_key: Option<&str>,
) -> PathBenchmarkResult {
    let api_key = api_key
        .map(|v| v.to_string())
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .or_else(|| std::env::var("COMPUTE_API_KEY").ok())
        .or_else(load_compute_code_api_key);

    let Some(api_key) = api_key else {
        return PathBenchmarkResult {
            name: "orchestrator".into(),
            first_token_ms: None,
            total_ms: None,
            completion_tokens: None,
            effective_tok_per_sec: None,
            profile: None,
            success: false,
            error: Some(
                "missing api key (pass --api-key, set OPENAI_API_KEY, or log into compute-code)"
                    .into(),
            ),
        };
    };

    let base_url = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.computenetwork.sh/v1".to_string());
    run_chat_stream_path_case(
        "orchestrator",
        &format!("{base_url}/chat/completions"),
        Some(model_name),
        Some(api_key.as_str()),
        true,
    )
    .await
}

async fn run_chat_stream_path_case(
    name: &str,
    url: &str,
    model_name: Option<&str>,
    api_key: Option<&str>,
    with_profile: bool,
) -> PathBenchmarkResult {
    let client = match reqwest::Client::builder().timeout(Duration::from_secs(180)).build() {
        Ok(client) => client,
        Err(e) => {
            return PathBenchmarkResult {
                name: name.into(),
                first_token_ms: None,
                total_ms: None,
                completion_tokens: None,
                effective_tok_per_sec: None,
                profile: None,
                success: false,
                error: Some(e.to_string()),
            };
        }
    };

    let prompt = "Write a concise explanation of why Apple Silicon memory bandwidth matters for local inference.";
    let mut request = client.post(url).json(&serde_json::json!({
        "model": model_name.unwrap_or("auto"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.2,
        "stream": true,
    }));
    if let Some(api_key) = api_key {
        request = request.bearer_auth(api_key);
    }
    if with_profile {
        request = request.header("x-compute-profile", "1");
    }

    let started = Instant::now();
    let resp = match request.send().await {
        Ok(resp) => resp,
        Err(e) => {
            return PathBenchmarkResult {
                name: name.into(),
                first_token_ms: None,
                total_ms: None,
                completion_tokens: None,
                effective_tok_per_sec: None,
                profile: None,
                success: false,
                error: Some(e.to_string()),
            };
        }
    };

    let mut profile = resp
        .headers()
        .get("x-compute-profile")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| serde_json::from_str::<BTreeMap<String, f64>>(s).ok());

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return PathBenchmarkResult {
            name: name.into(),
            first_token_ms: None,
            total_ms: None,
            completion_tokens: None,
            effective_tok_per_sec: None,
            profile,
            success: false,
            error: Some(format!("{status}: {body}")),
        };
    }

    let mut stream = resp.bytes_stream();
    let mut buf = Vec::new();
    let mut first_token_ms = None;
    let mut completion_tokens = None;
    let mut captured_content_tokens = 0u64;

    loop {
        match stream.next().await {
            Some(Ok(chunk)) => {
                buf.extend_from_slice(&chunk);
                while let Some(frame_end) = find_sse_frame_end(&buf) {
                    let frame = String::from_utf8_lossy(&buf[..frame_end]).to_string();
                    buf.drain(..frame_end + 2);
                    process_sse_frame(
                        &frame,
                        &started,
                        &mut first_token_ms,
                        &mut completion_tokens,
                        &mut captured_content_tokens,
                        &mut profile,
                    );
                }
            }
            Some(Err(e)) => {
                return PathBenchmarkResult {
                    name: name.into(),
                    first_token_ms,
                    total_ms: None,
                    completion_tokens,
                    effective_tok_per_sec: None,
                    profile,
                    success: false,
                    error: Some(e.to_string()),
                };
            }
            None => {
                if !buf.is_empty() {
                    let frame = String::from_utf8_lossy(&buf).to_string();
                    process_sse_frame(
                        &frame,
                        &started,
                        &mut first_token_ms,
                        &mut completion_tokens,
                        &mut captured_content_tokens,
                        &mut profile,
                    );
                }
                break;
            }
        }
    }

    let total_ms = started.elapsed().as_secs_f64() * 1000.0;
    let completion_tokens =
        completion_tokens.or_else(|| Some(captured_content_tokens).filter(|v| *v > 0));
    let effective_tok_per_sec =
        completion_tokens.map(|tokens| tokens as f64 / started.elapsed().as_secs_f64().max(0.001));

    PathBenchmarkResult {
        name: name.into(),
        first_token_ms,
        total_ms: Some(total_ms),
        completion_tokens,
        effective_tok_per_sec,
        profile,
        success: true,
        error: None,
    }
}

fn find_sse_frame_end(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|window| window == b"\n\n")
}

fn process_sse_frame(
    frame: &str,
    started: &Instant,
    first_token_ms: &mut Option<f64>,
    completion_tokens: &mut Option<u64>,
    captured_content_tokens: &mut u64,
    profile: &mut Option<BTreeMap<String, f64>>,
) {
    for line in frame.lines() {
        if !line.starts_with("data: ") {
            continue;
        }
        let payload = line.trim_start_matches("data: ").trim();
        if payload == "[DONE]" {
            continue;
        }
        let Ok(json) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        if json.get("object").and_then(|v| v.as_str()) == Some("compute.profile") {
            if let Some(next_profile) = json
                .get("profile")
                .cloned()
                .and_then(|value| serde_json::from_value::<BTreeMap<String, f64>>(value).ok())
            {
                *profile = Some(next_profile);
            }
            continue;
        }
        if first_token_ms.is_none() && has_visible_content_chunk(&json) {
            *first_token_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
        }
        if let Some(tokens) = json
            .get("usage")
            .and_then(|usage| usage.get("completion_tokens"))
            .and_then(|v| v.as_u64())
        {
            *completion_tokens = Some(tokens);
        }
        *captured_content_tokens +=
            extract_delta_text(&json).map(|text| estimate_token_count(text) as u64).unwrap_or(0);
    }
}

fn has_visible_content_chunk(json: &serde_json::Value) -> bool {
    extract_delta_text(json).map(|text| !text.trim().is_empty()).unwrap_or(false)
}

fn extract_delta_text<'a>(json: &'a serde_json::Value) -> Option<&'a str> {
    json.get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("delta"))
        .and_then(|delta| delta.get("content"))
        .and_then(|v| v.as_str())
        .or_else(|| {
            json.get("choices")
                .and_then(|choices| choices.as_array())
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("delta"))
                .and_then(|delta| delta.get("reasoning_content"))
                .and_then(|v| v.as_str())
        })
        .or_else(|| {
            json.get("choices")
                .and_then(|choices| choices.as_array())
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("message"))
                .and_then(|message| message.get("content"))
                .and_then(|v| v.as_str())
        })
        .or_else(|| {
            json.get("choices")
                .and_then(|choices| choices.as_array())
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("message"))
                .and_then(|message| message.get("reasoning_content"))
                .and_then(|v| v.as_str())
        })
}

fn estimate_token_count(text: &str) -> usize {
    text.split_whitespace().count().max(1)
}

async fn wait_for_health(port: u16, timeout_secs: u64) -> Result<()> {
    let client = reqwest::Client::builder().timeout(Duration::from_secs(2)).build()?;

    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(timeout_secs) {
        if let Ok(resp) = client.get(format!("http://127.0.0.1:{port}/health")).send().await
            && resp.status().is_success()
        {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    anyhow::bail!("llama-server did not become healthy within {timeout_secs}s")
}

fn stop_child(child: &mut Child) -> Result<()> {
    let _ = child.kill();
    let _ = child.wait();
    Ok(())
}

fn find_llama_server() -> Result<PathBuf> {
    if let Ok(output) = Command::new("which").arg("llama-server").output()
        && output.status.success()
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() {
            return Ok(PathBuf::from(path));
        }
    }

    for candidate in [
        "/usr/local/bin/llama-server",
        "/opt/homebrew/bin/llama-server",
        "~/.local/bin/llama-server",
    ] {
        let expanded = shellexpand::tilde(candidate).to_string();
        let path = PathBuf::from(expanded);
        if path.exists() {
            return Ok(path);
        }
    }

    anyhow::bail!("llama-server not found")
}

fn find_model_path(model_name: &str) -> Option<PathBuf> {
    let cache_dir = dirs::home_dir()?.join(".compute").join("models");
    if !cache_dir.exists() {
        return None;
    }

    let lower_model = model_name.to_lowercase();
    let segments: Vec<&str> = lower_model
        .split('-')
        .filter(|s| *s != "q4" && *s != "q8" && *s != "fp16" && *s != "q2")
        .collect();

    let entries = std::fs::read_dir(&cache_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_lowercase();
        if !name.ends_with(".gguf") {
            continue;
        }
        if segments.iter().all(|seg| name.contains(seg)) && verify_gguf_magic(&path) {
            return Some(path);
        }
    }

    let entries = std::fs::read_dir(&cache_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "gguf") && verify_gguf_magic(&path) {
            return Some(path);
        }
    }

    None
}

fn verify_gguf_magic(path: &Path) -> bool {
    let mut file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return false,
    };
    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_err() {
        return false;
    }
    magic == [0x47, 0x47, 0x55, 0x46]
}

fn load_compute_code_api_key() -> Option<String> {
    let home = dirs::home_dir()?;
    let path = home.join(".compute-code").join("credentials.json");
    let raw = std::fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    value
        .get("api_key")
        .or_else(|| value.get("apiKey"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}
