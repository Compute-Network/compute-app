use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

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
    let size = 64 * 1024 * 1024; // 64 MB
    let mut buf = vec![0u8; size];

    let start = Instant::now();
    let iterations = 4;
    for _ in 0..iterations {
        for (i, byte) in buf.iter_mut().enumerate() {
            *byte = (i & 0xFF) as u8;
        }
    }
    let elapsed = start.elapsed();
    let _ = buf[0]; // prevent optimization

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
