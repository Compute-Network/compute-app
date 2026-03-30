use anyhow::Result;
use serde::{Deserialize, Serialize};
use sysinfo::System;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub gpus: Vec<GpuInfo>,
    pub disk: DiskInfo,
    pub os: OsInfo,
    pub docker: DockerStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub brand: String,
    pub cores: usize,
    pub threads: usize,
    pub frequency_mhz: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_gb: f64,
    pub available_gb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_mb: u64,
    pub backend: GpuBackend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuBackend {
    Cuda,
    Metal,
    Cpu,
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackend::Cuda => write!(f, "CUDA"),
            GpuBackend::Metal => write!(f, "Metal"),
            GpuBackend::Cpu => write!(f, "CPU"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskInfo {
    pub total_gb: f64,
    pub available_gb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsInfo {
    pub name: String,
    pub version: String,
    pub arch: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerStatus {
    pub available: bool,
    pub version: Option<String>,
}

/// Detect all hardware information.
pub fn detect() -> HardwareInfo {
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu = detect_cpu(&sys);
    let memory = detect_memory(&sys);
    let gpus = detect_gpus();
    let disk = detect_disk();
    let os = detect_os();
    let docker = detect_docker();

    HardwareInfo { cpu, memory, gpus, disk, os, docker }
}

fn detect_cpu(sys: &System) -> CpuInfo {
    let cpus = sys.cpus();
    let brand = cpus.first().map(|c| c.brand().to_string()).unwrap_or_else(|| "Unknown".into());
    let frequency = cpus.first().map(|c| c.frequency()).unwrap_or(0);

    CpuInfo {
        brand,
        cores: sysinfo::System::physical_core_count().unwrap_or(0),
        threads: cpus.len(),
        frequency_mhz: frequency,
    }
}

fn detect_memory(sys: &System) -> MemoryInfo {
    MemoryInfo {
        total_gb: sys.total_memory() as f64 / 1_073_741_824.0,
        available_gb: sys.available_memory() as f64 / 1_073_741_824.0,
    }
}

fn detect_gpus() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();

    // Try Apple Silicon (macOS)
    #[cfg(target_os = "macos")]
    {
        if let Some(gpu) = detect_apple_gpu() {
            gpus.push(gpu);
        }
    }

    // Try NVIDIA (nvidia-smi)
    if gpus.is_empty()
        && let Ok(nvidia_gpus) = detect_nvidia_gpus()
    {
        gpus.extend(nvidia_gpus);
    }

    // CPU fallback
    if gpus.is_empty() {
        gpus.push(GpuInfo {
            name: "CPU (no GPU detected)".into(),
            vram_mb: 0,
            backend: GpuBackend::Cpu,
        });
    }

    gpus
}

#[cfg(target_os = "macos")]
fn detect_apple_gpu() -> Option<GpuInfo> {
    use std::process::Command;

    let output =
        Command::new("system_profiler").args(["SPDisplaysDataType", "-json"]).output().ok()?;

    let json: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let displays = json.get("SPDisplaysDataType")?.as_array()?;

    let display = displays.first()?;
    let name = display.get("sppci_model").and_then(|v| v.as_str()).unwrap_or("Apple GPU");

    // Apple Silicon unified memory — report total system memory as "VRAM"
    let sys = System::new_all();
    let vram_mb = sys.total_memory() / 1_048_576;

    Some(GpuInfo { name: name.to_string(), vram_mb, backend: GpuBackend::Metal })
}

fn detect_nvidia_gpus() -> Result<Vec<GpuInfo>> {
    use std::process::Command;

    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output()?;

    if !output.status.success() {
        anyhow::bail!("nvidia-smi failed");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();

    for line in stdout.lines() {
        let parts: Vec<&str> = line.splitn(2, ',').collect();
        if parts.len() == 2 {
            let name = parts[0].trim().to_string();
            let vram_mb: u64 = parts[1].trim().parse().unwrap_or(0);
            gpus.push(GpuInfo { name, vram_mb, backend: GpuBackend::Cuda });
        }
    }

    Ok(gpus)
}

fn detect_disk() -> DiskInfo {
    use sysinfo::Disks;
    let disks = Disks::new_with_refreshed_list();

    let mut total: u64 = 0;
    let mut available: u64 = 0;

    for disk in disks.list() {
        total += disk.total_space();
        available += disk.available_space();
    }

    DiskInfo {
        total_gb: total as f64 / 1_073_741_824.0,
        available_gb: available as f64 / 1_073_741_824.0,
    }
}

fn detect_os() -> OsInfo {
    OsInfo {
        name: System::name().unwrap_or_else(|| "Unknown".into()),
        version: System::os_version().unwrap_or_else(|| "Unknown".into()),
        arch: std::env::consts::ARCH.to_string(),
    }
}

fn detect_docker() -> DockerStatus {
    use std::process::Command;

    let output =
        Command::new("docker").args(["version", "--format", "{{.Server.Version}}"]).output();

    match output {
        Ok(out) if out.status.success() => DockerStatus {
            available: true,
            version: Some(String::from_utf8_lossy(&out.stdout).trim().to_string()),
        },
        _ => DockerStatus { available: false, version: None },
    }
}

/// Live system metrics for the dashboard.
#[derive(Debug, Clone, Default)]
pub struct LiveMetrics {
    pub cpu_usage: f32,
    pub memory_used_gb: f64,
    pub memory_total_gb: f64,
    pub gpu_temp: Option<u32>,
    pub gpu_usage: Option<f32>,
    pub gpu_vram_used_mb: Option<u64>,
    pub gpu_vram_total_mb: Option<u64>,
    pub gpu_power_watts: Option<u32>,
    pub gpu_power_limit_watts: Option<u32>,
}

/// Collect live metrics from the system.
pub fn collect_live_metrics(sys: &mut System) -> LiveMetrics {
    sys.refresh_all();

    let cpu_usage = sys.global_cpu_usage();
    let memory_used_gb = sys.used_memory() as f64 / 1_073_741_824.0;
    let memory_total_gb = sys.total_memory() as f64 / 1_073_741_824.0;

    let mut metrics =
        LiveMetrics { cpu_usage, memory_used_gb, memory_total_gb, ..Default::default() };

    // Try to get NVIDIA GPU metrics
    if let Ok(gpu_metrics) = collect_nvidia_metrics() {
        metrics.gpu_temp = Some(gpu_metrics.0);
        metrics.gpu_usage = Some(gpu_metrics.1);
        metrics.gpu_vram_used_mb = Some(gpu_metrics.2);
        metrics.gpu_vram_total_mb = Some(gpu_metrics.3);
        metrics.gpu_power_watts = Some(gpu_metrics.4);
        metrics.gpu_power_limit_watts = Some(gpu_metrics.5);
    }

    // Apple Silicon: unified memory is VRAM, read temp/power from ioreg
    #[cfg(target_os = "macos")]
    if metrics.gpu_temp.is_none() {
        let total_mb = (memory_total_gb * 1024.0) as u64;
        let used_mb = (memory_used_gb * 1024.0) as u64;
        metrics.gpu_vram_used_mb = Some(used_mb);
        metrics.gpu_vram_total_mb = Some(total_mb);

        let (temp, power) = collect_macos_metrics();
        metrics.gpu_temp = temp;
        if let Some(w) = power {
            metrics.gpu_power_watts = Some(w);
        }
    }

    metrics
}

fn collect_nvidia_metrics() -> Result<(u32, f32, u64, u64, u32, u32)> {
    use std::process::Command;

    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ])
        .output()?;

    if !output.status.success() {
        anyhow::bail!("nvidia-smi failed");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next().ok_or_else(|| anyhow::anyhow!("no output"))?;
    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

    if parts.len() < 6 {
        anyhow::bail!("unexpected nvidia-smi output");
    }

    Ok((
        parts[0].parse().unwrap_or(0),
        parts[1].parse().unwrap_or(0.0),
        parts[2].parse().unwrap_or(0),
        parts[3].parse().unwrap_or(0),
        parts[4].parse::<f32>().unwrap_or(0.0) as u32,
        parts[5].parse::<f32>().unwrap_or(0.0) as u32,
    ))
}

/// Read macOS system temp and power from AppleSmartBattery via ioreg.
/// Returns (temperature_celsius, power_watts).
#[cfg(target_os = "macos")]
fn collect_macos_metrics() -> (Option<u32>, Option<u32>) {
    use std::process::Command;

    let output = match Command::new("ioreg").args(["-r", "-n", "AppleSmartBattery"]).output() {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
        _ => return (None, None),
    };

    // Temperature: stored as centi-Kelvin (e.g. 3055 = 305.5K = 32.4°C)
    let temp = output
        .lines()
        .find(|l| l.contains("\"Temperature\""))
        .and_then(|l| l.split('=').nth(1))
        .and_then(|v| v.trim().parse::<u32>().ok())
        .map(|ck| ((ck as f64 / 10.0) - 273.15) as u32);

    // Power: voltage × current
    let voltage_mv = output
        .lines()
        .find(|l| l.contains("\"AppleRawBatteryVoltage\""))
        .and_then(|l| l.split('=').nth(1))
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(0);

    let power = output
        .lines()
        .find(|l| l.contains("\"InstantAmperage\""))
        .and_then(|l| l.split('=').nth(1))
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(|raw| {
            let current_ma = if raw > (1u64 << 63) {
                (raw.wrapping_neg()) as u64 // absolute value of negative
            } else {
                raw
            };
            ((voltage_mv * current_ma) / 1_000_000) as u32
        });

    (temp, power)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_returns_hardware() {
        let hw = detect();
        assert!(!hw.cpu.brand.is_empty());
        assert!(hw.cpu.cores > 0);
        assert!(hw.memory.total_gb > 0.0);
        assert!(!hw.gpus.is_empty());
        assert!(!hw.os.name.is_empty());
        assert!(!hw.os.arch.is_empty());
    }

    #[test]
    fn test_gpu_backend_display() {
        assert_eq!(format!("{}", GpuBackend::Cuda), "CUDA");
        assert_eq!(format!("{}", GpuBackend::Metal), "Metal");
        assert_eq!(format!("{}", GpuBackend::Cpu), "CPU");
    }

    #[test]
    fn test_hardware_serialization() {
        let hw = detect();
        let json = serde_json::to_string(&hw).unwrap();
        let parsed: HardwareInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.cpu.brand, hw.cpu.brand);
        assert_eq!(parsed.cpu.cores, hw.cpu.cores);
    }

    #[test]
    fn test_live_metrics_default() {
        let metrics = LiveMetrics::default();
        assert_eq!(metrics.cpu_usage, 0.0);
        assert_eq!(metrics.memory_used_gb, 0.0);
    }

    #[test]
    fn test_collect_live_metrics() {
        let mut sys = System::new_all();
        let metrics = collect_live_metrics(&mut sys);
        assert!(metrics.memory_total_gb > 0.0);
    }
}
