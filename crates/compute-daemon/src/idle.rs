use std::time::Duration;
use sysinfo::System;

/// Current idle state of the machine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IdleState {
    /// No input for threshold minutes + low CPU/GPU. Use maximum resources.
    Idle,
    /// Light use (browsing, docs). Use available GPU, throttle CPU.
    LightUse,
    /// Heavy use (gaming, rendering, compiling). Pause all workloads.
    HeavyUse,
    /// User manually paused.
    Paused,
}

impl std::fmt::Display for IdleState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IdleState::Idle => write!(f, "Idle"),
            IdleState::LightUse => write!(f, "Light Use"),
            IdleState::HeavyUse => write!(f, "Heavy Use"),
            IdleState::Paused => write!(f, "Paused"),
        }
    }
}

/// Idle detection system that monitors user activity and system load.
pub struct IdleDetector {
    idle_threshold: Duration,
    paused: bool,
    sys: System,
}

impl IdleDetector {
    pub fn new(idle_threshold_minutes: u32) -> Self {
        Self {
            idle_threshold: Duration::from_secs(idle_threshold_minutes as u64 * 60),
            paused: false,
            sys: System::new_all(),
        }
    }

    /// Set manual pause state.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Detect the current idle state.
    pub fn detect(&mut self) -> IdleState {
        if self.paused {
            return IdleState::Paused;
        }

        self.sys.refresh_all();

        let cpu_usage = self.sys.global_cpu_usage();
        let user_idle_time = get_user_idle_time();
        let on_battery = is_on_battery();

        // If on battery, treat as heavy use (don't run workloads)
        if on_battery {
            return IdleState::HeavyUse;
        }

        // Check user input idle time
        let is_user_idle = user_idle_time.map(|d| d >= self.idle_threshold).unwrap_or(false);

        if is_user_idle && cpu_usage < 30.0 {
            IdleState::Idle
        } else if cpu_usage > 80.0 {
            IdleState::HeavyUse
        } else {
            IdleState::LightUse
        }
    }

    /// Get resource limits based on current idle state.
    pub fn resource_limits(&mut self, max_gpu: u8, max_cpu: u8) -> ResourceLimits {
        match self.detect() {
            IdleState::Idle => {
                ResourceLimits { gpu_percent: max_gpu, cpu_percent: max_cpu, should_run: true }
            }
            IdleState::LightUse => ResourceLimits {
                gpu_percent: max_gpu,
                cpu_percent: (max_cpu as f32 * 0.5) as u8,
                should_run: true,
            },
            IdleState::HeavyUse | IdleState::Paused => {
                ResourceLimits { gpu_percent: 0, cpu_percent: 0, should_run: false }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResourceLimits {
    pub gpu_percent: u8,
    pub cpu_percent: u8,
    pub should_run: bool,
}

/// Get the time since last user input (keyboard/mouse).
#[cfg(target_os = "macos")]
fn get_user_idle_time() -> Option<Duration> {
    use std::process::Command;

    // Use ioreg to get HIDIdleTime (nanoseconds)
    let output = Command::new("ioreg").args(["-c", "IOHIDSystem", "-d", "4"]).output().ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.contains("HIDIdleTime") {
            // Extract the number
            let num_str: String = line.chars().filter(|c| c.is_ascii_digit()).collect();
            if let Ok(nanos) = num_str.parse::<u64>() {
                return Some(Duration::from_nanos(nanos));
            }
        }
    }

    None
}

#[cfg(target_os = "linux")]
fn get_user_idle_time() -> Option<Duration> {
    use std::process::Command;

    // Try xprintidle (X11)
    let output = Command::new("xprintidle").output().ok()?;
    if output.status.success() {
        let millis: u64 = String::from_utf8_lossy(&output.stdout).trim().parse().ok()?;
        return Some(Duration::from_millis(millis));
    }

    None
}

#[cfg(target_os = "windows")]
fn get_user_idle_time() -> Option<Duration> {
    // Windows: would use GetLastInputInfo via winapi
    // Placeholder until windows-sys dependency is added
    None
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
fn get_user_idle_time() -> Option<Duration> {
    None
}

/// Check if the machine is running on battery power.
#[cfg(target_os = "macos")]
fn is_on_battery() -> bool {
    use std::process::Command;

    let output = Command::new("pmset").args(["-g", "batt"]).output().ok();

    match output {
        Some(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            stdout.contains("Battery Power")
        }
        None => false,
    }
}

#[cfg(target_os = "linux")]
fn is_on_battery() -> bool {
    std::fs::read_to_string("/sys/class/power_supply/BAT0/status")
        .map(|s| s.trim() == "Discharging")
        .unwrap_or(false)
}

#[cfg(target_os = "windows")]
fn is_on_battery() -> bool {
    // Placeholder
    false
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
fn is_on_battery() -> bool {
    false
}
