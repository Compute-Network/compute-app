use crate::config;
use anyhow::{Context, Result};

/// Check if the daemon is currently running by checking the PID file.
pub fn is_running() -> bool {
    let pid = match read_pid() {
        Some(pid) => pid,
        None => return false,
    };

    // Check if process with this PID exists
    is_process_alive(pid)
}

/// Read the PID from the PID file.
pub fn read_pid() -> Option<u32> {
    let path = config::pid_file_path().ok()?;
    let contents = std::fs::read_to_string(path).ok()?;
    contents.trim().parse().ok()
}

/// Write the current process PID to the PID file.
pub fn write_pid() -> Result<()> {
    let path = config::pid_file_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, std::process::id().to_string())
        .with_context(|| "Failed to write PID file")?;
    Ok(())
}

/// Remove the PID file.
pub fn remove_pid() -> Result<()> {
    let path = config::pid_file_path()?;
    if path.exists() {
        std::fs::remove_file(&path)?;
    }
    Ok(())
}

/// Stop the running daemon by sending a signal.
pub fn stop_daemon() -> Result<()> {
    let pid = read_pid().ok_or_else(|| anyhow::anyhow!("Daemon is not running (no PID file)"))?;

    if !is_process_alive(pid) {
        remove_pid()?;
        anyhow::bail!(
            "Daemon PID file exists but process {} is not running (stale PID file removed)",
            pid
        );
    }

    // Send SIGTERM on Unix, TerminateProcess on Windows
    kill_process(pid)?;
    remove_pid()?;
    Ok(())
}

/// Get the uptime of the daemon by checking PID file modification time.
pub fn uptime() -> Option<std::time::Duration> {
    let path = config::pid_file_path().ok()?;
    let metadata = std::fs::metadata(path).ok()?;
    let created = metadata.modified().ok()?;
    std::time::SystemTime::now().duration_since(created).ok()
}

#[cfg(unix)]
fn is_process_alive(pid: u32) -> bool {
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

#[cfg(windows)]
fn is_process_alive(pid: u32) -> bool {
    use std::process::Command;
    Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/NH"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
        .unwrap_or(false)
}

#[cfg(unix)]
fn kill_process(pid: u32) -> Result<()> {
    let ret = unsafe { libc::kill(pid as i32, libc::SIGTERM) };
    if ret != 0 {
        anyhow::bail!(
            "Failed to send SIGTERM to process {}: {}",
            pid,
            std::io::Error::last_os_error()
        );
    }
    Ok(())
}

#[cfg(windows)]
fn kill_process(pid: u32) -> Result<()> {
    use std::process::Command;
    let output = Command::new("taskkill").args(["/PID", &pid.to_string(), "/F"]).output()?;
    if !output.status.success() {
        anyhow::bail!("Failed to kill process {pid}");
    }
    Ok(())
}
