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

/// Error returned when a single-instance lock cannot be acquired.
pub enum InstanceLockError {
    /// Another Compute instance already holds the lock.
    AlreadyRunning { pid: Option<u32> },
    /// An IO/OS error occurred while trying to acquire the lock.
    Io(anyhow::Error),
}

/// Guard that holds the single-instance lock for the lifetime of the process.
///
/// On Unix the lock is an advisory `flock` on the PID file; the kernel releases
/// it automatically when the process exits (even on crash or SIGKILL), so there
/// is never a stale lock to clean up. Dropping the guard also clears the PID file.
pub struct InstanceGuard {
    #[cfg(unix)]
    _file: std::fs::File,
}

impl Drop for InstanceGuard {
    fn drop(&mut self) {
        let _ = remove_pid();
    }
}

/// Acquire the single-instance lock so only one Compute app/daemon runs at a time.
///
/// Returns `AlreadyRunning` (with the holder's PID, if known) when another
/// instance is live. The returned guard must be kept alive for the whole run.
#[cfg(unix)]
pub fn acquire_single_instance() -> std::result::Result<InstanceGuard, InstanceLockError> {
    use std::os::unix::io::AsRawFd;

    let path = config::pid_file_path().map_err(InstanceLockError::Io)?;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // Rust opens files with O_CLOEXEC by default, so the lock fd is released on
    // exec() — this is what lets the session-expired flow relaunch into a fresh
    // process that can re-acquire the lock cleanly.
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|e| InstanceLockError::Io(e.into()))?;

    let ret = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if ret != 0 {
        let err = std::io::Error::last_os_error();
        if matches!(err.raw_os_error(), Some(code) if code == libc::EWOULDBLOCK) {
            return Err(InstanceLockError::AlreadyRunning { pid: read_pid() });
        }
        return Err(InstanceLockError::Io(err.into()));
    }

    // We hold the lock — record our PID so `compute status`/`stop` can find us.
    let _ = write_pid();
    Ok(InstanceGuard { _file: file })
}

/// Windows fallback: best-effort PID-file check (no atomic file lock).
#[cfg(windows)]
pub fn acquire_single_instance() -> std::result::Result<InstanceGuard, InstanceLockError> {
    if is_running() {
        return Err(InstanceLockError::AlreadyRunning { pid: read_pid() });
    }
    write_pid().map_err(InstanceLockError::Io)?;
    Ok(InstanceGuard {})
}

/// Stop the running daemon by sending a signal.
pub fn stop_daemon() -> Result<()> {
    let pid = read_pid().ok_or_else(|| anyhow::anyhow!("Daemon is not running (no PID file)"))?;

    // Never signal ourselves. Since the single-instance lock records the running
    // app's own PID, and the auto-updater calls this from inside that same app to
    // stop "the daemon" before installing, an unguarded kill would SIGTERM the
    // live process mid-update — i.e. the app would kill itself on launch.
    if pid == std::process::id() {
        return Ok(());
    }

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
