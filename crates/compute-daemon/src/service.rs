use anyhow::Result;
use std::path::PathBuf;

use crate::config;

/// Install compute as a system service (launchd on macOS, systemd on Linux).
pub fn install_service() -> Result<()> {
    #[cfg(target_os = "macos")]
    install_launchd()?;

    #[cfg(target_os = "linux")]
    install_systemd()?;

    #[cfg(target_os = "windows")]
    {
        println!("Windows service installation not yet supported.");
        println!("Run `compute start` manually or add to Task Scheduler.");
    }

    Ok(())
}

/// Uninstall the system service.
pub fn uninstall_service() -> Result<()> {
    #[cfg(target_os = "macos")]
    uninstall_launchd()?;

    #[cfg(target_os = "linux")]
    uninstall_systemd()?;

    Ok(())
}

/// Check if the service is installed.
pub fn is_service_installed() -> bool {
    #[cfg(target_os = "macos")]
    {
        launchd_plist_path().exists()
    }

    #[cfg(target_os = "linux")]
    {
        systemd_unit_path().exists()
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    false
}

// ---- macOS launchd ----

#[cfg(target_os = "macos")]
fn launchd_plist_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("~"))
        .join("Library/LaunchAgents/sh.computenetwork.compute.plist")
}

#[cfg(target_os = "macos")]
fn install_launchd() -> Result<()> {
    let binary =
        std::env::current_exe().unwrap_or_else(|_| PathBuf::from("/usr/local/bin/compute"));

    let logs_dir = config::logs_dir()?;
    std::fs::create_dir_all(&logs_dir)?;

    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>sh.computenetwork.compute</string>
    <key>ProgramArguments</key>
    <array>
        <string>{binary}</string>
        <string>start</string>
        <string>--daemon</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{stdout}</string>
    <key>StandardErrorPath</key>
    <string>{stderr}</string>
    <key>ProcessType</key>
    <string>Background</string>
</dict>
</plist>"#,
        binary = binary.display(),
        stdout = logs_dir.join("compute-stdout.log").display(),
        stderr = logs_dir.join("compute-stderr.log").display(),
    );

    let plist_path = launchd_plist_path();
    if let Some(parent) = plist_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(&plist_path, plist)?;

    // Load the service
    let output =
        std::process::Command::new("launchctl").args(["load", "-w"]).arg(&plist_path).output()?;

    if output.status.success() {
        println!("Service installed and started.");
        println!("Compute will now start automatically on login.");
        println!("Plist: {}", plist_path.display());
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        println!("Warning: launchctl load returned: {stderr}");
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn uninstall_launchd() -> Result<()> {
    let plist_path = launchd_plist_path();

    if plist_path.exists() {
        let _ = std::process::Command::new("launchctl")
            .args(["unload", "-w"])
            .arg(&plist_path)
            .output();

        std::fs::remove_file(&plist_path)?;
        println!("Service uninstalled.");
    } else {
        println!("No service installed.");
    }

    Ok(())
}

// ---- Linux systemd ----

#[cfg(target_os = "linux")]
fn systemd_unit_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("~"))
        .join(".config/systemd/user/compute.service")
}

#[cfg(target_os = "linux")]
fn install_systemd() -> Result<()> {
    let binary =
        std::env::current_exe().unwrap_or_else(|_| PathBuf::from("/usr/local/bin/compute"));

    let unit = format!(
        r#"[Unit]
Description=Compute - Decentralized GPU Infrastructure
After=network.target

[Service]
Type=simple
ExecStart={binary} start --daemon
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"#,
        binary = binary.display(),
    );

    let unit_path = systemd_unit_path();
    if let Some(parent) = unit_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(&unit_path, unit)?;

    // Reload and enable
    let _ = std::process::Command::new("systemctl").args(["--user", "daemon-reload"]).output();

    let output = std::process::Command::new("systemctl")
        .args(["--user", "enable", "--now", "compute.service"])
        .output()?;

    if output.status.success() {
        println!("Service installed and started.");
        println!("Compute will now start automatically on login.");
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        println!("Warning: systemctl returned: {stderr}");
    }

    Ok(())
}

#[cfg(target_os = "linux")]
fn uninstall_systemd() -> Result<()> {
    let unit_path = systemd_unit_path();

    if unit_path.exists() {
        let _ = std::process::Command::new("systemctl")
            .args(["--user", "disable", "--now", "compute.service"])
            .output();

        std::fs::remove_file(&unit_path)?;

        let _ = std::process::Command::new("systemctl").args(["--user", "daemon-reload"]).output();

        println!("Service uninstalled.");
    } else {
        println!("No service installed.");
    }

    Ok(())
}
