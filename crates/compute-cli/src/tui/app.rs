use anyhow::Result;
use if_addrs::{IfAddr, get_if_addrs};
use std::collections::BTreeSet;
use std::net::{Ipv4Addr, UdpSocket};
use tracing::warn;

use compute_daemon::config::Config;
use compute_daemon::hardware;
use compute_network::client::{
    NodeRegistration as OrchestratorNodeRegistration, OrchestratorClient,
};

use super::dashboard::{Dashboard, DashboardAction};
use super::onboarding::{OnboardingResult, OnboardingScreen};
use super::splash::{SplashAction, SplashScreen, UpdateGateOutcome};

pub struct NodeRegistrationResult {
    pub node_id: String,
    pub node_token: Option<String>,
}

/// Initialize the terminal and run the TUI application.
/// Flow: check config → onboarding (if needed) → splash → dashboard
pub fn run_splash_then_dashboard() -> Result<()> {
    // Single-instance lock: only one Compute app/daemon may run at a time.
    // Held for the whole session; the kernel releases it if this process dies.
    let _instance = match compute_daemon::daemon::acquire_single_instance() {
        Ok(guard) => guard,
        Err(compute_daemon::daemon::InstanceLockError::AlreadyRunning { pid }) => {
            eprintln!("Compute is already running (PID: {}).", pid.unwrap_or(0));
            eprintln!("Run `compute dashboard` to view it, or `compute stop` to stop it first.");
            return Ok(());
        }
        Err(compute_daemon::daemon::InstanceLockError::Io(e)) => return Err(e),
    };

    let hw = hardware::detect();
    let mut config = Config::load()?;

    let mut terminal = ratatui::init();
    let result = run_inner(&mut terminal, hw, &mut config);
    ratatui::restore();
    result
}

/// Run splash screen only (for `compute start`).
pub fn run_splash_only() -> Result<()> {
    let hw = hardware::detect();
    let assessment = compute_daemon::benchmark::assess_node_startup(&hw);

    let mut terminal = ratatui::init();
    let mut splash = SplashScreen::new(&hw, assessment);
    match splash.run_update_gate(&mut terminal)? {
        UpdateGateOutcome::Continue => {}
        UpdateGateOutcome::Exit => {
            ratatui::restore();
            return Ok(());
        }
        UpdateGateOutcome::Restart => {
            ratatui::restore();
            return relaunch_current_command();
        }
    }
    let _ = splash.run(&mut terminal);
    ratatui::restore();
    Ok(())
}

/// Run dashboard only (skip splash).
pub fn run_dashboard_only() -> Result<()> {
    let hw = hardware::detect();

    let mut terminal = ratatui::init();
    let mut dashboard = Dashboard::new(hw);
    // View-only dashboard has no embedded daemon, so it never reports a session
    // expiry; we don't act on the returned action here.
    let result = dashboard.run(&mut terminal);
    ratatui::restore();
    result.map(|_action| ())
}

fn run_inner(
    terminal: &mut ratatui::DefaultTerminal,
    hw: hardware::HardwareInfo,
    config: &mut Config,
) -> Result<()> {
    let assessment = compute_daemon::benchmark::assess_node_startup(&hw);

    // Check if wallet is configured — if not, show onboarding
    if config.wallet.public_address.is_empty() || config.wallet.node_token.is_empty() {
        let mut onboarding = OnboardingScreen::new();
        match onboarding.run(terminal)? {
            OnboardingResult::WalletSet(address) => {
                if let Ok(updated) = Config::load() {
                    *config = updated;
                } else {
                    config.wallet.public_address = address;
                }
            }
            OnboardingResult::Skipped => {
                // Continue without wallet — user can set it later
            }
            OnboardingResult::Quit => return Ok(()),
        }
    }

    // Enable file logging so daemon output is captured
    let _ = compute_daemon::logging::init(true);

    if !config.wallet.public_address.is_empty()
        && !config.wallet.node_token.is_empty()
        && config.wallet.node_id.is_empty()
    {
        register_node_blocking(config, &hw);
    }

    let mut splash = SplashScreen::new(&hw, assessment);
    match splash.run_update_gate(terminal)? {
        UpdateGateOutcome::Continue => {}
        UpdateGateOutcome::Exit => return Ok(()),
        UpdateGateOutcome::Restart => return relaunch_current_command(),
    }

    // Start daemon runtime in background DURING splash so there's no lag on transition.
    // Wrapped in an Arc so we can signal a graceful shutdown from this thread.
    let daemon_config = config.clone();
    let runtime = std::sync::Arc::new(compute_daemon::runtime::DaemonRuntime::with_hardware(
        daemon_config,
        hw.clone(),
    ));
    let state_rx = runtime.state_receiver();

    let runtime_for_thread = runtime.clone();
    let daemon_handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime");

        rt.block_on(async {
            if let Err(e) = runtime_for_thread.run().await {
                tracing::error!("Daemon error: {e}");
            }
        });
    });

    // Show splash (daemon boots in parallel)
    let mut splash = splash.with_daemon_state(state_rx.clone());
    let mut reauth_requested = false;
    match splash.run(terminal)? {
        SplashAction::ToDashboard => {
            let mut dashboard = Dashboard::with_daemon_state(hw.clone(), state_rx);
            match dashboard.run(terminal)? {
                DashboardAction::Quit => {}
                DashboardAction::Reauth => reauth_requested = true,
            }
        }
        SplashAction::Quit => {}
        SplashAction::Reauth => reauth_requested = true,
    }

    // Graceful shutdown: signal the runtime and wait for its event loop to return
    // so its Drop handlers kill the inference children (stage nodes, oMLX) instead
    // of orphaning them to PID 1.
    runtime.shutdown();
    let _ = daemon_handle.join();
    kill_llama_server();

    if reauth_requested {
        return reauthenticate_and_relaunch(&hw);
    }

    Ok(())
}

/// Handle a "session expired" re-authentication request from the TUI: drop out of
/// the alternate screen, run the browser wallet-login flow, persist the fresh
/// token, re-register the node, and relaunch into a clean process so every
/// component picks up the new session. On Unix the relaunch `exec`s and never
/// returns; the CLOEXEC lock fd is released so the new process re-acquires it.
fn reauthenticate_and_relaunch(hw: &hardware::HardwareInfo) -> Result<()> {
    ratatui::restore();
    println!("\n  Session expired — your node session has ended.");
    println!("  Re-authenticating your wallet...\n");

    let login = match crate::wallet_login_flow() {
        Ok(login) => login,
        Err(e) => {
            eprintln!("\n  Login failed: {e}");
            eprintln!("  Run `compute wallet login`, then start Compute again.\n");
            return Ok(());
        }
    };

    let mut config = Config::load().unwrap_or_default();
    config.wallet.public_address = login.wallet_address;
    config.wallet.node_token = login.node_token;
    if let Err(e) = config.save() {
        eprintln!("  Warning: could not save config: {e}");
    }

    // Re-register so the node_id matches the new session (login may mint a new one).
    if let Ok(rt) = tokio::runtime::Builder::new_current_thread().enable_all().build() {
        rt.block_on(async {
            if let Ok(Some(result)) = register_node_orchestrator(&config, hw).await {
                config.wallet.node_id = result.node_id;
                if let Some(token) = result.node_token {
                    config.wallet.node_token = token;
                }
                let _ = config.save();
            }
        });
    }

    println!("\n  Re-authenticated. Restarting Compute...\n");
    relaunch_current_command()
}

fn relaunch_current_command() -> Result<()> {
    let exe = std::env::current_exe()?;
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = std::process::Command::new(exe).args(args).exec();
        Err(err.into())
    }

    #[cfg(not(unix))]
    {
        std::process::Command::new(exe).args(args).spawn()?;
        std::process::exit(0);
    }
}

/// Kill any llama-server processes we spawned on port 8090.
fn kill_llama_server() {
    let _ = std::process::Command::new("pkill").args(["-f", "llama-server.*--port 8090"]).status();
}

pub async fn register_node_orchestrator(
    config: &Config,
    hw: &hardware::HardwareInfo,
) -> Result<Option<NodeRegistrationResult>> {
    if config.wallet.public_address.is_empty() || config.wallet.node_token.is_empty() {
        return Ok(None);
    }

    let gpu = hw.gpus.first();
    let gpu_model = gpu.map(|g| g.name.clone());
    let gpu_vram_mb = gpu.map(|g| g.vram_mb as u64);
    let gpu_backend = gpu.map(|g| match g.backend {
        hardware::GpuBackend::Cuda => "cuda".to_string(),
        hardware::GpuBackend::Metal => "metal".to_string(),
        hardware::GpuBackend::Cpu => "cpu".to_string(),
    });
    let tflops = gpu.map(|g| compute_daemon::benchmark::estimate_tflops(&g.name, g.vram_mb));
    let assessment = compute_daemon::benchmark::assess_node_startup(hw);
    let cpu_model = Some(hw.cpu.brand.clone());
    let cpu_cores = Some(hw.cpu.cores);
    let memory_mb = Some((hw.memory.total_gb * 1024.0) as u64);
    let os_str = Some(format!("{} {}", hw.os.name, hw.os.version));
    let ip_address = advertised_host(config);

    let mut client = OrchestratorClient::new(
        &config.network.orchestrator_url,
        Some(config.wallet.node_token.clone()),
    );
    let node = OrchestratorNodeRegistration {
        wallet_address: config.wallet.public_address.clone(),
        node_name: Some(config.node.name.clone()),
        gpu_model,
        gpu_vram_mb,
        gpu_backend,
        cpu_model,
        cpu_cores,
        memory_mb,
        os: os_str,
        app_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        region: Some(config.network.region.clone()),
        tflops_fp16: tflops,
        listen_port: Some(9090),
        ip_address,
        pipeline_capable: Some(assessment.split_capable),
        memory_bandwidth_gbps: Some(assessment.estimated_memory_bandwidth_gbps),
    };

    let node_id = client.register(&node).await?;
    Ok(Some(NodeRegistrationResult { node_id, node_token: client.node_token().map(str::to_owned) }))
}

fn detect_advertise_ip() -> Option<String> {
    let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).ok()?;
    socket.connect((Ipv4Addr::new(1, 1, 1, 1), 80)).ok()?;
    let addr = socket.local_addr().ok()?;
    match addr.ip() {
        std::net::IpAddr::V4(ip) if !ip.is_loopback() && !ip.is_unspecified() => {
            Some(ip.to_string())
        }
        _ => None,
    }
}

fn collect_advertise_ips() -> Vec<String> {
    let mut hosts = BTreeSet::new();

    if let Ok(ifaces) = get_if_addrs() {
        for iface in ifaces {
            let ip = match iface.addr {
                IfAddr::V4(v4) => v4.ip,
                IfAddr::V6(_) => continue,
            };
            if ip.is_loopback() || ip.is_unspecified() || ip.is_link_local() {
                continue;
            }
            if !ip.is_private() {
                continue;
            }
            hosts.insert(ip.to_string());
        }
    }

    if hosts.is_empty() {
        if let Some(ip) = detect_advertise_ip() {
            hosts.insert(ip);
        }
    }

    hosts.into_iter().collect()
}

fn advertised_host(config: &Config) -> Option<String> {
    let configured = config.network.advertise_host.trim();
    if !configured.is_empty() {
        return Some(configured.to_string());
    }
    collect_advertise_ips().into_iter().next()
}

fn register_node_blocking(config: &mut Config, hw: &hardware::HardwareInfo) {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build();
    let Ok(rt) = rt else {
        warn!("Failed to create runtime for node registration");
        return;
    };

    match rt.block_on(register_node_orchestrator(config, hw)) {
        Ok(Some(result)) => {
            tracing::info!("Node registered with orchestrator: {}", result.node_id);
            config.wallet.node_id = result.node_id;
            if let Some(node_token) = result.node_token {
                config.wallet.node_token = node_token;
            }
            if let Err(e) = config.save() {
                warn!("Failed to save node registration to config: {e}");
            }
        }
        Ok(None) => {}
        Err(e) => {
            warn!("Failed to register node with orchestrator: {e}");
        }
    }
}
