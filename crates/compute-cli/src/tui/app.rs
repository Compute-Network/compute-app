use anyhow::Result;
use tracing::warn;

use compute_daemon::config::Config;
use compute_daemon::hardware;
use compute_network::supabase::{NodeRow, SupabaseClient};

use super::dashboard::Dashboard;
use super::onboarding::{OnboardingResult, OnboardingScreen};
use super::splash::SplashScreen;

/// Initialize the terminal and run the TUI application.
/// Flow: check config → onboarding (if needed) → splash → dashboard
pub fn run_splash_then_dashboard() -> Result<()> {
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

    let mut terminal = ratatui::init();
    let mut splash = SplashScreen::new(&hw);
    let _ = splash.run(&mut terminal);
    ratatui::restore();
    Ok(())
}

/// Run dashboard only (skip splash).
pub fn run_dashboard_only() -> Result<()> {
    let hw = hardware::detect();

    let mut terminal = ratatui::init();
    let mut dashboard = Dashboard::new(hw);
    let result = dashboard.run(&mut terminal);
    ratatui::restore();
    result
}

fn run_inner(
    terminal: &mut ratatui::DefaultTerminal,
    hw: hardware::HardwareInfo,
    config: &mut Config,
) -> Result<()> {
    // Check if wallet is configured — if not, show onboarding
    if config.wallet.public_address.is_empty() {
        let mut onboarding = OnboardingScreen::new();
        match onboarding.run(terminal)? {
            OnboardingResult::WalletSet(address) => {
                config.wallet.public_address = address;
                compute_daemon::config::ensure_dirs()?;
                config.save()?;

                // Register with Supabase in background
                register_node_async(config, &hw);
            }
            OnboardingResult::Skipped => {
                // Continue without wallet — user can set it later
            }
            OnboardingResult::Quit => return Ok(()),
        }
    }

    // Start daemon runtime in background DURING splash so there's no lag on transition
    let daemon_config = config.clone();
    let runtime =
        compute_daemon::runtime::DaemonRuntime::with_hardware(daemon_config, hw.clone());
    let state_rx = runtime.state_receiver();

    let daemon_handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime");

        rt.block_on(async {
            if let Err(e) = runtime.run().await {
                tracing::error!("Daemon error: {e}");
            }
        });
    });

    // Show splash (daemon boots in parallel)
    let mut splash = SplashScreen::new(&hw);
    let continue_to_dashboard = splash.run(terminal)?;

    if continue_to_dashboard {
        let mut dashboard = Dashboard::with_daemon_state(hw, state_rx);
        dashboard.run(terminal)?;
    }

    // Clean shutdown: kill any llama-server we started
    kill_llama_server();

    drop(daemon_handle);

    Ok(())
}

/// Kill any llama-server processes we spawned on port 8090.
fn kill_llama_server() {
    let _ = std::process::Command::new("pkill")
        .args(["-f", "llama-server.*--port 8090"])
        .status();
}

/// Register the node with Supabase. Fires and forgets — non-blocking.
fn register_node_async(config: &Config, hw: &hardware::HardwareInfo) {
    let wallet = config.wallet.public_address.clone();
    let node_name = config.node.name.clone();
    let region = config.network.region.clone();
    let version = env!("CARGO_PKG_VERSION").to_string();

    let gpu = hw.gpus.first();
    let gpu_model = gpu.map(|g| g.name.clone());
    let gpu_vram_mb = gpu.map(|g| g.vram_mb as i64);
    let gpu_backend = gpu.map(|g| match g.backend {
        hardware::GpuBackend::Cuda => "cuda".to_string(),
        hardware::GpuBackend::Metal => "metal".to_string(),
        hardware::GpuBackend::Cpu => "cpu".to_string(),
    });
    let tflops = gpu.map(|g| compute_daemon::benchmark::estimate_tflops(&g.name, g.vram_mb));
    let cpu_model = Some(hw.cpu.brand.clone());
    let cpu_cores = Some(hw.cpu.cores as i32);
    let memory_mb = Some((hw.memory.total_gb * 1024.0) as i64);
    let os_str = Some(format!("{} {}", hw.os.name, hw.os.version));

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build();

        if let Ok(rt) = rt {
            rt.block_on(async {
                let client = SupabaseClient::new();
                let node = NodeRow {
                    id: None,
                    wallet_address: wallet,
                    node_name: Some(node_name),
                    status: Some("online".into()),
                    gpu_model,
                    gpu_vram_mb,
                    gpu_backend,
                    cpu_model,
                    cpu_cores,
                    memory_mb,
                    os: os_str,
                    app_version: Some(version),
                    region: Some(region),
                    tflops_fp16: tflops,
                };

                match client.register_node(&node).await {
                    Ok(id) => {
                        tracing::info!("Node registered with Supabase: {id}");
                        // Save node_id to config
                        if let Ok(mut cfg) = Config::load() {
                            cfg.wallet.node_id = id;
                            let _ = cfg.save();
                        }
                    }
                    Err(e) => {
                        warn!("Failed to register node with Supabase: {e}");
                    }
                }
            });
        }
    });
}
