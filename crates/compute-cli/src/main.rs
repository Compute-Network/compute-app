mod cli;
mod tui;

use anyhow::Result;
use clap::Parser;
use std::io::Seek;

use cli::{Cli, Commands, ConfigAction, ServiceAction, WalletAction};
use compute_daemon::config::{self, Config};
use compute_daemon::daemon;
use compute_daemon::hardware;
use compute_network::supabase::SupabaseClient;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Start) => cmd_start()?,
        Some(Commands::Stop) => cmd_stop()?,
        Some(Commands::Status) => cmd_status()?,
        Some(Commands::Dashboard) => cmd_dashboard()?,
        Some(Commands::Logs { lines, follow }) => cmd_logs(lines, follow)?,
        Some(Commands::Init) => cmd_init()?,
        Some(Commands::Config { action }) => cmd_config(action)?,
        Some(Commands::Wallet { action }) => cmd_wallet(action)?,
        Some(Commands::Earnings) => cmd_earnings()?,
        Some(Commands::Benchmark) => cmd_benchmark()?,
        Some(Commands::Hardware) => cmd_hardware()?,
        Some(Commands::Pipeline) => cmd_pipeline()?,
        Some(Commands::Nodes { all, limit }) => cmd_nodes(all, limit)?,
        Some(Commands::Update) => cmd_update()?,
        Some(Commands::Uninstall) => cmd_uninstall()?,
        Some(Commands::Doctor) => cmd_doctor()?,
        Some(Commands::Service { action }) => cmd_service(action)?,
        None => {
            // No subcommand — show splash then dashboard
            tui::app::run_splash_then_dashboard()?;
        }
    }

    Ok(())
}

fn cmd_start() -> Result<()> {
    if daemon::is_running() {
        println!("Daemon is already running (PID: {})", daemon::read_pid().unwrap_or(0));
        return Ok(());
    }

    // Check if wallet is configured — prompt if not
    let mut config_check = Config::load()?;
    if config_check.wallet.public_address.is_empty() {
        let mut terminal = ratatui::init();
        let mut onboarding = tui::onboarding::OnboardingScreen::new();
        let result = onboarding.run(&mut terminal);
        ratatui::restore();

        match result? {
            tui::onboarding::OnboardingResult::WalletSet(address) => {
                config_check.wallet.public_address = address;
                config::ensure_dirs()?;
                config_check.save()?;
            }
            tui::onboarding::OnboardingResult::Skipped => {}
            tui::onboarding::OnboardingResult::Quit => return Ok(()),
        }
    }

    // Show splash screen
    tui::app::run_splash_only()?;

    // Set up logging and dirs
    config::ensure_dirs()?;
    compute_daemon::logging::init(true)?;
    daemon::write_pid()?;

    let config = Config::load()?;
    let runtime = compute_daemon::runtime::DaemonRuntime::new(config.clone());

    println!("\n  Daemon started (PID: {})", std::process::id());
    println!("  Run `compute dashboard` to view live stats");
    println!("  Run `compute stop` to stop the daemon\n");

    // Run the daemon event loop
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        // Register with Supabase on startup
        if !config.wallet.public_address.is_empty() {
            register_node_supabase(&config).await;
        }

        if let Err(e) = runtime.run().await {
            eprintln!("Daemon error: {e}");
        }
    });

    daemon::remove_pid()?;
    Ok(())
}

async fn register_node_supabase(config: &Config) {
    use compute_network::supabase::{NodeRow, SupabaseClient};

    let hw = hardware::detect();
    let gpu = hw.gpus.first();

    let tflops = gpu.map(|g| compute_daemon::benchmark::estimate_tflops(&g.name, g.vram_mb));

    let node = NodeRow {
        id: None,
        wallet_address: config.wallet.public_address.clone(),
        node_name: Some(config.node.name.clone()),
        status: Some("online".into()),
        gpu_model: gpu.map(|g| g.name.clone()),
        gpu_vram_mb: gpu.map(|g| g.vram_mb as i64),
        gpu_backend: gpu.map(|g| match g.backend {
            hardware::GpuBackend::Cuda => "cuda".to_string(),
            hardware::GpuBackend::Metal => "metal".to_string(),
            hardware::GpuBackend::Cpu => "cpu".to_string(),
        }),
        cpu_model: Some(hw.cpu.brand.clone()),
        cpu_cores: Some(hw.cpu.cores as i32),
        memory_mb: Some((hw.memory.total_gb * 1024.0) as i64),
        os: Some(format!("{} {}", hw.os.name, hw.os.version)),
        app_version: Some(VERSION.to_string()),
        region: Some(config.network.region.clone()),
        tflops_fp16: tflops,
    };

    let client = SupabaseClient::new();
    match client.register_node(&node).await {
        Ok(id) => {
            println!("  Registered with network (node: {id})");
            // Persist node_id to config
            let mut updated_config = config.clone();
            updated_config.wallet.node_id = id;
            if let Err(e) = updated_config.save() {
                eprintln!("  Warning: Could not save node_id to config: {e}");
            }
        }
        Err(e) => eprintln!("  Warning: Could not register with network: {e}"),
    }
}

fn cmd_stop() -> Result<()> {
    match daemon::stop_daemon() {
        Ok(()) => println!("Daemon stopped."),
        Err(e) => println!("Error: {e}"),
    }
    Ok(())
}

fn cmd_status() -> Result<()> {
    let config = Config::load()?;

    if daemon::is_running() {
        let pid = daemon::read_pid().unwrap_or(0);
        let uptime =
            daemon::uptime().map(format_duration_short).unwrap_or_else(|| "unknown".into());

        println!("● ACTIVE  PID: {pid}  Uptime: {uptime}");
    } else {
        println!("○ STOPPED  Run `compute start` to begin");
    }

    // Show wallet/node info
    if !config.wallet.public_address.is_empty() {
        println!("  Wallet:  {}", config.wallet.public_address);
        if !config.wallet.node_id.is_empty() {
            println!("  Node ID: {}", config.wallet.node_id);
        }
    } else {
        println!("  Wallet:  not configured (run `compute init`)");
    }

    Ok(())
}

fn cmd_dashboard() -> Result<()> {
    tui::app::run_dashboard_only()
}

fn cmd_logs(lines: usize, follow: bool) -> Result<()> {
    let log_path = config::logs_dir()?.join("compute.log");

    if !log_path.exists() {
        println!("No log file found at {}", log_path.display());
        println!("Start the daemon first: `compute start`");
        return Ok(());
    }

    let contents = std::fs::read_to_string(&log_path)?;
    let all_lines: Vec<&str> = contents.lines().collect();
    let start = all_lines.len().saturating_sub(lines);

    for line in &all_lines[start..] {
        println!("{line}");
    }

    if follow {
        use std::io::Read;

        // Open the file and seek to end, then poll for new data
        let mut file = std::fs::File::open(&log_path)?;
        file.seek(std::io::SeekFrom::End(0))?;

        let mut buf = String::new();
        loop {
            buf.clear();
            match file.read_to_string(&mut buf) {
                Ok(0) => {
                    // No new data, sleep briefly
                    std::thread::sleep(std::time::Duration::from_millis(250));
                }
                Ok(_) => {
                    print!("{buf}");
                }
                Err(e) => {
                    eprintln!("Error reading log: {e}");
                    break;
                }
            }
        }
    }

    Ok(())
}

fn cmd_init() -> Result<()> {
    println!();
    println!("   ██████  ██████  ███    ███ ██████  ██    ██ ████████ ███████");
    println!("  ██      ██    ██ ████  ████ ██   ██ ██    ██    ██    ██");
    println!("  ██      ██    ██ ██ ████ ██ ██████  ██    ██    ██    █████");
    println!("  ██      ██    ██ ██  ██  ██ ██      ██    ██    ██    ██");
    println!("   ██████  ██████  ██      ██ ██       ██████     ██    ███████");
    println!();
    println!("  Welcome to Compute — Decentralized GPU Infrastructure");
    println!("  v{VERSION}");
    println!();

    // Detect hardware
    println!("  Detecting hardware...");
    let hw = hardware::detect();

    println!("  CPU:    {} ({} cores)", hw.cpu.brand, hw.cpu.cores);
    println!("  Memory: {:.1} GB", hw.memory.total_gb);

    for gpu in &hw.gpus {
        let vram = if gpu.vram_mb >= 1024 {
            format!("{}GB", gpu.vram_mb / 1024)
        } else {
            format!("{}MB", gpu.vram_mb)
        };
        println!("  GPU:    {} ({}, {})", gpu.name, vram, gpu.backend);
    }

    println!("  OS:     {} {} ({})", hw.os.name, hw.os.version, hw.os.arch);
    println!(
        "  Docker: {}",
        if hw.docker.available {
            format!("v{}", hw.docker.version.as_deref().unwrap_or("unknown"))
        } else {
            "not found".into()
        }
    );
    println!(
        "  Disk:   {:.0} GB available / {:.0} GB total",
        hw.disk.available_gb, hw.disk.total_gb
    );
    println!();

    // Create or load config
    let mut config = Config::default();

    // Ask for wallet address
    println!("  Enter your Solana wallet address (or press Enter to skip):");
    print!("  > ");
    use std::io::Write;
    std::io::stdout().flush()?;

    let mut wallet_input = String::new();
    std::io::stdin().read_line(&mut wallet_input)?;
    let wallet_input = wallet_input.trim();

    if !wallet_input.is_empty() {
        if compute_solana::is_valid_address(wallet_input) {
            config.wallet.public_address = wallet_input.to_string();
            println!("  ✓ Wallet address set");
        } else {
            println!(
                "  ⚠ Invalid address format — you can set it later with `compute wallet set <address>`"
            );
        }
    } else {
        println!("  Skipped — set later with `compute wallet set <address>`");
    }

    // Ask for node name
    println!();
    println!("  Enter a name for this node (or press Enter for '{}'):", config.node.name);
    print!("  > ");
    std::io::stdout().flush()?;

    let mut name_input = String::new();
    std::io::stdin().read_line(&mut name_input)?;
    let name_input = name_input.trim();

    if !name_input.is_empty() {
        config.node.name = name_input.to_string();
    }

    // Save config
    config::ensure_dirs()?;
    config.save()?;

    let config_path = config::config_file_path()?;
    println!();
    println!("  ✓ Config saved to {}", config_path.display());

    // Register with Supabase if wallet was set
    if !config.wallet.public_address.is_empty() {
        print!("  Registering with network... ");
        std::io::Write::flush(&mut std::io::stdout())?;
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(register_node_supabase(&config));
    }

    println!();
    println!("  Get started:");
    println!("    compute start       Start contributing compute");
    println!("    compute dashboard   View live stats");
    println!("    compute hardware    View hardware details");
    println!("    compute config show View configuration");
    println!();

    Ok(())
}

fn cmd_config(action: ConfigAction) -> Result<()> {
    match action {
        ConfigAction::Show => {
            let config = Config::load()?;
            let toml_str = toml::to_string_pretty(&config)?;
            println!("{toml_str}");
        }
        ConfigAction::Get { key } => {
            let config = Config::load()?;
            match config.get(&key) {
                Some(value) => println!("{value}"),
                None => println!("Unknown key: {key}"),
            }
        }
        ConfigAction::Set { key, value } => {
            let mut config = Config::load()?;
            config.set(&key, &value)?;
            config.save()?;
            println!("Set {key} = {value}");
        }
    }
    Ok(())
}

fn cmd_wallet(action: Option<WalletAction>) -> Result<()> {
    let config = Config::load()?;

    match action {
        Some(WalletAction::Set { address }) => {
            if !compute_solana::is_valid_address(&address) {
                println!("Invalid Solana address format");
                return Ok(());
            }
            let mut config = config;
            config.wallet.public_address = address.clone();
            config.save()?;
            println!("Wallet address set to: {address}");

            // Register/update with Supabase
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(register_node_supabase(&config));
        }
        None => {
            if config.wallet.public_address.is_empty() {
                println!("No wallet address configured");
                println!("Set one with: compute wallet set <address>");
            } else {
                println!("Address: {}", config.wallet.public_address);
                println!("(Balance checking not yet implemented)");
            }
        }
    }

    Ok(())
}

fn cmd_earnings() -> Result<()> {
    let config = Config::load()?;
    let wallet = &config.wallet.public_address;

    println!();
    println!("  EARNINGS");
    println!("  ────────────────────────────────");

    if wallet.is_empty() {
        println!("  No wallet configured. Run `compute init` first.");
        println!();
        return Ok(());
    }

    let rt = tokio::runtime::Runtime::new()?;
    let earnings = rt.block_on(async {
        let client = SupabaseClient::new();
        client.get_earnings(wallet).await
    });

    match earnings {
        Ok(e) => {
            println!("  Today       {:.2} $COMPUTE", e.today);
            println!("  This Week   {:.2} $COMPUTE", e.this_week);
            println!("  This Month  {:.2} $COMPUTE", e.this_month);
            println!("  All Time    {:.2} $COMPUTE", e.all_time);
            println!("  Pending     {:.2} $COMPUTE", e.pending);
            println!();
            println!("  Claim at: https://computenetwork.sh/dashboard/claim");
        }
        Err(e) => {
            println!("  Could not fetch earnings: {e}");
        }
    }
    println!();
    Ok(())
}

fn cmd_benchmark() -> Result<()> {
    use compute_daemon::benchmark;

    println!("\n  COMPUTE BENCHMARK\n");

    let hw = hardware::detect();

    println!("  HARDWARE");
    println!("  ────────────────────────────────");
    println!("  CPU:    {} ({} cores / {} threads)", hw.cpu.brand, hw.cpu.cores, hw.cpu.threads);
    println!(
        "  Memory: {:.1} GB total, {:.1} GB available",
        hw.memory.total_gb, hw.memory.available_gb
    );

    for gpu in &hw.gpus {
        let vram = if gpu.vram_mb >= 1024 {
            format!("{}GB", gpu.vram_mb / 1024)
        } else {
            format!("{}MB", gpu.vram_mb)
        };
        println!("  GPU:    {} ({}, {})", gpu.name, vram, gpu.backend);
    }

    println!(
        "  Disk:   {:.0} GB available / {:.0} GB total",
        hw.disk.available_gb, hw.disk.total_gb
    );
    println!(
        "  Docker: {}",
        if hw.docker.available {
            format!("v{}", hw.docker.version.as_deref().unwrap_or("unknown"))
        } else {
            "not available".into()
        }
    );

    // Run benchmarks
    println!();
    println!("  BENCHMARK");
    println!("  ────────────────────────────────");

    print!("  CPU (single)... ");
    std::io::Write::flush(&mut std::io::stdout())?;
    let (cpu_single, cpu_multi) = benchmark::bench_cpu();
    println!("{:.1} Mops", cpu_single);
    println!("  CPU (multi)...  {:.1} Mops ({:.1}x)", cpu_multi, cpu_multi / cpu_single);

    print!("  Memory...       ");
    std::io::Write::flush(&mut std::io::stdout())?;
    let mem_bw = benchmark::bench_memory();
    println!("{:.1} GB/s", mem_bw);

    print!("  Disk read...    ");
    std::io::Write::flush(&mut std::io::stdout())?;
    let disk = benchmark::bench_disk();
    println!("{:.0} MB/s", disk);

    let gpu_name = hw.gpus.first().map(|g| g.name.as_str()).unwrap_or("CPU");
    let gpu_vram = hw.gpus.first().map(|g| g.vram_mb).unwrap_or(0);
    let tflops = benchmark::estimate_tflops(gpu_name, gpu_vram);
    println!("  GPU (est.)...   {:.1} TFLOPS FP16", tflops);

    print!("  Network...      ");
    std::io::Write::flush(&mut std::io::stdout())?;
    let rt = tokio::runtime::Runtime::new()?;
    let download = rt.block_on(benchmark::bench_network_download());
    match download {
        Some(mbps) => println!("{:.1} Mbps download", mbps),
        None => println!("unavailable"),
    }

    println!();
    println!("  PIPELINE ELIGIBILITY");
    println!("  ────────────────────────────────");

    let has_gpu = hw.gpus.iter().any(|g| !matches!(g.backend, hardware::GpuBackend::Cpu));
    let has_enough_ram = hw.memory.total_gb >= 8.0;

    if has_gpu && has_enough_ram && hw.docker.available {
        println!("  ✓ Eligible for pipeline participation");
        println!("  ✓ GPU compute available");
        println!("  ✓ Docker available for workload isolation");
    } else {
        if !has_gpu {
            println!("  ⚠ No GPU detected — CPU-only pipeline stage");
        }
        if !has_enough_ram {
            println!("  ⚠ Less than 8GB RAM — may limit model sizes");
        }
        if !hw.docker.available {
            println!("  ⚠ Docker not found — required for workload containers");
        }
    }
    println!();

    Ok(())
}

fn cmd_hardware() -> Result<()> {
    let hw = hardware::detect();
    let json = serde_json::to_string_pretty(&hw)?;
    println!("{json}");
    Ok(())
}

fn cmd_pipeline() -> Result<()> {
    let config = Config::load()?;
    let wallet = &config.wallet.public_address;

    println!();
    println!("  PIPELINE STATUS");
    println!("  ────────────────────────────────");

    if wallet.is_empty() {
        println!("  No wallet configured. Run `compute init` first.");
        println!();
        return Ok(());
    }

    let rt = tokio::runtime::Runtime::new()?;
    let node = rt.block_on(async {
        let client = SupabaseClient::new();
        client.get_own_node(wallet).await
    });

    match node {
        Ok(Some(n)) => {
            if n.pipeline_id.is_some() {
                println!("  Status:   ● Active");
                if let (Some(s), Some(t)) = (n.pipeline_stage, n.pipeline_total_stages) {
                    println!("  Stage:    {} / {}", s, t);
                }
                if let Some(ref model) = n.model_name {
                    println!("  Model:    {model}");
                }
                println!("  Served:   {} requests", n.requests_served.unwrap_or(0));
                if let Some(tps) = n.tokens_per_second {
                    println!("  Speed:    {:.1} tok/s", tps);
                }
            } else {
                println!("  Status:   ○ Waiting for assignment");
            }
        }
        Ok(None) => {
            println!("  Node not registered. Start the daemon to register.");
        }
        Err(e) => {
            println!("  Could not fetch status: {e}");
        }
    }
    println!();

    Ok(())
}

fn cmd_nodes(all: bool, limit: usize) -> Result<()> {
    println!("\n  NETWORK NODES\n");

    let rt = tokio::runtime::Runtime::new()?;
    let nodes = rt.block_on(async {
        let client = SupabaseClient::new();
        client.list_nodes(!all, limit).await
    });

    match nodes {
        Ok(nodes) if nodes.is_empty() => {
            println!("  No {} nodes found.", if all { "" } else { "online " });
            println!("  Be the first! Run `compute start` to join the network.");
        }
        Ok(nodes) => {
            // Header
            println!(
                "  {:<12} {:<16} {:<8} {:<20} {:<8} {:<10}",
                "STATUS", "NAME", "REGION", "GPU", "TFLOPS", "UPTIME"
            );
            println!("  {}", "─".repeat(74));

            for node in &nodes {
                let status = match node.status.as_deref() {
                    Some("online") => "● online",
                    Some("idle") => "◐ idle",
                    Some("paused") => "◌ paused",
                    _ => "○ offline",
                };

                let name = node.node_name.as_deref().unwrap_or("-");
                let name_display = if name.len() > 14 { &name[..14] } else { name };

                let region = node.region.as_deref().unwrap_or("-");
                let region_display = if region.len() > 6 { &region[..6] } else { region };

                let gpu = node.gpu_model.as_deref().unwrap_or("-");
                let gpu_display = if gpu.len() > 18 { &gpu[..18] } else { gpu };

                let tflops =
                    node.tflops_fp16.map(|t| format!("{t:.1}")).unwrap_or_else(|| "-".into());

                let uptime = node
                    .uptime_seconds
                    .map(|s| format_duration_short(std::time::Duration::from_secs(s as u64)))
                    .unwrap_or_else(|| "-".into());

                println!(
                    "  {:<12} {:<16} {:<8} {:<20} {:<8} {:<10}",
                    status, name_display, region_display, gpu_display, tflops, uptime
                );
            }

            println!();
            println!("  {} nodes shown", nodes.len());
        }
        Err(e) => {
            println!("  Error fetching nodes: {e}");
            println!("  Check your internet connection.");
        }
    }

    println!();
    Ok(())
}

fn cmd_update() -> Result<()> {
    println!("\n  COMPUTE UPDATE\n");
    println!("  Current version: v{VERSION}");
    print!("  Checking for updates... ");
    std::io::Write::flush(&mut std::io::stdout())?;

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        match check_latest_release().await {
            Ok(Some((tag, download_url))) => {
                let latest = tag.trim_start_matches('v');
                if latest == VERSION {
                    println!("up to date");
                } else {
                    println!("v{latest} available");
                    println!();
                    println!("  Download: {download_url}");
                    println!();

                    // Attempt auto-update
                    print!("  Downloading... ");
                    std::io::Write::flush(&mut std::io::stdout()).ok();

                    match download_and_replace(&download_url).await {
                        Ok(()) => {
                            println!("done");
                            println!("  Updated to v{latest}");
                            println!("  Restart compute to use the new version.");
                        }
                        Err(e) => {
                            println!("failed");
                            println!("  Error: {e}");
                            println!("  Manual update: download from the URL above and replace the binary.");
                        }
                    }
                }
            }
            Ok(None) => {
                println!("no releases found");
                println!("  Check: https://github.com/Compute-Network/compute-app/releases");
            }
            Err(e) => {
                println!("failed");
                println!("  Error: {e}");
                println!("  Check: https://github.com/Compute-Network/compute-app/releases");
            }
        }
    });

    println!();
    Ok(())
}

/// Check the latest release from GitHub releases API.
/// Returns (tag_name, download_url) for the current platform.
async fn check_latest_release() -> Result<Option<(String, String)>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .user_agent(format!("compute-cli/{VERSION}"))
        .build()?;

    let resp = client
        .get("https://api.github.com/repos/Compute-Network/compute-app/releases/latest")
        .send()
        .await?;

    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }

    if !resp.status().is_success() {
        anyhow::bail!("GitHub API error: {}", resp.status());
    }

    let release: serde_json::Value = resp.json().await?;

    let tag = release["tag_name"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("No tag_name in release"))?
        .to_string();

    // Find the asset for this platform
    let target = current_target();
    let assets =
        release["assets"].as_array().ok_or_else(|| anyhow::anyhow!("No assets in release"))?;

    for asset in assets {
        let name = asset["name"].as_str().unwrap_or("");
        if name.contains(&target) {
            let url = asset["browser_download_url"].as_str().unwrap_or("").to_string();
            return Ok(Some((tag, url)));
        }
    }

    // If no platform-specific asset, return the release URL
    let html_url = release["html_url"]
        .as_str()
        .unwrap_or("https://github.com/Compute-Network/compute-app/releases")
        .to_string();

    Ok(Some((tag, html_url)))
}

/// Download a binary and replace the current executable.
async fn download_and_replace(url: &str) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .user_agent(format!("compute-cli/{VERSION}"))
        .build()?;

    let resp = client.get(url).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("Download failed: {}", resp.status());
    }

    let bytes = resp.bytes().await?;

    let current_exe = std::env::current_exe()?;
    let backup = current_exe.with_extension("old");

    // Rename current binary to .old, write new binary
    std::fs::rename(&current_exe, &backup)?;
    std::fs::write(&current_exe, &bytes)?;

    // Make executable on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(&current_exe, perms)?;
    }

    // Clean up backup
    let _ = std::fs::remove_file(&backup);

    Ok(())
}

/// Determine the current platform target string for asset matching.
fn current_target() -> String {
    let os = if cfg!(target_os = "macos") {
        "apple-darwin"
    } else if cfg!(target_os = "linux") {
        "unknown-linux"
    } else if cfg!(target_os = "windows") {
        "pc-windows"
    } else {
        "unknown"
    };

    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "unknown"
    };

    format!("{arch}-{os}")
}

fn cmd_uninstall() -> Result<()> {
    use std::io::{self, Write};

    println!("\n  UNINSTALL COMPUTE\n");
    println!("  This will remove:");
    if let Some(dir) = config::config_dir() {
        println!("    - Config & data:  {}", dir.display());
    }
    println!("    - System service (if installed)");
    println!();
    println!("  Your wallet and earnings are safe — they're on-chain.");
    println!();
    print!("  Continue? [y/N] ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    if !input.trim().eq_ignore_ascii_case("y") {
        println!("  Cancelled.");
        return Ok(());
    }

    // Stop daemon if running
    println!("  Stopping daemon...");
    let _ = cmd_stop();

    // Uninstall service if installed
    if compute_daemon::service::is_service_installed() {
        println!("  Removing system service...");
        let _ = compute_daemon::service::uninstall_service();
    }

    // Remove ~/.compute directory
    if let Some(dir) = config::config_dir() {
        if dir.exists() {
            println!("  Removing {}...", dir.display());
            std::fs::remove_dir_all(&dir)?;
        }
    }

    println!();
    println!("  Compute uninstalled.");
    println!("  To also remove the binary: rm $(which compute)");
    println!();
    Ok(())
}

fn cmd_doctor() -> Result<()> {
    println!("\n  COMPUTE DOCTOR\n");

    let hw = hardware::detect();
    let config = Config::load()?;

    // === Hardware ===
    println!("  HARDWARE");
    println!("  ────────────────────────────────");

    let has_gpu = hw.gpus.iter().any(|g| !matches!(g.backend, hardware::GpuBackend::Cpu));
    print_check("GPU detected", has_gpu);
    if has_gpu {
        for gpu in &hw.gpus {
            let vram = if gpu.vram_mb >= 1024 {
                format!("{}GB", gpu.vram_mb / 1024)
            } else {
                format!("{}MB", gpu.vram_mb)
            };
            println!("    {} ({}, {})", gpu.name, vram, gpu.backend);
        }
    }

    let enough_ram = hw.memory.total_gb >= 8.0;
    print_check(&format!("RAM >= 8GB ({:.1}GB detected)", hw.memory.total_gb), enough_ram);

    let enough_disk = hw.disk.available_gb > 10.0;
    print_check(
        &format!("Disk space > 10GB ({:.0}GB available)", hw.disk.available_gb),
        enough_disk,
    );

    print_check("Docker available", hw.docker.available);

    // Check for llama-server (required for inference)
    let llama_server_found = std::process::Command::new("which")
        .arg("llama-server")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
        || std::path::Path::new("/opt/homebrew/bin/llama-server").exists()
        || std::path::Path::new("/usr/local/bin/llama-server").exists();
    print_check("llama-server available", llama_server_found);

    // Check for model files
    let models_dir =
        dirs::home_dir().map(|h| h.join(".compute").join("models")).unwrap_or_default();
    let has_models = models_dir.exists()
        && std::fs::read_dir(&models_dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .any(|e| e.path().extension().map(|ext| ext == "gguf").unwrap_or(false))
            })
            .unwrap_or(false);
    print_check("GGUF model files present", has_models);
    if has_models && let Ok(entries) = std::fs::read_dir(&models_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            if entry.path().extension().map(|ext| ext == "gguf").unwrap_or(false) {
                let size_mb = entry.metadata().map(|m| m.len() / (1024 * 1024)).unwrap_or(0);
                println!("    {} ({size_mb}MB)", entry.file_name().to_string_lossy());
            }
        }
    }
    println!();

    // === Configuration ===
    println!("  CONFIGURATION");
    println!("  ────────────────────────────────");

    let config_ok = config::config_exists();
    print_check("Config file exists", config_ok);

    let wallet_ok = !config.wallet.public_address.is_empty();
    print_check("Wallet address configured", wallet_ok);
    if wallet_ok {
        let addr = &config.wallet.public_address;
        let short = if addr.len() > 12 {
            format!("{}...{}", &addr[..6], &addr[addr.len() - 4..])
        } else {
            addr.clone()
        };
        println!("    {short}");
    }

    let node_id_ok = !config.wallet.node_id.is_empty();
    print_check("Node registered with network", node_id_ok);
    println!();

    // === Runtime ===
    println!("  RUNTIME");
    println!("  ────────────────────────────────");

    let daemon_ok = daemon::is_running();
    print_check("Daemon running", daemon_ok);

    let service_ok = compute_daemon::service::is_service_installed();
    print_check("Auto-start service installed", service_ok);
    println!();

    // === Network ===
    println!("  NETWORK");
    println!("  ────────────────────────────────");

    let rt = tokio::runtime::Runtime::new()?;
    let (supabase_ok, node_count) = rt.block_on(async {
        let client = SupabaseClient::new();
        let healthy = client.health_check().await;
        let stats = if healthy { client.get_network_stats().await.ok() } else { None };
        (healthy, stats.map(|s| s.total_nodes).unwrap_or(0))
    });

    print_check("Supabase API reachable", supabase_ok);
    if supabase_ok {
        println!("    {node_count} nodes in network");
    }
    println!();

    // === Tips ===
    let mut tips = Vec::new();
    if !config_ok {
        tips.push("Run `compute init` to create your config file");
    }
    if !wallet_ok {
        tips.push("Run `compute init` or `compute wallet set <address>` to set your wallet");
    }
    if !node_id_ok && wallet_ok {
        tips.push("Run `compute start` to register with the network");
    }
    if !daemon_ok {
        tips.push("Run `compute start` to start the daemon");
    }
    if !service_ok && daemon_ok {
        tips.push("Run `compute service install` for auto-start on login");
    }
    if !hw.docker.available {
        tips.push("Install Docker from https://docker.com/get-started");
    }
    if !llama_server_found {
        tips.push("Install llama.cpp: brew install llama.cpp (macOS) or build from source");
    }
    if !has_models {
        tips.push("Place .gguf model files in ~/.compute/models/");
    }

    if !tips.is_empty() {
        println!("  RECOMMENDATIONS");
        println!("  ────────────────────────────────");
        for tip in tips {
            println!("  → {tip}");
        }
        println!();
    }

    Ok(())
}

fn print_check(label: &str, ok: bool) {
    if ok {
        println!("  ✓ {label}");
    } else {
        println!("  ✗ {label}");
    }
}

fn cmd_service(action: ServiceAction) -> Result<()> {
    match action {
        ServiceAction::Install => {
            compute_daemon::service::install_service()?;
        }
        ServiceAction::Uninstall => {
            compute_daemon::service::uninstall_service()?;
        }
        ServiceAction::Status => {
            if compute_daemon::service::is_service_installed() {
                println!("Service is installed (auto-start on login)");
            } else {
                println!("Service is not installed");
                println!("Install with: compute service install");
            }
        }
    }
    Ok(())
}

fn format_duration_short(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let mins = (secs % 3600) / 60;

    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {mins}m")
    } else {
        format!("{mins}m")
    }
}
