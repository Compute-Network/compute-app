mod cli;
mod tui;

use anyhow::Result;
use clap::Parser;
use std::io::Seek;

use cli::{Cli, Commands, ConfigAction, ServiceAction, WalletAction};
use compute_daemon::config::{self, Config};
use compute_daemon::daemon;
use compute_daemon::hardware;

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

    // Show splash screen
    tui::app::run_splash_only()?;

    // Set up logging and dirs
    config::ensure_dirs()?;
    compute_daemon::logging::init(true)?;
    daemon::write_pid()?;

    let config = Config::load()?;
    let runtime = compute_daemon::runtime::DaemonRuntime::new(config);

    println!("\n  Daemon started (PID: {})", std::process::id());
    println!("  Run `compute dashboard` to view live stats");
    println!("  Run `compute stop` to stop the daemon\n");

    // Run the daemon event loop
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        if let Err(e) = runtime.run().await {
            eprintln!("Daemon error: {e}");
        }
    });

    daemon::remove_pid()?;
    Ok(())
}

fn cmd_stop() -> Result<()> {
    match daemon::stop_daemon() {
        Ok(()) => println!("Daemon stopped."),
        Err(e) => println!("Error: {e}"),
    }
    Ok(())
}

fn cmd_status() -> Result<()> {
    if daemon::is_running() {
        let pid = daemon::read_pid().unwrap_or(0);
        let uptime =
            daemon::uptime().map(format_duration_short).unwrap_or_else(|| "unknown".into());

        println!("● ACTIVE  PID: {pid}  Uptime: {uptime}");
    } else {
        println!("○ STOPPED  Run `compute start` to begin");
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
    println!("  ██████╗ ██████╗ ███╗   ███╗██████╗ ██╗   ██╗████████╗███████╗");
    println!("  ██╔════╝██╔═══██╗████╗ ████║██╔══██╗██║   ██║╚══██╔══╝██╔════╝");
    println!("  ██║     ██║   ██║██╔████╔██║██████╔╝██║   ██║   ██║   █████╗  ");
    println!("  ██║     ██║   ██║██║╚██╔╝██║██╔═══╝ ██║   ██║   ██║   ██╔══╝  ");
    println!("  ╚██████╗╚██████╔╝██║ ╚═╝ ██║██║     ╚██████╔╝   ██║   ███████╗");
    println!("   ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝      ╚═════╝    ╚═╝   ╚══════╝");
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
    let e = compute_daemon::metrics::Earnings::mock();
    println!();
    println!("  EARNINGS");
    println!("  ────────────────────────────────");
    println!("  Today       {:.1} $COMPUTE  ≈ ${:.2}", e.today, e.today * e.usd_rate);
    println!("  This Week   {:.1} $COMPUTE  ≈ ${:.2}", e.this_week, e.this_week * e.usd_rate);
    println!("  This Month  {:.1} $COMPUTE  ≈ ${:.2}", e.this_month, e.this_month * e.usd_rate);
    println!("  All Time    {:.0} $COMPUTE", e.all_time);
    println!("  Pending     {:.1} $COMPUTE", e.pending);
    println!();
    println!("  (Mock data — connect to network for real earnings)");
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
    if !daemon::is_running() {
        println!("Daemon is not running. Start with `compute start`");
        return Ok(());
    }

    let p = compute_daemon::metrics::PipelineStatus::mock();
    println!();
    println!("  PIPELINE STATUS");
    println!("  ────────────────────────────────");
    if p.active {
        println!("  Status:   ● Active");
        if let (Some(s), Some(t)) = (p.stage, p.total_stages) {
            println!("  Stage:    {s} / {t}");
        }
        if let Some(ref model) = p.model {
            println!("  Model:    {model}");
        }
        println!("  Served:   {} requests", p.requests_served);
        println!("  Latency:  {:.0}ms avg", p.avg_latency_ms);
        println!("  Speed:    {:.1} tok/s", p.tokens_per_sec);
    } else {
        println!("  Status:   ○ Waiting for assignment");
    }
    println!();

    Ok(())
}

fn cmd_update() -> Result<()> {
    println!("Self-update not yet implemented.");
    println!("Current version: v{VERSION}");
    println!("Check https://github.com/Compute-Network/compute-app/releases for updates.");
    Ok(())
}

fn cmd_uninstall() -> Result<()> {
    println!("To uninstall Compute:");
    println!();
    println!("  1. Stop the daemon:  compute stop");
    println!("  2. Remove the binary: rm $(which compute)");
    if let Some(dir) = config::config_dir() {
        println!("  3. Remove config:    rm -rf {}", dir.display());
    }
    println!();
    println!("Your wallet and earnings are safe — they're on-chain.");
    Ok(())
}

fn cmd_doctor() -> Result<()> {
    println!("\n  COMPUTE DOCTOR\n");

    let hw = hardware::detect();

    // Check GPU
    let has_gpu = hw.gpus.iter().any(|g| !matches!(g.backend, hardware::GpuBackend::Cpu));
    print_check("GPU detected", has_gpu);

    // Check Docker
    print_check("Docker available", hw.docker.available);

    // Check config
    let config_ok = config::config_exists();
    print_check("Config file exists", config_ok);

    // Check daemon
    let daemon_ok = daemon::is_running();
    print_check("Daemon running", daemon_ok);

    // Check disk space
    let enough_disk = hw.disk.available_gb > 10.0;
    print_check("Disk space > 10GB", enough_disk);

    // Check RAM
    let enough_ram = hw.memory.total_gb >= 8.0;
    print_check("RAM >= 8GB", enough_ram);

    println!();

    if !config_ok {
        println!("  Tip: Run `compute init` to create your config file");
    }
    if !daemon_ok {
        println!("  Tip: Run `compute start` to start the daemon");
    }
    if !hw.docker.available {
        println!("  Tip: Install Docker from https://docker.com/get-started");
    }
    println!();

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
