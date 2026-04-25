use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::{
    DefaultTerminal, Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Sparkline, Wrap},
};

use compute_daemon::hardware::{self, HardwareInfo, LiveMetrics};
use compute_daemon::metrics::{Earnings, NetworkStats, PipelineStatus};

use super::globe::Globe;
use super::theme;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Overview,
    Logs,
    Config,
    Models,
}

struct ModelEntry {
    id: &'static str,
    label: &'static str,
    desc: &'static str,
    gguf_filename: &'static str,
    hf_url: &'static str,
    mlx_repo_id: Option<&'static str>,
    mlx_folder: Option<&'static str>,
    mlx_total_size_mb: Option<u64>,
}

fn model_entries() -> Vec<ModelEntry> {
    vec![
        ModelEntry {
            id: "auto",
            label: "Auto",
            desc: "Orchestrator selects the best model",
            gguf_filename: "",
            hf_url: "",
            mlx_repo_id: None,
            mlx_folder: None,
            mlx_total_size_mb: None,
        },
        ModelEntry {
            id: "gemma-4-26b-a4b-q4",
            label: "Gemma4 — 18GB (med)",
            desc: "MoE · 26B total, 4B active",
            gguf_filename: "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
            hf_url: "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
            mlx_repo_id: None,
            mlx_folder: None,
            mlx_total_size_mb: None,
        },
        ModelEntry {
            id: "gemma-4-e4b-q4",
            label: "Gemma4 — 3GB (small)",
            desc: "Fast · 4B params, multimodal",
            gguf_filename: "gemma-4-E4B-it-Q4_K_M.gguf",
            hf_url: "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf",
            mlx_repo_id: None,
            mlx_folder: None,
            mlx_total_size_mb: None,
        },
        // v0.4.4: qwen-3.6 unified id — daemon picks the format per host.
        // The TUI picker needs a single user-facing entry; compute-network's
        // ModelDefinition carries both the GGUF (Linux/Windows) and MLX
        // (Apple Silicon via oMLX) sources. Linux / Windows still download
        // the single GGUF file; Apple Silicon snapshots the full MLX repo
        // (`unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit`) into
        // `<cache>/mlx/Qwen3.6-35B-A3B-UD-MLX-4bit/`.
        ModelEntry {
            id: "qwen-3.6",
            label: "Qwen3.6 — 22GB (MoE)",
            desc: "35B-A3B MoE · auto MLX on Mac / GGUF elsewhere",
            gguf_filename: "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            hf_url: "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            mlx_repo_id: Some("unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"),
            mlx_folder: Some("Qwen3.6-35B-A3B-UD-MLX-4bit"),
            mlx_total_size_mb: Some(20_400),
        },
        ModelEntry {
            id: "gemma-4-e4b-q4-stage-head",
            label: "Split-Model-Test (head)",
            desc: "Pipeline head · layers 0-20",
            gguf_filename: "head-0-20.gguf",
            hf_url: "https://huggingface.co/ComputeNet-sh/gemma-4-e4b-q4-gguf-stages/resolve/main/gemma-4-e4b-q4-head-0-20.gguf",
            mlx_repo_id: None,
            mlx_folder: None,
            mlx_total_size_mb: None,
        },
        ModelEntry {
            id: "gemma-4-e4b-q4-stage-tail",
            label: "Split-Model-Test (tail)",
            desc: "Pipeline tail · layers 21-41",
            gguf_filename: "tail-21-41.gguf",
            hf_url: "https://huggingface.co/ComputeNet-sh/gemma-4-e4b-q4-gguf-stages/resolve/main/gemma-4-e4b-q4-tail-21-41.gguf",
            mlx_repo_id: None,
            mlx_folder: None,
            mlx_total_size_mb: None,
        },
        ModelEntry {
            id: "gemma-3-270m-q4-draft",
            label: "Gemma3 270M — 250MB (draft)",
            desc: "Speculative-decode helper for Gemma4 E4B",
            gguf_filename: "gemma-3-270m-it-Q4_K_M.gguf",
            hf_url: "https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf",
            mlx_repo_id: None,
            mlx_folder: None,
            mlx_total_size_mb: None,
        },
    ]
}

fn is_stage_entry(model_id: &str) -> bool {
    model_id.contains("-stage-head") || model_id.contains("-stage-tail")
}

/// Path where per-stage GGUF shards are cached. Mirrors
/// `stage_artifacts::gguf_shard_path` in compute-daemon so the TUI download
/// and the daemon's lazy download share the same cache.
fn stage_shard_path(filename: &str) -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_default()
        .join(".compute")
        .join("stages")
        .join("gemma-4-e4b-q4")
        .join(filename)
}

fn models_cache_dir() -> std::path::PathBuf {
    let config = compute_daemon::config::Config::load().unwrap_or_default();
    std::path::PathBuf::from(config.models.cache_dir)
}

fn is_apple_silicon() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

fn uses_mlx_snapshot(entry: &ModelEntry) -> bool {
    is_apple_silicon() && entry.mlx_repo_id.is_some() && entry.mlx_folder.is_some()
}

fn mlx_snapshot_path(entry: &ModelEntry) -> Option<std::path::PathBuf> {
    entry.mlx_folder.map(|folder| models_cache_dir().join("mlx").join(folder))
}

fn dir_size_bytes(root: &std::path::Path) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if let Ok(file_type) = entry.file_type() {
                if file_type.is_dir() {
                    stack.push(entry_path);
                } else if file_type.is_file() {
                    total = total.saturating_add(entry.metadata().map(|m| m.len()).unwrap_or(0));
                }
            }
        }
    }
    total
}

fn is_model_downloaded(model_id: &str) -> bool {
    if model_id == "auto" {
        return true;
    }
    if is_stage_entry(model_id) {
        for entry in model_entries() {
            if entry.id == model_id && !entry.gguf_filename.is_empty() {
                let path = stage_shard_path(entry.gguf_filename);
                if !path.exists() {
                    return false;
                }
                if let Ok(meta) = std::fs::metadata(&path) {
                    if meta.len() < 100 * 1024 * 1024 {
                        return false;
                    }
                }
                if let Ok(mut f) = std::fs::File::open(&path) {
                    use std::io::Read;
                    let mut magic = [0u8; 4];
                    if f.read_exact(&mut magic).is_err() || magic != [0x47, 0x47, 0x55, 0x46] {
                        return false;
                    }
                }
                return true;
            }
        }
        return false;
    }
    for entry in model_entries() {
        if entry.id == model_id && uses_mlx_snapshot(&entry) {
            let Some(folder) = entry.mlx_folder else {
                return false;
            };
            return compute_daemon::inference::stage_artifacts::mlx_model_cached(
                models_cache_dir().as_path(),
                folder,
            );
        }
        if entry.id == model_id && !entry.gguf_filename.is_empty() {
            let cache_dir = models_cache_dir();
            let path = cache_dir.join(entry.gguf_filename);
            if !path.exists() {
                return false;
            }
            // Check file is large enough to be a real model (> 100MB)
            if let Ok(meta) = std::fs::metadata(&path) {
                if meta.len() < 100 * 1024 * 1024 {
                    return false;
                }
            }
            // Verify GGUF magic header
            if let Ok(mut f) = std::fs::File::open(&path) {
                use std::io::Read;
                let mut magic = [0u8; 4];
                if f.read_exact(&mut magic).is_err() || magic != [0x47, 0x47, 0x55, 0x46] {
                    return false;
                }
            }
            return true;
        }
    }
    false
}

#[derive(Clone, Copy, PartialEq)]
enum ConfigItemKind {
    Text,
    Toggle,
    Choice, // cycles through options
}

struct ConfigItem {
    label: &'static str,
    key: &'static str,
    kind: ConfigItemKind,
}

const CONFIG_ITEMS: &[ConfigItem] = &[
    ConfigItem { label: "Node Name", key: "node.name", kind: ConfigItemKind::Text },
    ConfigItem { label: "Usage Priority", key: "node.max_gpu_usage", kind: ConfigItemKind::Choice },
    ConfigItem {
        label: "Pause on Battery",
        key: "node.pause_on_battery",
        kind: ConfigItemKind::Toggle,
    },
    ConfigItem {
        label: "Pause on Gaming",
        key: "node.pause_on_fullscreen",
        kind: ConfigItemKind::Toggle,
    },
    ConfigItem {
        label: "Prevent Sleep",
        key: "node.caffeinate_when_running",
        kind: ConfigItemKind::Toggle,
    },
    ConfigItem {
        label: "Model Auto-Downloads",
        key: "models.auto_download",
        kind: ConfigItemKind::Toggle,
    },
    ConfigItem {
        label: "Auto-start on Login",
        key: "service.autostart",
        kind: ConfigItemKind::Toggle,
    },
    ConfigItem { label: "Theme", key: "appearance.theme", kind: ConfigItemKind::Choice },
];

pub struct Dashboard {
    globe: Globe,
    hardware: HardwareInfo,
    earnings: Earnings,
    pipeline: PipelineStatus,
    network: NetworkStats,
    throughput_history: VecDeque<u64>,
    smoothed_tps: f64,
    last_throughput_push: Instant,
    sys: sysinfo::System,
    live_metrics: LiveMetrics,
    last_metrics_update: Instant,
    last_network_update: Instant,
    uptime_start: Instant,
    active_tab: Tab,
    log_lines: Vec<String>,
    log_scroll: usize,
    show_debug_logs: bool,
    last_log_update: Instant,
    config_selected: usize,
    config_editing: bool,
    config_edit_buffer: String,
    config_confirm: Option<String>, // "Are you sure?" for wallet changes
    daemon_state_rx: Option<tokio::sync::watch::Receiver<compute_daemon::runtime::DaemonState>>,
    models_selected: usize,
    model_downloads: BTreeMap<String, f64>, // model_id -> progress 0.0-1.0, negative = failed
    download_progress_rxs: Vec<std::sync::mpsc::Receiver<(String, f64)>>,
}

impl Dashboard {
    pub fn with_daemon_state(
        hardware: HardwareInfo,
        rx: tokio::sync::watch::Receiver<compute_daemon::runtime::DaemonState>,
    ) -> Self {
        let mut d = Self::new(hardware);
        d.daemon_state_rx = Some(rx);
        d
    }

    pub fn new(hardware: HardwareInfo) -> Self {
        let mut globe = Globe::new();

        // Fetch real nodes from the orchestrator for globe visualization
        let config = compute_daemon::config::Config::load().unwrap_or_default();
        let (network, node_regions) = fetch_network_and_nodes();
        if !node_regions.is_empty() {
            globe.set_nodes_from_regions(&node_regions);
            globe.set_my_position(Some(&config.network.region), &config.wallet.public_address);
        } else {
            globe.set_mock_nodes();
        }

        // Try to load recent log lines
        let log_lines = load_recent_logs(50);

        Self {
            globe,
            hardware,
            earnings: Earnings::default(),
            pipeline: PipelineStatus::default(),
            network,
            throughput_history: VecDeque::from(vec![0; 60]),
            smoothed_tps: 0.0,
            last_throughput_push: Instant::now(),
            sys: sysinfo::System::new_all(),
            live_metrics: LiveMetrics::default(),
            last_metrics_update: Instant::now(),
            last_network_update: Instant::now(),
            uptime_start: Instant::now(),
            active_tab: Tab::Overview,
            log_lines,
            log_scroll: 0,
            show_debug_logs: false,
            last_log_update: Instant::now(),
            config_selected: 0,
            config_editing: false,
            config_edit_buffer: String::new(),
            config_confirm: None,
            daemon_state_rx: None,
            models_selected: 0,
            model_downloads: BTreeMap::new(),
            download_progress_rxs: Vec::new(),
        }
    }

    /// Run the dashboard event loop.
    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
        let tick_rate = Duration::from_millis(100); // 10 fps

        loop {
            terminal.draw(|frame| self.draw(frame))?;

            if event::poll(tick_rate)?
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                match key.code {
                    // Config editing mode intercepts all keys
                    _ if self.config_editing && self.active_tab == Tab::Config => {
                        self.handle_config_key(key.code);
                        continue;
                    }
                    // Confirmation dialog
                    _ if self.config_confirm.is_some() && self.active_tab == Tab::Config => {
                        match key.code {
                            KeyCode::Char('y') | KeyCode::Char('Y') => {
                                if let Some(new_wallet) = self.config_confirm.take() {
                                    self.apply_wallet_change(&new_wallet);
                                }
                            }
                            _ => {
                                self.config_confirm = None;
                            }
                        }
                        continue;
                    }
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        return Ok(());
                    }
                    KeyCode::Char('p') => {
                        self.pipeline.active = !self.pipeline.active;
                    }
                    KeyCode::Char('1') => self.active_tab = Tab::Overview,
                    KeyCode::Char('2') | KeyCode::Char('l') => {
                        self.active_tab = Tab::Logs;
                        self.log_lines = load_recent_logs(100);
                    }
                    KeyCode::Char('3') => self.active_tab = Tab::Config,
                    KeyCode::Char('4') => self.active_tab = Tab::Models,
                    KeyCode::Char('c') => {
                        open_claim_page();
                    }
                    KeyCode::Up if self.active_tab == Tab::Logs => {
                        self.log_scroll = self.log_scroll.saturating_add(1);
                    }
                    KeyCode::Down if self.active_tab == Tab::Logs => {
                        self.log_scroll = self.log_scroll.saturating_sub(1);
                    }
                    KeyCode::End if self.active_tab == Tab::Logs => {
                        self.log_scroll = 0; // 0 = follow tail
                    }
                    KeyCode::Char('/') if self.active_tab == Tab::Logs => {
                        self.show_debug_logs = !self.show_debug_logs;
                        self.log_scroll = 0;
                    }
                    KeyCode::Up if self.active_tab == Tab::Config => {
                        if self.config_selected > 0 {
                            self.config_selected -= 1;
                        }
                    }
                    KeyCode::Down if self.active_tab == Tab::Config => {
                        if self.config_selected < CONFIG_ITEMS.len() - 1 {
                            self.config_selected += 1;
                        }
                    }
                    KeyCode::Enter if self.active_tab == Tab::Config => {
                        self.start_config_edit();
                    }
                    KeyCode::Up if self.active_tab == Tab::Models => {
                        if self.models_selected > 0 {
                            self.models_selected -= 1;
                        }
                    }
                    KeyCode::Down if self.active_tab == Tab::Models => {
                        let max = model_entries().len().saturating_sub(1);
                        if self.models_selected < max {
                            self.models_selected += 1;
                        }
                    }
                    KeyCode::Enter if self.active_tab == Tab::Models => {
                        self.select_model();
                    }
                    KeyCode::Char('d') if self.active_tab == Tab::Models => {
                        self.delete_model();
                    }
                    KeyCode::Tab => {
                        self.active_tab = match self.active_tab {
                            Tab::Overview => Tab::Logs,
                            Tab::Logs => Tab::Config,
                            Tab::Config => Tab::Models,
                            Tab::Models => Tab::Overview,
                        };
                        if self.active_tab == Tab::Logs {
                            self.log_lines = load_recent_logs(500);
                            self.log_scroll = 0;
                        }
                    }
                    _ => {}
                }
            }

            self.tick();
        }
    }

    fn tick(&mut self) {
        self.globe.tick();

        // Poll manual model download progress. Keep receivers independent so
        // users can queue several model downloads from the Models tab.
        let mut finished_receivers = Vec::new();
        for (idx, rx) in self.download_progress_rxs.iter().enumerate() {
            let mut finished = false;
            while let Ok((model_id, progress)) = rx.try_recv() {
                self.model_downloads.insert(model_id, progress);
                if progress >= 1.0 || progress < 0.0 {
                    finished = true;
                }
            }
            if finished {
                finished_receivers.push(idx);
            }
        }
        for idx in finished_receivers.into_iter().rev() {
            self.download_progress_rxs.remove(idx);
        }

        // Read from daemon state if available
        if let Some(ref rx) = self.daemon_state_rx {
            let state = rx.borrow().clone();
            self.live_metrics = state.live_metrics;
            self.earnings = state.earnings;
            for download in state.downloads {
                let progress = match download.phase {
                    compute_daemon::runtime::DownloadPhase::Downloading => download
                        .total_bytes
                        .filter(|total| *total > 0)
                        .map(|total| download.downloaded_bytes as f64 / total as f64)
                        .unwrap_or(0.01)
                        .clamp(0.01, 0.99),
                    compute_daemon::runtime::DownloadPhase::Complete => 1.0,
                    compute_daemon::runtime::DownloadPhase::Failed => -1.0,
                };
                self.model_downloads.insert(download.model_id, progress);
            }

            // Smooth toward target
            let target = state.pipeline.tokens_per_sec;
            if target > self.smoothed_tps {
                self.smoothed_tps += (target - self.smoothed_tps) * 0.3;
            } else {
                self.smoothed_tps += (target - self.smoothed_tps) * 0.15;
            }
            if self.smoothed_tps < 1.0 {
                self.smoothed_tps = 0.0;
            }

            // Push every 200ms (60 entries = 12 seconds visible)
            // O(1) VecDeque operations instead of O(n) Vec::remove(0)
            if self.last_throughput_push.elapsed() >= Duration::from_millis(200) {
                if self.throughput_history.len() >= 60 {
                    self.throughput_history.pop_front();
                }
                self.throughput_history.push_back(self.smoothed_tps as u64);
                self.last_throughput_push = Instant::now();
            }

            self.pipeline = state.pipeline;
        } else {
            // No daemon state — use local metrics collection
            if self.last_metrics_update.elapsed() > Duration::from_secs(2) {
                self.live_metrics = hardware::collect_live_metrics(&mut self.sys);
                self.last_metrics_update = Instant::now();
            }

            // Push one value every 500ms
            if self.last_throughput_push.elapsed() >= Duration::from_millis(500) {
                if self.throughput_history.len() >= 60 {
                    self.throughput_history.pop_front();
                }
                self.throughput_history.push_back(0);
                self.last_throughput_push = Instant::now();
            }
        }

        // Refresh logs every 2 seconds when on logs tab
        if self.active_tab == Tab::Logs && self.last_log_update.elapsed() > Duration::from_secs(2) {
            self.log_lines = load_recent_logs(500);
            self.last_log_update = Instant::now();
        }

        // Refresh network stats and nodes from the orchestrator every 60 seconds
        if self.last_network_update.elapsed() > Duration::from_secs(60) {
            let (network, node_regions) = fetch_network_and_nodes();
            self.network = network;
            if !node_regions.is_empty() {
                self.globe.set_nodes_from_regions(&node_regions);
            }
            self.last_network_update = Instant::now();
        }
    }

    fn draw(&self, frame: &mut Frame) {
        let full_area = frame.area();
        frame.render_widget(Clear, full_area);

        // Cap dimensions, align top-left
        let max_w: u16 = 160;
        let max_h: u16 = 45;
        let area = Rect {
            x: full_area.x,
            y: full_area.y,
            width: full_area.width.min(max_w),
            height: full_area.height.min(max_h),
        };

        // Responsive breakpoints:
        // Wide (>= 80 cols): side-by-side globe + content
        // Medium (50-79 cols): globe on top, content below
        // Narrow (< 50 cols): no globe, content only
        if area.width >= 80 {
            // Desktop: horizontal split
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(33), Constraint::Percentage(67)])
                .split(area);

            self.draw_globe_panel(frame, chunks[0]);
            self.draw_right_panel(frame, chunks[1]);
        } else if area.width >= 50 {
            // Tablet/vertical: globe on top, content below
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(15), Constraint::Min(10)])
                .split(area);

            // Globe centered in top section
            self.globe.render(chunks[0], frame.buffer_mut(), theme::palette());
            self.draw_right_panel(frame, chunks[1]);
        } else {
            // Narrow/mobile: content only, no globe
            self.draw_right_panel(frame, area);
        }
    }

    fn draw_globe_panel(&self, frame: &mut Frame, area: Rect) {
        let p = theme::palette();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(10),   // Globe
                Constraint::Length(4), // Network stats
            ])
            .split(area);

        // Globe
        let globe_block =
            Block::default().borders(Borders::RIGHT).border_style(Style::default().fg(p.dim));
        let inner = globe_block.inner(chunks[0]);
        frame.render_widget(globe_block, chunks[0]);
        self.globe.render(inner, frame.buffer_mut(), p);

        // Network stats below globe
        let stats_block = Block::default()
            .borders(Borders::RIGHT | Borders::TOP)
            .border_style(Style::default().fg(p.dim));
        let stats_inner = stats_block.inner(chunks[1]);
        frame.render_widget(stats_block, chunks[1]);

        let stats = Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    format!("  {:>6} ", format_number(self.network.total_nodes)),
                    Style::default().fg(p.text).add_modifier(Modifier::BOLD),
                ),
                Span::styled("nodes", Style::default().fg(p.dim)),
            ]),
            Line::from(vec![
                Span::styled("  ● ", Style::default().fg(p.success)),
                Span::styled(
                    format!("{:.0} PF", self.network.peak_petaflops),
                    Style::default().fg(p.text),
                ),
                Span::styled(" peak", Style::default().fg(p.dim)),
            ]),
        ]);
        frame.render_widget(stats, stats_inner);
    }

    fn draw_right_panel(&self, frame: &mut Frame, area: Rect) {
        match self.active_tab {
            Tab::Overview => self.draw_overview_panel(frame, area),
            Tab::Logs => self.draw_logs_panel(frame, area),
            Tab::Config => self.draw_config_panel(frame, area),
            Tab::Models => self.draw_models_panel(frame, area),
        }
    }

    fn draw_overview_panel(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5), // Header
                Constraint::Length(7), // Node info
                Constraint::Length(7), // Earnings
                Constraint::Length(8), // Workload
                Constraint::Min(4),    // Throughput sparkline
                Constraint::Length(1), // Keyboard shortcuts
            ])
            .split(area);

        self.draw_header(frame, chunks[0]);
        self.draw_node_info(frame, chunks[1]);
        self.draw_earnings(frame, chunks[2]);
        self.draw_workload(frame, chunks[3]);
        self.draw_throughput(frame, chunks[4]);
        self.draw_shortcuts(frame, chunks[5]);
    }

    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let p = theme::palette();
        let w = area.width as usize;
        let (status_color, status_text) = if self.pipeline.active
            && (self.pipeline.active_requests > 0 || self.pipeline.tokens_per_sec > 0.0)
        {
            (p.success, "ACTIVE")
        } else if self.pipeline.active {
            (p.success, "ONLINE")
        } else {
            (p.dim, "IDLE")
        };

        let tab_style = |tab: Tab| {
            if tab == self.active_tab {
                Style::default().fg(p.text).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(p.dim)
            }
        };

        let mut lines: Vec<Line> = Vec::new();
        lines.push(Line::from(""));

        // Title line — always fits
        lines.push(Line::from(vec![Span::styled(
            "  C O M P U T E",
            Style::default().fg(p.text).add_modifier(Modifier::BOLD),
        )]));

        // Subtitle + status: combine on one line if wide enough, split if narrow
        if w >= 55 {
            lines.push(Line::from(vec![
                Span::styled("  Decentralized GPU Infrastructure", Style::default().fg(p.dim)),
                Span::raw("  "),
                Span::styled("● ", Style::default().fg(status_color)),
                Span::styled(status_text, Style::default().fg(status_color)),
                Span::raw("  "),
                Span::styled(format!("v{VERSION}"), Style::default().fg(p.dim)),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled("● ", Style::default().fg(status_color)),
                Span::styled(status_text, Style::default().fg(status_color)),
                Span::raw("  "),
                Span::styled(format!("v{VERSION}"), Style::default().fg(p.dim)),
            ]));
        }

        // Tabs: use short labels on narrow screens
        if w >= 45 {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled("[1] OVERVIEW", tab_style(Tab::Overview)),
                Span::raw("  "),
                Span::styled("[2] LOGS", tab_style(Tab::Logs)),
                Span::raw("  "),
                Span::styled("[3] CONFIG", tab_style(Tab::Config)),
                Span::raw("  "),
                Span::styled("[4] MODELS", tab_style(Tab::Models)),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::raw(" "),
                Span::styled("[1]OVR", tab_style(Tab::Overview)),
                Span::raw(" "),
                Span::styled("[2]LOG", tab_style(Tab::Logs)),
                Span::raw(" "),
                Span::styled("[3]CFG", tab_style(Tab::Config)),
                Span::raw(" "),
                Span::styled("[4]MOD", tab_style(Tab::Models)),
            ]));
        }

        let header = Paragraph::new(lines).block(
            Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(p.dim)),
        );
        frame.render_widget(header, area);
    }

    fn draw_node_info(&self, frame: &mut Frame, area: Rect) {
        let p = theme::palette();
        let gpu_name = self
            .hardware
            .gpus
            .first()
            .map(|g| format!("{} ({})", g.name, format_vram(g.vram_mb)))
            .unwrap_or_else(|| "No GPU".into());

        let pipeline_stage =
            if let (Some(s), Some(t)) = (self.pipeline.stage, self.pipeline.total_stages) {
                format!("Pipeline Stage {}/{}", s + 1, t)
            } else {
                "Not assigned".into()
            };

        let uptime = format_duration(self.uptime_start.elapsed());

        let cpu_bar = progress_bar(self.live_metrics.cpu_usage as f64, 100.0, 10);

        let lines = vec![
            Line::from(Span::styled(
                "  NODE",
                Style::default().fg(p.dim).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("  GPU     ", Style::default().fg(p.dim)),
                Span::styled(&gpu_name, Style::default().fg(p.text)),
            ]),
            Line::from(vec![
                Span::styled("  Stage   ", Style::default().fg(p.dim)),
                Span::styled(&pipeline_stage, Style::default().fg(p.text)),
            ]),
            Line::from(vec![
                Span::styled("  Uptime  ", Style::default().fg(p.dim)),
                Span::styled(&uptime, Style::default().fg(p.text)),
            ]),
            Line::from(vec![
                Span::styled("  CPU     ", Style::default().fg(p.dim)),
                Span::styled(&cpu_bar, Style::default().fg(p.text)),
                Span::styled(
                    format!(" {:.0}%", self.live_metrics.cpu_usage),
                    Style::default().fg(p.dim),
                ),
            ]),
        ];

        let widget = Paragraph::new(lines).block(
            Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(p.dim)),
        );
        frame.render_widget(widget, area);
    }

    fn draw_earnings(&self, frame: &mut Frame, area: Rect) {
        let p = theme::palette();
        let e = &self.earnings;
        let lines = vec![
            Line::from(Span::styled(
                "  EARNINGS",
                Style::default().fg(p.dim).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Today      ", Style::default().fg(p.dim)),
                Span::styled(format!("{:.1} $COMPUTE", e.today), Style::default().fg(p.text)),
                Span::styled(
                    format!("    ≈ ${:.2}", e.today * e.usd_rate),
                    Style::default().fg(p.dim),
                ),
            ]),
            Line::from(vec![
                Span::styled("  This Week  ", Style::default().fg(p.dim)),
                Span::styled(format!("{:.1} $COMPUTE", e.this_week), Style::default().fg(p.text)),
                Span::styled(
                    format!("    ≈ ${:.2}", e.this_week * e.usd_rate),
                    Style::default().fg(p.dim),
                ),
            ]),
            Line::from(vec![
                Span::styled("  All Time   ", Style::default().fg(p.dim)),
                Span::styled(format!("{:.0} $COMPUTE", e.all_time), Style::default().fg(p.text)),
            ]),
            Line::from(vec![
                Span::styled("  Pending    ", Style::default().fg(p.dim)),
                Span::styled(format!("{:.1} $COMPUTE", e.pending), Style::default().fg(p.warning)),
                Span::styled("  [c]laim", Style::default().fg(p.dim)),
            ]),
        ];

        let widget = Paragraph::new(lines).block(
            Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(p.dim)),
        );
        frame.render_widget(widget, area);
    }

    fn draw_workload(&self, frame: &mut Frame, area: Rect) {
        let p = &self.pipeline;
        let theme = theme::palette();
        let model = p.model.as_deref().unwrap_or("None");
        let backend = if p.backend.trim().is_empty() { "auto" } else { p.backend.as_str() };

        let vram_line = if let (Some(used), Some(total)) =
            (self.live_metrics.gpu_vram_used_mb, self.live_metrics.gpu_vram_total_mb)
        {
            format!("{} / {} GB", used / 1024, total / 1024)
        } else if let Some(gpu) = self.hardware.gpus.first() {
            format!("-- / {} GB", gpu.vram_mb / 1024)
        } else {
            "N/A".into()
        };

        let temp_line =
            self.live_metrics.gpu_temp.map(|t| format!("{t}°C")).unwrap_or_else(|| "N/A".into());

        let power_line =
            match (self.live_metrics.gpu_power_watts, self.live_metrics.gpu_power_limit_watts) {
                (Some(p), Some(l)) => format!("{p}W / {l}W"),
                (Some(p), None) => format!("{p}W"),
                _ => "N/A".into(),
            };

        let lines = vec![
            Line::from(Span::styled(
                "  WORKLOAD",
                Style::default().fg(theme.dim).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Model     ", Style::default().fg(theme.dim)),
                Span::styled(model, Style::default().fg(theme.text)),
            ]),
            Line::from(vec![
                Span::styled("  Served    ", Style::default().fg(theme.dim)),
                Span::styled(
                    format!("{} requests", format_number(p.requests_served as u32)),
                    Style::default().fg(theme.text),
                ),
                Span::styled("    Active  ", Style::default().fg(theme.dim)),
                Span::styled(
                    format!("{}", p.active_requests),
                    Style::default().fg(if p.active_requests > 0 {
                        theme.success
                    } else {
                        theme.text
                    }),
                ),
            ]),
            Line::from(vec![
                Span::styled("  Latency   ", Style::default().fg(theme.dim)),
                Span::styled(
                    format!("{:.0}ms avg", p.avg_latency_ms),
                    Style::default().fg(theme.text),
                ),
            ]),
            Line::from(vec![
                Span::styled("  Backend   ", Style::default().fg(theme.dim)),
                Span::styled(backend, Style::default().fg(theme.text)),
            ]),
            Line::from(vec![
                Span::styled("  VRAM      ", Style::default().fg(theme.dim)),
                Span::styled(&vram_line, Style::default().fg(theme.text)),
            ]),
            Line::from(vec![
                Span::styled("  Temp      ", Style::default().fg(theme.dim)),
                Span::styled(&temp_line, Style::default().fg(theme.text)),
                Span::styled("    Power  ", Style::default().fg(theme.dim)),
                Span::styled(&power_line, Style::default().fg(theme.text)),
            ]),
        ];

        let widget = Paragraph::new(lines).block(
            Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(theme.dim)),
        );
        frame.render_widget(widget, area);
    }

    fn draw_throughput(&self, frame: &mut Frame, area: Rect) {
        let p = theme::palette();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(1),
                Constraint::Length(1),
            ])
            .split(area);

        let label = Paragraph::new(Line::from(Span::styled(
            "  THROUGHPUT",
            Style::default().fg(p.dim).add_modifier(Modifier::BOLD),
        )));
        frame.render_widget(label, chunks[0]);

        // Fixed scale: max 200, baseline offset 25 (always 1 visible block)
        let display_data: Vec<u64> = self.throughput_history.iter().map(|&v| v + 25).collect();

        let sparkline =
            Sparkline::default().data(&display_data).max(200).style(Style::default().fg(p.text));

        let sparkline_area =
            Rect { x: chunks[2].x + 2, width: chunks[2].width.saturating_sub(4), ..chunks[2] };
        frame.render_widget(sparkline, sparkline_area);

        // Show actual value (0 when idle, not 1)
        let current_tps = self.throughput_history.back().unwrap_or(&0);
        let tps_label = Paragraph::new(Line::from(vec![
            Span::raw("  "),
            Span::styled(
                format!("{current_tps} tok/s"),
                Style::default().fg(p.text).add_modifier(Modifier::BOLD),
            ),
        ]))
        .alignment(Alignment::Right);
        frame.render_widget(
            tps_label,
            Rect { width: chunks[3].width.saturating_sub(2), ..chunks[3] },
        );
    }

    fn draw_shortcuts(&self, frame: &mut Frame, area: Rect) {
        let w = area.width as usize;
        let dim = Style::default().fg(theme::palette().dim);

        let mut spans = vec![
            Span::styled(" [q]", dim),
            Span::styled("uit ", dim),
            Span::styled("[p]", dim),
            Span::styled("ause ", dim),
        ];

        if w >= 45 {
            spans.extend([Span::styled("[c]", dim), Span::styled("laim ", dim)]);
        }

        if w >= 50 {
            spans.extend([Span::styled("[Tab]", dim), Span::styled(" switch ", dim)]);
        }

        if w >= 35 {
            spans.extend([
                Span::styled("[1]", dim),
                Span::styled("-[4]", dim),
                Span::styled(" tabs", dim),
            ]);
        }

        if self.active_tab == Tab::Logs && w >= 58 {
            spans.extend([
                Span::styled(" [/]", dim),
                Span::styled(if self.show_debug_logs { "hide debug" } else { "show debug" }, dim),
            ]);
        }

        let shortcuts = Paragraph::new(Line::from(spans));
        frame.render_widget(shortcuts, area);
    }

    fn draw_logs_panel(&self, frame: &mut Frame, area: Rect) {
        let p = theme::palette();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5), // Header
                Constraint::Min(5),    // Log content
                Constraint::Length(1), // Shortcuts
            ])
            .split(area);

        self.draw_header(frame, chunks[0]);

        let visible_height = chunks[1].height as usize;

        if self.log_lines.is_empty() {
            // Generate live status lines from daemon state
            let mut lines = vec![
                Line::from(Span::styled(
                    "  LIVE STATUS",
                    Style::default().fg(p.dim).add_modifier(Modifier::BOLD),
                )),
                Line::from(""),
            ];

            if let Some(ref rx) = self.daemon_state_rx {
                let state = rx.borrow();
                let uptime = state.uptime_secs;
                let h = uptime / 3600;
                let m = (uptime % 3600) / 60;
                let s = uptime % 60;

                lines.push(Line::from(Span::styled(
                    format!("  [{h:02}:{m:02}:{s:02}] Daemon running"),
                    Style::default().fg(p.success),
                )));
                lines.push(Line::from(Span::styled(
                    format!("  [{h:02}:{m:02}:{s:02}] Idle state: {}", state.idle_state),
                    Style::default().fg(p.muted),
                )));
                lines.push(Line::from(Span::styled(
                    format!(
                        "  [{h:02}:{m:02}:{s:02}] CPU: {:.0}% | RAM: {:.1}GB",
                        state.live_metrics.cpu_usage, state.live_metrics.memory_used_gb
                    ),
                    Style::default().fg(p.muted),
                )));
                lines.push(Line::from(Span::styled(
                    format!("  [{h:02}:{m:02}:{s:02}] Inference: {}", state.inference_status),
                    Style::default().fg(p.muted),
                )));
                if state.pipeline.active {
                    lines.push(Line::from(Span::styled(
                        format!(
                            "  [{h:02}:{m:02}:{s:02}] Pipeline active: {:.1} tok/s",
                            state.pipeline.tokens_per_sec
                        ),
                        Style::default().fg(p.success),
                    )));
                }
            } else {
                lines.push(Line::from(Span::styled(
                    "  Daemon not connected",
                    Style::default().fg(p.dim),
                )));
            }

            let widget = Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .block(Block::default().borders(Borders::NONE));
            frame.render_widget(widget, chunks[1]);
        } else {
            // Show log file content with scrolling
            let wrap_width = chunks[1].width.saturating_sub(2).max(1) as usize;
            let mut visual_lines: Vec<Line> = Vec::new();
            for line in &self.log_lines {
                let lower = line.to_ascii_lowercase();
                if !self.show_debug_logs && is_debug_log_line(&lower) {
                    continue;
                }
                let color = if is_error_log_line(&lower) {
                    p.danger
                } else if is_warn_log_line(&lower) {
                    p.warning
                } else if is_debug_log_line(&lower) {
                    p.dim
                } else if is_info_log_line(&lower) {
                    p.muted
                } else {
                    p.dim
                };
                for wrapped in wrap_plain_line(&format!("  {line}"), wrap_width) {
                    visual_lines
                        .push(Line::from(Span::styled(wrapped, Style::default().fg(color))));
                }
            }

            let total = visual_lines.len();
            let max_scroll = total.saturating_sub(visible_height);
            let scroll = self.log_scroll.min(max_scroll);
            let end = if scroll == 0 { total } else { total.saturating_sub(scroll) };
            let start = end.saturating_sub(visible_height);

            let visible_lines: Vec<Line> = visual_lines[start..end].to_vec();

            let follow_label =
                if scroll > 0 { format!("scrolled +{}", scroll) } else { "following".into() };
            let debug_label = if self.show_debug_logs { "debug shown" } else { "debug hidden" };
            let scroll_indicator = format!(" ({follow_label} | {debug_label})");

            let logs = Paragraph::new(visible_lines)
                .block(
                    Block::default()
                        .borders(Borders::NONE)
                        .title(Span::styled(scroll_indicator, Style::default().fg(p.dim))),
                )
                .wrap(Wrap { trim: false });
            frame.render_widget(logs, chunks[1]);
        }

        self.draw_shortcuts(frame, chunks[2]);
    }

    /// Get a config value for display. Caller should cache the Config if calling multiple times.
    fn get_config_value_from(config: &compute_daemon::config::Config, key: &str) -> String {
        match key {
            "wallet.public_address" => {
                if config.wallet.public_address.is_empty() {
                    "(not set)".into()
                } else {
                    config.wallet.public_address.clone()
                }
            }
            "node.name" => config.node.name.clone(),
            "node.max_gpu_usage" => match config.node.max_gpu_usage {
                0..=40 => "Low".into(),
                41..=70 => "Medium".into(),
                _ => "High".into(),
            },
            "node.pause_on_battery" => {
                if config.node.pause_on_battery { "On" } else { "Off" }.into()
            }
            "node.pause_on_fullscreen" => {
                if config.node.pause_on_fullscreen { "On" } else { "Off" }.into()
            }
            "node.caffeinate_when_running" => {
                if config.node.caffeinate_when_running { "On" } else { "Off" }.into()
            }
            "models.auto_download" => if config.models.auto_download { "On" } else { "Off" }.into(),
            "service.autostart" => {
                if compute_daemon::service::is_service_installed() { "On" } else { "Off" }.into()
            }
            "network.region" => config.network.region.clone(),
            "appearance.theme" => match config.appearance.theme.as_str() {
                "system" => "System".into(),
                "light" => "Light".into(),
                _ => "Dark".into(),
            },
            _ => "?".into(),
        }
    }

    fn start_config_edit(&mut self) {
        let item = &CONFIG_ITEMS[self.config_selected];
        match item.kind {
            ConfigItemKind::Toggle => {
                // Toggle immediately
                let mut config = compute_daemon::config::Config::load().unwrap_or_default();
                match item.key {
                    "node.pause_on_battery" => {
                        config.node.pause_on_battery = !config.node.pause_on_battery
                    }
                    "node.pause_on_fullscreen" => {
                        config.node.pause_on_fullscreen = !config.node.pause_on_fullscreen
                    }
                    "node.caffeinate_when_running" => {
                        config.node.caffeinate_when_running = !config.node.caffeinate_when_running
                    }
                    "models.auto_download" => {
                        config.models.auto_download = !config.models.auto_download
                    }
                    "service.autostart" => {
                        if compute_daemon::service::is_service_installed() {
                            let _ = compute_daemon::service::uninstall_service();
                        } else {
                            let _ = compute_daemon::service::install_service();
                        }
                        return; // Don't save config for service
                    }
                    _ => {}
                }
                let _ = config.save();
            }
            ConfigItemKind::Choice => {
                // Cycle through options
                let mut config = compute_daemon::config::Config::load().unwrap_or_default();
                match item.key {
                    "node.max_gpu_usage" => {
                        config.node.max_gpu_usage = match config.node.max_gpu_usage {
                            0..=40 => 70,  // Low → Medium
                            41..=70 => 90, // Medium → High
                            _ => 30,       // High → Low
                        };
                        // Scale CPU proportionally
                        config.node.max_cpu_usage = match config.node.max_gpu_usage {
                            0..=40 => 25,
                            41..=70 => 50,
                            _ => 80,
                        };
                    }
                    "appearance.theme" => {
                        config.appearance.theme = match config.appearance.theme.as_str() {
                            "system" => "dark".into(),
                            "light" => "dark".into(),
                            "dark" => "light".into(),
                            _ => "system".into(),
                        };
                    }
                    _ => {}
                }
                let _ = config.save();
            }
            ConfigItemKind::Text => {
                self.config_editing = true;
                let config = compute_daemon::config::Config::load().unwrap_or_default();
                let current = Self::get_config_value_from(&config, item.key);
                self.config_edit_buffer =
                    if current == "(not set)" { String::new() } else { current };
            }
        }
    }

    fn handle_config_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc => {
                self.config_editing = false;
                self.config_edit_buffer.clear();
            }
            KeyCode::Enter => {
                let item = &CONFIG_ITEMS[self.config_selected];
                let value = self.config_edit_buffer.clone();
                self.config_editing = false;

                if item.key == "wallet.public_address" {
                    if !value.is_empty() && !compute_solana::is_valid_address(&value) {
                        return; // Invalid address, ignore
                    }
                    // Show confirmation
                    self.config_confirm = Some(value);
                } else {
                    self.apply_config_change(item.key, &value);
                }
                self.config_edit_buffer.clear();
            }
            KeyCode::Char(c) => {
                self.config_edit_buffer.push(c);
            }
            KeyCode::Backspace => {
                self.config_edit_buffer.pop();
            }
            _ => {}
        }
    }

    fn apply_config_change(&self, key: &str, value: &str) {
        let mut config = compute_daemon::config::Config::load().unwrap_or_default();
        match key {
            "node.name" => config.node.name = value.to_string(),
            "network.region" => config.network.region = value.to_string(),
            _ => {}
        }
        let _ = config.save();
    }

    fn apply_wallet_change(&self, new_wallet: &str) {
        let mut config = compute_daemon::config::Config::load().unwrap_or_default();
        config.wallet.public_address = new_wallet.to_string();
        config.wallet.node_id.clear();
        config.wallet.node_token.clear();
        let _ = config.save();
    }

    fn draw_config_panel(&self, frame: &mut Frame, area: Rect) {
        let p = theme::palette();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5), // Header
                Constraint::Min(5),    // Config content
                Constraint::Length(1), // Shortcuts
            ])
            .split(area);

        self.draw_header(frame, chunks[0]);

        let mut lines = vec![
            Line::from(Span::styled(
                "  SETTINGS",
                Style::default().fg(p.dim).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ];

        // Confirmation dialog
        if let Some(ref wallet) = self.config_confirm {
            lines.push(Line::from(Span::styled(
                "  Are you sure you want to change your wallet address?",
                Style::default().fg(p.warning),
            )));
            lines.push(Line::from(Span::styled(
                format!("  New: {wallet}"),
                Style::default().fg(p.text),
            )));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "  [Y] Confirm   [N] Cancel",
                Style::default().fg(p.dim),
            )));
            let widget = Paragraph::new(lines);
            frame.render_widget(widget, chunks[1]);
            self.draw_shortcuts(frame, chunks[2]);
            return;
        }

        // Load config once for all items (not once per item — was doing 7 disk reads per frame!)
        let cached_config = compute_daemon::config::Config::load().unwrap_or_default();

        for (i, item) in CONFIG_ITEMS.iter().enumerate() {
            let is_selected = i == self.config_selected;
            let value = if self.config_editing && is_selected {
                format!("{}█", self.config_edit_buffer)
            } else {
                Self::get_config_value_from(&cached_config, item.key)
            };

            let arrow = if is_selected { "▸ " } else { "  " };
            let label_color = if is_selected { p.text } else { p.dim };
            let value_color = if is_selected {
                if self.config_editing { p.warning } else { p.text }
            } else {
                p.muted
            };

            let hint = match item.kind {
                ConfigItemKind::Toggle if is_selected => "  ↵ toggle",
                ConfigItemKind::Choice if is_selected => "  ↵ cycle",
                ConfigItemKind::Text if is_selected => "  ↵ edit",
                _ => "",
            };

            lines.push(Line::from(vec![
                Span::styled(
                    format!("{arrow}{:<22}", item.label),
                    Style::default().fg(label_color),
                ),
                Span::styled(value, Style::default().fg(value_color)),
                Span::styled(hint, Style::default().fg(p.dim)),
            ]));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  ↑↓ Navigate   ↵ Edit/Toggle   Esc Cancel",
            Style::default().fg(p.dim),
        )));

        let config_widget = Paragraph::new(lines);
        frame.render_widget(config_widget, chunks[1]);

        self.draw_shortcuts(frame, chunks[2]);
    }

    fn select_model(&mut self) {
        let models = model_entries();
        let entry = match models.get(self.models_selected) {
            Some(e) => e,
            None => return,
        };

        // "auto" doesn't need a download
        if entry.id == "auto" {
            let mut config = compute_daemon::config::Config::load().unwrap_or_default();
            config.models.active_model = "auto".into();
            let _ = config.save();
            return;
        }

        // Already downloaded → just activate
        if is_model_downloaded(entry.id) {
            let mut config = compute_daemon::config::Config::load().unwrap_or_default();
            config.models.active_model = entry.id.into();
            let _ = config.save();
            return;
        }

        // Not downloaded → start download in background
        if matches!(self.model_downloads.get(entry.id), Some(progress) if *progress >= 0.0 && *progress < 1.0)
        {
            return; // this model is already downloading
        }

        let (tx, rx) = std::sync::mpsc::channel();
        self.download_progress_rxs.push(rx);
        self.model_downloads.insert(entry.id.into(), 0.0);

        let url = entry.hf_url.to_string();
        let filename = entry.gguf_filename.to_string();
        let model_id = entry.id.to_string();
        let is_stage = is_stage_entry(entry.id);
        let mlx_repo_id = entry.mlx_repo_id.map(str::to_string);
        let mlx_folder = entry.mlx_folder.map(str::to_string);
        let mlx_total_size_mb = entry.mlx_total_size_mb;
        let download_mlx = uses_mlx_snapshot(entry);

        std::thread::spawn(move || {
            if download_mlx {
                let Some(repo_id) = mlx_repo_id else {
                    let _ = tx.send((model_id, -1.0));
                    return;
                };
                let Some(folder) = mlx_folder else {
                    let _ = tx.send((model_id, -1.0));
                    return;
                };
                let cache_dir = models_cache_dir();
                let dest = cache_dir.join("mlx").join(&folder);
                let total_bytes =
                    mlx_total_size_mb.unwrap_or(0).saturating_mul(1024).saturating_mul(1024);
                let stop_progress = Arc::new(AtomicBool::new(false));
                let progress_handle = if total_bytes > 0 {
                    let stop = stop_progress.clone();
                    let tx_progress = tx.clone();
                    let model_id_progress = model_id.clone();
                    let dest_progress = dest.clone();
                    Some(std::thread::spawn(move || {
                        let mut last_bucket = -1i32;
                        while !stop.load(Ordering::Relaxed) {
                            let bytes = dir_size_bytes(&dest_progress);
                            if bytes > 0 {
                                let pct = (bytes as f64 / total_bytes as f64).clamp(0.01, 0.99);
                                let bucket = (pct * 100.0).floor() as i32;
                                if bucket != last_bucket {
                                    let _ = tx_progress.send((model_id_progress.clone(), pct));
                                    last_bucket = bucket;
                                }
                            }
                            std::thread::sleep(std::time::Duration::from_millis(750));
                        }
                    }))
                } else {
                    None
                };
                let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                    Ok(rt) => rt,
                    Err(_) => {
                        stop_progress.store(true, Ordering::Relaxed);
                        if let Some(handle) = progress_handle {
                            let _ = handle.join();
                        }
                        let _ = tx.send((model_id, -1.0));
                        return;
                    }
                };

                let result =
                    rt.block_on(compute_daemon::inference::stage_artifacts::ensure_mlx_snapshot(
                        cache_dir.as_path(),
                        &repo_id,
                        &folder,
                    ));
                stop_progress.store(true, Ordering::Relaxed);
                if let Some(handle) = progress_handle {
                    let _ = handle.join();
                }

                match result {
                    Ok(_) => {
                        let _ = tx.send((model_id.clone(), 1.0));
                        let mut config = compute_daemon::config::Config::load().unwrap_or_default();
                        config.models.active_model = model_id;
                        let _ = config.save();
                    }
                    Err(err) => {
                        tracing::warn!("[download] MLX snapshot failed: {err}");
                        let _ = tx.send((model_id, -1.0));
                    }
                }
                return;
            }

            let (_cache_dir, dest, tmp) = if is_stage {
                let dest = stage_shard_path(&filename);
                let parent = dest.parent().unwrap_or(std::path::Path::new("")).to_path_buf();
                let _ = std::fs::create_dir_all(&parent);
                let tmp = dest.with_extension("tmp");
                (parent, dest, tmp)
            } else {
                let dir = models_cache_dir();
                let _ = std::fs::create_dir_all(&dir);
                let dest = dir.join(&filename);
                let tmp = dest.with_extension("tmp");
                (dir, dest, tmp)
            };

            let client = match reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(7200))
                .connect_timeout(std::time::Duration::from_secs(30))
                .build()
            {
                Ok(c) => c,
                Err(_) => {
                    let _ = tx.send((model_id, -1.0));
                    return;
                }
            };

            // Resume support: check if a partial .tmp file exists
            let mut downloaded: u64 = std::fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);

            // Retry loop: on network interruption (sleep/wake), retry up to 10 times
            let max_retries = 10;
            let mut total: u64 = 0;

            for attempt in 0..=max_retries {
                // Build request with Range header for resume
                let mut req = client.get(&url);
                if downloaded > 0 {
                    req = req.header("Range", format!("bytes={downloaded}-"));
                    tracing::info!(
                        "[download] Resuming from {:.1} MB (attempt {})",
                        downloaded as f64 / 1048576.0,
                        attempt + 1
                    );
                }

                let resp = match req.send() {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!("[download] Request failed: {e}");
                        if attempt < max_retries {
                            std::thread::sleep(std::time::Duration::from_secs(
                                2u64.pow(attempt.min(5)),
                            ));
                            continue;
                        }
                        let _ = tx.send((model_id, -1.0));
                        return;
                    }
                };

                let status = resp.status();
                if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
                    // File is already complete or server doesn't support range
                    // Check if tmp is actually complete by re-downloading fresh
                    downloaded = 0;
                    let _ = std::fs::remove_file(&tmp);
                    continue;
                }

                if !status.is_success() && status != reqwest::StatusCode::PARTIAL_CONTENT {
                    if attempt < max_retries {
                        std::thread::sleep(std::time::Duration::from_secs(2));
                        continue;
                    }
                    let _ = tx.send((model_id, -1.0));
                    return;
                }

                // Determine total file size
                if status == reqwest::StatusCode::PARTIAL_CONTENT {
                    // Parse Content-Range: bytes 12345-99999/100000
                    if let Some(range) = resp.headers().get("content-range") {
                        if let Ok(s) = range.to_str() {
                            if let Some(slash) = s.rfind('/') {
                                total = s[slash + 1..].parse().unwrap_or(0);
                            }
                        }
                    }
                } else {
                    total = resp.content_length().unwrap_or(0);
                    // Fresh download (not partial) — reset
                    downloaded = 0;
                }

                // Open file for append (resume) or create (fresh)
                use std::io::{Read, Write};
                let mut file =
                    match std::fs::OpenOptions::new().create(true).append(true).open(&tmp) {
                        Ok(f) => f,
                        Err(_) => {
                            let _ = tx.send((model_id, -1.0));
                            return;
                        }
                    };

                let mut reader = std::io::BufReader::with_capacity(1024 * 1024, resp);
                let mut buf = vec![0u8; 256 * 1024]; // 256KB chunks
                let mut stall_count = 0u32;

                loop {
                    let n = match reader.read(&mut buf) {
                        Ok(0) => break, // Stream finished
                        Ok(n) => {
                            stall_count = 0;
                            n
                        }
                        Err(e) => {
                            // Network interruption (e.g. Mac sleep) — break to retry
                            tracing::warn!("[download] Read error: {e}");
                            stall_count += 1;
                            if stall_count >= 3 {
                                break; // Will retry from outer loop
                            }
                            std::thread::sleep(std::time::Duration::from_secs(1));
                            continue;
                        }
                    };
                    let _ = file.write_all(&buf[..n]);
                    downloaded += n as u64;

                    // Send progress every ~1MB
                    if total > 0 && downloaded % (1024 * 1024) < n as u64 {
                        let pct = downloaded as f64 / total as f64;
                        let _ = tx.send((model_id.clone(), pct));
                    }
                }

                let _ = file.flush();
                drop(file);

                // Check if download is complete
                if total > 0 && downloaded >= total {
                    break; // Success — exit retry loop
                }

                // Incomplete — retry after backoff
                if attempt < max_retries {
                    tracing::info!(
                        "[download] Incomplete ({:.1}/{:.1} MB), retrying in {}s...",
                        downloaded as f64 / 1048576.0,
                        total as f64 / 1048576.0,
                        2u64.pow(attempt.min(5))
                    );
                    let _ = tx.send((model_id.clone(), downloaded as f64 / total.max(1) as f64));
                    std::thread::sleep(std::time::Duration::from_secs(2u64.pow(attempt.min(5))));
                } else {
                    let _ = tx.send((model_id, -1.0));
                    return;
                }
            }

            // Verify download size matches expected (total is from Content-Range or Content-Length)
            if total > 0 {
                if let Ok(meta) = std::fs::metadata(&tmp) {
                    if meta.len() != total {
                        tracing::warn!(
                            "[download] Size mismatch: expected {} bytes, got {}",
                            total,
                            meta.len()
                        );
                        let _ = std::fs::remove_file(&tmp);
                        let _ = tx.send((model_id, -1.0));
                        return;
                    }
                }
            }

            // Verify GGUF magic header (stage shards are plain GGUF files)
            {
                use std::io::Read;
                if let Ok(mut f) = std::fs::File::open(&tmp) {
                    let mut magic = [0u8; 4];
                    if f.read_exact(&mut magic).is_err() || magic != [0x47, 0x47, 0x55, 0x46] {
                        let _ = std::fs::remove_file(&tmp);
                        let _ = tx.send((model_id, -1.0));
                        return;
                    }
                }
            }
            let _ = std::fs::rename(&tmp, &dest);
            let _ = tx.send((model_id.clone(), 1.0));
            // Only flip active_model for full-model entries — stage shards are
            // a capability advertisement, not a default backend selection.
            if !is_stage {
                let mut config = compute_daemon::config::Config::load().unwrap_or_default();
                config.models.active_model = model_id;
                let _ = config.save();
            }
        });
    }

    fn delete_model(&mut self) {
        let models = model_entries();
        let entry = match models.get(self.models_selected) {
            Some(e) => e,
            None => return,
        };

        if entry.id == "auto" || entry.gguf_filename.is_empty() {
            return;
        }

        if matches!(self.model_downloads.get(entry.id), Some(progress) if *progress >= 0.0 && *progress < 1.0)
        {
            return;
        }

        if uses_mlx_snapshot(entry) {
            if let Some(path) = mlx_snapshot_path(entry) {
                let _ = std::fs::remove_dir_all(&path);
            }
            let mut config = compute_daemon::config::Config::load().unwrap_or_default();
            if config.models.active_model == entry.id {
                config.models.active_model = "auto".into();
                let _ = config.save();
            }
            return;
        }

        let path = if is_stage_entry(entry.id) {
            stage_shard_path(entry.gguf_filename)
        } else {
            models_cache_dir().join(entry.gguf_filename)
        };

        let tmp = path.with_extension("tmp");
        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_file(&path);

        // If we just deleted the active full model, fall back to auto.
        if !is_stage_entry(entry.id) {
            let mut config = compute_daemon::config::Config::load().unwrap_or_default();
            if config.models.active_model == entry.id {
                config.models.active_model = "auto".into();
                let _ = config.save();
            }
        }
    }

    fn draw_models_panel(&self, frame: &mut Frame, area: Rect) {
        let p = theme::palette();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5), // Header (reuse shared header)
                Constraint::Min(0),
                Constraint::Length(2),
            ])
            .split(area);

        self.draw_header(frame, chunks[0]);

        // Model list
        let config = compute_daemon::config::Config::load().unwrap_or_default();
        let active = config.models.active_model;
        let models = model_entries();

        let mut lines: Vec<Line> = Vec::new();
        lines.push(Line::from(Span::styled(
            "  MODELS",
            Style::default().fg(p.dim).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));

        for (i, entry) in models.iter().enumerate() {
            let is_selected = i == self.models_selected;
            let is_active = entry.id == active.as_str();
            let downloaded = is_model_downloaded(entry.id);

            let arrow = if is_selected { "▸ " } else { "  " };

            let check = if is_active {
                " ●"
            } else if !downloaded && entry.id != "auto" {
                " ↓"
            } else {
                "  "
            };

            let style = if !downloaded && entry.id != "auto" {
                // Grey out undownloaded models
                if is_selected {
                    Style::default().fg(p.dim).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(p.dim)
                }
            } else if is_selected {
                Style::default().fg(p.text).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(p.muted)
            };

            let check_style = if is_active {
                Style::default().fg(p.success)
            } else if !downloaded && entry.id != "auto" {
                Style::default().fg(p.dim)
            } else {
                Style::default().fg(p.dim)
            };

            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(arrow, style),
                Span::styled(entry.label, style),
                Span::styled(check, check_style),
            ]));

            let desc_text = if !downloaded && entry.id != "auto" {
                format!("{} · not downloaded", entry.desc)
            } else {
                entry.desc.to_string()
            };
            lines.push(Line::from(vec![
                Span::raw("      "),
                Span::styled(desc_text, Style::default().fg(p.dim)),
            ]));
            lines.push(Line::from(""));
        }

        // Download progress
        for (model_id, progress) in &self.model_downloads {
            let bar_width = 30;
            if *progress < 0.0 {
                lines.push(Line::from(Span::styled(
                    format!("  Download failed: {}", model_id),
                    Style::default().fg(p.danger),
                )));
            } else {
                let filled = (*progress * bar_width as f64).clamp(0.0, bar_width as f64) as usize;
                let empty = bar_width - filled;
                let pct = *progress * 100.0;
                lines.push(Line::from(vec![Span::styled(
                    format!(
                        "  Downloading {}  [{}{}] {:.0}%",
                        model_id,
                        "█".repeat(filled),
                        "░".repeat(empty),
                        pct
                    ),
                    Style::default().fg(p.text),
                )]));
            }
        }

        let list = Paragraph::new(lines).wrap(Wrap { trim: false });
        frame.render_widget(list, chunks[1]);

        // Footer
        let footer = Paragraph::new(Line::from(vec![
            Span::raw("  "),
            Span::styled("↑↓", Style::default().fg(p.dim)),
            Span::styled(" navigate  ", Style::default().fg(p.dim)),
            Span::styled("↵", Style::default().fg(p.dim)),
            Span::styled(" select/download  ", Style::default().fg(p.dim)),
            Span::styled("d", Style::default().fg(p.dim)),
            Span::styled(" delete", Style::default().fg(p.dim)),
        ]));
        frame.render_widget(footer, chunks[2]);
    }
}

fn load_recent_logs(n: usize) -> Vec<String> {
    let log_path = compute_daemon::config::logs_dir().ok().map(|d| d.join("compute.log"));

    match log_path {
        Some(path) if path.exists() => {
            let contents = std::fs::read_to_string(&path).unwrap_or_default();
            let lines: Vec<String> = contents.lines().map(String::from).collect();
            let start = lines.len().saturating_sub(n);
            lines[start..].to_vec()
        }
        _ => Vec::new(), // Empty = show live daemon status instead
    }
}

fn wrap_plain_line(line: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut wrapped = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;

    for ch in line.chars() {
        if current_width >= width {
            wrapped.push(current);
            current = String::new();
            current_width = 0;
        }
        current.push(ch);
        current_width += 1;
    }

    if current.is_empty() {
        wrapped.push(String::new());
    } else {
        wrapped.push(current);
    }

    wrapped
}

fn is_error_log_line(lower: &str) -> bool {
    lower.contains(" error ") || lower.contains(" error:")
}

fn is_warn_log_line(lower: &str) -> bool {
    lower.contains(" warn ") || lower.contains(" warn:")
}

fn is_debug_log_line(lower: &str) -> bool {
    lower.contains(" debug ") || lower.contains(" debug:")
}

fn is_info_log_line(lower: &str) -> bool {
    lower.contains(" info ") || lower.contains(" info:")
}

// --- Helpers ---

fn format_vram(vram_mb: u64) -> String {
    if vram_mb >= 1024 { format!("{}GB", vram_mb / 1024) } else { format!("{vram_mb}MB") }
}

fn format_number(n: u32) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{},{:03}", n / 1_000, n % 1_000)
    } else {
        n.to_string()
    }
}

fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let mins = (secs % 3600) / 60;

    let s = secs % 60;

    if days > 0 {
        format!("{days}d {hours}h {mins}m")
    } else if hours > 0 {
        format!("{hours}h {mins}m")
    } else if mins > 0 {
        format!("{mins}m {s}s")
    } else {
        format!("{s}s")
    }
}

fn progress_bar(value: f64, max: f64, width: usize) -> String {
    let ratio = (value / max).clamp(0.0, 1.0);
    let filled = (ratio * width as f64).round() as usize;
    let empty = width - filled;
    format!("{}{}", "▓".repeat(filled), "░".repeat(empty))
}

/// Open the claim page in the user's default browser.
fn open_claim_page() {
    let config = compute_daemon::config::Config::load().unwrap_or_default();
    let wallet = &config.wallet.public_address;

    let url = if wallet.is_empty() {
        "https://computenetwork.sh/?connect=true".to_string()
    } else {
        format!("https://computenetwork.sh/?connect=true&wallet={wallet}")
    };

    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(&url).spawn();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("xdg-open").arg(&url).spawn();
    }
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd").args(["/C", "start", &url]).spawn();
    }
}

/// Fetch network stats and online nodes from the orchestrator.
/// Returns (stats, list of (wallet, region) for globe).
fn fetch_network_and_nodes() -> (NetworkStats, Vec<(String, Option<String>)>) {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build();
    match rt {
        Ok(rt) => rt.block_on(async {
            let client = compute_network::client::OrchestratorClient::new(
                "https://api.computenetwork.sh",
                None,
            );

            let stats = match client.get_network_stats().await {
                Ok(s) => NetworkStats {
                    total_nodes: s.total_nodes as u32,
                    peak_petaflops: s.total_nodes as f64 * 0.066,
                },
                Err(_) => NetworkStats::default(),
            };

            let nodes = match client.get_online_nodes().await {
                Ok(n) => n.into_iter().map(|node| (node.wallet_address, node.region)).collect(),
                Err(_) => Vec::new(),
            };

            (stats, nodes)
        }),
        Err(_) => (NetworkStats::default(), Vec::new()),
    }
}
