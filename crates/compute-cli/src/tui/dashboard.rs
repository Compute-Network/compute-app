use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::{
    DefaultTerminal, Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Sparkline},
};

use compute_daemon::hardware::{self, HardwareInfo, LiveMetrics};
use compute_daemon::metrics::{Earnings, NetworkStats, PipelineStatus};

use super::globe::Globe;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Overview,
    Logs,
    Config,
}

pub struct Dashboard {
    globe: Globe,
    hardware: HardwareInfo,
    earnings: Earnings,
    pipeline: PipelineStatus,
    network: NetworkStats,
    throughput_history: Vec<u64>,
    sys: sysinfo::System,
    live_metrics: LiveMetrics,
    last_metrics_update: Instant,
    last_network_update: Instant,
    uptime_start: Instant,
    active_tab: Tab,
    log_lines: Vec<String>,
    daemon_state_rx: Option<tokio::sync::watch::Receiver<compute_daemon::runtime::DaemonState>>,
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

        // Fetch real nodes from Supabase for globe visualization
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
            earnings: Earnings::mock(),
            pipeline: PipelineStatus::mock(),
            network,
            throughput_history: generate_mock_throughput(),
            sys: sysinfo::System::new_all(),
            live_metrics: LiveMetrics::default(),
            last_metrics_update: Instant::now(),
            last_network_update: Instant::now(),
            uptime_start: Instant::now(),
            active_tab: Tab::Overview,
            log_lines,
            daemon_state_rx: None,
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
                    KeyCode::Char('c') => {
                        open_claim_page();
                    }
                    KeyCode::Tab => {
                        self.active_tab = match self.active_tab {
                            Tab::Overview => Tab::Logs,
                            Tab::Logs => Tab::Config,
                            Tab::Config => Tab::Overview,
                        };
                        if self.active_tab == Tab::Logs {
                            self.log_lines = load_recent_logs(100);
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

        // Read from daemon state if available
        if let Some(ref rx) = self.daemon_state_rx {
            let state = rx.borrow().clone();
            self.live_metrics = state.live_metrics;
            self.earnings.pending = state.earnings.pending;

            if self.throughput_history.len() > 60 {
                self.throughput_history.remove(0);
            }
            // Use tokens_per_sec from daemon state (updated by inference manager)
            let tps = state.pipeline.tokens_per_sec as u64;
            self.throughput_history.push(tps);

            self.pipeline = state.pipeline;
        } else {
            // No daemon state — use local metrics collection
            if self.last_metrics_update.elapsed() > Duration::from_secs(2) {
                self.live_metrics = hardware::collect_live_metrics(&mut self.sys);
                self.last_metrics_update = Instant::now();
            }

            // Simulated throughput
            if self.throughput_history.len() > 60 {
                self.throughput_history.remove(0);
            }
            let last = *self.throughput_history.last().unwrap_or(&40);
            let jitter = (rand_simple() * 10.0 - 5.0) as i64;
            let new_val = (last as i64 + jitter).clamp(0, 80) as u64;
            self.throughput_history.push(new_val);
        }

        // Refresh network stats and nodes from Supabase every 60 seconds
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
            self.globe.render(chunks[0], frame.buffer_mut());
            self.draw_right_panel(frame, chunks[1]);
        } else {
            // Narrow/mobile: content only, no globe
            self.draw_right_panel(frame, area);
        }
    }

    fn draw_globe_panel(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(10),   // Globe
                Constraint::Length(4), // Network stats
            ])
            .split(area);

        // Globe
        let globe_block = Block::default()
            .borders(Borders::RIGHT)
            .border_style(Style::default().fg(Color::DarkGray));
        let inner = globe_block.inner(chunks[0]);
        frame.render_widget(globe_block, chunks[0]);
        self.globe.render(inner, frame.buffer_mut());

        // Network stats below globe
        let stats_block = Block::default()
            .borders(Borders::RIGHT | Borders::TOP)
            .border_style(Style::default().fg(Color::DarkGray));
        let stats_inner = stats_block.inner(chunks[1]);
        frame.render_widget(stats_block, chunks[1]);

        let stats = Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    format!("  {:>6} ", format_number(self.network.total_nodes)),
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ),
                Span::styled("nodes", Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("  ● ", Style::default().fg(Color::Green)),
                Span::styled(
                    format!("{:.0} PF", self.network.peak_petaflops),
                    Style::default().fg(Color::White),
                ),
                Span::styled(" peak", Style::default().fg(Color::DarkGray)),
            ]),
        ]);
        frame.render_widget(stats, stats_inner);
    }

    fn draw_right_panel(&self, frame: &mut Frame, area: Rect) {
        match self.active_tab {
            Tab::Overview => self.draw_overview_panel(frame, area),
            Tab::Logs => self.draw_logs_panel(frame, area),
            Tab::Config => self.draw_config_panel(frame, area),
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
        let w = area.width as usize;
        let (status_color, status_text) =
            if self.pipeline.active && self.pipeline.tokens_per_sec > 0.0 {
                (Color::Green, "ACTIVE")
            } else if self.pipeline.active {
                (Color::Green, "ONLINE")
            } else {
                (Color::DarkGray, "IDLE")
            };

        let tab_style = |tab: Tab| {
            if tab == self.active_tab {
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            }
        };

        let mut lines: Vec<Line> = Vec::new();
        lines.push(Line::from(""));

        // Title line — always fits
        lines.push(Line::from(vec![Span::styled(
            "  C O M P U T E",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        )]));

        // Subtitle + status: combine on one line if wide enough, split if narrow
        if w >= 55 {
            lines.push(Line::from(vec![
                Span::styled(
                    "  Decentralized GPU Infrastructure",
                    Style::default().fg(Color::DarkGray),
                ),
                Span::raw("  "),
                Span::styled("● ", Style::default().fg(status_color)),
                Span::styled(status_text, Style::default().fg(status_color)),
                Span::raw("  "),
                Span::styled(format!("v{VERSION}"), Style::default().fg(Color::DarkGray)),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled("● ", Style::default().fg(status_color)),
                Span::styled(status_text, Style::default().fg(status_color)),
                Span::raw("  "),
                Span::styled(format!("v{VERSION}"), Style::default().fg(Color::DarkGray)),
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
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::raw(" "),
                Span::styled("[1]OVR", tab_style(Tab::Overview)),
                Span::raw(" "),
                Span::styled("[2]LOG", tab_style(Tab::Logs)),
                Span::raw(" "),
                Span::styled("[3]CFG", tab_style(Tab::Config)),
            ]));
        }

        let header = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        frame.render_widget(header, area);
    }

    fn draw_node_info(&self, frame: &mut Frame, area: Rect) {
        let gpu_name = self
            .hardware
            .gpus
            .first()
            .map(|g| format!("{} ({})", g.name, format_vram(g.vram_mb)))
            .unwrap_or_else(|| "No GPU".into());

        let pipeline_stage =
            if let (Some(s), Some(t)) = (self.pipeline.stage, self.pipeline.total_stages) {
                format!("Pipeline Stage {s}/{t}")
            } else {
                "Not assigned".into()
            };

        let uptime = format_duration(self.uptime_start.elapsed());

        let cpu_bar = progress_bar(self.live_metrics.cpu_usage as f64, 100.0, 10);

        let lines = vec![
            Line::from(Span::styled(
                "  NODE",
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("  GPU     ", Style::default().fg(Color::DarkGray)),
                Span::styled(&gpu_name, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  Stage   ", Style::default().fg(Color::DarkGray)),
                Span::styled(&pipeline_stage, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  Uptime  ", Style::default().fg(Color::DarkGray)),
                Span::styled(&uptime, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  CPU     ", Style::default().fg(Color::DarkGray)),
                Span::styled(&cpu_bar, Style::default().fg(Color::White)),
                Span::styled(
                    format!(" {:.0}%", self.live_metrics.cpu_usage),
                    Style::default().fg(Color::DarkGray),
                ),
            ]),
        ];

        let widget = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        frame.render_widget(widget, area);
    }

    fn draw_earnings(&self, frame: &mut Frame, area: Rect) {
        let e = &self.earnings;
        let lines = vec![
            Line::from(Span::styled(
                "  EARNINGS",
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Today      ", Style::default().fg(Color::DarkGray)),
                Span::styled(format!("{:.1} $COMPUTE", e.today), Style::default().fg(Color::White)),
                Span::styled(
                    format!("    ≈ ${:.2}", e.today * e.usd_rate),
                    Style::default().fg(Color::DarkGray),
                ),
            ]),
            Line::from(vec![
                Span::styled("  This Week  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.1} $COMPUTE", e.this_week),
                    Style::default().fg(Color::White),
                ),
                Span::styled(
                    format!("    ≈ ${:.2}", e.this_week * e.usd_rate),
                    Style::default().fg(Color::DarkGray),
                ),
            ]),
            Line::from(vec![
                Span::styled("  All Time   ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.0} $COMPUTE", e.all_time),
                    Style::default().fg(Color::White),
                ),
            ]),
            Line::from(vec![
                Span::styled("  Pending    ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.1} $COMPUTE", e.pending),
                    Style::default().fg(Color::Yellow),
                ),
                Span::styled("  [c]laim", Style::default().fg(Color::DarkGray)),
            ]),
        ];

        let widget = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        frame.render_widget(widget, area);
    }

    fn draw_workload(&self, frame: &mut Frame, area: Rect) {
        let p = &self.pipeline;
        let model = p.model.as_deref().unwrap_or("None");

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
            self.live_metrics.gpu_temp.map(|t| format!("{t}°C")).unwrap_or_else(|| "--".into());

        let power_line =
            match (self.live_metrics.gpu_power_watts, self.live_metrics.gpu_power_limit_watts) {
                (Some(p), Some(l)) => format!("{p}W / {l}W"),
                _ => "--".into(),
            };

        let lines = vec![
            Line::from(Span::styled(
                "  WORKLOAD",
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Model     ", Style::default().fg(Color::DarkGray)),
                Span::styled(model, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  Served    ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{} requests", format_number(p.requests_served as u32)),
                    Style::default().fg(Color::White),
                ),
            ]),
            Line::from(vec![
                Span::styled("  Latency   ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.0}ms avg", p.avg_latency_ms),
                    Style::default().fg(Color::White),
                ),
            ]),
            Line::from(vec![
                Span::styled("  VRAM      ", Style::default().fg(Color::DarkGray)),
                Span::styled(&vram_line, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled("  Temp      ", Style::default().fg(Color::DarkGray)),
                Span::styled(&temp_line, Style::default().fg(Color::White)),
                Span::styled("    Power  ", Style::default().fg(Color::DarkGray)),
                Span::styled(&power_line, Style::default().fg(Color::White)),
            ]),
        ];

        let widget = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        frame.render_widget(widget, area);
    }

    fn draw_throughput(&self, frame: &mut Frame, area: Rect) {
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
            Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD),
        )));
        frame.render_widget(label, chunks[0]);

        // Ensure a visible baseline — minimum value is 1, max scales to data
        let display_data: Vec<u64> =
            self.throughput_history.iter().map(|&v| if v == 0 { 1 } else { v }).collect();

        let data_max = display_data.iter().copied().max().unwrap_or(1).max(8);

        let sparkline = Sparkline::default()
            .data(&display_data)
            .max(data_max)
            .style(Style::default().fg(Color::White));

        let sparkline_area =
            Rect { x: chunks[2].x + 2, width: chunks[2].width.saturating_sub(4), ..chunks[2] };
        frame.render_widget(sparkline, sparkline_area);

        // Show actual value (0 when idle, not 1)
        let current_tps = self.throughput_history.last().unwrap_or(&0);
        let tps_label = Paragraph::new(Line::from(vec![
            Span::raw("  "),
            Span::styled(
                format!("{current_tps} tok/s"),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
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
        let dim = Style::default().fg(Color::DarkGray);

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
                Span::styled("-[3]", dim),
                Span::styled(" tabs", dim),
            ]);
        }

        let shortcuts = Paragraph::new(Line::from(spans));
        frame.render_widget(shortcuts, area);
    }

    fn draw_logs_panel(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5), // Header
                Constraint::Min(5),    // Log content
                Constraint::Length(1), // Shortcuts
            ])
            .split(area);

        self.draw_header(frame, chunks[0]);

        // Log lines
        let visible_height = chunks[1].height as usize;
        let start = self.log_lines.len().saturating_sub(visible_height);
        let visible_lines: Vec<Line> = self.log_lines[start..]
            .iter()
            .map(|line| {
                Line::from(Span::styled(format!("  {line}"), Style::default().fg(Color::Gray)))
            })
            .collect();

        let logs = Paragraph::new(visible_lines).block(Block::default().borders(Borders::NONE));
        frame.render_widget(logs, chunks[1]);

        self.draw_shortcuts(frame, chunks[2]);
    }

    fn draw_config_panel(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5), // Header
                Constraint::Min(5),    // Config content
                Constraint::Length(1), // Shortcuts
            ])
            .split(area);

        self.draw_header(frame, chunks[0]);

        // Show current config
        let config = compute_daemon::config::Config::load().unwrap_or_default();
        let lines = vec![
            Line::from(Span::styled(
                "  CONFIGURATION",
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            config_line("  node.name", &config.node.name),
            config_line("  node.max_gpu_usage", &format!("{}%", config.node.max_gpu_usage)),
            config_line("  node.max_cpu_usage", &format!("{}%", config.node.max_cpu_usage)),
            config_line(
                "  node.idle_threshold",
                &format!("{} min", config.node.idle_threshold_minutes),
            ),
            config_line("  node.pause_on_battery", &config.node.pause_on_battery.to_string()),
            Line::from(""),
            config_line(
                "  wallet.public_address",
                if config.wallet.public_address.is_empty() {
                    "(not set)"
                } else {
                    &config.wallet.public_address
                },
            ),
            Line::from(""),
            config_line("  network.orchestrator", &config.network.orchestrator_url),
            config_line("  network.region", &config.network.region),
            Line::from(""),
            config_line("  logging.level", &config.logging.level),
            Line::from(""),
            Line::from(Span::styled(
                "  Edit with: compute config set <key> <value>",
                Style::default().fg(Color::DarkGray),
            )),
        ];

        let config_widget = Paragraph::new(lines);
        frame.render_widget(config_widget, chunks[1]);

        self.draw_shortcuts(frame, chunks[2]);
    }
}

fn config_line(key: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{key:<30}"), Style::default().fg(Color::DarkGray)),
        Span::styled(value.to_string(), Style::default().fg(Color::White)),
    ])
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
        _ => vec!["  No log file found. Start the daemon to generate logs.".into()],
    }
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

    if days > 0 {
        format!("{days}d {hours}h {mins}m")
    } else if hours > 0 {
        format!("{hours}h {mins}m")
    } else {
        format!("{mins}m")
    }
}

fn progress_bar(value: f64, max: f64, width: usize) -> String {
    let ratio = (value / max).clamp(0.0, 1.0);
    let filled = (ratio * width as f64).round() as usize;
    let empty = width - filled;
    format!("{}{}", "▓".repeat(filled), "░".repeat(empty))
}

fn generate_mock_throughput() -> Vec<u64> {
    vec![0; 40] // Start with zeros — sparkline renders 1-block baseline
}

/// Simple deterministic pseudo-random for throughput jitter. No external dep needed.
fn rand_simple() -> f64 {
    use std::time::SystemTime;
    let nanos =
        SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().subsec_nanos();
    (nanos % 1000) as f64 / 1000.0
}

/// Open the claim page in the user's default browser.
fn open_claim_page() {
    let config = compute_daemon::config::Config::load().unwrap_or_default();
    let wallet = &config.wallet.public_address;

    let url = if wallet.is_empty() {
        "https://computenetwork.sh/dashboard/claim".to_string()
    } else {
        format!("https://computenetwork.sh/dashboard/claim?wallet={wallet}")
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

/// Fetch network stats and online nodes from Supabase.
/// Returns (stats, list of (wallet, region) for globe).
fn fetch_network_and_nodes() -> (NetworkStats, Vec<(String, Option<String>)>) {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build();
    match rt {
        Ok(rt) => rt.block_on(async {
            let client = compute_network::supabase::SupabaseClient::new();

            let stats = match client.get_network_stats().await {
                Ok(s) => NetworkStats {
                    total_nodes: s.total_nodes as u32,
                    peak_petaflops: s.total_nodes as f64 * 0.066,
                },
                Err(_) => NetworkStats::mock(),
            };

            let nodes = match client.get_online_nodes().await {
                Ok(n) => n.into_iter().map(|node| (node.wallet_address, node.region)).collect(),
                Err(_) => Vec::new(),
            };

            (stats, nodes)
        }),
        Err(_) => (NetworkStats::mock(), Vec::new()),
    }
}
