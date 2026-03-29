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
    ConfigItem {
        label: "Wallet Address",
        key: "wallet.public_address",
        kind: ConfigItemKind::Text,
    },
    ConfigItem { label: "Node Name", key: "node.name", kind: ConfigItemKind::Text },
    ConfigItem { label: "Usage Priority", key: "node.max_gpu_usage", kind: ConfigItemKind::Choice },
    ConfigItem {
        label: "Pause on Battery",
        key: "node.pause_on_battery",
        kind: ConfigItemKind::Toggle,
    },
    ConfigItem {
        label: "Pause on Fullscreen",
        key: "node.pause_on_fullscreen",
        kind: ConfigItemKind::Toggle,
    },
    ConfigItem {
        label: "Auto-start on Login",
        key: "service.autostart",
        kind: ConfigItemKind::Toggle,
    },
    ConfigItem { label: "Region", key: "network.region", kind: ConfigItemKind::Text },
    ConfigItem { label: "Log Level", key: "logging.level", kind: ConfigItemKind::Choice },
];

pub struct Dashboard {
    globe: Globe,
    hardware: HardwareInfo,
    earnings: Earnings,
    pipeline: PipelineStatus,
    network: NetworkStats,
    throughput_history: Vec<u64>,
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
    last_log_update: Instant,
    config_selected: usize,
    config_editing: bool,
    config_edit_buffer: String,
    config_confirm: Option<String>, // "Are you sure?" for wallet changes
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
            earnings: Earnings::default(),
            pipeline: PipelineStatus::default(),
            network,
            throughput_history: vec![0; 60],
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
            last_log_update: Instant::now(),
            config_selected: 0,
            config_editing: false,
            config_edit_buffer: String::new(),
            config_confirm: None,
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
                    KeyCode::Char('c') => {
                        open_claim_page();
                    }
                    KeyCode::Up if self.active_tab == Tab::Logs => {
                        if self.log_scroll > 0 {
                            self.log_scroll -= 1;
                        }
                    }
                    KeyCode::Down if self.active_tab == Tab::Logs => {
                        self.log_scroll += 1;
                    }
                    KeyCode::End if self.active_tab == Tab::Logs => {
                        self.log_scroll = 0; // 0 = follow tail
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
                    KeyCode::Tab => {
                        self.active_tab = match self.active_tab {
                            Tab::Overview => Tab::Logs,
                            Tab::Logs => Tab::Config,
                            Tab::Config => Tab::Overview,
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

        // Read from daemon state if available
        if let Some(ref rx) = self.daemon_state_rx {
            let state = rx.borrow().clone();
            self.live_metrics = state.live_metrics;
            self.earnings.pending = state.earnings.pending;

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
            if self.last_throughput_push.elapsed() >= Duration::from_millis(200) {
                if self.throughput_history.len() >= 60 {
                    self.throughput_history.remove(0);
                }
                self.throughput_history.push(self.smoothed_tps as u64);
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
                    self.throughput_history.remove(0);
                }
                self.throughput_history.push(0);
                self.last_throughput_push = Instant::now();
            }
        }

        // Refresh logs every 2 seconds when on logs tab
        if self.active_tab == Tab::Logs && self.last_log_update.elapsed() > Duration::from_secs(2) {
            self.log_lines = load_recent_logs(500);
            self.last_log_update = Instant::now();
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

        // Fixed scale: max 200, baseline offset 25 (always 1 visible block)
        let display_data: Vec<u64> = self.throughput_history.iter().map(|&v| v + 25).collect();

        let sparkline = Sparkline::default()
            .data(&display_data)
            .max(200)
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

        let visible_height = chunks[1].height as usize;

        if self.log_lines.is_empty() {
            // Generate live status lines from daemon state
            let mut lines = vec![
                Line::from(Span::styled(
                    "  LIVE STATUS",
                    Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD),
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
                    Style::default().fg(Color::Green),
                )));
                lines.push(Line::from(Span::styled(
                    format!("  [{h:02}:{m:02}:{s:02}] Idle state: {}", state.idle_state),
                    Style::default().fg(Color::Gray),
                )));
                lines.push(Line::from(Span::styled(
                    format!(
                        "  [{h:02}:{m:02}:{s:02}] CPU: {:.0}% | RAM: {:.1}GB",
                        state.live_metrics.cpu_usage, state.live_metrics.memory_used_gb
                    ),
                    Style::default().fg(Color::Gray),
                )));
                lines.push(Line::from(Span::styled(
                    format!("  [{h:02}:{m:02}:{s:02}] Inference: {}", state.inference_status),
                    Style::default().fg(Color::Gray),
                )));
                if state.pipeline.active {
                    lines.push(Line::from(Span::styled(
                        format!(
                            "  [{h:02}:{m:02}:{s:02}] Pipeline active: {:.1} tok/s",
                            state.pipeline.tokens_per_sec
                        ),
                        Style::default().fg(Color::Green),
                    )));
                }
            } else {
                lines.push(Line::from(Span::styled(
                    "  Daemon not connected",
                    Style::default().fg(Color::DarkGray),
                )));
            }

            let widget = Paragraph::new(lines).block(Block::default().borders(Borders::NONE));
            frame.render_widget(widget, chunks[1]);
        } else {
            // Show log file content with scrolling
            let total = self.log_lines.len();
            let end =
                if self.log_scroll == 0 { total } else { total.saturating_sub(self.log_scroll) };
            let start = end.saturating_sub(visible_height);

            let visible_lines: Vec<Line> = self.log_lines[start..end]
                .iter()
                .map(|line| {
                    let color = if line.contains("ERROR") || line.contains("error") {
                        Color::Red
                    } else if line.contains("WARN") || line.contains("warn") {
                        Color::Yellow
                    } else if line.contains("INFO") || line.contains("info") {
                        Color::Gray
                    } else {
                        Color::DarkGray
                    };
                    Line::from(Span::styled(format!("  {line}"), Style::default().fg(color)))
                })
                .collect();

            let scroll_indicator = if self.log_scroll > 0 {
                format!(" (scrolled +{})", self.log_scroll)
            } else {
                " (following)".into()
            };

            let logs = Paragraph::new(visible_lines).block(
                Block::default()
                    .borders(Borders::NONE)
                    .title(Span::styled(scroll_indicator, Style::default().fg(Color::DarkGray))),
            );
            frame.render_widget(logs, chunks[1]);
        }

        self.draw_shortcuts(frame, chunks[2]);
    }

    fn get_config_value(&self, key: &str) -> String {
        let config = compute_daemon::config::Config::load().unwrap_or_default();
        match key {
            "wallet.public_address" => {
                if config.wallet.public_address.is_empty() {
                    "(not set)".into()
                } else {
                    config.wallet.public_address
                }
            }
            "node.name" => config.node.name,
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
            "service.autostart" => {
                if compute_daemon::service::is_service_installed() { "On" } else { "Off" }.into()
            }
            "network.region" => config.network.region,
            "logging.level" => config.logging.level,
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
                    "logging.level" => {
                        config.logging.level = match config.logging.level.as_str() {
                            "info" => "debug".into(),
                            "debug" => "warn".into(),
                            "warn" => "error".into(),
                            _ => "info".into(),
                        };
                    }
                    _ => {}
                }
                let _ = config.save();
            }
            ConfigItemKind::Text => {
                self.config_editing = true;
                let current = self.get_config_value(item.key);
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
        config.wallet.node_id.clear(); // Reset node_id, will re-register
        let _ = config.save();

        // Register with Supabase in background
        if !new_wallet.is_empty() {
            let wallet = new_wallet.to_string();
            std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread().enable_all().build();
                if let Ok(rt) = rt {
                    rt.block_on(async {
                        let client = compute_network::supabase::SupabaseClient::new();
                        let node = compute_network::supabase::NodeRow {
                            id: None,
                            wallet_address: wallet,
                            node_name: None,
                            status: Some("online".into()),
                            gpu_model: None,
                            gpu_vram_mb: None,
                            gpu_backend: None,
                            cpu_model: None,
                            cpu_cores: None,
                            memory_mb: None,
                            os: None,
                            app_version: Some(env!("CARGO_PKG_VERSION").into()),
                            region: None,
                            tflops_fp16: None,
                        };
                        match client.register_node(&node).await {
                            Ok(id) => {
                                if let Ok(mut cfg) = compute_daemon::config::Config::load() {
                                    cfg.wallet.node_id = id;
                                    let _ = cfg.save();
                                }
                            }
                            Err(e) => tracing::warn!("Re-registration failed: {e}"),
                        }
                    });
                }
            });
        }
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

        let mut lines = vec![
            Line::from(Span::styled(
                "  SETTINGS",
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ];

        // Confirmation dialog
        if let Some(ref wallet) = self.config_confirm {
            lines.push(Line::from(Span::styled(
                "  Are you sure you want to change your wallet address?",
                Style::default().fg(Color::Yellow),
            )));
            lines.push(Line::from(Span::styled(
                format!("  New: {wallet}"),
                Style::default().fg(Color::White),
            )));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "  [Y] Confirm   [N] Cancel",
                Style::default().fg(Color::DarkGray),
            )));
            let widget = Paragraph::new(lines);
            frame.render_widget(widget, chunks[1]);
            self.draw_shortcuts(frame, chunks[2]);
            return;
        }

        for (i, item) in CONFIG_ITEMS.iter().enumerate() {
            let is_selected = i == self.config_selected;
            let value = if self.config_editing && is_selected {
                format!("{}█", self.config_edit_buffer)
            } else {
                self.get_config_value(item.key)
            };

            let arrow = if is_selected { "▸ " } else { "  " };
            let label_color = if is_selected { Color::White } else { Color::DarkGray };
            let value_color = if is_selected {
                if self.config_editing { Color::Yellow } else { Color::White }
            } else {
                Color::Gray
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
                Span::styled(hint, Style::default().fg(Color::DarkGray)),
            ]));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  ↑↓ Navigate   ↵ Edit/Toggle   Esc Cancel",
            Style::default().fg(Color::DarkGray),
        )));

        let config_widget = Paragraph::new(lines);
        frame.render_widget(config_widget, chunks[1]);

        self.draw_shortcuts(frame, chunks[2]);
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
