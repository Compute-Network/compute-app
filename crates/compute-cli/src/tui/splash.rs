use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::{
    DefaultTerminal, Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Clear, Paragraph},
};
use tokio::sync::watch;

use compute_daemon::benchmark::StartupAssessment;
use compute_daemon::runtime::{DaemonState, DownloadPhase, DownloadStatus};

use super::globe::Globe;
use super::theme;

// Main logo — compact block
const LOGO_MAIN: &[&str] = &[
    " ██████  ██████  ███    ███ ██████  ██    ██ ████████ ███████",
    "██      ██    ██ ████  ████ ██   ██ ██    ██    ██    ██",
    "██      ██    ██ ██ ████ ██ ██████  ██    ██    ██    █████",
    "██      ██    ██ ██  ██  ██ ██      ██    ██    ██    ██",
    " ██████  ██████  ██      ██ ██       ██████     ██    ███████",
];

// Small logo (< 55 chars) — spaced text
const LOGO_SMALL: &[&str] = &["C O M P U T E"];

const VERSION: &str = env!("CARGO_PKG_VERSION");

const SPINNER_CHARS: &[char] = &['/', '-', '\\', '|'];

/// Steps shown during the startup splash.
#[derive(Clone)]
struct StartupStep {
    label: String,
    done: bool,
    result: Option<String>,
}

pub struct SplashScreen {
    globe: Globe,
    steps: Vec<StartupStep>,
    current_step: usize,
    start_time: Instant,
    step_timer: Instant,
    logo_visible_chars: usize,
    phase: SplashPhase,
    /// Tracks llama-server install progress (None = not installing)
    llama_install_rx: Option<mpsc::Receiver<Result<(), String>>>,
    /// Whether llama-server was found or successfully installed
    llama_ready: bool,
    /// Spinner frame counter for rotating / animation
    spinner_frame: usize,
    /// Timer for spinner animation
    spinner_timer: Instant,
    /// Tracks model loading progress
    model_load_rx: Option<mpsc::Receiver<Result<String, String>>>,
    /// Whether model is loaded and serving
    model_loaded: Arc<AtomicBool>,
    /// Experimental stage mode skips local llama-server health gating.
    stage_mode_enabled: bool,
    /// Whether a local model is present and worth waiting on.
    has_local_model: bool,
    /// Step index that may start the llama-server install.
    llama_status_step: usize,
    /// Step index that may wait for model/runtime readiness.
    model_step: usize,
    /// Live daemon state, used to show startup model download progress.
    daemon_state_rx: Option<watch::Receiver<DaemonState>>,
}

#[derive(PartialEq)]
enum SplashPhase {
    GlobeFadeIn,
    StepsRunning,
    /// Waiting for llama-server install to complete
    InstallingLlama,
    /// Waiting for model to load into llama-server
    LoadingModel,
    Complete,
}

impl SplashScreen {
    pub fn new(
        hardware_info: &compute_daemon::hardware::HardwareInfo,
        assessment: StartupAssessment,
    ) -> Self {
        let mut globe = Globe::new();
        globe.set_mock_nodes();

        let gpu_name = hardware_info
            .gpus
            .first()
            .map(|g| format!("{} ({})", g.name, format_vram(g.vram_mb)))
            .unwrap_or_else(|| "No GPU detected".into());

        // Fetch network stats from the orchestrator (non-blocking with timeout)
        let node_count = fetch_node_count();

        // Check if llama-server is available
        let llama_found = find_llama_server();

        let hardware_result = format!(
            "{} threads · {} RAM",
            hardware_info.cpu.threads.max(hardware_info.cpu.cores),
            format_gb(hardware_info.memory.total_gb)
        );
        let benchmark_result = format!(
            "score {}/100 · {} · {:.1} TFLOPS",
            assessment.score, assessment.tier, assessment.estimated_tflops_fp16
        );
        let split_result = if assessment.split_capable {
            format!(
                "yes · {} · {}",
                assessment.split_role,
                assessment.split_model_label.as_deref().unwrap_or("split model")
            )
        } else {
            format!("no · {}", assessment.split_reason)
        };

        let mut steps = vec![
            StartupStep {
                label: "Detecting hardware...".into(),
                done: false,
                result: Some(hardware_result),
            },
            StartupStep { label: format!("GPU: {gpu_name}"), done: false, result: None },
            StartupStep {
                label: "Benchmarking node...".into(),
                done: false,
                result: Some(benchmark_result),
            },
            StartupStep {
                label: "Assigning model...".into(),
                done: false,
                result: Some(format!(
                    "{} · {}",
                    assessment.assigned_model_id, assessment.assigned_model_label
                )),
            },
            StartupStep {
                label: "Checking split-node potential...".into(),
                done: false,
                result: Some(split_result),
            },
            StartupStep { label: "Checking llama-server...".into(), done: false, result: None },
        ];

        let llama_status_step = steps.len();
        if llama_found {
            steps.push(StartupStep {
                label: "llama-server ready".into(),
                done: false,
                result: None,
            });
        } else {
            steps.push(StartupStep {
                label: "Installing llama-server...".into(),
                done: false,
                result: None,
            });
        }

        // Show the model the daemon will actually pre-warm (from config), not just the first file
        let config = compute_daemon::config::Config::load().unwrap_or_default();
        let stage_mode_enabled = config.experimental.stage_mode_enabled;
        let first_model_path = find_first_model();
        let has_local_model = first_model_path.is_some();
        let model_label = if stage_mode_enabled {
            format!("orchestrator target {}", assessment.assigned_model_id)
        } else if config.models.active_model != "auto" {
            config.models.active_model.clone()
        } else if !has_local_model {
            "no local model cached yet".into()
        } else {
            // Same logic as runtime.rs pre-warm: prefer small Gemma, then larger Gemma, then Qwen.
            let downloaded = first_model_path
                .and_then(|p| {
                    std::path::Path::new(&p)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string())
                })
                .unwrap_or_default();
            downloaded
        };

        let model_step = steps.len();
        steps.push(StartupStep {
            label: if stage_mode_enabled {
                format!("Preparing inference runtime: {}...", model_label)
            } else {
                format!("Loading model: {}...", model_label)
            },
            done: false,
            result: None,
        });

        steps.push(StartupStep {
            label: "Connecting to network...".into(),
            done: false,
            result: None,
        });
        steps.push(StartupStep {
            label: format!("{} nodes online", format_count(node_count)),
            done: false,
            result: None,
        });

        let model_loaded = Arc::new(AtomicBool::new(false));

        Self {
            globe,
            steps,
            current_step: 0,
            start_time: Instant::now(),
            step_timer: Instant::now(),
            logo_visible_chars: 0,
            phase: SplashPhase::GlobeFadeIn,
            llama_install_rx: None,
            llama_ready: llama_found,
            spinner_frame: 0,
            spinner_timer: Instant::now(),
            model_load_rx: None,
            model_loaded,
            stage_mode_enabled,
            has_local_model,
            llama_status_step,
            model_step,
            daemon_state_rx: None,
        }
    }

    pub fn with_daemon_state(mut self, state_rx: watch::Receiver<DaemonState>) -> Self {
        self.daemon_state_rx = Some(state_rx);
        self
    }

    /// Run the splash screen animation. Returns true if user wants to continue to dashboard.
    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<bool> {
        let tick_rate = Duration::from_millis(50); // 20 fps

        loop {
            terminal.draw(|frame| self.draw(frame))?;

            // Handle input
            if event::poll(tick_rate)?
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(false),
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        return Ok(false);
                    }
                    KeyCode::Char('i') | KeyCode::Char('I') => {
                        // Open llama-server docs page
                        open_url("https://computenetwork.sh/docs/llama-server");
                    }
                    KeyCode::Enter if self.phase == SplashPhase::Complete => {
                        return Ok(true);
                    }
                    // Any key during animation speeds it up (but not during install)
                    _ if self.phase == SplashPhase::StepsRunning => {
                        self.skip_to_complete();
                    }
                    _ => {}
                }
            }

            // Update state
            self.tick();

            // Auto-complete after all steps done + brief pause
            if self.phase == SplashPhase::Complete
                && self.start_time.elapsed() > Duration::from_secs(4)
                && !self.has_active_download()
            {
                return Ok(true);
            }
        }
    }

    fn tick(&mut self) {
        self.globe.tick();

        let elapsed = self.start_time.elapsed();

        // Spinner animation — rotate every 100ms
        if self.spinner_timer.elapsed() > Duration::from_millis(100) {
            self.spinner_frame = (self.spinner_frame + 1) % SPINNER_CHARS.len();
            self.spinner_timer = Instant::now();
        }

        // Logo typewriter — runs fast, concurrently with steps
        let total_chars: usize = LOGO_MAIN.iter().map(|l| l.len()).sum();
        if self.logo_visible_chars < total_chars {
            // ~20 chars per tick at 20fps = full logo in ~0.5s
            self.logo_visible_chars = (self.logo_visible_chars + 20).min(total_chars);
        }

        // Steps run concurrently with the logo animation
        match self.phase {
            SplashPhase::GlobeFadeIn => {
                if elapsed > Duration::from_millis(200) {
                    self.phase = SplashPhase::StepsRunning;
                    self.step_timer = Instant::now();
                }
            }
            SplashPhase::StepsRunning => {
                if self.current_step < self.steps.len() {
                    let step_delay = self.step_delay(self.current_step);
                    if self.step_timer.elapsed() > step_delay {
                        if self.current_step == self.llama_status_step && !self.llama_ready {
                            self.steps[self.current_step].done = false;
                            self.start_llama_install();
                            self.phase = SplashPhase::InstallingLlama;
                            return;
                        }

                        if self.current_step == self.model_step && self.stage_mode_enabled {
                            self.model_loaded.store(true, Ordering::Relaxed);
                            self.steps[self.current_step].label =
                                "Inference runtime ready".into();
                            self.steps[self.current_step].done = true;
                            self.current_step += 1;
                            self.step_timer = Instant::now();
                            return;
                        }

                        if self.current_step == self.model_step && !self.has_local_model {
                            self.model_loaded.store(true, Ordering::Relaxed);
                            self.steps[self.current_step].label =
                                "No local model cached yet".into();
                            self.steps[self.current_step].done = true;
                            self.current_step += 1;
                            self.step_timer = Instant::now();
                            return;
                        }

                        if self.current_step == self.model_step
                            && !self.model_loaded.load(Ordering::Relaxed)
                        {
                            self.start_model_health_poll();
                            self.phase = SplashPhase::LoadingModel;
                            return;
                        }

                        self.steps[self.current_step].done = true;
                        self.current_step += 1;
                        self.step_timer = Instant::now();
                    }
                } else {
                    self.phase = SplashPhase::Complete;
                }
            }
            SplashPhase::LoadingModel => {
                if let Some(ref rx) = self.model_load_rx {
                    if let Ok(result) = rx.try_recv() {
                        match result {
                            Ok(model_name) => {
                                self.model_loaded.store(true, Ordering::Relaxed);
                                self.steps[self.current_step].label =
                                    format!("Model ready: {model_name}");
                                self.steps[self.current_step].done = true;
                                self.current_step += 1;
                                self.step_timer = Instant::now();
                                self.phase = SplashPhase::StepsRunning;
                            }
                            Err(e) => {
                                self.steps[self.current_step].label = format!("Model failed: {e}");
                                self.steps[self.current_step].done = true;
                                self.current_step += 1;
                                self.step_timer = Instant::now();
                                self.phase = SplashPhase::StepsRunning;
                            }
                        }
                    }
                }
            }
            SplashPhase::InstallingLlama => {
                // Poll install progress
                if let Some(ref rx) = self.llama_install_rx {
                    if let Ok(result) = rx.try_recv() {
                        match result {
                            Ok(()) => {
                                self.llama_ready = true;
                                self.steps[self.current_step].label =
                                    "llama-server installed".into();
                                self.steps[self.current_step].done = true;
                                self.current_step += 1;
                                self.step_timer = Instant::now();
                                self.phase = SplashPhase::StepsRunning;
                            }
                            Err(e) => {
                                self.steps[self.current_step].label = format!("llama-server: {e}");
                                self.steps[self.current_step].done = true;
                                self.current_step += 1;
                                self.step_timer = Instant::now();
                                self.phase = SplashPhase::StepsRunning;
                            }
                        }
                    }
                }
            }
            SplashPhase::Complete => {}
        }
    }

    fn step_delay(&self, step_index: usize) -> Duration {
        if step_index == 0 {
            Duration::from_millis(200)
        } else if step_index == self.llama_status_step.saturating_sub(1) {
            Duration::from_millis(300)
        } else if step_index == self.model_step {
            Duration::from_millis(100)
        } else if step_index > self.model_step {
            Duration::from_millis(350)
        } else {
            Duration::from_millis(180)
        }
    }

    fn start_llama_install(&mut self) {
        let (tx, rx) = mpsc::channel();
        self.llama_install_rx = Some(rx);

        std::thread::spawn(move || {
            let result = if cfg!(target_os = "macos") {
                std::process::Command::new("brew")
                    .args(["install", "llama.cpp"])
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .status()
            } else {
                // On Linux/Windows, can't auto-install
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "install manually from github.com/ggerganov/llama.cpp",
                ))
            };

            let _ = tx.send(match result {
                Ok(status) if status.success() => Ok(()),
                Ok(status) => Err(format!("brew exited with code {status}")),
                Err(e) => Err(format!("{e}")),
            });
        });
    }

    fn start_model_health_poll(&mut self) {
        let (tx, rx) = mpsc::channel();
        self.model_load_rx = Some(rx);

        std::thread::spawn(move || {
            let model_name = find_first_model()
                .and_then(|p| {
                    std::path::Path::new(&p)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string())
                })
                .unwrap_or_else(|| "unknown".into());

            // Poll llama-server /health until the daemon has it ready (120s timeout)
            let start = Instant::now();
            loop {
                if start.elapsed() > Duration::from_secs(120) {
                    let _ = tx.send(Err("Model load timed out (120s)".into()));
                    return;
                }
                std::thread::sleep(Duration::from_millis(500));

                let Ok(resp) = reqwest::blocking::Client::new()
                    .get("http://127.0.0.1:8090/health")
                    .timeout(Duration::from_secs(2))
                    .send()
                else {
                    continue;
                };

                if resp.status().is_success() {
                    let _ = tx.send(Ok(model_name));
                    return;
                }
            }
        });
    }

    fn skip_to_complete(&mut self) {
        // Don't skip if we're in the middle of installing llama or waiting for model
        if self.phase == SplashPhase::InstallingLlama || self.phase == SplashPhase::LoadingModel {
            return;
        }
        for step in &mut self.steps {
            step.done = true;
        }
        self.current_step = self.steps.len();
        let total_chars: usize = LOGO_MAIN.iter().map(|l| l.len()).sum();
        self.logo_visible_chars = total_chars;
        self.phase = SplashPhase::Complete;
    }

    fn visible_download(&self) -> Option<DownloadStatus> {
        let state = self.daemon_state_rx.as_ref()?.borrow().clone();
        state
            .downloads
            .iter()
            .find(|download| download.phase == DownloadPhase::Downloading)
            .cloned()
            .or_else(|| {
                state
                    .downloads
                    .iter()
                    .find(|download| download.phase == DownloadPhase::Failed)
                    .cloned()
            })
    }

    fn has_active_download(&self) -> bool {
        self.visible_download().is_some_and(|download| download.phase == DownloadPhase::Downloading)
    }

    fn draw(&self, frame: &mut Frame) {
        let full_area = frame.area();
        frame.render_widget(Clear, full_area);
        let palette = theme::palette();

        // Cap dimensions, align top-left
        let max_w: u16 = 160;
        let max_h: u16 = 45;
        let area = Rect {
            x: full_area.x,
            y: full_area.y,
            width: full_area.width.min(max_w),
            height: full_area.height.min(max_h),
        };

        // Responsive breakpoints
        if area.width >= 80 {
            // Desktop: side-by-side
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(33), Constraint::Percentage(67)])
                .split(area);

            self.globe.render(chunks[0], frame.buffer_mut(), palette);
            self.draw_splash_content(frame, chunks[1]);
        } else if area.width >= 50 {
            // Vertical: globe on top, content below
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(12), Constraint::Min(8)])
                .split(area);

            self.globe.render(chunks[0], frame.buffer_mut(), palette);
            self.draw_splash_content(frame, chunks[1]);
        } else {
            // Narrow: content only
            self.draw_splash_content(frame, area);
        }
    }

    fn draw_splash_content(&self, frame: &mut Frame, area: Rect) {
        let p = theme::palette();
        let w = area.width as usize;

        // Pick logo variant based on available width
        let logo_data: &[&str] = if w >= 50 { LOGO_MAIN } else { LOGO_SMALL };
        let logo_height = logo_data.len() as u16 + 1; // +1 breathing room

        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),           // top padding
                Constraint::Length(logo_height), // logo
                Constraint::Length(2),           // tagline + version
                Constraint::Length(1),           // spacer
                Constraint::Min(6),              // steps
                Constraint::Length(3),           // bottom message + [i] hint
            ])
            .split(area);

        // Logo with typewriter animation
        let total_chars: usize = logo_data.iter().map(|l| l.len()).sum();
        let visible = self.logo_visible_chars.min(total_chars);

        let mut logo_lines = Vec::new();
        let mut chars_shown = 0;
        for &line in logo_data {
            if chars_shown >= visible {
                logo_lines.push(Line::from(""));
            } else {
                let line_visible = (visible - chars_shown).min(line.len());
                let visible_text: String = line.chars().take(line_visible).collect();
                logo_lines.push(Line::from(Span::styled(
                    visible_text,
                    Style::default().fg(p.text).add_modifier(Modifier::BOLD),
                )));
                chars_shown += line.len();
            }
        }

        let logo = Paragraph::new(logo_lines);
        frame.render_widget(logo, right_chunks[1]);

        // Tagline + version
        let tagline = Paragraph::new(vec![
            Line::from(Span::styled(
                "Decentralized GPU Infrastructure",
                Style::default().fg(p.dim),
            )),
            Line::from(Span::styled(format!("v{VERSION}"), Style::default().fg(p.dim))),
        ]);
        frame.render_widget(tagline, right_chunks[2]);

        // Steps
        let spinner = SPINNER_CHARS[self.spinner_frame];
        let mut step_lines = Vec::new();
        for (i, step) in self.steps.iter().enumerate() {
            let is_installing =
                i == self.current_step && self.phase == SplashPhase::InstallingLlama;

            let is_loading_model =
                i == self.current_step && self.phase == SplashPhase::LoadingModel;

            let (icon, color) = if step.done {
                ("  ✓ ".to_string(), p.success)
            } else if is_installing
                || is_loading_model
                || (i == self.current_step && self.phase == SplashPhase::StepsRunning)
            {
                (format!("  {spinner} "), p.warning)
            } else {
                ("    ".to_string(), p.dim)
            };

            if i <= self.current_step || step.done {
                let mut spans = vec![
                    Span::styled(icon, Style::default().fg(color)),
                    Span::styled(&step.label, Style::default().fg(p.muted)),
                ];
                if let Some(result) = step.result.as_deref() {
                    spans.push(Span::styled(" ", Style::default().fg(p.dim)));
                    spans.push(Span::styled(result, Style::default().fg(p.dim)));
                }
                step_lines.push(Line::from(spans));
            }
        }
        let steps_widget = Paragraph::new(step_lines);
        frame.render_widget(steps_widget, right_chunks[4]);

        // Bottom message
        if let Some(download) = self.visible_download() {
            let msg = Paragraph::new(download_lines(&download));
            frame.render_widget(msg, right_chunks[5]);
        } else if self.phase == SplashPhase::Complete {
            let msg = Paragraph::new(vec![
                Line::from(Span::styled(
                    "  Daemon started. Earning $COMPUTE...",
                    Style::default().fg(p.success),
                )),
                Line::from(Span::styled("  [i] What is llama-server?", Style::default().fg(p.dim))),
            ]);
            frame.render_widget(msg, right_chunks[5]);
        } else if self.phase == SplashPhase::InstallingLlama {
            let msg = Paragraph::new(vec![
                Line::from(Span::styled(
                    "  Installing llama-server via Homebrew...",
                    Style::default().fg(p.warning),
                )),
                Line::from(Span::styled("  [i] What is llama-server?", Style::default().fg(p.dim))),
            ]);
            frame.render_widget(msg, right_chunks[5]);
        } else if !self.has_local_model && !self.stage_mode_enabled {
            let msg = Paragraph::new(vec![
                Line::from(Span::styled(
                    "  No model cached yet. Download one from Models after startup.",
                    Style::default().fg(p.warning),
                )),
                Line::from(Span::styled("  [i] What is llama-server?", Style::default().fg(p.dim))),
            ]);
            frame.render_widget(msg, right_chunks[5]);
        }
    }
}

fn format_vram(vram_mb: u64) -> String {
    if vram_mb >= 1024 { format!("{}GB", vram_mb / 1024) } else { format!("{vram_mb}MB") }
}

fn format_gb(value: f64) -> String {
    if value >= 10.0 { format!("{value:.0}GB") } else { format!("{value:.1}GB") }
}

fn download_lines(download: &DownloadStatus) -> Vec<Line<'static>> {
    let p = theme::palette();
    match download.phase {
        DownloadPhase::Downloading => {
            let progress = download
                .total_bytes
                .filter(|total| *total > 0)
                .map(|total| download.downloaded_bytes as f64 / total as f64)
                .unwrap_or(0.0)
                .clamp(0.0, 0.995);
            vec![
                Line::from(Span::styled(
                    format!("  Downloading model: {}", download.model_id),
                    Style::default().fg(p.warning),
                )),
                Line::from(Span::styled(
                    format!(
                        "  {} {:>3.0}% {}",
                        progress_bar(progress, 18),
                        progress * 100.0,
                        format_download_size(download)
                    ),
                    Style::default().fg(p.muted),
                )),
                Line::from(Span::styled(
                    "  Enter opens dashboard; download continues.",
                    Style::default().fg(p.dim),
                )),
            ]
        }
        DownloadPhase::Failed => vec![
            Line::from(Span::styled(
                format!("  Model download failed: {}", download.model_id),
                Style::default().fg(p.danger),
            )),
            Line::from(Span::styled(
                format!("  {}", download.error.as_deref().unwrap_or("unknown error")),
                Style::default().fg(p.dim),
            )),
            Line::from(Span::styled(
                "  Open Models after startup to retry.",
                Style::default().fg(p.dim),
            )),
        ],
        DownloadPhase::Complete => vec![Line::from(Span::styled(
            format!("  Model downloaded: {}", download.model_id),
            Style::default().fg(p.success),
        ))],
    }
}

fn progress_bar(progress: f64, width: usize) -> String {
    let filled = (progress * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("[{}{}]", "#".repeat(filled), "-".repeat(empty))
}

fn format_download_size(download: &DownloadStatus) -> String {
    let downloaded = format_bytes(download.downloaded_bytes);
    match download.total_bytes {
        Some(total) if total > 0 => format!("{downloaded}/{}", format_bytes(total)),
        _ => downloaded,
    }
}

fn format_bytes(bytes: u64) -> String {
    const GB: f64 = 1_073_741_824.0;
    const MB: f64 = 1_048_576.0;
    if bytes >= 1_073_741_824 {
        format!("{:.1}GB", bytes as f64 / GB)
    } else {
        format!("{:.0}MB", bytes as f64 / MB)
    }
}

/// Fetch the total node count from the orchestrator. Returns 0 on failure.
/// Uses a short timeout so the splash screen isn't delayed.
fn fetch_node_count() -> u64 {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build();
    match rt {
        Ok(rt) => rt.block_on(async {
            let client = compute_network::client::OrchestratorClient::new(
                "https://api.computenetwork.sh",
                None,
            );
            match client.get_network_stats().await {
                Ok(stats) => stats.total_nodes,
                Err(_) => 0,
            }
        }),
        Err(_) => 0,
    }
}

/// Check if llama-server is available on the system.
/// Find the first .gguf model in ~/.compute/models/
fn find_first_model() -> Option<String> {
    let cache_dir = dirs::home_dir()?.join(".compute").join("models");
    let entries = std::fs::read_dir(&cache_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".gguf") {
            return Some(entry.path().to_string_lossy().to_string());
        }
    }
    None
}

fn find_llama_server() -> bool {
    if let Ok(output) = std::process::Command::new("which").arg("llama-server").output()
        && output.status.success()
    {
        return true;
    }

    let candidates = ["/usr/local/bin/llama-server", "/opt/homebrew/bin/llama-server"];
    candidates.iter().any(|p| std::path::Path::new(p).exists())
}

/// Open a URL in the default browser.
fn open_url(url: &str) {
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(url).spawn();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    }
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd").args(["/C", "start", url]).spawn();
    }
}

/// Format a count with comma separators.
fn format_count(n: u64) -> String {
    if n == 0 {
        return "0".into();
    }
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
