use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::{
    DefaultTerminal, Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
};

use super::globe::Globe;

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

/// Steps shown during the startup splash.
#[derive(Clone)]
struct StartupStep {
    label: String,
    done: bool,
    #[allow(dead_code)]
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
}

#[derive(PartialEq)]
enum SplashPhase {
    GlobeFadeIn,
    StepsRunning,
    Complete,
}

impl SplashScreen {
    pub fn new(hardware_info: &compute_daemon::hardware::HardwareInfo) -> Self {
        let mut globe = Globe::new();
        globe.set_mock_nodes();

        let gpu_name = hardware_info
            .gpus
            .first()
            .map(|g| format!("{} ({})", g.name, format_vram(g.vram_mb)))
            .unwrap_or_else(|| "No GPU detected".into());

        // Fetch network stats from Supabase (non-blocking with timeout)
        let node_count = fetch_node_count();

        let steps = vec![
            StartupStep { label: "Detecting hardware...".into(), done: false, result: None },
            StartupStep { label: format!("GPU: {gpu_name}"), done: false, result: None },
            StartupStep { label: "Connecting to network...".into(), done: false, result: None },
            StartupStep {
                label: format!("{} nodes online", format_count(node_count)),
                done: false,
                result: None,
            },
        ];

        Self {
            globe,
            steps,
            current_step: 0,
            start_time: Instant::now(),
            step_timer: Instant::now(),
            logo_visible_chars: 0,
            phase: SplashPhase::GlobeFadeIn,
        }
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
                    KeyCode::Enter if self.phase == SplashPhase::Complete => {
                        return Ok(true);
                    }
                    // Any key during animation speeds it up
                    _ if self.phase != SplashPhase::Complete => {
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
            {
                return Ok(true);
            }
        }
    }

    fn tick(&mut self) {
        self.globe.tick();

        let elapsed = self.start_time.elapsed();

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
                    let step_delay = match self.current_step {
                        0 => Duration::from_millis(200),
                        1 => Duration::from_millis(150),
                        2 => Duration::from_millis(400),
                        _ => Duration::from_millis(200),
                    };
                    if self.step_timer.elapsed() > step_delay {
                        self.steps[self.current_step].done = true;
                        self.current_step += 1;
                        self.step_timer = Instant::now();
                    }
                } else {
                    self.phase = SplashPhase::Complete;
                }
            }
            SplashPhase::Complete => {}
        }
    }

    fn skip_to_complete(&mut self) {
        for step in &mut self.steps {
            step.done = true;
        }
        self.current_step = self.steps.len();
        let total_chars: usize = LOGO_MAIN.iter().map(|l| l.len()).sum();
        self.logo_visible_chars = total_chars;
        self.phase = SplashPhase::Complete;
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

        // Responsive breakpoints
        if area.width >= 80 {
            // Desktop: side-by-side
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(33), Constraint::Percentage(67)])
                .split(area);

            self.globe.render(chunks[0], frame.buffer_mut());
            self.draw_splash_content(frame, chunks[1]);
        } else if area.width >= 50 {
            // Vertical: globe on top, content below
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(12), Constraint::Min(8)])
                .split(area);

            self.globe.render(chunks[0], frame.buffer_mut());
            self.draw_splash_content(frame, chunks[1]);
        } else {
            // Narrow: content only
            self.draw_splash_content(frame, area);
        }
    }

    fn draw_splash_content(&self, frame: &mut Frame, area: Rect) {
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
                Constraint::Length(2),           // bottom message
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
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
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
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(Span::styled(format!("v{VERSION}"), Style::default().fg(Color::DarkGray))),
        ]);
        frame.render_widget(tagline, right_chunks[2]);

        // Steps
        let mut step_lines = Vec::new();
        for (i, step) in self.steps.iter().enumerate() {
            let (icon, color) = if step.done {
                ("  ✓ ", Color::Green)
            } else if i == self.current_step && self.phase == SplashPhase::StepsRunning {
                ("  ◌ ", Color::Yellow)
            } else {
                ("    ", Color::DarkGray)
            };

            if i <= self.current_step || step.done {
                step_lines.push(Line::from(vec![
                    Span::styled(icon, Style::default().fg(color)),
                    Span::styled(&step.label, Style::default().fg(Color::Gray)),
                ]));
            }
        }
        let steps_widget = Paragraph::new(step_lines);
        frame.render_widget(steps_widget, right_chunks[4]);

        // Bottom message
        if self.phase == SplashPhase::Complete {
            let msg = Paragraph::new(Line::from(Span::styled(
                "  Daemon started. Earning $COMPUTE...",
                Style::default().fg(Color::Green),
            )));
            frame.render_widget(msg, right_chunks[5]);
        }
    }
}

fn format_vram(vram_mb: u64) -> String {
    if vram_mb >= 1024 { format!("{}GB", vram_mb / 1024) } else { format!("{vram_mb}MB") }
}

/// Fetch the total node count from Supabase. Returns 0 on failure.
/// Uses a short timeout so the splash screen isn't delayed.
fn fetch_node_count() -> u64 {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build();
    match rt {
        Ok(rt) => rt.block_on(async {
            let client = compute_network::supabase::SupabaseClient::new();
            match client.get_network_stats().await {
                Ok(stats) => stats.total_nodes,
                Err(_) => 0,
            }
        }),
        Err(_) => 0,
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
