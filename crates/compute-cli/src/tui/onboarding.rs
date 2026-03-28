use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::{
    DefaultTerminal, Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

use super::globe::Globe;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Onboarding result — the wallet address entered by the user.
pub enum OnboardingResult {
    /// User entered a valid wallet address.
    WalletSet(String),
    /// User skipped onboarding.
    Skipped,
    /// User quit the application.
    Quit,
}

/// Onboarding screen state.
pub struct OnboardingScreen {
    globe: Globe,
    input: String,
    cursor_pos: usize,
    error_message: Option<String>,
    success: bool,
    registering: bool,
    step: OnboardingStep,
    cursor_blink: bool,
    last_blink: Instant,
}

#[derive(PartialEq)]
enum OnboardingStep {
    Welcome,
    WalletInput,
    Registering,
    Done,
}

impl OnboardingScreen {
    pub fn new() -> Self {
        let mut globe = Globe::new();
        globe.set_mock_nodes();

        Self {
            globe,
            input: String::new(),
            cursor_pos: 0,
            error_message: None,
            success: false,
            registering: false,
            step: OnboardingStep::Welcome,
            cursor_blink: true,
            last_blink: Instant::now(),
        }
    }

    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> anyhow::Result<OnboardingResult> {
        let tick_rate = Duration::from_millis(50);

        loop {
            terminal.draw(|frame| self.draw(frame))?;

            if event::poll(tick_rate)?
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                match key.code {
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        return Ok(OnboardingResult::Quit);
                    }
                    KeyCode::Esc => {
                        return Ok(OnboardingResult::Quit);
                    }
                    _ => {
                        if let Some(result) = self.handle_key(key.code, key.modifiers) {
                            return Ok(result);
                        }
                    }
                }
            }

            self.tick();
        }
    }

    fn handle_key(
        &mut self,
        code: KeyCode,
        _modifiers: KeyModifiers,
    ) -> Option<OnboardingResult> {
        match self.step {
            OnboardingStep::Welcome => {
                match code {
                    KeyCode::Enter => {
                        self.step = OnboardingStep::WalletInput;
                    }
                    KeyCode::Char('s') | KeyCode::Char('S') => {
                        return Some(OnboardingResult::Skipped);
                    }
                    _ => {}
                }
                None
            }
            OnboardingStep::WalletInput => {
                match code {
                    KeyCode::Char(c) => {
                        self.input.insert(self.cursor_pos, c);
                        self.cursor_pos += 1;
                        self.error_message = None;
                    }
                    KeyCode::Backspace => {
                        if self.cursor_pos > 0 {
                            self.cursor_pos -= 1;
                            self.input.remove(self.cursor_pos);
                            self.error_message = None;
                        }
                    }
                    KeyCode::Delete => {
                        if self.cursor_pos < self.input.len() {
                            self.input.remove(self.cursor_pos);
                            self.error_message = None;
                        }
                    }
                    KeyCode::Left => {
                        if self.cursor_pos > 0 {
                            self.cursor_pos -= 1;
                        }
                    }
                    KeyCode::Right => {
                        if self.cursor_pos < self.input.len() {
                            self.cursor_pos += 1;
                        }
                    }
                    KeyCode::Home => {
                        self.cursor_pos = 0;
                    }
                    KeyCode::End => {
                        self.cursor_pos = self.input.len();
                    }
                    KeyCode::Enter => {
                        if self.input.trim().is_empty() {
                            return Some(OnboardingResult::Skipped);
                        }
                        let address = self.input.trim().to_string();
                        if compute_solana::is_valid_address(&address) {
                            self.step = OnboardingStep::Registering;
                            self.registering = true;
                            // Registration happens in the main flow after we return
                            // For now, mark as done immediately — the caller handles async registration
                            self.step = OnboardingStep::Done;
                            self.success = true;
                            return Some(OnboardingResult::WalletSet(address));
                        } else {
                            self.error_message = Some("Invalid Solana address. Must be 32-44 base58 characters.".into());
                        }
                    }
                    _ => {}
                }
                None
            }
            OnboardingStep::Registering => None,
            OnboardingStep::Done => {
                if code == KeyCode::Enter {
                    let addr = self.input.trim().to_string();
                    Some(OnboardingResult::WalletSet(addr))
                } else {
                    None
                }
            }
        }
    }

    fn tick(&mut self) {
        self.globe.tick();

        // Cursor blink every 500ms
        if self.last_blink.elapsed() > Duration::from_millis(500) {
            self.cursor_blink = !self.cursor_blink;
            self.last_blink = Instant::now();
        }
    }

    fn draw(&self, frame: &mut Frame) {
        let full_area = frame.area();

        let max_w: u16 = 160;
        let max_h: u16 = 45;
        let area = Rect {
            x: full_area.x,
            y: full_area.y,
            width: full_area.width.min(max_w),
            height: full_area.height.min(max_h),
        };

        if area.width >= 80 {
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(33), Constraint::Percentage(67)])
                .split(area);

            self.globe.render(chunks[0], frame.buffer_mut());
            self.draw_content(frame, chunks[1]);
        } else if area.width >= 50 {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(12), Constraint::Min(8)])
                .split(area);

            self.globe.render(chunks[0], frame.buffer_mut());
            self.draw_content(frame, chunks[1]);
        } else {
            self.draw_content(frame, area);
        }
    }

    fn draw_content(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // top padding
                Constraint::Length(3),  // header
                Constraint::Length(1),  // spacer
                Constraint::Min(10),   // main content
                Constraint::Length(2),  // bottom help
            ])
            .split(area);

        // Header
        let header = Paragraph::new(vec![
            Line::from(Span::styled(
                "  SETUP",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                format!("  v{VERSION}"),
                Style::default().fg(Color::DarkGray),
            )),
        ]);
        frame.render_widget(header, chunks[1]);

        // Main content based on step
        match self.step {
            OnboardingStep::Welcome => self.draw_welcome(frame, chunks[3]),
            OnboardingStep::WalletInput | OnboardingStep::Registering => {
                self.draw_wallet_input(frame, chunks[3]);
            }
            OnboardingStep::Done => self.draw_done(frame, chunks[3]),
        }

        // Bottom help
        let help_text = match self.step {
            OnboardingStep::Welcome => "  [Enter] Continue  [S] Skip  [Esc] Quit",
            OnboardingStep::WalletInput => "  [Enter] Submit  [Enter on empty] Skip  [Esc] Quit",
            OnboardingStep::Registering => "  Registering...",
            OnboardingStep::Done => "  [Enter] Continue",
        };
        let help = Paragraph::new(Line::from(Span::styled(
            help_text,
            Style::default().fg(Color::DarkGray),
        )));
        frame.render_widget(help, chunks[4]);
    }

    fn draw_welcome(&self, frame: &mut Frame, area: Rect) {
        let lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                "  Welcome to Compute",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  Decentralized GPU Infrastructure",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(""),
            Line::from(""),
            Line::from(Span::styled(
                "  To start earning $COMPUTE, you need to link",
                Style::default().fg(Color::Gray),
            )),
            Line::from(Span::styled(
                "  your Solana wallet address. This is the address",
                Style::default().fg(Color::Gray),
            )),
            Line::from(Span::styled(
                "  where your rewards will be sent.",
                Style::default().fg(Color::Gray),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  No private keys needed — just your public address.",
                Style::default().fg(Color::DarkGray),
            )),
        ];
        frame.render_widget(Paragraph::new(lines), area);
    }

    fn draw_wallet_input(&self, frame: &mut Frame, area: Rect) {
        let inner_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),  // label
                Constraint::Length(3),  // input box
                Constraint::Length(2),  // error/status
                Constraint::Min(0),    // remaining
            ])
            .split(area);

        // Label
        let label = Paragraph::new(vec![
            Line::from(Span::styled(
                "  SOLANA WALLET ADDRESS",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )),
        ]);
        frame.render_widget(label, inner_chunks[0]);

        // Input box — calculate visible area
        let input_area = Rect {
            x: inner_chunks[1].x + 2,
            y: inner_chunks[1].y,
            width: inner_chunks[1].width.saturating_sub(4),
            height: inner_chunks[1].height,
        };

        // Build the display string with cursor
        let display_text = if self.input.is_empty() && self.step == OnboardingStep::WalletInput {
            if self.cursor_blink {
                "█".to_string()
            } else {
                " ".to_string()
            }
        } else {
            let mut text = self.input.clone();
            if self.step == OnboardingStep::WalletInput && self.cursor_blink {
                if self.cursor_pos >= text.len() {
                    text.push('█');
                } else {
                    text.insert(self.cursor_pos, '█');
                }
            }
            text
        };

        let border_color = if self.error_message.is_some() {
            Color::Red
        } else if self.success
            || (!self.input.is_empty() && compute_solana::is_valid_address(self.input.trim()))
        {
            Color::Green
        } else {
            Color::DarkGray
        };

        let input_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color));

        let input_text = Paragraph::new(Line::from(Span::styled(
            display_text,
            Style::default().fg(Color::White),
        )))
        .block(input_block);

        frame.render_widget(input_text, input_area);

        // Error or validation message
        let msg_area = inner_chunks[2];
        if let Some(ref err) = self.error_message {
            let err_msg = Paragraph::new(Line::from(Span::styled(
                format!("  {err}"),
                Style::default().fg(Color::Red),
            )));
            frame.render_widget(err_msg, msg_area);
        } else if !self.input.is_empty() && compute_solana::is_valid_address(self.input.trim()) {
            let ok_msg = Paragraph::new(Line::from(Span::styled(
                "  Valid Solana address",
                Style::default().fg(Color::Green),
            )));
            frame.render_widget(ok_msg, msg_area);
        } else if !self.input.is_empty() {
            let hint = Paragraph::new(Line::from(Span::styled(
                format!("  {} / 32-44 characters", self.input.trim().len()),
                Style::default().fg(Color::DarkGray),
            )));
            frame.render_widget(hint, msg_area);
        }

        if self.registering {
            let reg_msg = Paragraph::new(Line::from(Span::styled(
                "  Registering with network...",
                Style::default().fg(Color::Yellow),
            )));
            frame.render_widget(reg_msg, inner_chunks[3]);
        }
    }

    fn draw_done(&self, frame: &mut Frame, area: Rect) {
        let lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                "  Node registered successfully",
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(Span::styled(
                format!("  Wallet: {}", self.input.trim()),
                Style::default().fg(Color::Gray),
            )),
        ];
        frame.render_widget(Paragraph::new(lines), area);
    }
}
