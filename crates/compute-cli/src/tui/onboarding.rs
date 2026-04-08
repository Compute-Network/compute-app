use std::sync::mpsc;
use std::time::{Duration, Instant};

use base64::Engine;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use serde::{de::DeserializeOwned, Deserialize};
use ratatui::{
    DefaultTerminal, Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

use compute_daemon::config::{self, Config};

use super::globe::Globe;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const API_BASE: &str = "https://api.computenetwork.sh";

#[derive(Debug, Deserialize)]
struct DeviceStartResult {
    device_code: String,
    auth_url: String,
    expires_in: u64,
    poll_interval: u64,
}

#[derive(Debug, Deserialize)]
struct DevicePollResult {
    status: String,
    node_token: Option<String>,
    wallet_address: Option<String>,
}

#[derive(Debug)]
struct WalletAuthResult {
    wallet_address: String,
    node_token: String,
}

#[derive(Debug, Deserialize)]
struct NodeTokenClaims {
    #[serde(rename = "walletAddress")]
    wallet_address: Option<String>,
}

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
    error_message: Option<String>,
    success: bool,
    step: OnboardingStep,
    llama_found: bool,
    install_rx: Option<mpsc::Receiver<Result<(), String>>>,
    install_status: Option<String>,
    auth_rx: Option<mpsc::Receiver<Result<WalletAuthResult, String>>>,
    auth_status: Option<String>,
}

#[derive(PartialEq)]
enum OnboardingStep {
    Welcome,
    WalletInput,
    DependencyCheck,
    Installing,
    Done,
}

impl OnboardingScreen {
    pub fn new() -> Self {
        let mut globe = Globe::new();
        globe.set_mock_nodes();

        Self {
            globe,
            input: String::new(),
            error_message: None,
            success: false,
            step: OnboardingStep::Welcome,
            llama_found: find_llama_server(),
            install_rx: None,
            install_status: None,
            auth_rx: None,
            auth_status: None,
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

            if let Some(result) = self.tick() {
                return Ok(result);
            }
        }
    }

    fn handle_key(&mut self, code: KeyCode, _modifiers: KeyModifiers) -> Option<OnboardingResult> {
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
                    KeyCode::Enter => {
                        if self.auth_rx.is_none() {
                            self.start_wallet_auth();
                        }
                    }
                    KeyCode::Char('s') | KeyCode::Char('S') => return Some(OnboardingResult::Skipped),
                    _ => {}
                }
                None
            }
            OnboardingStep::DependencyCheck => {
                // Install failed — Enter to retry
                if matches!(code, KeyCode::Enter) {
                    self.start_llama_install();
                    self.step = OnboardingStep::Installing;
                }
                None
            }
            OnboardingStep::Installing => None,
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

    fn tick(&mut self) -> Option<OnboardingResult> {
        self.globe.tick();

        if self.step == OnboardingStep::WalletInput
            && let Some(ref rx) = self.auth_rx
            && let Ok(result) = rx.try_recv()
        {
            self.auth_rx = None;
            match result {
                Ok(auth) => {
                    self.input = auth.wallet_address.clone();
                    self.auth_status = Some("Wallet connected".into());
                    self.success = true;

                    if let Ok(mut config) = Config::load() {
                        config.wallet.public_address = auth.wallet_address.clone();
                        config.wallet.node_token = auth.node_token;
                        let _ = config::ensure_dirs();
                        let _ = config.save();
                    }

                    if self.llama_found {
                        self.step = OnboardingStep::Done;
                        return Some(OnboardingResult::WalletSet(auth.wallet_address));
                    }

                    self.start_llama_install();
                    self.step = OnboardingStep::Installing;
                }
                Err(e) => {
                    self.error_message = Some(e);
                    self.auth_status = None;
                }
            }
        }

        // Poll install progress
        if self.step == OnboardingStep::Installing
            && let Some(ref rx) = self.install_rx
            && let Ok(result) = rx.try_recv()
        {
            match result {
                Ok(()) => {
                    self.llama_found = true;
                    self.install_status = Some("Installed successfully".into());
                    self.step = OnboardingStep::Done;
                    self.success = true;
                    let addr = self.input.trim().to_string();
                    return Some(OnboardingResult::WalletSet(addr));
                }
                Err(e) => {
                    self.install_status = Some(format!("Install failed: {e}"));
                    self.step = OnboardingStep::DependencyCheck;
                }
            }
        }

        None
    }

    fn start_wallet_auth(&mut self) {
        let (tx, rx) = mpsc::channel();
        self.auth_rx = Some(rx);
        self.auth_status = Some("Opening browser for wallet login...".into());
        self.error_message = None;

        std::thread::spawn(move || {
            let result = (|| -> Result<WalletAuthResult, String> {
                let client = reqwest::blocking::Client::builder()
                    .timeout(Duration::from_secs(15))
                    .build()
                    .map_err(|e| e.to_string())?;

                let start = client
                    .post(format!("{API_BASE}/v1/auth/device/start"))
                    .json(&serde_json::json!({ "purpose": "node_session" }))
                    .send()
                    .map_err(|e| e.to_string())?
                    .error_for_status()
                    .map_err(|e| e.to_string())?
                    .json::<DeviceStartResult>()
                    .map_err(|e| e.to_string())?;

                open_url(&start.auth_url);

                let started_at = Instant::now();
                loop {
                    if started_at.elapsed().as_secs() >= start.expires_in {
                        return Err("Wallet login expired. Press Enter to try again.".into());
                    }

                    let res = client
                        .get(format!("{API_BASE}/v1/auth/device/poll/{}", start.device_code))
                        .send()
                        .map_err(|e| e.to_string())?;

                    if res.status() == reqwest::StatusCode::ACCEPTED {
                        std::thread::sleep(Duration::from_secs(start.poll_interval.max(1)));
                        continue;
                    }
                    if res.status() == reqwest::StatusCode::GONE
                        || res.status() == reqwest::StatusCode::NOT_FOUND
                    {
                        return Err("Wallet login expired. Press Enter to try again.".into());
                    }

                    let body = res
                        .error_for_status()
                        .map_err(|e| e.to_string())?
                        .json::<DevicePollResult>()
                        .map_err(|e| e.to_string())?;

                    if body.status == "complete" {
                        let node_token = body
                            .node_token
                            .ok_or_else(|| "Missing node token in auth response".to_string())?;
                        let wallet_address = body
                            .wallet_address
                            .or_else(|| wallet_address_from_node_token(&node_token))
                            .ok_or_else(|| "Missing wallet address in auth response".to_string())?;
                        return Ok(WalletAuthResult {
                            wallet_address,
                            node_token,
                        });
                    }

                    std::thread::sleep(Duration::from_secs(start.poll_interval.max(1)));
                }
            })();

            let _ = tx.send(result);
        });
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

    fn start_llama_install(&mut self) {
        let (tx, rx) = mpsc::channel();
        self.install_rx = Some(rx);
        self.install_status = Some("Installing llama.cpp via Homebrew...".into());

        std::thread::spawn(move || {
            let result = std::process::Command::new("brew")
                .args(["install", "llama.cpp"])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .status();

            let _ = tx.send(match result {
                Ok(status) if status.success() => Ok(()),
                Ok(status) => Err(format!("brew exited with code {status}")),
                Err(e) => Err(format!(
                    "Homebrew not found: {e}. Install manually: https://github.com/ggerganov/llama.cpp"
                )),
            });
        });
    }

    fn draw_content(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // top padding
                Constraint::Length(3), // header
                Constraint::Length(1), // spacer
                Constraint::Min(10),   // main content
                Constraint::Length(2), // bottom help
            ])
            .split(area);

        // Header
        let header = Paragraph::new(vec![
            Line::from(Span::styled(
                "  SETUP",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(format!("  v{VERSION}"), Style::default().fg(Color::DarkGray))),
        ]);
        frame.render_widget(header, chunks[1]);

        // Main content based on step
        match self.step {
            OnboardingStep::Welcome => self.draw_welcome(frame, chunks[3]),
            OnboardingStep::WalletInput => {
                self.draw_wallet_input(frame, chunks[3]);
            }
            OnboardingStep::DependencyCheck | OnboardingStep::Installing => {
                self.draw_dependency_check(frame, chunks[3]);
            }
            OnboardingStep::Done => self.draw_done(frame, chunks[3]),
        }

        // Bottom help
        let help_text = match self.step {
            OnboardingStep::Welcome => "  [Enter] Continue  [S] Skip  [Esc] Quit",
            OnboardingStep::WalletInput => "  [Enter] Open wallet login  [S] Skip  [Esc] Quit",
            OnboardingStep::DependencyCheck => "  [Enter] Retry  [Esc] Quit",
            OnboardingStep::Installing => "  Installing...",
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
                "  To start earning $COMPUTE, you need to connect",
                Style::default().fg(Color::Gray),
            )),
            Line::from(Span::styled(
                "  your Solana wallet in the browser. Compute will",
                Style::default().fg(Color::Gray),
            )),
            Line::from(Span::styled(
                "  create your wallet account and bind this node.",
                Style::default().fg(Color::Gray),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  No private keys touch the CLI.",
                Style::default().fg(Color::DarkGray),
            )),
        ];
        frame.render_widget(Paragraph::new(lines), area);
    }

    fn draw_wallet_input(&self, frame: &mut Frame, area: Rect) {
        let inner_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Length(5),
                Constraint::Length(2),
                Constraint::Min(0),    // remaining
            ])
            .split(area);

        // Label
        let label = Paragraph::new(vec![Line::from(Span::styled(
            "  CONNECT WALLET",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ))]);
        frame.render_widget(label, inner_chunks[0]);

        let border_color = if self.error_message.is_some() {
            Color::Red
        } else if self.success {
            Color::Green
        } else {
            Color::DarkGray
        };

        let input_block =
            Block::default().borders(Borders::ALL).border_style(Style::default().fg(border_color));

        let status = self.auth_status.as_deref().unwrap_or("Press Enter to authenticate in your browser.");
        let input_text = Paragraph::new(vec![
            Line::from(Span::styled(
                "  Compute uses wallet auth like compute-code.",
                Style::default().fg(Color::Gray),
            )),
            Line::from(Span::styled(
                format!("  {status}"),
                Style::default().fg(Color::White),
            )),
        ])
        .block(input_block);

        frame.render_widget(input_text, inner_chunks[1]);

        // Error or validation message
        let msg_area = inner_chunks[2];
        if let Some(ref err) = self.error_message {
            let err_msg = Paragraph::new(Line::from(Span::styled(
                format!("  {err}"),
                Style::default().fg(Color::Red),
            )));
            frame.render_widget(err_msg, msg_area);
        } else if self.success && !self.input.is_empty() {
            let ok_msg = Paragraph::new(Line::from(Span::styled(
                format!("  Connected wallet: {}", self.input.trim()),
                Style::default().fg(Color::Green),
            )));
            frame.render_widget(ok_msg, msg_area);
        }
    }

    fn draw_dependency_check(&self, frame: &mut Frame, area: Rect) {
        let mut lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                "  DEPENDENCIES",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ];

        if self.step == OnboardingStep::Installing {
            let status = self.install_status.as_deref().unwrap_or("Installing...");
            lines.push(Line::from(Span::styled(
                format!("  {status}"),
                Style::default().fg(Color::Yellow),
            )));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "  This may take a minute...",
                Style::default().fg(Color::DarkGray),
            )));
        } else {
            // Only shown after a failed install attempt
            if let Some(ref status) = self.install_status {
                lines.push(Line::from(Span::styled(
                    format!("  {status}"),
                    Style::default().fg(Color::Red),
                )));
                lines.push(Line::from(""));
            }

            if cfg!(target_os = "macos") {
                lines.push(Line::from(Span::styled(
                    "  Press [Enter] to retry, or install manually:",
                    Style::default().fg(Color::Gray),
                )));
                lines.push(Line::from(Span::styled(
                    "  brew install llama.cpp",
                    Style::default().fg(Color::White),
                )));
            } else {
                lines.push(Line::from(Span::styled(
                    "  Install manually and restart:",
                    Style::default().fg(Color::Gray),
                )));
                lines.push(Line::from(Span::styled(
                    "  https://github.com/ggerganov/llama.cpp",
                    Style::default().fg(Color::White),
                )));
            }
        }

        frame.render_widget(Paragraph::new(lines), area);
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

/// Check if llama-server is available on the system.
fn find_llama_server() -> bool {
    // Check PATH
    if let Ok(output) = std::process::Command::new("which").arg("llama-server").output()
        && output.status.success()
    {
        return true;
    }

    // Check common locations
    let candidates = ["/usr/local/bin/llama-server", "/opt/homebrew/bin/llama-server"];
    candidates.iter().any(|p| std::path::Path::new(p).exists())
}

fn open_url(url: &str) {
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(url).spawn();
    }
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn();
    }
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    }
}

fn wallet_address_from_node_token(token: &str) -> Option<String> {
    let payload = token.split('.').nth(1)?;
    let claims = decode_jwt_payload::<NodeTokenClaims>(payload)?;
    claims.wallet_address
}

fn decode_jwt_payload<T: DeserializeOwned>(payload: &str) -> Option<T> {
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(payload).ok()?;
    serde_json::from_slice(&bytes).ok()
}
