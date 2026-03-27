use anyhow::Result;

use compute_daemon::hardware;

use super::dashboard::Dashboard;
use super::splash::SplashScreen;

/// Initialize the terminal and run the TUI application.
pub fn run_splash_then_dashboard() -> Result<()> {
    let hw = hardware::detect();

    let mut terminal = ratatui::init();
    let result = run_inner(&mut terminal, hw);
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

fn run_inner(terminal: &mut ratatui::DefaultTerminal, hw: hardware::HardwareInfo) -> Result<()> {
    // Show splash first
    let mut splash = SplashScreen::new(&hw);
    let continue_to_dashboard = splash.run(terminal)?;

    if continue_to_dashboard {
        // Transition to dashboard
        let mut dashboard = Dashboard::new(hw);
        dashboard.run(terminal)?;
    }

    Ok(())
}
