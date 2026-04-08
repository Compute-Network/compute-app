use ratatui::style::Color;

use compute_daemon::config::Config;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThemeMode {
    System,
    Dark,
    Light,
}

#[derive(Clone, Copy)]
pub struct Palette {
    pub text: Color,
    pub muted: Color,
    pub dim: Color,
    pub success: Color,
    pub warning: Color,
    pub danger: Color,
    pub globe_outline: Color,
    pub globe_land: Color,
    pub globe_nodes: Color,
    pub globe_me: Color,
}

impl ThemeMode {
    pub fn from_config_value(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "system" => Self::System,
            "light" => Self::Light,
            _ => Self::Dark,
        }
    }

    pub fn as_config_value(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Dark => "dark",
            Self::Light => "light",
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            Self::System => Self::Dark,
            Self::Dark => Self::Light,
            Self::Light => Self::System,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::System => "System",
            Self::Dark => "Dark",
            Self::Light => "Light",
        }
    }
}

pub fn current_mode() -> ThemeMode {
    let config = Config::load().unwrap_or_default();
    ThemeMode::from_config_value(&config.appearance.theme)
}

pub fn palette() -> Palette {
    palette_for(current_mode())
}

pub fn palette_for(mode: ThemeMode) -> Palette {
    match resolve_mode(mode) {
        ThemeMode::Dark => Palette {
            text: Color::White,
            muted: Color::Gray,
            dim: Color::DarkGray,
            success: Color::Green,
            warning: Color::Yellow,
            danger: Color::Red,
            globe_outline: Color::DarkGray,
            globe_land: Color::White,
            globe_nodes: Color::Green,
            globe_me: Color::Yellow,
        },
        ThemeMode::Light => Palette {
            text: Color::Black,
            muted: Color::DarkGray,
            dim: Color::Gray,
            success: Color::Green,
            warning: Color::Yellow,
            danger: Color::Red,
            globe_outline: Color::Gray,
            globe_land: Color::Black,
            globe_nodes: Color::Green,
            globe_me: Color::Blue,
        },
        ThemeMode::System => unreachable!("system mode should be resolved before palette selection"),
    }
}

fn resolve_mode(mode: ThemeMode) -> ThemeMode {
    match mode {
        ThemeMode::System => detect_system_mode(),
        other => other,
    }
}

fn detect_system_mode() -> ThemeMode {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("defaults")
            .args(["read", "-g", "AppleInterfaceStyle"])
            .output();
        if let Ok(output) = output {
            if output.status.success() {
                let value = String::from_utf8_lossy(&output.stdout).to_ascii_lowercase();
                if value.contains("dark") {
                    return ThemeMode::Dark;
                }
            }
        }
        return ThemeMode::Light;
    }

    #[cfg(not(target_os = "macos"))]
    {
        let gtk = std::env::var("GTK_THEME").unwrap_or_default().to_ascii_lowercase();
        let color_scheme = std::env::var("COLORFGBG").unwrap_or_default().to_ascii_lowercase();
        if gtk.contains("dark") || color_scheme.ends_with(";0") {
            ThemeMode::Dark
        } else {
            ThemeMode::Light
        }
    }
}
