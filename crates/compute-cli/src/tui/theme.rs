use ratatui::style::Color;

use compute_daemon::config::Config;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThemeMode {
    Dark,
    Light,
}

#[derive(Clone, Copy)]
pub struct Palette {
    pub background: Color,
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
            "light" => Self::Light,
            _ => Self::Dark,
        }
    }

    pub fn as_config_value(self) -> &'static str {
        match self {
            Self::Dark => "dark",
            Self::Light => "light",
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            Self::Dark => Self::Light,
            Self::Light => Self::Dark,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
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
    match mode {
        ThemeMode::Dark => Palette {
            background: Color::Black,
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
            background: Color::White,
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
    }
}
