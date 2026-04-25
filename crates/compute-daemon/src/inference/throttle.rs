//! Resource throttling and pipeline failover.
//!
//! Monitors system idle state and controls the inference engine accordingly:
//! - When the user is idle: full compute power available
//! - When the user is active: reduce or pause inference
//! - When heavy use detected: gracefully exit pipeline
//!
//! Also handles process priority management and VRAM budgeting.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::config::NodeConfig;
use crate::idle::{IdleDetector, IdleState};

/// Throttle policy based on idle state transitions.
#[derive(Debug, Clone, PartialEq)]
pub enum ThrottleAction {
    /// Full speed — user is idle, all resources available.
    FullSpeed,
    /// Reduced — user is lightly active, limit GPU/CPU usage.
    Reduced { gpu_percent: u8, cpu_percent: u8 },
    /// Pause — user is heavily active, stop inference after current token.
    Pause,
    /// Exit pipeline — sustained heavy use, tell orchestrator to reassign our layers.
    ExitPipeline,
}

/// Manages resource throttling for the inference engine.
pub struct ThrottleController {
    config: NodeConfig,
    idle_detector: IdleDetector,
    current_action: ThrottleAction,
    /// How long the node has been in heavy use continuously.
    heavy_use_since: Option<Instant>,
    /// How long to wait in heavy use before exiting pipeline.
    exit_threshold: Duration,
    /// Whether the inference engine is currently paused.
    paused: Arc<AtomicBool>,
    /// Nice value to set on inference process.
    nice_value: i32,
}

impl ThrottleController {
    pub fn new(config: NodeConfig) -> Self {
        let idle_detector = IdleDetector::new(config.idle_threshold_minutes);

        Self {
            config,
            idle_detector,
            current_action: ThrottleAction::FullSpeed,
            heavy_use_since: None,
            exit_threshold: Duration::from_secs(30), // Exit pipeline after 30s of heavy use
            paused: Arc::new(AtomicBool::new(false)),
            nice_value: 19, // Lowest priority
        }
    }

    /// Check system state and return the appropriate throttle action.
    /// Call this every 1-2 seconds.
    pub fn check(&mut self) -> ThrottleAction {
        let idle_state = self.idle_detector.detect();

        let action = match idle_state {
            IdleState::Idle => {
                self.heavy_use_since = None;
                ThrottleAction::FullSpeed
            }
            IdleState::LightUse => {
                self.heavy_use_since = None;
                ThrottleAction::Reduced {
                    gpu_percent: self.config.max_gpu_usage,
                    cpu_percent: (self.config.max_cpu_usage as f32 * 0.5) as u8,
                }
            }
            IdleState::HeavyUse | IdleState::Paused => {
                // Track how long we've been in heavy use
                let start = self.heavy_use_since.get_or_insert(Instant::now());
                if start.elapsed() > self.exit_threshold {
                    ThrottleAction::ExitPipeline
                } else {
                    ThrottleAction::Pause
                }
            }
        };

        if action != self.current_action {
            info!(
                "Throttle action changed: {:?} → {:?} (idle_state={idle_state})",
                self.current_action, action
            );
        }

        self.current_action = action.clone();
        action
    }

    /// Get the current throttle action without re-checking.
    pub fn current_action(&self) -> &ThrottleAction {
        &self.current_action
    }

    /// Get whether inference should be paused.
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    /// Get the shared paused flag (for the inference engine to check).
    pub fn paused_flag(&self) -> Arc<AtomicBool> {
        self.paused.clone()
    }

    /// Apply the throttle action to the system.
    pub fn apply(&mut self, action: &ThrottleAction) {
        match action {
            ThrottleAction::FullSpeed => {
                self.paused.store(false, Ordering::SeqCst);
                self.nice_value = 10; // Medium-low priority when idle
                debug!("Throttle: full speed, nice={}", self.nice_value);
            }
            ThrottleAction::Reduced { gpu_percent, cpu_percent } => {
                self.paused.store(false, Ordering::SeqCst);
                self.nice_value = 15; // Lower priority
                debug!(
                    "Throttle: reduced (GPU {}%, CPU {}%), nice={}",
                    gpu_percent, cpu_percent, self.nice_value
                );
            }
            ThrottleAction::Pause => {
                self.paused.store(true, Ordering::SeqCst);
                self.nice_value = 19; // Lowest priority
                debug!("Throttle: paused, nice={}", self.nice_value);
            }
            ThrottleAction::ExitPipeline => {
                self.paused.store(true, Ordering::SeqCst);
                self.nice_value = 19;
                warn!("Throttle: exit pipeline requested (sustained heavy use)");
            }
        }
    }

    /// Get the recommended nice value for the inference process.
    pub fn nice_value(&self) -> i32 {
        self.nice_value
    }

    /// Set the exit threshold (how long heavy use before exiting pipeline).
    pub fn set_exit_threshold(&mut self, duration: Duration) {
        self.exit_threshold = duration;
    }
}

/// VRAM budget calculator.
/// Determines how much VRAM to reserve for inference vs user apps.
pub struct VramBudget {
    /// Total VRAM available in MB.
    pub total_mb: u64,
    /// VRAM reserved for the OS and user applications.
    pub reserved_mb: u64,
    /// VRAM available for inference.
    pub available_mb: u64,
}

impl VramBudget {
    /// Calculate VRAM budget given total VRAM and a usage percentage cap.
    pub fn calculate(total_vram_mb: u64, max_usage_percent: u8) -> Self {
        let max_pct = max_usage_percent.min(95) as u64; // Never use more than 95%
        let available = (total_vram_mb * max_pct) / 100;
        let reserved = total_vram_mb - available;

        Self { total_mb: total_vram_mb, reserved_mb: reserved, available_mb: available }
    }

    /// Check if a shard fits within the VRAM budget.
    pub fn can_fit(&self, shard_vram_mb: u64) -> bool {
        shard_vram_mb <= self.available_mb
    }

    /// Get the maximum number of model layers that fit, given VRAM per layer.
    pub fn max_layers(&self, vram_per_layer_mb: u64) -> u32 {
        if vram_per_layer_mb == 0 {
            return 0;
        }
        (self.available_mb / vram_per_layer_mb) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vram_budget() {
        let budget = VramBudget::calculate(24576, 90); // 24GB, 90% usage
        assert_eq!(budget.total_mb, 24576);
        assert_eq!(budget.available_mb, 22118); // 90% of 24576
        assert_eq!(budget.reserved_mb, 2458);
    }

    #[test]
    fn test_vram_budget_cap_at_95() {
        let budget = VramBudget::calculate(24576, 100); // 100% → capped at 95%
        assert_eq!(budget.available_mb, (24576 * 95) / 100);
    }

    #[test]
    fn test_vram_can_fit() {
        let budget = VramBudget::calculate(24576, 90);
        assert!(budget.can_fit(20000));
        assert!(!budget.can_fit(24000));
    }

    #[test]
    fn test_vram_max_layers() {
        let budget = VramBudget::calculate(24576, 90); // ~22GB available
        assert_eq!(budget.max_layers(200), 110); // 22118 / 200 = 110
        assert_eq!(budget.max_layers(500), 44); // 22118 / 500 = 44
        assert_eq!(budget.max_layers(0), 0);
    }

    #[test]
    fn test_throttle_controller_initial() {
        let config = NodeConfig {
            name: "test".into(),
            max_gpu_usage: 90,
            max_cpu_usage: 50,
            idle_threshold_minutes: 5,
            pause_on_battery: true,
            pause_on_fullscreen: true,
            caffeinate_when_running: true,
        };
        let controller = ThrottleController::new(config);
        assert_eq!(*controller.current_action(), ThrottleAction::FullSpeed);
        assert!(!controller.is_paused());
    }

    #[test]
    fn test_throttle_apply_pause() {
        let config = NodeConfig {
            name: "test".into(),
            max_gpu_usage: 90,
            max_cpu_usage: 50,
            idle_threshold_minutes: 5,
            pause_on_battery: true,
            pause_on_fullscreen: true,
            caffeinate_when_running: true,
        };
        let mut controller = ThrottleController::new(config);

        controller.apply(&ThrottleAction::Pause);
        assert!(controller.is_paused());
        assert_eq!(controller.nice_value(), 19);

        controller.apply(&ThrottleAction::FullSpeed);
        assert!(!controller.is_paused());
        assert_eq!(controller.nice_value(), 10);
    }
}
