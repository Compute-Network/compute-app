use serde::{Deserialize, Serialize};

use crate::hardware::{GpuBackend, HardwareInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageAccelerationTargetPreference {
    Auto,
    Cpu,
    Metal,
    Cuda,
    Vulkan,
    DirectMl,
}

impl StageAccelerationTargetPreference {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" | "reference" | "reference_cpu" | "reference-cpu" => Self::Cpu,
            "metal" => Self::Metal,
            "cuda" => Self::Cuda,
            "vulkan" => Self::Vulkan,
            "directml" | "direct_ml" | "direct-ml" | "dml" => Self::DirectMl,
            _ => Self::Auto,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::DirectMl => "directml",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageAccelerationProviderPreference {
    Auto,
    CpuRef,
    Ggml,
}

impl StageAccelerationProviderPreference {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu-ref" | "cpu_ref" | "cpuref" | "reference" | "reference-cpu" | "reference_cpu" => {
                Self::CpuRef
            }
            "ggml" => Self::Ggml,
            _ => Self::Auto,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::CpuRef => "cpu-ref",
            Self::Ggml => "ggml",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageAccelerationTarget {
    Cpu,
    Metal,
    Cuda,
    Vulkan,
    DirectMl,
}

impl StageAccelerationTarget {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::DirectMl => "directml",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageExecutionProvider {
    ReferenceCpu,
    Ggml,
}

impl StageExecutionProvider {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ReferenceCpu => "cpu-ref",
            Self::Ggml => "ggml",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageAccelerationPlan {
    pub requested_target: StageAccelerationTargetPreference,
    pub requested_provider: StageAccelerationProviderPreference,
    pub desired_target: Option<StageAccelerationTarget>,
    pub desired_provider: StageExecutionProvider,
    pub active_provider: StageExecutionProvider,
    pub active_target: StageAccelerationTarget,
    pub reason: String,
}

impl StageAccelerationPlan {
    pub fn for_real_forward(
        hw: &HardwareInfo,
        requested_target: StageAccelerationTargetPreference,
        requested_provider: StageAccelerationProviderPreference,
    ) -> Self {
        let desired_target = match requested_target {
            StageAccelerationTargetPreference::Auto => auto_target(hw),
            StageAccelerationTargetPreference::Cpu => Some(StageAccelerationTarget::Cpu),
            StageAccelerationTargetPreference::Metal => Some(StageAccelerationTarget::Metal),
            StageAccelerationTargetPreference::Cuda => Some(StageAccelerationTarget::Cuda),
            StageAccelerationTargetPreference::Vulkan => Some(StageAccelerationTarget::Vulkan),
            StageAccelerationTargetPreference::DirectMl => Some(StageAccelerationTarget::DirectMl),
        };
        let desired_provider = match requested_provider {
            StageAccelerationProviderPreference::Auto => match desired_target {
                Some(StageAccelerationTarget::Cpu) | None => StageExecutionProvider::ReferenceCpu,
                Some(
                    StageAccelerationTarget::Metal
                    | StageAccelerationTarget::Cuda
                    | StageAccelerationTarget::Vulkan
                    | StageAccelerationTarget::DirectMl,
                ) => StageExecutionProvider::Ggml,
            },
            StageAccelerationProviderPreference::CpuRef => StageExecutionProvider::ReferenceCpu,
            StageAccelerationProviderPreference::Ggml => StageExecutionProvider::Ggml,
        };

        let reason = match (desired_provider, desired_target) {
            (StageExecutionProvider::ReferenceCpu, Some(StageAccelerationTarget::Cpu) | None) => {
                "CPU reference execution selected explicitly".to_string()
            }
            (provider, Some(target)) => format!(
                "Requested stage provider `{}` targeting `{}`; current real_forward runtime remains on the defended `cpu-ref` path until that provider lands",
                provider.as_str(),
                target.as_str()
            ),
            (provider, None) => format!(
                "Requested stage provider `{}` with auto target selection; no stage-local accelerator target is active yet, so runtime remains on the defended `cpu-ref` path",
                provider.as_str()
            ),
        };

        Self {
            requested_target,
            requested_provider,
            desired_target,
            desired_provider,
            active_provider: StageExecutionProvider::ReferenceCpu,
            active_target: StageAccelerationTarget::Cpu,
            reason,
        }
    }

    pub fn summary_label(&self) -> String {
        match (self.desired_provider, self.desired_target) {
            (provider, Some(target))
                if provider != self.active_provider || target != StageAccelerationTarget::Cpu =>
            {
                format!(
                    "{} -> {}/{}",
                    self.active_provider.as_str(),
                    provider.as_str(),
                    target.as_str()
                )
            }
            (provider, None) if provider != self.active_provider => {
                format!("{} -> {}", self.active_provider.as_str(), provider.as_str())
            }
            _ => self.active_provider.as_str().to_string(),
        }
    }

    pub fn desired_target_or_cpu(&self) -> StageAccelerationTarget {
        self.desired_target.unwrap_or(StageAccelerationTarget::Cpu)
    }

    pub fn allows_cpu_fallback(&self) -> bool {
        matches!(
            self.requested_target,
            StageAccelerationTargetPreference::Auto | StageAccelerationTargetPreference::Cpu
        ) && matches!(
            self.requested_provider,
            StageAccelerationProviderPreference::Auto | StageAccelerationProviderPreference::CpuRef
        )
    }
}

fn auto_target(hw: &HardwareInfo) -> Option<StageAccelerationTarget> {
    if hw.gpus.iter().any(|gpu| matches!(gpu.backend, GpuBackend::Metal)) {
        return Some(StageAccelerationTarget::Metal);
    }
    if hw.gpus.iter().any(|gpu| matches!(gpu.backend, GpuBackend::Cuda)) {
        return Some(StageAccelerationTarget::Cuda);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{CpuInfo, DiskInfo, DockerStatus, GpuInfo, MemoryInfo, OsInfo};

    fn test_hw(gpus: Vec<GpuInfo>) -> HardwareInfo {
        HardwareInfo {
            cpu: CpuInfo { brand: "test".into(), cores: 8, threads: 16, frequency_mhz: 3000 },
            memory: MemoryInfo { total_gb: 32.0, available_gb: 16.0 },
            gpus,
            disk: DiskInfo { total_gb: 1000.0, available_gb: 500.0 },
            os: OsInfo { name: "TestOS".into(), version: "1".into(), arch: "test".into() },
            docker: DockerStatus { available: false, version: None },
        }
    }

    #[test]
    fn parses_preference_aliases() {
        assert_eq!(
            StageAccelerationTargetPreference::parse("auto"),
            StageAccelerationTargetPreference::Auto
        );
        assert_eq!(
            StageAccelerationTargetPreference::parse("cpu"),
            StageAccelerationTargetPreference::Cpu
        );
        assert_eq!(
            StageAccelerationTargetPreference::parse("reference_cpu"),
            StageAccelerationTargetPreference::Cpu
        );
        assert_eq!(
            StageAccelerationTargetPreference::parse("metal"),
            StageAccelerationTargetPreference::Metal
        );
        assert_eq!(
            StageAccelerationTargetPreference::parse("cuda"),
            StageAccelerationTargetPreference::Cuda
        );
        assert_eq!(
            StageAccelerationTargetPreference::parse("direct-ml"),
            StageAccelerationTargetPreference::DirectMl
        );
        assert_eq!(
            StageAccelerationProviderPreference::parse("auto"),
            StageAccelerationProviderPreference::Auto
        );
        assert_eq!(
            StageAccelerationProviderPreference::parse("cpu-ref"),
            StageAccelerationProviderPreference::CpuRef
        );
        assert_eq!(
            StageAccelerationProviderPreference::parse("ggml"),
            StageAccelerationProviderPreference::Ggml
        );
    }

    #[test]
    fn auto_prefers_metal_when_present() {
        let hw = test_hw(vec![GpuInfo {
            name: "Apple GPU".into(),
            vram_mb: 24576,
            backend: GpuBackend::Metal,
        }]);
        let plan = StageAccelerationPlan::for_real_forward(
            &hw,
            StageAccelerationTargetPreference::Auto,
            StageAccelerationProviderPreference::Auto,
        );
        assert_eq!(plan.desired_target, Some(StageAccelerationTarget::Metal));
        assert_eq!(plan.desired_provider, StageExecutionProvider::Ggml);
        assert_eq!(plan.active_provider, StageExecutionProvider::ReferenceCpu);
        assert_eq!(plan.summary_label(), "cpu-ref -> ggml/metal");
    }

    #[test]
    fn auto_prefers_cuda_when_present() {
        let hw = test_hw(vec![GpuInfo {
            name: "RTX".into(),
            vram_mb: 24576,
            backend: GpuBackend::Cuda,
        }]);
        let plan = StageAccelerationPlan::for_real_forward(
            &hw,
            StageAccelerationTargetPreference::Auto,
            StageAccelerationProviderPreference::Auto,
        );
        assert_eq!(plan.desired_target, Some(StageAccelerationTarget::Cuda));
        assert_eq!(plan.desired_provider, StageExecutionProvider::Ggml);
        assert_eq!(plan.summary_label(), "cpu-ref -> ggml/cuda");
    }

    #[test]
    fn auto_falls_back_to_cpu_without_gpu_target() {
        let hw =
            test_hw(vec![GpuInfo { name: "CPU".into(), vram_mb: 0, backend: GpuBackend::Cpu }]);
        let plan = StageAccelerationPlan::for_real_forward(
            &hw,
            StageAccelerationTargetPreference::Auto,
            StageAccelerationProviderPreference::Auto,
        );
        assert_eq!(plan.desired_target, None);
        assert_eq!(plan.desired_provider, StageExecutionProvider::ReferenceCpu);
        assert_eq!(plan.active_target, StageAccelerationTarget::Cpu);
        assert_eq!(plan.summary_label(), "cpu-ref");
    }

    #[test]
    fn explicit_cross_platform_target_is_preserved() {
        let hw = test_hw(vec![]);
        let plan = StageAccelerationPlan::for_real_forward(
            &hw,
            StageAccelerationTargetPreference::Vulkan,
            StageAccelerationProviderPreference::Auto,
        );
        assert_eq!(plan.desired_target, Some(StageAccelerationTarget::Vulkan));
        assert_eq!(plan.desired_provider, StageExecutionProvider::Ggml);
        assert_eq!(plan.summary_label(), "cpu-ref -> ggml/vulkan");
        assert!(!plan.allows_cpu_fallback());
    }

    #[test]
    fn auto_mode_allows_cpu_fallback() {
        let plan = StageAccelerationPlan::for_real_forward(
            &test_hw(vec![]),
            StageAccelerationTargetPreference::Auto,
            StageAccelerationProviderPreference::Auto,
        );
        assert!(plan.allows_cpu_fallback());
        assert_eq!(plan.desired_target_or_cpu(), StageAccelerationTarget::Cpu);
    }

    #[test]
    fn explicit_provider_request_disables_cpu_fallback() {
        let plan = StageAccelerationPlan::for_real_forward(
            &test_hw(vec![]),
            StageAccelerationTargetPreference::Auto,
            StageAccelerationProviderPreference::Ggml,
        );
        assert_eq!(plan.desired_provider, StageExecutionProvider::Ggml);
        assert!(!plan.allows_cpu_fallback());
        assert_eq!(plan.summary_label(), "cpu-ref -> ggml");
    }
}
