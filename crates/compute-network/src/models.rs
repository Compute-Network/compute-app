use serde::{Deserialize, Serialize};

/// Supported model catalog with sharding metadata.
/// Each model defines its layer count, VRAM requirements, and shard configurations.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCatalog {
    pub models: Vec<ModelDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefinition {
    /// Unique model identifier (e.g., "llama-3.1-70b-q4").
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Model family.
    pub family: ModelFamily,
    /// Total transformer layers.
    pub total_layers: u32,
    /// VRAM per layer in MB (approximate, at the specified quantization).
    pub vram_per_layer_mb: u64,
    /// Total model size in MB.
    pub total_size_mb: u64,
    /// Quantization level.
    pub quantization: Quantization,
    /// Minimum total VRAM across all pipeline nodes.
    pub min_total_vram_mb: u64,
    /// Recommended number of pipeline stages.
    pub recommended_stages: u32,
    /// HuggingFace model ID or download URL.
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelFamily {
    Llama,
    DeepSeek,
    Qwen,
    Mistral,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Quantization {
    FP16,
    Q8,
    Q4,
    Q2,
}

impl std::fmt::Display for Quantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Quantization::FP16 => write!(f, "FP16"),
            Quantization::Q8 => write!(f, "Q8"),
            Quantization::Q4 => write!(f, "Q4"),
            Quantization::Q2 => write!(f, "Q2"),
        }
    }
}

/// A pre-computed shard — a contiguous range of layers saved as a standalone model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardConfig {
    pub model_id: String,
    pub shard_index: u32,
    pub start_layer: u32,
    pub end_layer: u32,
    pub size_mb: u64,
    /// Download URL or local cache path.
    pub source: String,
    /// SHA256 checksum for verification.
    pub checksum: Option<String>,
}

impl ModelCatalog {
    /// Built-in model catalog with common configurations.
    pub fn default_catalog() -> Self {
        Self {
            models: vec![
                ModelDefinition {
                    id: "llama-3.1-8b-q4".into(),
                    name: "Llama 3.1 8B (Q4)".into(),
                    family: ModelFamily::Llama,
                    total_layers: 32,
                    vram_per_layer_mb: 150,
                    total_size_mb: 4800,
                    quantization: Quantization::Q4,
                    min_total_vram_mb: 6000,
                    recommended_stages: 1,
                    source: "meta-llama/Llama-3.1-8B".into(),
                },
                ModelDefinition {
                    id: "llama-3.1-70b-q4".into(),
                    name: "Llama 3.1 70B (Q4)".into(),
                    family: ModelFamily::Llama,
                    total_layers: 80,
                    vram_per_layer_mb: 500,
                    total_size_mb: 40000,
                    quantization: Quantization::Q4,
                    min_total_vram_mb: 42000,
                    recommended_stages: 3,
                    source: "meta-llama/Llama-3.1-70B".into(),
                },
                ModelDefinition {
                    id: "llama-3.1-70b-fp16".into(),
                    name: "Llama 3.1 70B (FP16)".into(),
                    family: ModelFamily::Llama,
                    total_layers: 80,
                    vram_per_layer_mb: 1750,
                    total_size_mb: 140000,
                    quantization: Quantization::FP16,
                    min_total_vram_mb: 144000,
                    recommended_stages: 5,
                    source: "meta-llama/Llama-3.1-70B".into(),
                },
                ModelDefinition {
                    id: "deepseek-r1-q4".into(),
                    name: "DeepSeek R1 671B (Q4 MoE)".into(),
                    family: ModelFamily::DeepSeek,
                    total_layers: 61,
                    vram_per_layer_mb: 2800,
                    total_size_mb: 170000,
                    quantization: Quantization::Q4,
                    min_total_vram_mb: 180000,
                    recommended_stages: 8,
                    source: "deepseek-ai/DeepSeek-R1".into(),
                },
                ModelDefinition {
                    id: "qwen-2.5-72b-q4".into(),
                    name: "Qwen 2.5 72B (Q4)".into(),
                    family: ModelFamily::Qwen,
                    total_layers: 80,
                    vram_per_layer_mb: 520,
                    total_size_mb: 41600,
                    quantization: Quantization::Q4,
                    min_total_vram_mb: 43000,
                    recommended_stages: 3,
                    source: "Qwen/Qwen2.5-72B".into(),
                },
                ModelDefinition {
                    id: "mistral-7b-q4".into(),
                    name: "Mistral 7B (Q4)".into(),
                    family: ModelFamily::Mistral,
                    total_layers: 32,
                    vram_per_layer_mb: 130,
                    total_size_mb: 4200,
                    quantization: Quantization::Q4,
                    min_total_vram_mb: 5000,
                    recommended_stages: 1,
                    source: "mistralai/Mistral-7B-v0.3".into(),
                },
            ],
        }
    }

    /// Find a model by ID.
    pub fn find(&self, model_id: &str) -> Option<&ModelDefinition> {
        self.models.iter().find(|m| m.id == model_id)
    }

    /// List all models that can fit in the given total VRAM (across pipeline).
    pub fn models_for_vram(&self, total_vram_mb: u64) -> Vec<&ModelDefinition> {
        self.models.iter().filter(|m| m.min_total_vram_mb <= total_vram_mb).collect()
    }
}

impl ModelDefinition {
    /// Generate shard configs for a given number of pipeline stages.
    pub fn shard_for_stages(&self, num_stages: u32) -> Vec<ShardConfig> {
        let layers_per_stage = self.total_layers / num_stages;
        let remainder = self.total_layers % num_stages;

        let mut shards = Vec::new();
        let mut current_layer = 0;

        for i in 0..num_stages {
            let extra = if i < remainder { 1 } else { 0 };
            let num_layers = layers_per_stage + extra;
            let end_layer = current_layer + num_layers - 1;

            shards.push(ShardConfig {
                model_id: self.id.clone(),
                shard_index: i,
                start_layer: current_layer,
                end_layer,
                size_mb: num_layers as u64 * self.vram_per_layer_mb,
                source: format!(
                    "{}/shard-{}-layers-{}-{}",
                    self.source, i, current_layer, end_layer
                ),
                checksum: None,
            });

            current_layer = end_layer + 1;
        }

        shards
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_catalog() {
        let catalog = ModelCatalog::default_catalog();
        assert!(!catalog.models.is_empty());
        assert!(catalog.find("llama-3.1-70b-q4").is_some());
        assert!(catalog.find("nonexistent").is_none());
    }

    #[test]
    fn test_models_for_vram() {
        let catalog = ModelCatalog::default_catalog();

        // 8GB VRAM should fit small models
        let small = catalog.models_for_vram(8000);
        assert!(!small.is_empty());
        assert!(small.iter().any(|m| m.id == "mistral-7b-q4"));

        // 48GB should fit 70B models
        let medium = catalog.models_for_vram(48000);
        assert!(medium.iter().any(|m| m.id == "llama-3.1-70b-q4"));
    }

    #[test]
    fn test_shard_for_stages() {
        let catalog = ModelCatalog::default_catalog();
        let model = catalog.find("llama-3.1-70b-q4").unwrap();

        let shards = model.shard_for_stages(4);
        assert_eq!(shards.len(), 4);

        // Total layers should match
        let total: u32 = shards.iter().map(|s| s.end_layer - s.start_layer + 1).sum();
        assert_eq!(total, 80);

        // First shard starts at 0
        assert_eq!(shards[0].start_layer, 0);

        // Last shard ends at 79
        assert_eq!(shards[3].end_layer, 79);

        // Contiguous
        for i in 1..shards.len() {
            assert_eq!(shards[i].start_layer, shards[i - 1].end_layer + 1);
        }
    }

    #[test]
    fn test_shard_uneven_split() {
        let catalog = ModelCatalog::default_catalog();
        let model = catalog.find("llama-3.1-8b-q4").unwrap();

        // 32 layers / 3 stages = 11, 11, 10
        let shards = model.shard_for_stages(3);
        assert_eq!(shards.len(), 3);

        let total: u32 = shards.iter().map(|s| s.end_layer - s.start_layer + 1).sum();
        assert_eq!(total, 32);
    }
}
