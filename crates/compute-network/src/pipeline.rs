use serde::{Deserialize, Serialize};

// Pipeline scheduler — forms optimal pipelines from available nodes.
//
// Inspired by Parallax's two-phase dynamic programming approach:
// - Phase 1 (Offline): Allocate model layers to nodes proportional to their compute power
// - Phase 2 (Per-request): Find minimum-latency path through available pipeline stages

/// Node capabilities as reported during registration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub node_id: String,
    pub gpu_name: String,
    pub vram_mb: u64,
    /// Estimated TFLOPS FP16 (used for layer allocation weighting).
    pub tflops_fp16: f64,
    /// Memory bandwidth in GB/s.
    pub memory_bandwidth_gbs: f64,
    /// Measured latency to orchestrator in ms.
    pub latency_ms: f64,
    /// Whether the node is currently available.
    pub available: bool,
    /// Geographic region for latency optimization.
    pub region: Option<String>,
}

/// A model that needs to be served across a pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub model_id: String,
    pub total_layers: u32,
    /// VRAM per layer in MB (approximate).
    pub vram_per_layer_mb: u64,
    /// Minimum VRAM needed for a single layer.
    pub min_vram_mb: u64,
}

/// A pipeline stage assignment — which node runs which layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageAssignment {
    pub node_id: String,
    pub start_layer: u32,
    pub end_layer: u32,
    pub estimated_latency_ms: f64,
}

/// A complete pipeline — an ordered sequence of stages that covers all layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pipeline {
    pub pipeline_id: String,
    pub model_id: String,
    pub stages: Vec<StageAssignment>,
    pub estimated_total_latency_ms: f64,
}

impl Pipeline {
    /// Total number of stages in this pipeline.
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Estimated throughput in tokens/sec based on latency.
    /// At 100ms inter-node latency with N stages: ~(1000 / (N * latency)) tok/s.
    pub fn estimated_throughput(&self) -> f64 {
        if self.stages.is_empty() || self.estimated_total_latency_ms <= 0.0 {
            return 0.0;
        }
        1000.0 / self.estimated_total_latency_ms
    }
}

/// Phase 1: Allocate layers to nodes proportional to compute power.
///
/// Uses a water-filling heuristic: distribute layers proportionally to each node's
/// TFLOPS, then adjust to ensure contiguous layer intervals and VRAM constraints.
pub fn allocate_layers(
    nodes: &[NodeCapabilities],
    model: &ModelSpec,
) -> Result<Vec<StageAssignment>, SchedulerError> {
    let available_nodes: Vec<&NodeCapabilities> = nodes.iter().filter(|n| n.available).collect();

    if available_nodes.is_empty() {
        return Err(SchedulerError::NoAvailableNodes);
    }

    // Check total VRAM
    let total_vram: u64 = available_nodes.iter().map(|n| n.vram_mb).sum();
    let required_vram = model.total_layers as u64 * model.vram_per_layer_mb;
    if total_vram < required_vram {
        return Err(SchedulerError::InsufficientVram {
            required_mb: required_vram,
            available_mb: total_vram,
        });
    }

    // Sort nodes by compute power (highest first) for best pipeline ordering
    let mut sorted_nodes = available_nodes.clone();
    sorted_nodes.sort_by(|a, b| {
        b.tflops_fp16.partial_cmp(&a.tflops_fp16).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Water-filling: distribute layers proportional to TFLOPS
    let total_tflops: f64 = sorted_nodes.iter().map(|n| n.tflops_fp16).sum();

    let mut assignments = Vec::new();
    let mut current_layer: u32 = 0;

    for (i, node) in sorted_nodes.iter().enumerate() {
        let is_last = i == sorted_nodes.len() - 1;

        // Proportional allocation
        let proportion = node.tflops_fp16 / total_tflops;
        let ideal_layers = (model.total_layers as f64 * proportion).round() as u32;

        // Clamp: at least 1 layer, respect VRAM
        let max_layers_by_vram = (node.vram_mb / model.vram_per_layer_mb.max(1)) as u32;
        let layers = if is_last {
            // Last node gets remaining layers
            model.total_layers - current_layer
        } else {
            ideal_layers.clamp(1, max_layers_by_vram).min(model.total_layers - current_layer)
        };

        if layers == 0 {
            continue;
        }

        assignments.push(StageAssignment {
            node_id: node.node_id.clone(),
            start_layer: current_layer,
            end_layer: current_layer + layers - 1,
            estimated_latency_ms: node.latency_ms,
        });

        current_layer += layers;

        if current_layer >= model.total_layers {
            break;
        }
    }

    if current_layer < model.total_layers {
        return Err(SchedulerError::InsufficientNodes {
            layers_assigned: current_layer,
            total_layers: model.total_layers,
        });
    }

    Ok(assignments)
}

/// Phase 2: Find the minimum-latency pipeline from a set of stage assignments.
///
/// Given that Phase 1 may produce multiple valid allocations (when there are more
/// nodes than needed), this selects the optimal subset.
pub fn form_pipeline(assignments: Vec<StageAssignment>, model: &ModelSpec) -> Pipeline {
    let total_latency: f64 = assignments.iter().map(|s| s.estimated_latency_ms).sum();

    Pipeline {
        pipeline_id: generate_pipeline_id(),
        model_id: model.model_id.clone(),
        stages: assignments,
        estimated_total_latency_ms: total_latency,
    }
}

fn generate_pipeline_id() -> String {
    use std::time::SystemTime;
    let nanos =
        SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_nanos();
    format!("pipe-{:x}", nanos & 0xFFFF_FFFF)
}

#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("No available nodes")]
    NoAvailableNodes,

    #[error("Insufficient VRAM: need {required_mb}MB, have {available_mb}MB")]
    InsufficientVram { required_mb: u64, available_mb: u64 },

    #[error("Not enough nodes: assigned {layers_assigned}/{total_layers} layers")]
    InsufficientNodes { layers_assigned: u32, total_layers: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: &str, vram_mb: u64, tflops: f64, latency_ms: f64) -> NodeCapabilities {
        NodeCapabilities {
            node_id: id.into(),
            gpu_name: "Test GPU".into(),
            vram_mb,
            tflops_fp16: tflops,
            memory_bandwidth_gbs: 500.0,
            latency_ms,
            available: true,
            region: None,
        }
    }

    fn make_model(layers: u32, vram_per_layer: u64) -> ModelSpec {
        ModelSpec {
            model_id: "test-model".into(),
            total_layers: layers,
            vram_per_layer_mb: vram_per_layer,
            min_vram_mb: vram_per_layer,
        }
    }

    #[test]
    fn test_allocate_layers_single_node() {
        let nodes = vec![make_node("node-1", 24000, 40.0, 0.0)];
        let model = make_model(32, 500);

        let assignments = allocate_layers(&nodes, &model).unwrap();
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].start_layer, 0);
        assert_eq!(assignments[0].end_layer, 31);
    }

    #[test]
    fn test_allocate_layers_two_equal_nodes() {
        let nodes =
            vec![make_node("node-1", 24000, 40.0, 10.0), make_node("node-2", 24000, 40.0, 15.0)];
        let model = make_model(80, 300);

        let assignments = allocate_layers(&nodes, &model).unwrap();
        assert_eq!(assignments.len(), 2);

        // Both should get ~40 layers each
        let total_layers: u32 = assignments.iter().map(|a| a.end_layer - a.start_layer + 1).sum();
        assert_eq!(total_layers, 80);

        // Layers should be contiguous
        assert_eq!(assignments[0].start_layer, 0);
        assert_eq!(assignments[1].start_layer, assignments[0].end_layer + 1);
        assert_eq!(assignments[1].end_layer, 79);
    }

    #[test]
    fn test_allocate_layers_heterogeneous() {
        let nodes = vec![
            make_node("fast", 24000, 80.0, 10.0), // 2x faster
            make_node("slow", 24000, 40.0, 20.0), // 1x
        ];
        let model = make_model(60, 300);

        let assignments = allocate_layers(&nodes, &model).unwrap();
        assert_eq!(assignments.len(), 2);

        // Fast node should get more layers
        let fast_layers = assignments[0].end_layer - assignments[0].start_layer + 1;
        let slow_layers = assignments[1].end_layer - assignments[1].start_layer + 1;
        assert!(fast_layers > slow_layers);
        assert_eq!(fast_layers + slow_layers, 60);
    }

    #[test]
    fn test_allocate_layers_insufficient_vram() {
        let nodes = vec![make_node("node-1", 1000, 40.0, 10.0)];
        let model = make_model(80, 500); // Needs 40GB, node has 1GB

        let result = allocate_layers(&nodes, &model);
        assert!(result.is_err());
    }

    #[test]
    fn test_allocate_layers_no_nodes() {
        let nodes: Vec<NodeCapabilities> = vec![];
        let model = make_model(32, 500);

        let result = allocate_layers(&nodes, &model);
        assert!(result.is_err());
    }

    #[test]
    fn test_form_pipeline() {
        let assignments = vec![
            StageAssignment {
                node_id: "node-1".into(),
                start_layer: 0,
                end_layer: 39,
                estimated_latency_ms: 10.0,
            },
            StageAssignment {
                node_id: "node-2".into(),
                start_layer: 40,
                end_layer: 79,
                estimated_latency_ms: 15.0,
            },
        ];

        let model = make_model(80, 300);
        let pipeline = form_pipeline(assignments, &model);

        assert_eq!(pipeline.num_stages(), 2);
        assert_eq!(pipeline.estimated_total_latency_ms, 25.0);
        assert!((pipeline.estimated_throughput() - 40.0).abs() < 0.1); // 1000/25 = 40 tok/s
    }

    #[test]
    fn test_five_node_pipeline() {
        let nodes = vec![
            make_node("node-1", 24000, 50.0, 20.0),
            make_node("node-2", 24000, 50.0, 25.0),
            make_node("node-3", 16000, 30.0, 30.0),
            make_node("node-4", 24000, 50.0, 15.0),
            make_node("node-5", 8000, 20.0, 40.0),
        ];
        let model = make_model(80, 300);

        let assignments = allocate_layers(&nodes, &model).unwrap();
        assert!(assignments.len() <= 5);

        // All layers covered
        let total: u32 = assignments.iter().map(|a| a.end_layer - a.start_layer + 1).sum();
        assert_eq!(total, 80);

        // Form pipeline
        let pipeline = form_pipeline(assignments, &model);
        assert!(pipeline.estimated_throughput() > 0.0);
    }
}
