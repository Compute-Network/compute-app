//! Integration test: multi-node pipeline passing activations over QUIC.
//!
//! Simulates the full pipeline flow:
//! 1. Nodes bind to local addresses
//! 2. Pipeline scheduler allocates layers
//! 3. Activations flow through stages via QUIC
//! 4. Control messages for setup and teardown

use std::net::SocketAddr;

use compute_network::pipeline::{ModelSpec, NodeCapabilities, allocate_layers, form_pipeline};
use compute_network::transport::node::TransportNode;
use compute_network::transport::protocol::{
    ActivationPayload, ControlMessage, PipelineMessage, PingMessage, PongMessage, TensorDtype,
};

/// Simulate a 3-stage pipeline: scheduler allocates layers, then activations flow through.
#[tokio::test]
async fn test_three_stage_pipeline() {
    // Define the model
    let model = ModelSpec {
        model_id: "llama-3.1-8b-q4".into(),
        total_layers: 32,
        vram_per_layer_mb: 200,
        min_vram_mb: 200,
    };

    // Define 3 nodes with different capabilities
    let nodes = vec![
        NodeCapabilities {
            node_id: "node-a".into(),
            gpu_name: "RTX 4090".into(),
            vram_mb: 24576,
            tflops_fp16: 82.6,
            memory_bandwidth_gbs: 1008.0,
            latency_ms: 5.0,
            available: true,
            region: Some("us-east".into()),
        },
        NodeCapabilities {
            node_id: "node-b".into(),
            gpu_name: "RTX 3090".into(),
            vram_mb: 24576,
            tflops_fp16: 35.6,
            memory_bandwidth_gbs: 936.0,
            latency_ms: 8.0,
            available: true,
            region: Some("us-east".into()),
        },
        NodeCapabilities {
            node_id: "node-c".into(),
            gpu_name: "M3 Max".into(),
            vram_mb: 36864,
            tflops_fp16: 14.2,
            memory_bandwidth_gbs: 400.0,
            latency_ms: 12.0,
            available: true,
            region: Some("us-west".into()),
        },
    ];

    // Phase 1: Allocate layers
    let stages = allocate_layers(&nodes, &model).expect("Layer allocation should succeed");
    assert_eq!(stages.len(), 3);

    // Verify all layers are covered (end_layer is inclusive)
    let total_assigned: u32 = stages.iter().map(|s| s.end_layer - s.start_layer + 1).sum();
    assert_eq!(total_assigned, model.total_layers);

    // Verify contiguous assignment
    assert_eq!(stages[0].start_layer, 0);
    for i in 1..stages.len() {
        assert_eq!(stages[i].start_layer, stages[i - 1].end_layer + 1);
    }
    assert_eq!(stages.last().unwrap().end_layer, model.total_layers - 1);

    // Phase 2: Form pipeline
    let pipeline = form_pipeline(stages, &model);
    assert_eq!(pipeline.stages.len(), 3);
    assert!(pipeline.estimated_total_latency_ms > 0.0);

    // Phase 3: QUIC transport — two-hop activation forwarding (stage 0 → 1 → 2)
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let node_1 = TransportNode::bind(addr).await.unwrap();
    let node_2 = TransportNode::bind(addr).await.unwrap();

    let addr_1 = node_1.listen_addr();
    let addr_2 = node_2.listen_addr();

    let stage_1 = pipeline.stages[1].clone();
    let stage_2 = pipeline.stages[2].clone();

    // Spawn node 2 (final stage) — accepts activations from node 1
    let node_2_handle = tokio::spawn(async move {
        let peer = node_2.accept().await.unwrap();

        // Receive control: layer assignment
        let msg = peer.recv_activations().await.unwrap();
        match &msg {
            PipelineMessage::Control(ControlMessage::AssignLayers {
                start_layer,
                end_layer,
                ..
            }) => {
                assert_eq!(*start_layer, stage_2.start_layer);
                assert_eq!(*end_layer, stage_2.end_layer);
            }
            _ => panic!("Expected AssignLayers, got {msg:?}"),
        }

        // Receive activations
        let msg = peer.recv_activations().await.unwrap();
        match &msg {
            PipelineMessage::Activations(act) => {
                assert_eq!(act.request_id, "req-001");
                assert_eq!(act.seq_position, 1); // Incremented by stage 1
                assert_eq!(act.shape, vec![1, 1, 4096]);
            }
            _ => panic!("Expected Activations, got {msg:?}"),
        }

        // Final stage done — in real pipeline would generate tokens here
        node_2.close();
    });

    // Spawn node 1 (middle stage) — accepts from node 0, forwards to node 2
    let node_1_handle = tokio::spawn(async move {
        let peer_from_0 = node_1.accept().await.unwrap();

        // Receive control: layer assignment
        let msg = peer_from_0.recv_activations().await.unwrap();
        match &msg {
            PipelineMessage::Control(ControlMessage::AssignLayers {
                start_layer,
                end_layer,
                ..
            }) => {
                assert_eq!(*start_layer, stage_1.start_layer);
                assert_eq!(*end_layer, stage_1.end_layer);
            }
            _ => panic!("Expected AssignLayers, got {msg:?}"),
        }

        // Receive activations from stage 0
        let msg = peer_from_0.recv_activations().await.unwrap();
        let activations = match msg {
            PipelineMessage::Activations(act) => {
                assert_eq!(act.request_id, "req-001");
                assert_eq!(act.seq_position, 0);
                act
            }
            _ => panic!("Expected Activations"),
        };

        // "Process" layers and forward to stage 2 with incremented position
        let forward = PipelineMessage::Activations(ActivationPayload {
            request_id: activations.request_id,
            seq_position: activations.seq_position + 1,
            batch_index: activations.batch_index,
            shape: activations.shape,
            data: activations.data,
            dtype: activations.dtype,
        });

        // Connect to node 2 and forward
        let peer_to_2 = node_1.connect(addr_2).await.unwrap();

        // Send layer assignment then activations to node 2
        let assign_2 = PipelineMessage::Control(ControlMessage::AssignLayers {
            model_id: "llama-3.1-8b-q4".into(),
            start_layer: stage_2.start_layer,
            end_layer: stage_2.end_layer,
            total_layers: 32,
        });
        peer_to_2.send_activations(&assign_2).await.unwrap();
        peer_to_2.send_activations(&forward).await.unwrap();

        // Wait for node 2 to finish before closing
        node_2_handle.await.unwrap();
        peer_to_2.close();
        node_1.close();
    });

    // Node 0 (first stage) — sends activations to node 1
    let node_0 = TransportNode::bind(addr).await.unwrap();
    let conn_to_1 = node_0.connect(addr_1).await.unwrap();

    // Send layer assignment to node 1
    let assign_1 = PipelineMessage::Control(ControlMessage::AssignLayers {
        model_id: "llama-3.1-8b-q4".into(),
        start_layer: stage_1.start_layer,
        end_layer: stage_1.end_layer,
        total_layers: 32,
    });
    conn_to_1.send_activations(&assign_1).await.unwrap();

    // Create mock activations (simulating output of layers 0..N)
    let activations = PipelineMessage::Activations(ActivationPayload {
        request_id: "req-001".into(),
        seq_position: 0,
        batch_index: 0,
        shape: vec![1, 1, 4096],
        data: vec![0u8; 8192], // 4096 * 2 bytes (f16)
        dtype: TensorDtype::Float16,
    });
    conn_to_1.send_activations(&activations).await.unwrap();

    // Wait for pipeline to complete
    node_1_handle.await.unwrap();

    conn_to_1.close();
    node_0.close();
}

/// Test ping-pong latency measurement between two nodes.
#[tokio::test]
async fn test_latency_ping_pong() {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let node_a = TransportNode::bind(addr).await.unwrap();
    let node_b = TransportNode::bind(addr).await.unwrap();

    let addr_b = node_b.listen_addr();

    // Use a channel to coordinate shutdown
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();

    let node_b_handle = tokio::spawn(async move {
        let peer = node_b.accept().await.unwrap();

        // Receive ping via uni stream
        let msg = peer.recv_activations().await.unwrap();
        match msg {
            PipelineMessage::Ping(ping) => {
                assert_eq!(ping.node_id, "node-a");
                assert_eq!(ping.timestamp_ms, 1000);
            }
            _ => panic!("Expected Ping"),
        }

        // Send pong back via uni stream
        let pong = PipelineMessage::Pong(PongMessage {
            node_id: "node-b".into(),
            timestamp_ms: 1000,
            latency_ms: Some(2),
        });
        peer.send_activations(&pong).await.unwrap();

        // Wait for node_a to signal it received the pong
        let _ = rx.await;
        node_b.close();
    });

    let conn = node_a.connect(addr_b).await.unwrap();

    let ping = PipelineMessage::Ping(PingMessage {
        node_id: "node-a".into(),
        timestamp_ms: 1000,
    });
    conn.send_activations(&ping).await.unwrap();

    // Receive pong from node_b
    let pong = conn.recv_activations().await.unwrap();
    match pong {
        PipelineMessage::Pong(p) => {
            assert_eq!(p.node_id, "node-b");
            assert_eq!(p.latency_ms, Some(2));
        }
        _ => panic!("Expected Pong"),
    }

    // Signal node_b it can close
    let _ = tx.send(());
    node_b_handle.await.unwrap();
    conn.close();
    node_a.close();
}

/// Test pipeline teardown: control release message.
#[tokio::test]
async fn test_pipeline_teardown() {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let node_a = TransportNode::bind(addr).await.unwrap();
    let node_b = TransportNode::bind(addr).await.unwrap();

    let addr_b = node_b.listen_addr();

    let node_b_handle = tokio::spawn(async move {
        let peer = node_b.accept().await.unwrap();
        let msg = peer.recv_activations().await.unwrap();
        match msg {
            PipelineMessage::Control(ControlMessage::Release { reason }) => {
                assert_eq!(reason, "pipeline complete");
            }
            _ => panic!("Expected Release control message"),
        }
        node_b.close();
    });

    let conn = node_a.connect(addr_b).await.unwrap();
    let release = PipelineMessage::Control(ControlMessage::Release {
        reason: "pipeline complete".into(),
    });
    conn.send_activations(&release).await.unwrap();

    node_b_handle.await.unwrap();
    conn.close();
    node_a.close();
}
