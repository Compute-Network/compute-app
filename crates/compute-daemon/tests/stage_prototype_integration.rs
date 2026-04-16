use std::net::SocketAddr;

use compute_daemon::config::Config;
use compute_daemon::hardware::HardwareInfo;
use compute_daemon::stage_runtime::{
    StagePrototypeResponse, StagePrototypeSpec, start_stage_prototype_chain,
    start_stage_prototype_with_bind_addr,
};
use compute_network::models::ModelCatalog;
use compute_network::transport::node::TransportNode;
use compute_network::transport::protocol::{
    ActivationPayload, ControlMessage, PipelineMessage, TensorDtype, TokenPayload,
};

/// Local 2-node LAN harness for the stage-based prototype path.
///
/// Flow:
/// 1. Client sends one activation-shaped payload to the head stage.
/// 2. Head forwards it to the tail stage over QUIC.
/// 3. Tail returns token payload back to the head.
/// 4. Head returns the tokens to the original client.
///
/// This does not prove model correctness yet.
/// It proves the intended transport shape for the head/tail prototype.
#[tokio::test]
async fn test_two_stage_head_tail_roundtrip() {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();

    let head_node = TransportNode::bind(addr).await.unwrap();
    let tail_node = TransportNode::bind(addr).await.unwrap();
    let client_node = TransportNode::bind(addr).await.unwrap();

    let head_addr = head_node.listen_addr();
    let tail_addr = tail_node.listen_addr();
    let (tail_done_tx, tail_done_rx) = tokio::sync::oneshot::channel::<()>();
    let (client_done_tx, client_done_rx) = tokio::sync::oneshot::channel::<()>();

    let tail_task = tokio::spawn(async move {
        let peer = tail_node.accept().await.unwrap();

        let assign = peer.recv_activations().await.unwrap();
        match assign {
            PipelineMessage::Control(ControlMessage::AssignLayers {
                model_id,
                start_layer,
                end_layer,
                total_layers,
            }) => {
                assert_eq!(model_id, "gemma-4-e4b-q4");
                assert_eq!(start_layer, 14);
                assert_eq!(end_layer, 27);
                assert_eq!(total_layers, 28);
            }
            other => panic!("expected AssignLayers, got {other:?}"),
        }

        let activation = peer.recv_activations().await.unwrap();
        match activation {
            PipelineMessage::Activations(act) => {
                assert_eq!(act.request_id, "req-stage-1");
                assert_eq!(act.seq_position, 0);
                assert_eq!(act.batch_index, 0);
                assert_eq!(act.shape, vec![1, 1, 2048]);
            }
            other => panic!("expected Activations, got {other:?}"),
        }

        let tokens = PipelineMessage::Tokens(TokenPayload {
            request_id: "req-stage-1".into(),
            tokens: vec![42, 43],
            is_finished: true,
            text: Some("OK".into()),
            timings: None,
        });
        peer.send_activations(&tokens).await.unwrap();
        let _ = tail_done_rx.await;
        peer.close();
        tail_node.close();
    });

    let head_task = tokio::spawn(async move {
        let client_peer = head_node.accept().await.unwrap();

        let activation = client_peer.recv_activations().await.unwrap();
        let activation = match activation {
            PipelineMessage::Activations(act) => act,
            other => panic!("expected Activations from client, got {other:?}"),
        };

        let tail_peer = head_node.connect(tail_addr).await.unwrap();

        let assign_tail = PipelineMessage::Control(ControlMessage::AssignLayers {
            model_id: "gemma-4-e4b-q4".into(),
            start_layer: 14,
            end_layer: 27,
            total_layers: 28,
        });
        tail_peer.send_activations(&assign_tail).await.unwrap();
        tail_peer.send_activations(&PipelineMessage::Activations(activation)).await.unwrap();

        let response = tail_peer.recv_activations().await.unwrap();
        match response {
            PipelineMessage::Tokens(tokens) => {
                assert_eq!(tokens.request_id, "req-stage-1");
                assert_eq!(tokens.tokens, vec![42, 43]);
                assert!(tokens.is_finished);
                assert_eq!(tokens.text.as_deref(), Some("OK"));

                client_peer.send_activations(&PipelineMessage::Tokens(tokens)).await.unwrap();
                let _ = client_done_rx.await;
            }
            other => panic!("expected Tokens from tail, got {other:?}"),
        }

        tail_peer.close();
        client_peer.close();
        head_node.close();
    });

    let head_conn = client_node.connect(head_addr).await.unwrap();
    let request = PipelineMessage::Activations(ActivationPayload {
        request_id: "req-stage-1".into(),
        seq_position: 0,
        batch_index: 0,
        shape: vec![1, 1, 2048],
        data: vec![0u8; 4096],
        dtype: TensorDtype::Float16,
    });
    head_conn.send_activations(&request).await.unwrap();

    let reply = head_conn.recv_activations().await.unwrap();
    match reply {
        PipelineMessage::Tokens(tokens) => {
            assert_eq!(tokens.request_id, "req-stage-1");
            assert_eq!(tokens.tokens, vec![42, 43]);
            assert!(tokens.is_finished);
            assert_eq!(tokens.text.as_deref(), Some("OK"));
        }
        other => panic!("expected Tokens from head, got {other:?}"),
    }

    let _ = client_done_tx.send(());
    let _ = tail_done_tx.send(());
    head_task.await.unwrap();
    tail_task.await.unwrap();
    head_conn.close();
    client_node.close();
}

#[tokio::test]
async fn test_three_stage_runtime_chain_roundtrip() {
    let mut config = Config::default();
    config.experimental.stage_backend = "prototype".to_string();
    let hw = HardwareInfo::empty();

    let tail = start_stage_prototype_with_bind_addr(
        &config,
        &hw,
        StagePrototypeSpec {
            pipeline_id: "proto-3".into(),
            model_name: "gemma-4-e4b-q4".into(),
            shard_id: "tail".into(),
            start_layer: 20,
            end_layer: 27,
            stage_index: 2,
            total_stages: 3,
            upstream_addr: None,
            downstream_addr: None,
        },
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap();

    let middle = start_stage_prototype_with_bind_addr(
        &config,
        &hw,
        StagePrototypeSpec {
            pipeline_id: "proto-3".into(),
            model_name: "gemma-4-e4b-q4".into(),
            shard_id: "middle".into(),
            start_layer: 10,
            end_layer: 19,
            stage_index: 1,
            total_stages: 3,
            upstream_addr: None,
            downstream_addr: Some(tail.listen_addr().to_string()),
        },
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap();

    let head = start_stage_prototype_with_bind_addr(
        &config,
        &hw,
        StagePrototypeSpec {
            pipeline_id: "proto-3".into(),
            model_name: "gemma-4-e4b-q4".into(),
            shard_id: "head".into(),
            start_layer: 0,
            end_layer: 9,
            stage_index: 0,
            total_stages: 3,
            upstream_addr: None,
            downstream_addr: Some(middle.listen_addr().to_string()),
        },
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap();

    let result = head
        .client()
        .complete_prompt("req-3-stage".into(), "Hello".into(), None, Vec::new())
        .await
        .unwrap();

    assert_eq!(result.model_name, "gemma-4-e4b-q4");
    assert_eq!(result.finish_reason, "stop");
    assert!(result.prompt_tokens > 0);
    assert!(result.completion_tokens > 0);
    assert!(result.total_tokens >= result.prompt_tokens + result.completion_tokens);
    assert!(result.content.contains("Prototype stage completion for gemma-4-e4b-q4"));
    assert!(result.content.contains("0-9 -> 10-19"));

    head.stop().await;
    middle.stop().await;
    tail.stop().await;
}

#[tokio::test]
async fn test_four_stage_runtime_chain_roundtrip() {
    let mut config = Config::default();
    config.experimental.stage_backend = "prototype".to_string();
    let hw = HardwareInfo::empty();
    let model = ModelCatalog::default_catalog().find("gemma-4-e4b-q4").unwrap().clone();
    let ranges = model
        .shard_for_stages(4)
        .into_iter()
        .map(|shard| (shard.start_layer, shard.end_layer))
        .collect::<Vec<_>>();

    let result = run_stage_runtime_chain_roundtrip(&config, &hw, &model.id, &ranges).await;
    let expected_trace = expected_stage_trace(&ranges);

    assert_eq!(result.model_name, model.id);
    assert_eq!(result.finish_reason, "stop");
    assert!(result.prompt_tokens > 0);
    assert!(result.completion_tokens > 0);
    assert!(!result.completion_token_ids.is_empty());
    assert!(result.total_tokens >= result.prompt_tokens + result.completion_tokens);
    assert!(result.content.contains("Prototype stage completion"));
    assert!(result.content.contains(&expected_trace));
}

async fn run_stage_runtime_chain_roundtrip(
    config: &Config,
    hw: &HardwareInfo,
    model_name: &str,
    ranges: &[(u32, u32)],
) -> StagePrototypeResponse {
    let mut handles = start_stage_prototype_chain(
        config,
        hw,
        &format!("proto-{}-stage", ranges.len()),
        model_name,
        ranges,
    )
    .await
    .unwrap();
    let result = handles[0]
        .client()
        .complete_prompt(format!("req-{}-stage", ranges.len()), "Hello".into(), None, Vec::new())
        .await
        .unwrap();

    while let Some(handle) = handles.pop() {
        handle.stop().await;
    }

    result
}

fn expected_stage_trace(ranges: &[(u32, u32)]) -> String {
    ranges[..ranges.len().saturating_sub(1)]
        .iter()
        .map(|(start_layer, end_layer)| format!("{start_layer}-{end_layer}"))
        .collect::<Vec<_>>()
        .join(" -> ")
}
