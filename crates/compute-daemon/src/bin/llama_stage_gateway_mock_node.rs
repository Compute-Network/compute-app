use anyhow::{Context, Result, bail};
use llama_stage_backend::{
    LLAMA_STAGE_PROTOCOL_VERSION, StageGatewayRequest, StageGatewayResponse, StageNodeInfo,
};
use std::env;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};

#[derive(Clone, Copy)]
enum Scenario {
    ProtocolMismatch,
    ModelMismatch,
    UnusableGateway,
}

impl Scenario {
    fn parse(raw: &str) -> Result<Self> {
        match raw {
            "protocol-mismatch" => Ok(Self::ProtocolMismatch),
            "model-mismatch" => Ok(Self::ModelMismatch),
            "unusable-gateway" => Ok(Self::UnusableGateway),
            other => bail!("unknown scenario: {other}"),
        }
    }
}

struct Args {
    bind_addr: String,
    scenario: Scenario,
}

fn parse_args() -> Result<Args> {
    let mut bind_addr = "127.0.0.1:0".to_string();
    let mut scenario = None;

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--bind" => {
                bind_addr = it.next().context("missing value for --bind")?;
            }
            "--scenario" => {
                let raw = it.next().context("missing value for --scenario")?;
                scenario = Some(Scenario::parse(&raw)?);
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args { bind_addr, scenario: scenario.context("missing --scenario")? })
}

fn info_response(protocol_version: u32, model_id: &str) -> StageGatewayResponse {
    StageGatewayResponse::Info {
        protocol_version,
        head_info: StageNodeInfo {
            protocol_version,
            model_id: model_id.to_string(),
            stage_id: "mock-head".to_string(),
            start_layer: 0,
            end_layer: 20,
            is_head: true,
            is_tail: false,
            spec_decode_v1: false,
        },
        tail_info: StageNodeInfo {
            protocol_version,
            model_id: model_id.to_string(),
            stage_id: "mock-tail".to_string(),
            start_layer: 21,
            end_layer: 41,
            is_head: false,
            is_tail: true,
            spec_decode_v1: false,
        },
        reconnect_after_prompt: false,
    }
}

fn scenario_response(scenario: Scenario, request: StageGatewayRequest) -> StageGatewayResponse {
    match request {
        StageGatewayRequest::Info => match scenario {
            Scenario::ProtocolMismatch => {
                info_response(LLAMA_STAGE_PROTOCOL_VERSION + 1, "gemma-4-e4b-q4")
            }
            Scenario::ModelMismatch => info_response(LLAMA_STAGE_PROTOCOL_VERSION, "wrong-model"),
            Scenario::UnusableGateway => StageGatewayResponse::Error {
                message: "head endpoint 127.0.0.1:9201 is not marked as a head stage".to_string(),
            },
        },
        _ => StageGatewayResponse::Error { message: "mock gateway only supports info".to_string() },
    }
}

fn handle_stream(stream: TcpStream, scenario: Scenario) -> Result<()> {
    stream.set_nodelay(true)?;
    let reader_stream = stream.try_clone()?;
    let mut reader = BufReader::new(reader_stream);
    let mut writer = stream;
    let mut line = String::new();

    loop {
        line.clear();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            break;
        }

        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<StageGatewayRequest>(line.trim()) {
            Ok(request) => scenario_response(scenario, request),
            Err(err) => StageGatewayResponse::Error { message: format!("invalid request: {err}") },
        };

        serde_json::to_writer(&mut writer, &response)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let listener = TcpListener::bind(&args.bind_addr)
        .with_context(|| format!("binding {}", args.bind_addr))?;
    eprintln!("listening={}", listener.local_addr()?);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(err) = handle_stream(stream, args.scenario) {
                    eprintln!("connection error: {err}");
                }
            }
            Err(err) => eprintln!("accept error: {err}"),
        }
    }

    Ok(())
}
