use anyhow::{Context, Result, bail};
use llama_stage_backend::{
    RemoteStageGateway, StageGatewayRequest, StageGatewayResponse, handle_stage_gateway_request,
};
use std::env;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};

#[derive(Debug, Clone)]
struct Args {
    bind_addr: String,
    head_addr: String,
    tail_addr: String,
    reconnect_after_prompt: bool,
}

fn parse_args() -> Result<Args> {
    let mut bind_addr = "127.0.0.1:0".to_string();
    let mut head_addr = None;
    let mut tail_addr = None;
    let mut reconnect_after_prompt = false;

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--bind" => {
                bind_addr = it.next().context("missing value for --bind")?;
            }
            "--head" => {
                head_addr = Some(it.next().context("missing value for --head")?);
            }
            "--tail" => {
                tail_addr = Some(it.next().context("missing value for --tail")?);
            }
            "--reconnect-after-prompt" => reconnect_after_prompt = true,
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        bind_addr,
        head_addr: head_addr.context("missing --head")?,
        tail_addr: tail_addr.context("missing --tail")?,
        reconnect_after_prompt,
    })
}

fn handle_stream(stream: TcpStream, gateway: &mut RemoteStageGateway) -> Result<()> {
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

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<StageGatewayRequest>(trimmed) {
            Ok(request) => handle_stage_gateway_request(gateway, request),
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
    let mut gateway =
        RemoteStageGateway::connect(&args.head_addr, &args.tail_addr, args.reconnect_after_prompt)?;

    let listener = TcpListener::bind(&args.bind_addr)
        .with_context(|| format!("binding {}", args.bind_addr))?;
    eprintln!("listening={}", listener.local_addr()?);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(err) = handle_stream(stream, &mut gateway) {
                    eprintln!("connection error: {err}");
                }
            }
            Err(err) => eprintln!("accept error: {err}"),
        }
    }

    Ok(())
}
