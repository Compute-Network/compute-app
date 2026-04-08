# Stage Prototype LAN Test

## Goal

Bring up a 2-machine experimental stage pipeline for the deterministic prototype
backend before attempting a real model-stage backend.

This is a transport and control-plane validation path, not a correctness benchmark
against real model layers.

## Install

On each machine:

```bash
curl -fsSL https://computenetwork.sh/install.sh | sh
```

Verify:

```bash
compute --version
```

## Required Auth

On each machine:

```bash
compute wallet login
```

## Required Config

On each machine:

```bash
compute config set experimental.stage_mode_enabled true
compute config set experimental.stage_backend prototype
compute config set models.active_model auto
```

Verify:

```bash
compute config show
```

Look for:

- `experimental.stage_mode_enabled = true`
- `experimental.stage_backend = "prototype"`

Optional real-execution spike:

```bash
compute config set experimental.stage_backend tail-llama
```

`tail-llama` keeps the same head/tail QUIC stage path, but the tail stage runs a
real local completion instead of returning the deterministic prototype text.
This is still not true model-shard execution; it is the first real inference step
inside the stage transport.

## Start

On each machine:

```bash
compute start
```

## Orchestrator Requirement

The orchestrator must have:

- `EXPERIMENTAL_STAGE_MODE=true`

Without that, nodes will stay on the normal single-node path.

## Expected Behavior

When the scheduler forms the fixed Gemma prototype pipeline:

- one node should be assigned shard `0-13`
- one node should be assigned shard `14-27`
- the daemon log should show `stage-prototype` status
- the stage runtime should log its listen address and backend `prototype`

## Current Limitations

- stage mode only supports non-streaming requests right now
- the prototype backend returns deterministic synthetic completions
- `tail-llama` requires the tail node to have the model downloaded locally
- `tail-llama` is a real-execution spike, not true layer-sharded inference
- this validates routing, transport, and stage ownership, not real model math

## What To Test

1. Both machines authenticate and register cleanly.
2. Both machines accept experimental stage assignments.
3. Head stage connects to downstream tail stage.
4. A non-streaming request completes through the prototype stage path.
5. Cancelling/streaming should fail explicitly rather than silently falling back.

## Not Ready Yet

Do not treat these as solved yet:

- real hidden-state forward between model shards
- meaningful performance numbers
- real model correctness across stage boundaries
- public internet deployment
