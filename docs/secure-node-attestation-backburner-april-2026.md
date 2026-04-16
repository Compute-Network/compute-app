# Secure Node Attestation Back Burner

## Why this exists

We discussed stronger privacy guarantees for untrusted node operators after
finding Apple-oriented examples that use managed device attestation, Secure
Enclave roots of trust, hardened runtime, and periodic re-verification.

The core product question is valid:

> If a prompt runs on someone else's machine, can that machine owner read it?

For the current network, the honest answer is still "yes, in the general case."
Recent transport hardening removed avoidable plaintext leaks between stages, but
it does not stop the executing node from seeing the request or generated output.

This note captures the current thinking so it can be resumed later without
polluting the active Gemma forward-path work.

## Current status

- Stage transport privacy is improved:
  - hidden-state forwarding no longer carries plaintext prompt text in the
    relevant paths
  - stage token payloads no longer forward plaintext completion text by default
  - prototype hidden-state bytes are no longer directly derived from raw prompt
    bytes
- Full untrusted-node confidentiality is not solved.
- We should not claim PCC-like guarantees for the general node pool.

## Apple-specific path

Apple has the cleanest story today.

Relevant capabilities:

- Managed Device Attestation on Apple silicon
- Secure Enclave rooted identity
- attestation freshness / nonce binding
- attested SIP / Secure Boot / kernel-extension posture
- hardened runtime and code-signing enforcement

This suggests a viable future "attested Mac node" tier with requirements like:

- Apple silicon only
- managed device enrollment
- Secure Boot enabled
- SIP enabled
- no third-party kernel extensions
- signed node binary
- no dynamic plugin loading or debug entitlements in the inference path
- per-session or per-request attestation, not just a loose heartbeat

This would be a real security improvement, but only for a curated Apple tier.

## Linux path

Linux does not have one unified Apple-like answer. The closest realistic options
are:

- TPM-backed measured boot
- Secure Boot
- remote attestation over PCR values
- tightly managed immutable images
- optionally confidential-compute environments on supported hardware

This can support a stronger managed Linux tier, but only if we control the OS
image and runtime environment aggressively. Generic community Linux nodes should
not be treated as equivalent to attested Apple nodes.

## Windows path

Windows also has relevant building blocks, but the integration story is weaker
for our use case:

- TPM-backed device identity
- Secure Boot
- managed device posture
- virtualization-based security / HVCI style settings

As with Linux, this is viable only for a managed higher-trust tier. It is not a
drop-in answer for arbitrary third-party Windows nodes.

## Cross-platform conclusion

There is no single cross-platform mechanism today that gives us a clean,
uniform, PCC-like privacy guarantee across:

- macOS
- Linux
- Windows

The realistic future model is a trust-tier architecture.

## Proposed trust tiers

### Tier 0: Standard nodes

Use case:

- broad cross-platform pool
- low-friction participation

Guarantee:

- no strong confidentiality claim against the node operator

### Tier 1: Managed attested nodes

Use case:

- curated higher-trust pool
- sensitive jobs that need stronger hardware/software posture guarantees

Possible implementations:

- Apple silicon + managed device attestation
- Linux + TPM measured boot + immutable image + remote attestation
- Windows + TPM / Secure Boot / managed posture attestation

Guarantee:

- stronger device-state assurance
- still weaker and less uniform than Apple PCC

### Tier 2: Confidential-compute nodes

Use case:

- highest-trust hosted environments
- likely Linux-first

Possible implementations:

- confidential VMs / enclave-backed execution on supported hardware

Guarantee:

- closest non-Apple path to protecting request data from the host/operator

Tradeoff:

- much higher operational complexity
- weaker consumer-device compatibility

## Important design caution

Periodic re-verification alone is not enough.

If we ever make a serious privacy claim for a secure-node tier, the better
pattern is:

1. attest the node for a specific session or request
2. bind request encryption to that attestation result
3. only release prompt material to nodes that meet the policy

That is meaningfully stronger than a generic "node checked in five minutes ago"
model.

## Recommended future path

When this comes back off the shelf:

1. Define formal trust tiers in product and protocol terms.
2. Build an Apple-attested secure-node prototype first.
3. Keep Linux and Windows in the standard pool until a real attested design
   exists for each.
4. Treat confidential compute as the long-term answer for the highest-trust
   hosted tier, not as a requirement for the general node network.

## What not to do

- Do not claim full prompt/output privacy for the general cross-platform node
  pool.
- Do not treat Apple-specific attestation as evidence that Linux and Windows are
  solved.
- Do not rely on heartbeat-style attestation alone for sensitive request
  protection.

## Resume point

If this work is resumed later, start by writing a concrete trust-tier spec:

- what each tier guarantees
- what hardware/software is required
- how attestation is verified
- how keys are issued
- what the user-facing privacy claims actually are
