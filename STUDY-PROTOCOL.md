# Study protocol

This repository documents **one measured serving recipe** for Inkling-2-Mini on
a single DGX Spark. It is not a leaderboard entry and not a claim that the
recipe is optimal for every workload.

## Fixed platform

- Hardware: one NVIDIA DGX Spark GB10, 128 GB unified memory.
- Runtime: `vllm/vllm-openai:inkling-2-mini` (arm64), vLLM
  `0.1.dev25855+g7cb5dae22`.
- Model: `thinkingmachines/Inkling-2-Mini`. The final matrix was measured at
  revision `b85c3cfa615fc413cc9af3379f491b963f3b586a`.
- KV cache: FP8 (e4m3), 3,095,742 tokens at the measured start.
- Served window: 409,600 tokens per request.

## Throughput matrix

- Prompts are built from exact token IDs, not text-and-hope token counts.
- Every request starts with a unique nonce so prefix caching cannot turn a
  cell into a cache-hit measurement.
- The matrix uses raw `/v1/completions` with `ignore_eos` and a forced output
  budget. Chat-template effects, reasoning blocks and tool parsing are
  therefore **not** part of the throughput number.
- `OUTPUT_TOKENS=512` keeps the sweep tractable on one Spark. The reference
  GLM repository used 2,048; this is a shorter-window view, not an inflated
  one. Decode is measured inside the common interval where all requests are
  emitting, not from total wall time.
- Cells whose aggregate prompt tokens exceed 2.8M are skipped rather than
  letting the scheduler queue or retract them. A requested C8 at 400K is
  3.2M and is intentionally absent.
- TTFT is measured from request start to the first streamed usage event.
  Effective prefill includes queueing and any mixed decode work until the last
  request in the wave reaches its first token.

## Quality and feature checks

- Forced-output throughput is not a quality score.
- `smoke.py` is bounded acceptance, not an evaluation suite: arithmetic, JSON
  schema, one tool-call round trip, one image, Anthropic Messages, prompt
  reuse, a 120k needle, the reasoning-effort scale and eight concurrent
  requests.
- FP8 KV is lossy. The checks above are a sanity gate; they do not establish
  parity with BF16 KV on long-context or reasoning-heavy workloads.
- Reasoning is on by default (effort 0.9). A small `max_tokens` budget can be
  consumed by the reasoning block and return no final content.

## Repeatability and reporting

- A difference below 5% between repeated key cells is treated as noise.
- Every published matrix cell has a raw event file; `common_window.py`
  recomputes the table without rerunning inference.
- `audit.py` checks exact prompt/output counts, successful completion, common
  decode windows and manifest hashes.
- Missing measurements remain unknown. No synthetic number is substituted.

## Capacity is not context

The KV pool holds 3,095,742 tokens. A single request can use up to 409,600;
eight requests cannot each use 409,600. At 400K the pool fits seven full
sessions; `max-num-seqs 8` is the admission cap, not a reservation.

## Open questions

- Full 1,048,576-token window under FP8 KV (the pool is large enough in
  principle; it has not been qualified end to end).
- FP8-KV quality on a real long-context benchmark beyond single-needle
  retrieval.
- A one-hour soak, multi-user agent traces, and rollback exercise.
- Multi-image and video input.
- Whether pinning the model revision in the recipe changes startup or draft
  loading behavior.
