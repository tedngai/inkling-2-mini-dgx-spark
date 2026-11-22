# Results

Published receipts for the tuned Inkling-2-Mini FP8-KV recipe. The benchmark
code that produced them is in `../benchmarks/`.

| File | Contents |
|---|---|
| `launch.json` | Model revision, tokenizer/config hashes, image digest, exact running command, cache configuration and host state. The matrix hashes this file. |
| `config.json` | Short published configuration extracted from the launch. |
| `inkling-2-mini-summary.json` | One row per completed matrix cell. |
| `inkling-2-mini-table.md` | Rendered matrix table (same cells as the README). |
| `matrix-<timestamp>/` | Raw per-request token events, per-cell summaries and the rendered table. |
| `feature-acceptance.json` | Health, arithmetic, JSON schema, tool call/result, single-image vision, Anthropic Messages, prompt reuse and needle checks. |
| `needle-120k.json` | Long-context retrieval receipt (120,015 prompt tokens). |
| `reasoning-effort.json` | Reasoning-token counts for `low`, `high`, `xhigh`. |
| `concurrency-8.json` | Eight simultaneous requests, wall time and per-request latency. |
| `chat-spec-ab.json` | Chat prose vs matrix-prompt DSpark acceptance; DSpark 7 vs no-spec. |
| `external-4x-small.json` | Comparison with AidenLab's 4× Spark Inkling-2-Small SGLang recipe. |
| `sglang-ab.json` | Same-hardware vLLM vs SGLang comparison (earlier revision). |
| `bf16-vs-fp8.json` | BF16 vs FP8 KV capacity, speed and quality probes. |
| `kv-tuning.json` | Prefill-batch and utilization sweep that produced the 3.10M-token pool. |
| `long-context.json` | 362,661-token and 120K needle receipts. |
| `protocol-audit.json` | Structural audit: exact prompt/output counts, no failed requests, matching manifest hashes. Status `pass`, with warnings for seven large-context cells that have no positive common decode window (staggered prefill). |

All throughput cells use exact-length token-ID prompts, unique nonce prefixes
(no prefix-cache reuse), forced output and greedy sampling. They are
performance tests, not quality scores.

The final matrix was measured at model revision
`b85c3cfa615fc413cc9af3379f491b963f3b586a`. Earlier tuning rows in
`../EVAL-REPORT.md` were measured at `ed40e7da625f594fc9c4990f5f66be24873c22d7`;
the Hub's `main` moved between those runs.
