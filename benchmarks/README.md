# Benchmarks

All scripts are Python 3.10+ and use only the standard library plus
`tokenizers` (see `../requirements.txt`). They talk to the running vLLM server
over its OpenAI-compatible API.

| Script | Purpose |
|---|---|
| `capture_env.py` | Writes `results/launch.json` and `results/config.json`. The matrix hashes `launch.json` so every cell is tied to one launch identity. |
| `matrix.py` | Sends exact-length token-ID prompts at the requested input sizes and concurrency, saves raw streamed token events, and writes `summary.json` + `TABLE.md`. |
| `common_window.py` | Recomputes summaries from saved raw events; use it to audit a published matrix. |
| `smoke.py` | Bounded feature checks: arithmetic, JSON schema, tool call + result, single-image vision, Anthropic Messages, prompt reuse, 120k needle, reasoning-effort scale and 8-way concurrency. |

## Matrix environment

| Variable | Default | Meaning |
|---|---|---|
| `API_BASE` | `http://127.0.0.1:8000` | vLLM OpenAI endpoint. |
| `MODEL` | `inkling-2-mini` | Served model ID. |
| `TOKENIZER_PATH` | auto-detect | `tokenizer.json` from the HF cache's `refs/main` snapshot. |
| `PREFILL_SIZES` | `512,2048,8192,32768,131072,200000,400000` | Input lengths. |
| `CONCURRENCIES` | `1,2,4,8` | Requested burst sizes. |
| `OUTPUT_TOKENS` | `512` | Forced output tokens per request. |
| `IGNORE_EOS` | `1` | Generate exactly the budget. |
| `MAX_PROMPT_TOKENS` | `400000` | Per-request cap; leaves room in the 409,600-token window. |
| `MAX_AGGREGATE_PROMPT_TOKENS` | `2800000` | Skips cells above the 3.10M-token KV pool's headroom. |
| `CASE_TIMEOUT_S` | `3600` | Per-request socket timeout. |
| `RESULTS_DIR` | `results/matrix-<timestamp>` | Output directory. |
| `CONTINUE_ON_ERROR` | unset | Set to `1` to keep going after a failed wave. |

## Why exact token IDs

Sending text and trusting the tokenizer count makes prompt-length comparisons
soft. `matrix.py` builds a prompt from a unique nonce, a repeated reference
filler and a fixed instruction, trims it to the requested number of token IDs,
and verifies the server-reported `prompt_tokens`. The nonce also prevents the
prefix cache from turning a cell into a cache-hit measurement.

The matrix uses raw `/v1/completions`, not the chat template. That gives exact
prompt and output lengths and measures decode without the model's reasoning
block. It is a performance test, not a quality score; the chat-based numbers in
`../EVAL-REPORT.md` are not directly comparable to this table.

## Raw event shape

Each request record contains cumulative `{seconds, count}` events from
`stream_options.continuous_usage_stats`, plus `first_token_s`, `finished_s`,
`ttft_s`, `decode_window_tps`, `output_tokens` and any error. The summary is
derived only from those events; `common_window.py` can reproduce it.

## Reproduce

```bash
python3 benchmarks/capture_env.py
OUTPUT_TOKENS=512 \
PREFILL_SIZES=512,2048,8192,32768,131072,200000,400000 \
CONCURRENCIES=1,2,4,8 \
python3 benchmarks/matrix.py
./benchmarks/finalize.sh results/matrix-<timestamp>
python3 benchmarks/smoke.py
```
