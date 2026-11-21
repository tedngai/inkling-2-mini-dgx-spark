# Inkling-2-Mini on 1 × DGX Spark

Run **Thinking Machines' Inkling-2-Mini** on a single NVIDIA DGX Spark (GB10,
128 GB unified memory). The measured recipe serves the native NVFP4/MXFP8
checkpoint through the day-0 **vLLM** image, with an **FP8 KV cache**, DSpark
speculative decoding, eight admission slots and a **409,600-token** served
window.

The repository contains the pinned launcher, the exact recipe, a reproducible
benchmark harness and the measured result set. Loading, speed, KV capacity and
feature correctness are reported as separate milestones.

## 1. Current status

| Item | Verified result or current limit |
| --- | --- |
| Model precision | `thinkingmachines/Inkling-2-Mini`: NVFP4 experts, MXFP8 last four layers. No local requantization. |
| Hardware | 1 × DGX Spark GB10, 128 GB unified memory. TP1. |
| Engine | vLLM `0.1.dev25855+g7cb5dae22.cu130` in `vllm/vllm-openai@sha256:c2dc759f…d1449e`. |
| API model ID | `inkling-2-mini` |
| Served context | **409,600 tokens** per request. Native window is 1,048,576; it is not enabled in this recipe. |
| KV capacity | **3,095,742 tokens** with FP8 KV at 0.90 utilization — **7.56×** a 400K window. |
| Admission slots | `max-num-seqs 8`; eight simultaneous requests verified. |
| Speculative decoding | DSpark, 7 draft tokens, probabilistic draft sampling. |
| Speed | See §4. Short-prompt decode ~94 tok/s at C1; 10K-token chat decode ~68–75 tok/s; C8 throughput scales with the matrix. |
| Long context | 120,015-token needle pass; 362,661-token prompt pass on the earlier BF16 recipe; 400K cells in the final matrix. |
| Features | Text, JSON schema, tool calls, single-image vision, Anthropic Messages, reasoning-effort control; 13/13 smoke checks. |
| Reliability | One boot; no one-hour soak, rollback or multi-user agent trace yet. |

**Scope:** the publishable claim is the measured single-Spark vLLM recipe. The
SGLang comparison and earlier BF16/FP8 experiments are in
[`EVAL-REPORT.md`](EVAL-REPORT.md); they are not merged into the final matrix.

## 2. Start the service

Requirements: one DGX Spark with ~110 GiB free unified memory, Docker with the
NVIDIA runtime, and network access to Docker Hub and Hugging Face.

```bash
git clone https://github.com/tedngai/inkling-2-mini-dgx-spark.git
cd inkling-2-mini-dgx-spark
./start.sh --setup          # first run: image + model, then start
```

On later boots:

```bash
./start.sh                  # background
DAEMON=0 ./start.sh         # foreground
```

`start.sh` uses `spark-vllm-docker` at `0bdc9c9` and the recipe in
[`recipe/inkling-2-mini.yaml`](recipe/inkling-2-mini.yaml). Set
`SPARK_VLLM_DIR` to reuse an existing checkout.

The API binds to port **8000**; there is no authentication in this recipe.

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini",
       "messages":[{"role":"user","content":"What is 19 + 23?"}],
       "max_tokens":256,
       "reasoning_effort":"low"}'
```

Stop with:

```bash
./stop.sh
```

## 3. Configuration

The published profile is the tuned recipe, not the upstream stock defaults.

| Setting | Published value | Upstream vLLM recipe | Why |
| --- | --- | --- | --- |
| `max-model-len` | **409600** | 262144 | Uses more of the model's 1M window. |
| `kv-cache-dtype` | **fp8** | auto (BF16) | ~45% more KV tokens in the same memory budget. |
| `max-num-batched-tokens` | **2048** | unset (2048 baseline) | Minimizes vLLM's hybrid-cache padding; 8192/4096 shrink the pool. |
| `max-num-seqs` | **8** | unset | Matches the ~7.5× 400K KV capacity; 8 concurrent verified. |
| `gpu-memory-utilization` | **0.90** | unset (0.90 default) | Leaves ~7 GiB host headroom on a 128 GB Spark. |
| DSpark | 7 tokens, probabilistic | same | Day-0 recipe. |
| Prefix caching | on | on | Keeps repeated system prompts cheap. |
| Parsers | `inkling` reasoning + tool | same | Day-0 recipe. |

**Capacity is not the same as context.** The KV pool holds 3,095,742 tokens.
A single request can use up to 409,600; eight requests cannot each use 409,600.
At 400K, seven full sessions fit in the pool; `max-num-seqs 8` is the admission
cap, not a reservation.

**Thinking control.** The chat template maps `reasoning_effort` to
`none 0.0`, `minimal 0.1`, `low 0.2`, `medium 0.7`, `high 0.9`,
`xhigh 0.99`, `max 0.99`. The default is `high`. `xhigh` and `max` are
identical, and `none` does not fully disable reasoning. A small `max_tokens`
budget can be consumed by the reasoning block and return no final content;
give chat requests headroom.

## 4. Measured speed

<!-- MATRIX_TABLE -->

## 5. KV tuning history

The final pool came from three changes, measured separately:

| Config | KV tokens | Concurrency @ 400K | KV memory |
|---|---:|---:|---:|
| BF16 KV, 8192 batch, 0.85 | 1,029,348 | 2.51× | 80.52 GiB |
| FP8 KV, 8192 batch, 0.85 | 1,487,307 | 3.63× | 80.41 GiB |
| FP8 KV, 4096 batch, 0.85 | 2,219,441 | 5.42× | 81.50 GiB |
| FP8 KV, 2048 batch, 0.85 | 2,897,565 | 7.07× | 82.07 GiB |
| **FP8 KV, 2048 batch, 0.90, 8 slots (published)** | **3,095,742** | **7.56×** | **88.14 GiB** |

FP8 KV did not reduce the memory footprint; vLLM filled the same budget with
more tokens. The batch change is the larger effect: at 8192 vLLM's hybrid-cache
planner padded layer groups (up to 350% waste in one group), while 2048 removed
the padding warnings. The cost is TTFT: ~3.9 s vs ~3.4 s on a 10K prompt.

Details and the BF16/FP8 quality probes are in [`EVAL-REPORT.md`](EVAL-REPORT.md).

## 6. Feature acceptance

| Check | Result | Evidence |
| --- | --- | --- |
| Health / models | pass | `/health` 200; `inkling-2-mini` listed |
| Arithmetic | pass | `19 + 23` → `42` |
| JSON schema | pass | valid city/temperature/conditions object |
| Tool call + result | pass | `get_weather({"city":"Paris"})` round trip |
| Single-image vision | pass | red square → `Red` |
| Anthropic count_tokens | pass | 17 input tokens |
| Anthropic messages | pass | one-word reply |
| Prompt reuse, 32K | pass | 12.54 s → 0.48 s |
| Needle, 120K | pass | 120,015 prompt tokens, `ZEBRA-9174` |
| Reasoning-effort scale | pass | low 133 / high 299 / xhigh 326 tokens |
| 8-way concurrency | pass | wall 3.0 s for 8 requests |

These are smoke checks, not quality scores.

## 7. Reproduce benchmarks and stop

Capture the launch identity first; the matrix hashes `results/launch.json`.

```bash
python3 benchmarks/capture_env.py
```

Run the main matrix (512 forced output tokens; cells above the KV guard are
skipped):

```bash
OUTPUT_TOKENS=512 \
PREFILL_SIZES=512,2048,8192,32768,131072,200000,400000 \
CONCURRENCIES=1,2,4,8 \
python3 benchmarks/matrix.py
```

Recompute the table from saved token events and audit:

```bash
./benchmarks/finalize.sh results/matrix-<timestamp>
```

Run the feature checks only when the matrix is idle:

```bash
python3 benchmarks/smoke.py
```

Stop the service:

```bash
./stop.sh
```

The harness writes:

- `results/launch.json` — model revision, tokenizer/config hashes, image
  digest, exact command, KV configuration, host state.
- `results/matrix-<timestamp>/<size>-c<n>.json` — raw per-request events.
- `results/matrix-<timestamp>/summary.json` — one row per completed cell.
- `results/matrix-<timestamp>/TABLE.md` — the rendered table.
- `results/feature-acceptance.json`, `needle-120k.json`,
  `reasoning-effort.json`, `concurrency-8.json`, `protocol-audit.json`.

## 8. Provenance

| Component | Pinned source / attribution |
| --- | --- |
| Model | [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), measured at `b85c3cfa615fc413cc9af3379f491b963f3b586a`. |
| Runtime image | `vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e`, vLLM `0.1.dev25855+g7cb5dae22.cu130`. |
| Launcher | [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) at `0bdc9c9`; recipe copied into `recipe/`. |
| SGLang comparison | `lmsysorg/sglang:inkling-2-mini`, recorded in `EVAL-REPORT.md`. |
| Repository pattern | [tedngai/glm-5.3-flash-2x-dgx-spark](https://github.com/tedngai/glm-5.3-flash-2x-dgx-spark). |
| Hardware | 1 × NVIDIA DGX Spark GB10, 128 GB unified memory. |

Weights are downloaded at runtime and retain their upstream license. They are
not included in this repository.

## Study protocol and limitations

The workload, ranking rules and evidence audit are in
[`STUDY-PROTOCOL.md`](STUDY-PROTOCOL.md). The short version:

- A throughput winner requires repeatable matched measurements; this matrix is
  a baseline, not a final sweep.
- Differences under 5% between repeated key cells are treated as noise.
- Forced-output tokens are not quality scores.
- FP8 KV is lossy; the quality gate is three deterministic prompts plus
  needles, not a benchmark suite.
- The 400K window, single-image vision, one tool round trip and the Anthropic
  surface are qualified only to the extent shown above. Native 1M context,
  multi-image/video, a one-hour soak and rollback remain open.
