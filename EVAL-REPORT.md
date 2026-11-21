# Evaluation report

This is the experiment log behind the published recipe. It separates
**throughput** (matrix), **capacity** (KV pool), **feature acceptance** (smoke)
and **quality** (small deterministic probes). None of the throughput cells are
quality scores.

## 1. Revision note

The model repository's `main` moved during the work:

| Revision | Used for |
|---|---|
| `ed40e7da625f594fc9c4990f5f66be24873c22d7` | First launch, vLLM/SGLang A/B, BF16-vs-FP8 KV A/B, first long-context checks |
| `b85c3cfa615fc413cc9af3379f491b963f3b586a` | Final recipe, final matrix, smoke checks, reasoning-effort checks |

The final `results/launch.json` records `b85c3cfa...`. The comparison rows
below that used `ed40...` are labelled; no difference is attributed to the
revision, and the throughput of the final recipe should be read from the matrix,
not from the earlier A/B.

## 2. vLLM vs SGLang on the same Spark (`ed40`)

Both engines served the same checkpoint on one GB10, one request at a time,
with 9.4K–10K-token realistic prompts, 1,024 output tokens, greedy sampling and
DSpark enabled. Six prompts per engine.

| Metric | vLLM | SGLang |
|---|---:|---:|
| Decode, mean | 63.4 tok/s | **65.3 tok/s** |
| Decode, median | 63.7 tok/s | **66.4 tok/s** |
| TTFT, mean | 4.04 s | **2.86 s** |
| Configured context | 262,144 | 1,048,576 |
| KV pool at the measured start | 790,632 tokens (BF16) | 493,056 tokens (BF16) |
| Effective single-request max | 262,144 | ~493K (KV-limited) |

Decode was a tie within noise. SGLang's real advantage was prefill: ~1.2 s
lower TTFT. SGLang's 1M context setting was larger than its 493K KV pool, so
the practical context gap was ~1.9×, not 4×. The cookbook's published 63 tok/s
used a simulated 3.5-token DSpark acceptance; our measured SGLang run used real
acceptance and landed at 65.3.

The vLLM recipe was kept because the target workload is one Spark, the vLLM
image is the day-0 reference, and the FP8-KV tuning below recovered the
capacity gap while keeping decode at parity.

## 3. BF16 vs FP8 KV (`ed40`, 0.85 utilization, 8,192 batch)

| | BF16 KV | FP8 KV |
|---|---:|---:|
| KV pool | 1,029,348 tokens | **1,487,307 tokens** |
| KV memory reserved | 80.52 GiB | 80.41 GiB |
| Per token | ~82 KiB | ~55 KiB |
| Concurrency @ 400K | 2.51× | **3.63×** |
| Decode, three prompts | 67.9 / 63.8 / 70.3 | 71.7 / 75.9 / 63.0 |
| TTFT, 10K prompt | ~3.2 s | ~3.2–3.4 s |

FP8 KV did **not** lower the memory footprint: vLLM filled the same budget
with more tokens. It added ~4% decode and ~45% capacity.

Quality probes (math, palindrome code, syllogism) were correct under both.
FP8 outputs were more verbose: reasoning tokens rose 264→299, 978→1,199 and
989→1,458 on the three probes. This is a real behavioral difference, not a
free lunch.

## 4. Prefill batch and KV pool (`ed40`, FP8, 400K window)

`max_num_batched_tokens` controls prefill chunk size and, indirectly, vLLM's
hybrid-cache padding. The pool changed far more than the activation saving:

| Batch | KV tokens | Concurrency @ 400K | KV memory | TTFT, 10K | Decode |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 1,487,307 | 3.63× | 80.41 GiB | ~3.4 s | ~70 tok/s |
| 4,096 | 2,219,441 | 5.42× | 81.50 GiB | not measured | not measured |
| 2,048 | **2,897,565** | **7.07×** | 82.07 GiB | ~3.9 s | ~75 tok/s |

At 8,192 vLLM logged padding groups of +6, +5 and +7 layers, the last "up to
350% waste". At 2,048 those padding warnings disappeared and per-token KV fell
to ~30 KiB. The cost is ~0.5 s extra TTFT on a 10K prompt from more chunks.

Quality at 2,048 vs 8,192, same deterministic probes: **byte-identical**
reasoning and content on all three short prompts. At 120k context the output
path diverged slightly: a 200-token budget was consumed by reasoning at 2,048,
while 600 tokens still returned the correct needle. Chunked prefill is
mathematically equivalent but not guaranteed bit-exact at long context.

## 5. Utilization and admission

| Config | KV tokens | KV memory | Concurrency @ 400K | Host available |
|---|---:|---:|---:|---:|
| 0.85, 4 slots | 2,897,565 | 82.07 GiB | 7.07× | ~13 GiB |
| 0.90, 8 slots | **3,095,742** | 88.14 GiB | **7.56×** | ~7 GiB |

Eight simultaneous short requests all returned HTTP 200 in 3.0–7.7 s wall
(depending on output budget), confirming the admission slot is exercised rather
than merely configured. Host headroom at 0.90 is ~7 GiB; this is the reason the
recipe stops at 0.90 rather than 1.0.

## 6. Reasoning effort (`b85`)

The chat template maps `reasoning_effort` to a numeric level:

| Level | Value | Measured reasoning tokens |
|---|---:|---:|
| none | 0.0 | not re-measured at `b85` |
| minimal | 0.1 | not re-measured at `b85` |
| low | 0.2 | 133 |
| medium | 0.7 | not re-measured at `b85` |
| high | 0.9 | 299 |
| xhigh | 0.99 | 326 |
| max | 0.99 | same as xhigh |

The default is 0.9. `xhigh` and `max` are identical. An earlier full sweep at
`ed40` measured none 101, minimal 85, low 85, medium 240, high 285, xhigh 429,
max 429; the model card warns that minimum effort does not fully disable
reasoning, and the `none` row supports that.

A separate `thinking_token_budget` field hard-caps reasoning tokens; a budget
of 50 produced exactly 50 reasoning tokens in the earlier test.

## 7. Feature acceptance (`b85`)

`smoke.py` passed **13/13** checks on one boot:

| Check | Result |
|---|---|
| Health / models | pass |
| Arithmetic | pass (`42`) |
| JSON schema | pass |
| Tool call request + round trip | pass |
| Single-image red square | pass (`Red`) |
| Anthropic count_tokens | pass |
| Anthropic messages | pass (`Hello`) |
| Prompt reuse, 32k | pass (12.54 s → 0.48 s) |
| Needle, 120,015 tokens | pass (`ZEBRA-9174`) |
| Reasoning-effort scale | pass (133 / 299 / 326) |
| 8-way concurrency | pass (wall 3.0 s) |

These are smoke checks, not a quality evaluation.

## 8. Context receipts

- **362,661-token** prompt accepted under the earlier BF16/8,192 recipe:
  244.5 s end-to-end, ~1,483 tok/s prefill.
- **120,941-token** needle passed under FP8/8,192.
- **120,015-token** needle passed under the final FP8/2,048 recipe.
- The final matrix includes a **400,000-token** single request and a
  400,000-token C2/C4 wave (see `results/inkling-2-mini-table.md`).

## 9. Limitations

- FP8 KV is lossy and the image warns that accuracy can drop without a proper
  scaling factor. The quality gate is three deterministic prompts plus needles,
  not a benchmark suite.
- The matrix uses raw completions with forced output; it does not measure
  chat-template or reasoning overhead.
- One boot, no one-hour soak, no multi-user agent trace, no rollback exercise.
- Multi-image/video input and the full 1,048,576-token window are not
  qualified.
- The model repository is not pinned by the recipe; re-check
  `results/launch.json` before comparing a new run to these numbers.
