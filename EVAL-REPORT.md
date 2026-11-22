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
capacity gap while keeping decode at parity. Raw samples:
`results/sglang-ab.json`.

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

Raw capacity, speed and quality notes: `results/bf16-vs-fp8.json`. Quality
probes (math, palindrome code, syllogism) were correct under both. FP8 outputs
were more verbose: reasoning tokens rose 264→299, 978→1,199 and
989→1,458 on the three probes. This is a real behavioral difference, not a
free lunch.

### BF16 KV option (`b85`, 0.90, 2,048)

| | FP8 KV | BF16 KV |
|---|---:|---:|
| KV tokens | 3,106,964 | **1,895,843** |
| Concurrency @ 400K | 7.59× | 4.63× |
| Chat prose, 3 prompts | 37.2 tok/s mean | 35.0 tok/s mean |
| Coding answers, 3 prompts | 49.5 tok/s mean | 50.2 tok/s mean |

BF16 keeps exact KV values and still fits a 400K request with ~4.6×
concurrency. The cost is ~39% of the KV pool and ~6% chat-prose decode;
coding-answer decode is unchanged within noise. Use
`recipe/inkling-2-mini-bf16.yaml` to serve it. Receipt:
`results/bf16-kv.json`.

## 4. Prefill batch and KV pool (`ed40`, FP8, 400K window)

`max_num_batched_tokens` controls prefill chunk size and, indirectly, vLLM's
hybrid-cache padding. The pool changed far more than the activation saving:

| Batch | KV tokens | Concurrency @ 400K | KV memory | TTFT, 10K | Decode |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 1,487,307 | 3.63× | 80.41 GiB | ~3.4 s | ~70 tok/s |
| 4,096 | 2,219,441 | 5.42× | 81.50 GiB | not measured | not measured |
| 2,048 | **2,897,565** | **7.07×** | 82.07 GiB | ~3.9 s | ~75 tok/s |

Raw sweep: `results/kv-tuning.json`. At 8,192 vLLM logged padding groups of +6,
+5 and +7 layers, the last "up to 350% waste". At 2,048 those padding warnings disappeared and per-token KV fell
to ~30 KiB. The cost is ~0.5 s extra TTFT on a 10K prompt from more chunks.

Quality at 2,048 vs 8,192, same deterministic probes: **byte-identical**
reasoning and content on all three short prompts. At 120k context the output
path diverged slightly: a 200-token budget was consumed by reasoning at 2,048,
while 600 tokens still returned the correct needle. Chunked prefill is
mathematically equivalent but not guaranteed bit-exact at long context.

### 400K/C1 at 8,192 vs 2,048 (b85, 0.90 utilization)

One targeted retest checked whether a larger prefill budget fixes the 400K
TTFT:

| Batch | KV tokens | Concurrency @ 400K | TTFT | Effective prefill | Decode |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 1,612,400 | 3.94× | **322.4 s** | 1,240.6 tok/s | 41.5 tok/s |
| 2,048 | 3,095,742 | 7.56× | 357.5 s | 1,118.8 tok/s | 47.8 tok/s |

8,192 bought only ~10% lower TTFT and cut the KV pool by ~48%. Not adopted;
the published recipe stays at 2,048. The test is a single 400K/C1 receipt, not
a new matrix.

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
  Earlier receipts: `results/long-context.json`.

## 9. Final matrix (`b85`)

The published matrix has **27 cells and 97 requests**, with zero failed
requests; `results/protocol-audit.json` reports `status: pass`. The full table
is in the README and `results/inkling-2-mini-table.md`.

| Observation | Result |
|---|---|
| C1 decode, 512–32K | 76.3–94.1 tok/s |
| C1 decode, 131K / 200K / 400K | 66.0 / 59.4 / 47.8 tok/s |
| C8 aggregate, 512 / 2K / 8K | 280.7 / 348.4 / 291.5 tok/s |
| Effective prefill, 512–32K | 2,058–2,785 tok/s |
| Effective prefill, 200K / 400K C1 | 1,675 / 1,119 tok/s |
| 400K TTFT, C1 / C2 / C4 | 357.5 / 540.7 / 907.4 s |

Seven cells have **no positive common decode window** (32K/C8, 131K/C4,
131K/C8, 200K/C4, 200K/C8, 400K/C2, 400K/C4). This is a load-shape result,
not a failure: the 2,048-token prefill budget and long prompts stagger the
requests, so peak overlap is 1–6 and not all streams emit in a shared
interval. The aggregate columns are left empty rather than synthesized; the
per-request decode medians are still defined:

| Cell | Peak overlap | Per-request decode median |
|---|---:|---:|
| 32K / C8 | 6 | 12.1 tok/s |
| 131K / C4 | 2 | 6.6 tok/s |
| 131K / C8 | 3 | 5.4 tok/s |
| 200K / C4 | 1 | 4.4 tok/s |
| 200K / C8 | 2 | 4.0 tok/s |
| 400K / C2 | 1 | 25.3 tok/s |
| 400K / C4 | 1 | 4.0 tok/s |

Those medians are low where a request decodes while another request is
prefilling; the last request often decodes at ~40–50 tok/s once the shared
prefill has finished.

## 10. Chat vs benchmark: DSpark acceptance

A live chat request measured 27–34 tok/s while the matrix reports 76–94 tok/s
at C1. The difference is DSpark acceptance, not hardware.

| Workload | Decode | Accepted tokens/step | Draft acceptance |
|---|---:|---:|---:|
| Matrix-style raw code completion | **90.2 tok/s** | **5.00** | **71.4%** |
| Chat prose + reasoning, DSpark 7 | 39.8 tok/s | 1.42 | 20.4% |
| Chat prose + reasoning, DSpark 7 | 37.2 tok/s | 1.31 | 18.7% |
| Chat prose + reasoning, DSpark 7 | 34.5 tok/s | 1.09 | 15.5% |
| Same three prompts, DSpark off | **37.4 tok/s** | — | — |

The matrix prompt is a repetitive code module, which the draft model predicts
well. Open-ended chat and reasoning prose is much less predictable, so most
of the seven drafts are rejected and decode settles around 35–40 tok/s.

**Drafts cannot be reduced.** The checkpoint enforces exactly seven:

```
ValueError: Inkling2 DSpark requires num_speculative_tokens = 7
           (the checkpoint's block size - 1)
```

**DSpark off is not a chat win.** Across three chat prompts, DSpark 7 averaged
37.2 tok/s and no-spec averaged 37.4 tok/s — a tie. No-spec raises the KV pool
from 3.10M to 3.70M tokens (7.59× to 9.03×) and makes chat latency more
consistent, but gives up the 90 tok/s code path. The published recipe keeps
DSpark 7.

Receipt: `results/chat-spec-ab.json`.

## 11. External comparison: Inkling-2-Small on 4× Spark

[AidenLab's recipe](https://aidenle.com/recipes/inkling-2-small-sglang-4x-dgx-spark/)
publishes 55–56 tok/s for one stream. It is a different model, node count,
engine and workload:

| | Current: Mini, 1× Spark | External: Small, 4× Spark |
|---|---|---|
| Model / active parameters | 26B / 5.4B effective | 594B / 17B |
| Hardware | 1× GB10, TP1 | 4× GB10, TP4/EP4 |
| Engine | vLLM day-0 image | SGLang dev image + MIT hooks |
| Context / KV pool | 400K / 3.10M tokens | 524K / 793K tokens |
| Decode, coding answers | **46.6–55.1 tok/s** (mean 49.5) | **55–56 tok/s** |
| Decode, chat prose | 37.2 tok/s | not published |
| Raw code completion | 90.2 tok/s | not published |
| Aggregate | C8 281–348 tok/s, short raw prompts | 92–107 / 131–137 at C4/C8 |
| Cold prefill | 1,119–2,785 tok/s | 3,110 / 2,786 tok/s at 30K / 120K |

The 55–56 number is **16 coding questions with 400-token answers at
temperature 1.0**, not chat. On comparable coding questions the current Mini
measures 46.6–55.1, so the gap is small.

Why four Sparks can beat one on a much larger model:

- Inkling-2-Small has 17B active parameters spread over four nodes — roughly
  4.25B of active work per node on average, versus 5.4B effective active work
  per token for Mini on one Spark.
- The external recipe splits all 384 experts into quarter-width slices across
  nodes; its measured single-stream gain is 48.3 → 60.0 tok/s (+24%). The
  current vLLM recipe has no equivalent hook.
- Coding answers keep DSpark acceptance high. Chat/reasoning prose collapses
  it to 1.1–1.4 tokens/step, which is why the same current server prints
  ~32–37 tok/s in chat.

Receipt: `results/external-4x-small.json`.

## 12. Limitations

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
