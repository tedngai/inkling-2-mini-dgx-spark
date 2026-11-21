# Restart and rebuild

## Prerequisites

- One NVIDIA DGX Spark (GB10, 128 GB unified memory), arm64.
- Docker with the NVIDIA runtime; the container is started with `--gpus all`
  and `--privileged` by the launcher.
- ~35 GB free for the image and ~25 GB for the checkpoint/cache.
- Network access to Docker Hub and Hugging Face (or pre-seeded caches).

The recipe is single-node. There is no worker and no SSH setup.

## First run

```bash
git clone https://github.com/tedngai/inkling-2-mini-dgx-spark.git
cd inkling-2-mini-dgx-spark
./start.sh --setup
```

`--setup` pulls `vllm/vllm-openai:inkling-2-mini` (arm64) and downloads
`thinkingmachines/Inkling-2-Mini` (~19 GB) into
`~/.cache/huggingface/hub`. It then starts the server in the background on
port 8000.

Later starts:

```bash
./start.sh
DAEMON=0 ./start.sh     # foreground
./stop.sh
```

`start.sh` uses `spark-vllm-docker` at `0bdc9c9`. Set `SPARK_VLLM_DIR` to an
existing checkout instead of cloning into the repository directory.

## Verify

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models
```

Expected model id: `inkling-2-mini`, `max_model_len: 409600`.

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini",
       "messages":[{"role":"user","content":"What is 19 + 23?"}],
       "max_tokens":256,
       "reasoning_effort":"low"}'
```

The model reasons by default. If a reply has empty `content`, inspect the
`reasoning` field and give it a larger `max_tokens` budget.

## Capture the launch identity

```bash
python3 benchmarks/capture_env.py
python3 benchmarks/smoke.py
```

`capture_env.py` records the model revision, tokenizer/config hashes, image
digest, exact command and KV-pool numbers in `results/launch.json` and
`results/config.json`.

## Memory fallbacks

The published profile uses `gpu_memory_utilization: 0.90`, which leaves about
7 GiB of host memory available and a 3.10M-token FP8 KV pool. If another
service shares the Spark:

| Setting | KV pool | Host headroom |
|---|---:|---:|
| 0.90 (published) | 3.10M tokens | ~7 GiB |
| 0.88 | ~2.95M tokens | ~9 GiB |
| 0.85 | 2.90M tokens | ~13 GiB |

Set `gpu_memory_utilization` in `recipe/inkling-2-mini.yaml` and restart.

## Logs

```bash
docker logs -f vllm_node
docker stats --no-stream vllm_node
```

The container is named `vllm_node`. `/health` returns 200 with an empty body;
use `/v1/models` and `/metrics` for detail.

## Clean restart

```bash
./stop.sh
./start.sh
```

`stop.sh` stops and removes the container. The model and image stay cached.
