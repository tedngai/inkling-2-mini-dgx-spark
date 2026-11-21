#!/usr/bin/env python3
"""Capture the identity and configuration of the running Inkling-2-Mini recipe.

Writes `results/launch.json` (hashed by the matrix) and `results/config.json`
(the short published configuration). No secrets are read or printed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
RESULTS.mkdir(parents=True, exist_ok=True)
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
CONTAINER = os.environ.get("CONTAINER", "vllm_node")
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
MODEL_REPO = "thinkingmachines/Inkling-2-Mini"
CACHE_MODEL = HF_HOME / "hub" / "models--thinkingmachines--Inkling-2-Mini"


def run(command: str, timeout: int = 30) -> str:
    try:
        return subprocess.run(
            command, shell=True, check=True, text=True, capture_output=True,
            timeout=timeout,
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001 - capture, do not fail the report
        return f"<unavailable: {error}>"


def get_json(path: str) -> dict:
    with urllib.request.urlopen(API_BASE + path, timeout=20) as response:
        return json.loads(response.read())


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


revision = (CACHE_MODEL / "refs" / "main").read_text().strip()
snapshot = CACHE_MODEL / "snapshots" / revision
tokenizer_file = snapshot / "tokenizer.json"
config_file = snapshot / "config.json"

model = get_json("/v1/models")["data"][0]
try:
    version = get_json("/version").get("version")
except Exception:  # noqa: BLE001
    version = "<unavailable>"
metrics = urllib.request.urlopen(API_BASE + "/metrics", timeout=20).read().decode()
cache_line = next(
    (line for line in metrics.splitlines() if line.startswith("vllm:cache_config_info{")),
    "",
)
cache_info = dict(re.findall(r'(\w+)="([^"]*)"', cache_line))
processes = run(f"docker exec {CONTAINER} ps -eo pid,args")
command = next(
    (line.split(None, 1)[1] for line in processes.splitlines() if "vllm serve" in line),
    "<not found>",
)
image_id = run(f"docker inspect {CONTAINER} --format '{{{{.Image}}}}'")
image_digest = run(
    f"docker image inspect {image_id} --format '{{{{index .RepoDigests 0}}}}'"
)
image_tag = run(
    f"docker inspect {CONTAINER} --format '{{{{.Config.Image}}}}'"
)
mem = run("free -m")
uname = run("uname -a")
hostname = run("hostname")
gpu = run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
container_stats = run(f"docker stats --no-stream --format '{{{{.MemUsage}}}}' {CONTAINER}")

launch = {
    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "hostname": hostname,
    "uname": uname,
    "hardware": "NVIDIA DGX Spark GB10, 128 GB unified memory",
    "gpu_query": gpu,
    "model": model["id"],
    "model_root": model.get("root"),
    "model_repo": MODEL_REPO,
    "model_revision": revision,
    "model_snapshot": str(snapshot),
    "config_sha256": sha256(config_file),
    "tokenizer_sha256": sha256(tokenizer_file),
    "image_tag": image_tag,
    "image_id": image_id,
    "image_digest": image_digest,
    "vllm_version": version,
    "command": command,
    "cache_config_info": cache_info,
    "container_mem_usage": container_stats,
    "host_mem_free_m": mem,
}

config = {
    "model": MODEL_REPO,
    "model_revision": revision,
    "served_model_name": model["id"],
    "max_model_len": model.get("max_model_len"),
    "kv_cache_dtype": cache_info.get("cache_dtype"),
    "kv_cache_size_tokens": int(cache_info.get("kv_cache_size_tokens", 0) or 0),
    "kv_cache_max_concurrency_at_max_len": float(
        cache_info.get("kv_cache_max_concurrency", 0) or 0
    ),
    "gpu_memory_utilization": cache_info.get("gpu_memory_utilization"),
    "max_num_batched_tokens": 2048,
    "max_num_seqs": 8,
    "enable_prefix_caching": cache_info.get("enable_prefix_caching") == "True",
    "speculative_config": {
        "method": "dspark",
        "num_speculative_tokens": 7,
        "draft_sample_method": "probabilistic",
    },
    "reasoning_effort_levels": {
        "none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7,
        "high": 0.9, "xhigh": 0.99, "max": 0.99,
    },
    "reasoning_effort_default": 0.9,
}

(RESULTS / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
(RESULTS / "config.json").write_text(json.dumps(config, indent=2) + "\n")
print(json.dumps({
    "launch": str(RESULTS / "launch.json"),
    "config": str(RESULTS / "config.json"),
    "model_revision": revision,
    "vllm_version": version,
    "kv_cache_size_tokens": config["kv_cache_size_tokens"],
    "kv_cache_max_concurrency": config["kv_cache_max_concurrency_at_max_len"],
}, indent=2))
