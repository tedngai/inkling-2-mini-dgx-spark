#!/usr/bin/env bash
# Stop the Inkling-2-Mini server on this Spark.
set -euo pipefail
docker stop vllm_node >/dev/null 2>&1 || true
docker rm -f vllm_node >/dev/null 2>&1 || true
echo "stopped vllm_node"
