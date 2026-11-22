#!/usr/bin/env bash
# Start Inkling-2-Mini on one DGX Spark.
#
#   ./start.sh              start in the background
#   DAEMON=0 ./start.sh     foreground (launcher log on the terminal)
#   ./start.sh --setup      pull the image / download the model first, then start
#
# The launcher is pinned because the recipe runner and image conventions move
# quickly. Set SPARK_VLLM_DIR to use an existing checkout instead.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_PIN="0bdc9c9"
IMAGE="vllm/vllm-openai:inkling-2-mini"
MODEL="thinkingmachines/Inkling-2-Mini"
CACHE="$HOME/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini"

if [[ -n "${SPARK_VLLM_DIR:-}" ]]; then
  LAUNCHER_DIR="$SPARK_VLLM_DIR"
elif [[ -x "$ROOT/../spark-vllm-docker/run-recipe.sh" ]]; then
  LAUNCHER_DIR="$ROOT/../spark-vllm-docker"
else
  LAUNCHER_DIR="$ROOT/spark-vllm-docker"
fi

if [[ ! -x "$LAUNCHER_DIR/run-recipe.sh" ]]; then
  echo "cloning spark-vllm-docker at $LAUNCHER_PIN into $LAUNCHER_DIR"
  git clone https://github.com/eugr/spark-vllm-docker.git "$LAUNCHER_DIR"
  git -C "$LAUNCHER_DIR" checkout "$LAUNCHER_PIN"
fi

SETUP=0
if [[ "${1:-}" == "--setup" ]]; then
  SETUP=1
  shift
fi

if [[ "$SETUP" == "1" ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker pull "$IMAGE"
fi

if [[ "$SETUP" == "1" ]] || [[ ! -f "$CACHE/refs/main" ]]; then
  "$LAUNCHER_DIR/hf-download.sh" "$MODEL"
fi

if curl -sf --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  if [[ "${FORCE:-0}" != "1" ]]; then
    echo "A server is already answering on port 8000." >&2
    echo "Stop it with ./stop.sh first, or set FORCE=1 to let the launcher decide." >&2
    exit 1
  fi
fi

DAEMON_ARGS=(-d)
if [[ "${DAEMON:-1}" == "0" ]]; then
  DAEMON_ARGS=()
fi

RECIPE="${RECIPE:-$ROOT/recipe/inkling-2-mini.yaml}"
echo "recipe: $RECIPE"
echo "launcher: $LAUNCHER_DIR ($(git -C "$LAUNCHER_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown))"
exec "$LAUNCHER_DIR/run-recipe.sh" \
  "$RECIPE" \
  --solo "${DAEMON_ARGS[@]}" -- "$@"
