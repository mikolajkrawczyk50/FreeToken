#!/usr/bin/env bash
# ==============================================================================
# Fast launcher for FreeToken serving Ornith-1.5-35B on AMD ROCm (RDNA2 / gfx1030)
# ==============================================================================
set -euo pipefail

# Hardware & Toolchain exports for AMD Radeon RX 6800 (gfx1030)
export PYTHONUNBUFFERED=1
export PYTORCH_ROCM_ARCH=gfx1030
export FREETOKEN_ROCM_ARCH=gfx1030
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Model Paths
FTW_MODEL_PATH="/home/user/Ornith-1.5-35B-FTW"
GGUF_MODEL_PATH="/home/user/Ornith-1.5-35B-Q4_K_M.gguf"

if [ -d "$FTW_MODEL_PATH" ]; then
    MODEL_PATH="$FTW_MODEL_PATH"
    echo "[FreeToken] Using FTW fast-path checkpoint: $MODEL_PATH"
elif [ -f "$GGUF_MODEL_PATH" ]; then
    MODEL_PATH="$GGUF_MODEL_PATH"
    echo "[FreeToken] Using GGUF checkpoint: $MODEL_PATH"
else
    echo "[Error] Neither $FTW_MODEL_PATH nor $GGUF_MODEL_PATH found!"
    exit 1
fi

PORT="${PORT:-8000}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-64000}"
PREFILL_LENGTH="${PREFILL_LENGTH:-4096}"

# Check for existing process on port
if lsof -Pi :"$PORT" -sTCP:LISTEN -t >/dev/null ; then
    echo "[FreeToken] Port $PORT is already in use by PID $(lsof -Pi :$PORT -sTCP:LISTEN -t). Please stop it first."
    exit 1
fi

echo "[FreeToken] Starting server on port $PORT (Context: $CONTEXT_TOKENS, Prefill Chunk: $PREFILL_LENGTH)..."

exec python3 -m freetoken.cli serve \
  --model-path "$MODEL_PATH" \
  --served-model-name "Ornith-1.5-35B-Q4_K_M.gguf" \
  --dtype float16 \
  --moe-backend offload \
  --attention-backend triton \
  --cuda-graph-max-bs 0 \
  --memory-ratio 0.88 \
  --max-prefill-length "$PREFILL_LENGTH" \
  --kv-reserve-tokens "$CONTEXT_TOKENS" \
  --max-running-req 1 \
  --enable-special-token-ckpt \
  --tool-call-parser qwen3_coder \
  --port "$PORT"
