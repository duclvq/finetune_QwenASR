#!/usr/bin/env bash
# Build the qwen-asr-serve argument list from environment variables and exec it.
# qwen-asr-serve == `vllm serve` (+ model-class registration), so all flags below are vLLM 0.14.0 flags.
# Extra container arguments (docker compose run asr --enforce-eager ...) are appended verbatim.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/model}             # local dir (bind mount) or a HF repo id
SERVED_NAME=${SERVED_NAME:-qwen3-asr-malay}  # name clients put in "model"
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8100}
QUANT=${QUANT:-fp8}                          # fp8 | none
DTYPE=${DTYPE:-bfloat16}                     # bfloat16 (Ampere+) | float16 (Turing/Volta) | auto
MAX_SEQS=${MAX_SEQS:-96}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_BATCHED=${MAX_BATCHED:-16384}
GPU_MEM=${GPU_MEM:-0.9}
EXTRA_ARGS=${EXTRA_ARGS:-}                   # free-form extra vllm flags, e.g. "--enforce-eager --kv-cache-dtype fp8"

if [[ "$MODEL_PATH" == /* && ! -f "$MODEL_PATH/config.json" ]]; then
  echo "[entrypoint] $MODEL_PATH/config.json not found: is MODEL_DIR in deploy/.env pointing at the merged checkpoint?" >&2
  exit 1
fi

args=(
  "$MODEL_PATH"
  --served-model-name "$SERVED_NAME"
  --host "$HOST" --port "$PORT"
  --dtype "$DTYPE"
  --gpu-memory-utilization "$GPU_MEM"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_SEQS"
  --max-num-batched-tokens "$MAX_BATCHED"
  --limit-mm-per-prompt '{"audio":1}'
)

case "${QUANT,,}" in
  fp8)            args+=(--quantization fp8) ;;   # online weight-only FP8 of the LLM linears; audio encoder stays $DTYPE
  none|bf16|"")   ;;                              # plain $DTYPE weights
  *) echo "[entrypoint] QUANT must be 'fp8' or 'none', got '$QUANT'" >&2; exit 2 ;;
esac

if [[ -n "${API_KEY:-}" ]]; then
  args+=(--api-key "$API_KEY")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206  # intentional word splitting
  extra=($EXTRA_ARGS)
  args+=("${extra[@]}")
fi

args+=("$@")

# log the command with the key masked
printable=("${args[@]}")
for i in "${!printable[@]}"; do
  if [[ "${printable[$i]}" == "--api-key" ]]; then printable[$((i+1))]="****"; fi
done
echo "[entrypoint] exec qwen-asr-serve ${printable[*]}"

exec qwen-asr-serve "${args[@]}"
