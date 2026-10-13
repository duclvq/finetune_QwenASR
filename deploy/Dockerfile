# syntax=docker/dockerfile:1
# Qwen3-ASR-1.7B (Malay LoRA merged) served with vLLM 0.14.0 through qwen-asr-serve.
#
# qwen-asr-serve (qwen_asr/cli/serve.py) only registers the Qwen3ASRForConditionalGeneration
# model class with vLLM and then calls `vllm serve` with argv passed through unchanged, so every
# flag in entrypoint.sh is a plain vLLM 0.14.0 `vllm serve` flag.
#
# Base image: official vLLM OpenAI server, Python 3.12, torch 2.9.1, CUDA 12.x, ENTRYPOINT ["vllm","serve"].
FROM vllm/vllm-openai:v0.14.0

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_OFFLINE=0

# ffmpeg: librosa decodes wav/flac/ogg through soundfile, but falls back to audioread+ffmpeg for
# mp3/m4a/webm uploads. Drop this layer if you only ever send wav.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# qwen-asr 0.0.6 hard-pins vllm==0.14.0 (already in the image) and also drags in gradio, flask,
# accelerate, soynlp, sox, pytz, qwen-omni-utils which the server never imports. Install the
# package without dependencies, then add only what `import qwen_asr.cli.serve` needs at runtime:
#   transformers==4.57.6  hard pin in qwen-asr; inside vLLM 0.14.0's own range (>=4.56,<5)
#   nagisa==0.2.11        imported at package import time by qwen_asr/inference/qwen3_forced_aligner.py
#   librosa, soundfile    qwen_asr/inference/utils.py, and vLLM's /v1/audio/transcriptions decoder
# torch/vllm are never touched: nothing here depends on them with a different pin.
RUN set -eux; \
    if command -v uv >/dev/null 2>&1; then PIP="uv pip install --system --no-cache-dir"; \
    else PIP="python3 -m pip install --no-cache-dir"; fi; \
    $PIP --no-deps "qwen-asr==0.0.6"; \
    $PIP "transformers==4.57.6" "nagisa==0.2.11" librosa soundfile; \
    python3 -c "import qwen_asr.core.transformers_backend, nagisa, librosa, soundfile, transformers, vllm; \
                print('transformers', transformers.__version__, 'vllm', vllm.__version__)"

COPY entrypoint.sh /entrypoint.sh
# strip CRLF in case the file was checked out on Windows without .gitattributes
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# model is bind-mounted read-only here by docker-compose.yml
VOLUME ["/model", "/root/.cache/huggingface"]
EXPOSE 8100

ENTRYPOINT ["/entrypoint.sh"]
