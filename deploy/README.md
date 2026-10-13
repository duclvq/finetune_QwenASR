# Serving the Malay Qwen3-ASR model with vLLM (Docker Compose)

Everything in this folder is self-contained: copy `deploy/` plus the merged checkpoint to a Linux box
with an NVIDIA GPU, run `docker compose up -d --build`, and you get an OpenAI-compatible speech API
(`/v1/audio/transcriptions`, `/v1/chat/completions`) backed by vLLM 0.14.0.

| file | role |
|---|---|
| `Dockerfile` | `vllm/vllm-openai:v0.14.0` + `qwen-asr==0.0.6` (model class registration) + ffmpeg |
| `entrypoint.sh` | turns env vars into the `qwen-asr-serve` / `vllm serve` command line |
| `docker-compose.yml` | service `asr`: GPU reservation, read-only model mount, HF cache, healthcheck |
| `.env.example` | all tunables with their defaults; copy to `.env` |

`qwen-asr-serve` is a 10-line shim (`qwen_asr/cli/serve.py`): it registers
`Qwen3ASRForConditionalGeneration` with vLLM's `ModelRegistry` and then calls `vllm serve` with the
argv untouched. So every flag below is a stock vLLM 0.14.0 flag, and anything you put in `EXTRA_ARGS`
goes straight to vLLM.

## 1. Get the merged checkpoint onto the GPU box

The model to serve is the LoRA adapter merged into the base and saved as a plain checkpoint
(`config.json`, `model.safetensors` ~3.9 GB, tokenizer + `preprocessor_config.json`). It lives at
`outputs/lora-1.7b-malay50h-babel/merged/` on the training machine.

Either copy it:

```bash
rsync -avP outputs/lora-1.7b-malay50h-babel/merged/  gpu-box:/data/models/qwen3-asr-malay/
```

or rebuild it on the GPU box from the adapter (needs the base `models/Qwen3-ASR-1.7B`, `peft` and
`qwen-asr` installed; the merge is the first thing `eval_quant.py` does, and `--max_samples 1` keeps
the eval that follows to a single clip):

```bash
python eval_quant.py --adapter outputs/lora-1.7b-malay50h-babel/adapter --configs bf16 --max_samples 1
# -> outputs/lora-1.7b-malay50h-babel/merged/
```

Equivalent one-liner without the eval script:

```python
from qwen_asr import Qwen3ASRModel
from peft import PeftModel
import torch
w = Qwen3ASRModel.from_pretrained("models/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cpu")
w.model = PeftModel.from_pretrained(w.model, "outputs/lora-1.7b-malay50h-babel/adapter").merge_and_unload()
gc = w.model.generation_config; gc.temperature = None; gc.top_p = None; gc.top_k = None
w.model.save_pretrained("outputs/lora-1.7b-malay50h-babel/merged", safe_serialization=True)
w.processor.save_pretrained("outputs/lora-1.7b-malay50h-babel/merged")
```

(The generation-config reset mirrors `eval_quant.py::merge_adapter`: upstream ships sampling
parameters together with `do_sample=False`, which `save_pretrained` rejects.)

## 2. Run

```bash
cd deploy
cp .env.example .env            # set MODEL_DIR to the checkpoint directory, optionally API_KEY
docker compose up -d --build    # first build pulls the ~10 GB vLLM image
docker compose logs -f asr      # wait for "Application startup complete"
curl -s localhost:8100/health -o /dev/null -w '%{http_code}\n'   # 200 when ready
curl -s localhost:8100/v1/models | python -m json.tool
```

Prerequisites on the host: NVIDIA driver, Docker >= 24 with the NVIDIA Container Toolkit
(`nvidia-ctk runtime configure --runtime=docker`), Docker Compose v2. The compose file reserves one
GPU via `deploy.resources.reservations.devices`; to pin a particular card replace `count: 1` with
`device_ids: ["1"]`.

Resulting command (from `entrypoint.sh`, defaults shown):

```
qwen-asr-serve /model --served-model-name qwen3-asr-malay --host 0.0.0.0 --port 8100 \
  --dtype bfloat16 --gpu-memory-utilization 0.9 --max-model-len 4096 --max-num-seqs 96 \
  --max-num-batched-tokens 16384 --limit-mm-per-prompt '{"audio":1}' --quantization fp8 [--api-key ...]
```

## 3. Call it

### `/v1/audio/transcriptions` (OpenAI style, multipart)

```bash
curl -s http://localhost:8100/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F model=qwen3-asr-malay \
  -F language=ms \
  -F response_format=json \
  -F file=@sample.wav
# {"text":"language Malay<asr_text>saya nak pergi ke pasar"}
```

- Any sample rate / channel count is fine: vLLM decodes with librosa and resamples to 16 kHz. wav/flac/ogg
  need nothing extra; mp3/m4a/webm go through ffmpeg (installed in the image).
- Clips longer than the processor's `chunk_length` (30 s) are split into chunks and the transcripts
  concatenated, so long files work but sentence boundaries at chunk edges may suffer.
- `language` is an ISO-639-1 code validated against vLLM's Whisper language table
  (`ms` = Malay, `vi` = Vietnamese, `en`, ...). Unknown codes are rejected with HTTP 400; omitting it
  is allowed (the model then detects the language itself).
- **The response text starts with a language tag: `language Malay<asr_text>`.** That is the native
  Qwen3-ASR output format (the model was fine-tuned with it) and vLLM returns generated tokens
  verbatim. Strip it client-side:

  ```python
  import re
  def strip_prefix(text: str) -> str:
      return re.sub(r"^language [^<]*<asr_text>", "", text, count=1).strip()
  ```

  The `qwen_asr` package does the same in `qwen_asr.parse_asr_output`.
- Alternative, read from `qwen_asr/core/vllm_backend/qwen3_asr.py::get_generation_prompt`: the Qwen
  backend ignores `language` when building the prompt, but if the vLLM extension field
  `to_language=ms` is sent it pre-fills `language Malay<asr_text>` into the assistant turn. The
  response then contains only the transcript and the language is forced instead of detected. This
  was taken from the source and not exercised end to end; test before relying on it.
- `temperature` defaults to 0 (greedy) on this endpoint, which is what the evals used.

### `/v1/chat/completions` with `audio_url`

```bash
B64=$(base64 -w0 sample.wav)
curl -s http://localhost:8100/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3-asr-malay\",\"temperature\":0,\"max_tokens\":400,
       \"messages\":[{\"role\":\"user\",\"content\":[
         {\"type\":\"audio_url\",\"audio_url\":{\"url\":\"data:audio/wav;base64,$B64\"}}]}]}"
```

`audio_url.url` may also be an `http(s)://` link reachable from the container. The chat template in
the checkpoint (`chat_template.jinja`) places the audio placeholder; one audio per request
(`--limit-mm-per-prompt '{"audio":1}'`). The assistant text carries the same
`language Malay<asr_text>` prefix.

### OpenAI Python SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8100/v1", api_key=API_KEY or "none")
r = client.audio.transcriptions.create(model="qwen3-asr-malay", file=open("sample.wav", "rb"), language="ms")
print(strip_prefix(r.text))
```

## 4. QUANT: why `fp8`, and why there is no `q6`

| `QUANT` | what vLLM does | GPU |
|---|---|---|
| `fp8` (default) | `--quantization fp8`: converts the LLM's linear weights to FP8 (e4m3) at load time, per-channel scales, dynamic per-token activation scaling. The audio encoder stays bf16 (the Qwen backend builds the encoder's linears without `quant_config`). About half the LLM weight memory, same or better throughput. | native FP8 tensor cores on compute capability >= 8.9 (L4, RTX 40xx/50xx, H100/H200, B200). On Ampere (8.0/8.6: A100, A10, A30, RTX 30xx) vLLM falls back to the FP8-Marlin weight-only kernel: works, same memory saving, dequantizes to bf16 for the GEMM. Not available on Turing/Volta. |
| `none` | plain `--dtype bfloat16` weights, no quantization. Set `DTYPE=float16` on Turing (T4, RTX 20xx), which has no bf16. | any GPU vLLM supports |

The request was for a "q6" build. There is no honest way to ship one:

- Our q6 result in `eval_quant.py` was a **round-to-nearest simulation** of 6-bit weights done in
  transformers: the weights were quantized to 6 bits and immediately dequantized back to bf16, so it
  measured accuracy only. Memory and speed were those of bf16.
- **HQQ has no 6-bit packing**: it stores 6-bit tensors unpacked (8-bit containers), so a "q6" HQQ
  checkpoint saves no memory over int8.
- **vLLM has no 6-bit weight format for this architecture.** Its sub-8-bit paths (GPTQ, AWQ,
  compressed-tensors W4A16/W8A16, bitsandbytes NF4, GGUF) are 4- or 8-bit, and none of them has a
  loader for `Qwen3ASRForConditionalGeneration` (the model class is registered at runtime by
  `qwen-asr`, not shipped with vLLM). GGUF Q6_K would additionally need a converter for this
  architecture.
- The int8 numbers we measured came from **HQQ int8 in transformers**, which ran about **2x slower**
  than bf16 because HQQ dequantizes in unfused kernels. That is not what vLLM does: vLLM's FP8
  path uses fused CUTLASS/Marlin GEMMs and is as fast as or faster than bf16.

So `fp8` is the near-lossless 8-bit option that actually reduces memory and keeps throughput; `none`
is the exact bf16 model from the eval. If a sub-8-bit build is ever required, the realistic route is
an AWQ/GPTQ 4-bit export plus a vLLM quantization-aware loader for this model class, which is a
separate piece of work.

Memory: the bf16 checkpoint is 4.08 GB on disk (encoder + 1.7B LLM); with `fp8` roughly 2.5 GB of
weights. The rest of `GPU_MEM` x VRAM goes to KV cache (`--max-model-len 4096` x `--max-num-seqs 96`
is an upper bound; vLLM lowers concurrency if the cache is too small). An 8 GB card runs the server
with `GPU_MEM=0.85`; a 24 GB card has room to raise `MAX_SEQS`.

## 5. Tunables (`.env`)

| var | default | meaning |
|---|---|---|
| `MODEL_DIR` | `../outputs/lora-1.7b-malay50h-babel/merged` | host path bind-mounted read-only at `/model` (relative to `deploy/`) |
| `MODEL_PATH` | `/model` | what the server loads; set to a HF repo id to serve from the hub instead of the mount |
| `SERVED_NAME` | `qwen3-asr-malay` | value clients send in `model` |
| `PORT` | `8100` | host and container port |
| `QUANT` | `fp8` | `fp8` or `none` (see above) |
| `DTYPE` | `bfloat16` | `--dtype`; `float16` on Turing |
| `MAX_SEQS` / `MAX_MODEL_LEN` / `MAX_BATCHED` | `96` / `4096` / `16384` | vLLM scheduler limits, same as the Vietnamese track's `serve_vllm.sh` |
| `GPU_MEM` | `0.9` | `--gpu-memory-utilization` (fraction of VRAM vLLM reserves). Lower it when sharing the GPU |
| `API_KEY` | empty | when set, `/v1/*` requires `Authorization: Bearer <key>`; `/health` stays open |
| `EXTRA_ARGS` | empty | appended `vllm serve` flags, e.g. `--enforce-eager` (faster startup, slower decode) or `--kv-cache-dtype fp8` |
| `HF_TOKEN` | inherited from the host shell if set | only needed for gated hub repos |
| `HF_CACHE` | named volume `hf-cache` | HF cache mount; only used when `MODEL_PATH` is a hub id |

`docker compose run --rm asr --help` prints vLLM's full flag list (arguments after the service name
are appended by `entrypoint.sh`).

## 6. Serving the Vietnamese full model with the same files

Nothing in the image is Malay-specific. The Vietnamese track (git branch `vi`) served its full
fine-tune `outputs/full-1.7b/final` (also on the hub as `duclvQ/qwen3-asr-1.7b-vimd-full`) with the
same flags; only the model and the served name change:

```bash
# local checkpoint
MODEL_DIR=/data/models/qwen3-asr-vi SERVED_NAME=qwen3-asr-vi PORT=8101 docker compose -p asr-vi up -d
# or straight from the hub (first start downloads into the hf-cache volume)
MODEL_PATH=duclvQ/qwen3-asr-1.7b-vimd-full SERVED_NAME=qwen3-asr-vi PORT=8101 docker compose -p asr-vi up -d
```

Clients then send `language=vi` and strip the `language Vietnamese<asr_text>` prefix. To run both
models on one box use a different `-p` project name, `container_name` and `PORT` for each; each
container reserves one GPU, so either give them separate cards (`device_ids`) or share one by setting
`GPU_MEM=0.45` on both.

## 7. Operations

- Logs: `docker compose logs -f asr`. Health: `curl localhost:8100/health` (200 ok, 503 engine dead);
  the compose healthcheck polls it every 30 s after a 10-minute start grace period.
- Restart policy is `unless-stopped`, so the API comes back after a reboot.
- Metrics: `GET /metrics` (Prometheus) is on by default in vLLM.
- Upgrade the model: replace the files under `MODEL_DIR`, then `docker compose restart asr`
  (the mount is read-only; no rebuild needed). Rebuild only when `Dockerfile`/`entrypoint.sh` change.
- First start on a fresh box is slow: image pull (~10 GB), then weight load, FP8 conversion and CUDA
  graph capture (a few minutes). `EXTRA_ARGS=--enforce-eager` skips graph capture at a decode-speed cost.

## 8. Not verified here

This folder was written and validated on a Windows machine without a Linux GPU: `docker compose
config`, `bash -n entrypoint.sh` and `docker build --check` pass, but the image was not built or run.
Things to confirm on the GPU box at first start:

1. `pip`/`uv` resolution inside `vllm/vllm-openai:v0.14.0` (Python 3.12): `transformers==4.57.6`
   is within vLLM 0.14.0's declared range and `nagisa==0.2.11` has cp312 manylinux wheels, so no
   compilation is expected, but the image's exact preinstalled set was not inspected.
2. `--quantization fp8` on the fine-tuned checkpoint: re-run a quick WER check through the API
   against the bf16 numbers (`QUANT=none`) before switching production traffic.
3. The `to_language` trick in section 3.

## 9. Measured on an RTX 5060 Ti 16 GB (Docker Desktop / WSL2, FP8, GPU_MEM=0.6)

`python deploy/bench.py --levels 1 2 4 8 16 32 64` (24 babel_test clips of 2-30 s, 48-64 requests per level, raw in
`outputs/bench_api/`). Engine init 172 s (CUDA-graph capture), 10.6 GB VRAM, first request ~46 s warm-up.

| Clip length | avg RTF | p50 RTF | p95 RTF | p50 latency |
|---|---|---|---|---|
| 2-5 s | 0.056 | 0.049 | 0.080 | 0.21 s |
| 5-15 s | 0.045 | 0.043 | 0.054 | 0.36 s |
| 15-30 s | 0.046 | 0.046 | 0.058 | 1.19 s |

| Concurrent | avg RTF | p95 RTF | p95 latency | throughput (x real time) | req/s | GPU util | power |
|---|---|---|---|---|---|---|---|
| 1 | 0.049 | 0.077 | 1.50 s | 21 | 1.7 | 94% | 118 W |
| 8 | 0.052 | 0.084 | 1.63 s | 151 | 11.6 | 99% | 119 W |
| 16 | 0.058 | 0.091 | 1.74 s | 222 | 17.1 | 99% | 120 W |
| 32 | 0.069 | 0.105 | 1.80 s | 286 | 22.0 | 99% | 119 W |
| 64 | 0.086 | 0.151 | 1.92 s | 347 | 30.4 | 90% | 122 W |

Compute-bound (99% SM utilization from 2 streams, 120 W of a 150 W limit); p95 RTF stays under 0.5 at every level
tried. Whole-file requests, not the streaming gateway, and short runs per level.
