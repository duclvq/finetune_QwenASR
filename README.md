# Fine-tuning Qwen3-ASR-1.7B for Malay and Malay-English code-switch

LoRA fine-tuning of [Qwen3-ASR-1.7B](https://github.com/QwenLM/Qwen3-ASR) on Malay, with the data pipelines,
evaluation (in-domain, Revolab benchmark, synthetic and real code-switch), results and a Colab notebook for training
without a local GPU. Branches:

| Branch | Content |
|---|---|
| `main` | Malay track (this README, [docs/malay.md](docs/malay.md)) |
| `vi` | Vietnamese track (its own README, report and results live only on that branch) |

The trainer and evaluator are shared; `vi` has the older copies plus the Vietnamese-only tools. Nothing on `main` reports Vietnamese numbers.

**Best system:** `lora-1.7b-malay50h-babel` (83h: 50h cleaned Malay + code-switch + English + 12h real conversational
Malay). WER:

| System | in-domain test (1,994) | Revolab benchmark (820, 12 domains) | synthetic code-switch (221) | real code-switch, unseen speaker (377) |
|---|---|---|---|---|
| base Qwen3-ASR-1.7B | 52.19 | 15.01 | 25.69 | 17.94 |
| lora-mix (63h) | 37.35 | **13.38** | 8.69 | 14.76 |
| **lora-babel (83h)** | **37.03** | 13.39 | 8.69 | **7.83** |

A Malay-only LoRA forgets English and loses to base out of domain; mixing in code-switched and English speech fixes
that. About a third of lora-babel's gain on real code-switch is label-convention matching (spelling variants,
spelled-out numbers), the rest is recognition; one test speaker only. Full tables, ablations, analysis and next
steps: [docs/malay.md](docs/malay.md).

## Repository layout

```
train_qwen_asr.py      LoRA / QLoRA / full trainer (--mode), per-step train_log.jsonl, auto-resume from the last checkpoint
eval_wer.py            WER/CER of base + adapters on a prepared split; writes <system>_predictions.jsonl + summary.json
score_malay.py         re-scores eval_wer.py predictions with Revolab's Malay text normalizer + number spelling
eval_revolab.py        Revolab/ASR-Benchmark-Public (820 clips, 12 domains), per-domain WER with the benchmark's scoring
eval_quant.py          merges a LoRA adapter into the base (outputs/<run>/merged), then sweeps HQQ int8/int4/... for WER and memory
plot_malay.py          regenerates every figure in docs/malay.md from results/malay/ into docs/figures/
tb_sync.py             mirrors outputs/*/train_log.jsonl into TensorBoard, live
smoke_test.py          loads the base model and transcribes one clip (transformers or vllm backend)

data_prep/
  build_malay_dataset.py   malaya-speech parquet -> prepared format (train/valid/test), up to N hours, resumable per shard
  subset_malay.py          random N-hour train subset of a prepared build
  vad_stats.py             Silero VAD speech seconds per clip -> <split>/vad.jsonl
  clean_malay_subset.py    drops clips whose label is too short for the speech, tops the subset back up
  build_malay_mix.py       + Synth-Manglish + cs_pilot code-switch + FLEURS English (tagged language English); carves cs_test
  build_malay_babel.py     + BabelSpeech, per-speaker cap, whole speakers held out as babel_test

scripts/
  run_malay.sh             the whole chain: data builds -> best LoRA run -> four evaluations (skips finished stages)
  setup.sh, env.sh         Linux venv (uv), model download, GPU selection

deploy/
  Dockerfile, docker-compose.yml, entrypoint.sh, .env.example   vLLM 0.14 + qwen-asr serving, FP8 or bf16, healthcheck
  bench_offline.py         Part 2 baseline: transformers bf16, one clip at a time, RTF per clip-length bucket
  bench.py                 served-API load test: RTF / latency / throughput per concurrency level, GPU sampling
  eval_api.py              transcribes a whole split through the API in eval_wer.py format (then score_malay.py)
  README.md                how to run, call, tune; measured baseline, concurrency table, bottleneck

notebooks/colab_train_malay_babel.ipynb   resume training + evaluate on a Colab A100 from Drive checkpoints
docs/malay.md              full report (datasets, runs, results, quantization, serving, bottleneck, next steps)
docs/figures/              charts used in the report (plot_malay.py)
docs/bao_cao_tong_hop.md   consolidated report in Vietnamese
results/malay/             every run's train log / summary, every test prediction, quantization and serving summaries
requirements.txt           pinned Python deps (torch installed separately)
```

**How the pieces fit.** `data_prep/` turns raw corpora into the prepared format below (one `audio.bin` + `meta.jsonl`
per split under `<data>/`). `train_qwen_asr.py` reads a prepared `train/` and `valid/` and writes
`outputs/<run>/adapter` (plus checkpoints, `train_log.jsonl`, `summary.json`). `eval_wer.py` loads base + adapter and
writes per-clip predictions for any split; `score_malay.py` and `eval_revolab.py` score them. `eval_quant.py` merges
the adapter into `outputs/<run>/merged`, which is what `deploy/` serves; `deploy/bench*.py` and `eval_api.py` measure
the served model. Everything reported lives under `results/`; `outputs/` is the working directory and is not in git.
Python scripts stay at the root on purpose: they import each other (`eval_wer` -> `train_qwen_asr`,
`score_malay` -> `eval_wer`, `deploy/*` -> `eval_wer`) and are run as `python <script>.py` from the root.

## Setup

**Windows / conda (what the local runs used):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128    # match your CUDA
pip install -r requirements.txt
export USE_TF=0 CPU_AFFINITY=0-11 CUDA_MEM_FRACTION=0.8                  # see "Known behaviour"
```

**Linux:** `scripts/setup.sh` (uv venv, qwen-asr, model download), then `source scripts/env.sh`.

**Colab:** open `notebooks/colab_train_malay_babel.ipynb` (bf16 GPU: L4 / A100). It clones this repo, mounts Drive,
copies the prepared data to local disk, resumes training from the checkpoints on Drive and runs all evaluations.

Model weights: `Qwen/Qwen3-ASR-1.7B` from the Hub; scripts default to `models/Qwen3-ASR-1.7B`, pass `--model_path`
(local dir or Hub id). The Revolab dataset is gated: `huggingface-cli login` first (Colab reads an `HF_TOKEN`
secret). `score_malay.py` and `eval_revolab.py` need a clone of
[revolab-asr-benchmark](https://github.com/revolab-sdn-bhd/revolab-asr-benchmark) for its text normalizer: set
`REVOLAB_DIR` to it.

## Data

Every builder writes the same layout, which the trainer and evaluator memmap directly:

```
<data>/<split>/audio.bin    int16 mono 16 kHz samples, all clips concatenated
<data>/<split>/meta.jsonl   one line per clip: {"offset", "length", "text", ...}
```

Splits: `train`, `valid`, `test`, plus `cs_test` (synthetic code-switch) and `babel_test` (BabelSpeech speaker
spk001). Clips are <= 30 s. Keep `audio.bin` on local disk: memmap through a network or Drive mount is far too slow.

| Step | Script | Output used in the report |
|---|---|---|
| 1 | `build_malay_dataset.py` | `malay_prepared`: up to 500h of `mesolitica/malaya-speech-malay-stt`, fixed-seed valid/test |
| 2 | `subset_malay.py` | `malay_50h`: random 50h train subset |
| 3 | `vad_stats.py`, `clean_malay_subset.py` | `malay_50h_clean`: drop clips whose label is truncated (Silero VAD), top up to 50h |
| 4 | `build_malay_mix.py` | `malay_50h_mix`: + Synth-Manglish + cs_pilot code-switch + 5h FLEURS English (`language English`) |
| 5 | `build_malay_babel.py` | `malay_50h_babel`: + BabelSpeech real conversational Malay (12h), spk001 held out as `babel_test` |

Default source paths point at the training PC (`D:/hf_datasets/...`, `D:/data/...`); override with the flags or the
variables at the top of `scripts/run_malay.sh`. External inputs: the malaya-speech parquet shards, Synth-Manglish,
the cs_pilot clips (machine-labelled, train only), FLEURS en_us, BabelSpeech Malay 50h.

## Running

**Train** (auto-resumes if `--output_dir` has a checkpoint; a resume must keep the checkpoint's `--batch_size`):
```bash
python train_qwen_asr.py --mode lora --model_path <model> --data D:/data/malay_50h_babel --lang Malay \
    --output_dir outputs/lora-1.7b-malay50h-babel --epochs 2 --lr 1e-4 --batch_size 4 --grad_acc 4 --eval_steps 200 --save_steps 200
python train_qwen_asr.py --mode lora --init_adapter outputs/<run>/adapter ...   # continue a finished adapter, fresh LR schedule
python tb_sync.py & tensorboard --logdir outputs/tb                              # live curves
```
Outputs: `outputs/<run>/adapter`, `checkpoint-*`, `train_log.jsonl` (every step: loss, LR, grad norm, step time,
GPU memory), `summary.json`, `args.json`.

**Evaluate:**
```bash
python eval_wer.py --model_path <model> --data <data> --lang Malay --runs <run>... --out outputs/eval_malay50h            # in-domain
python eval_wer.py ... --split cs_test    --out outputs/eval_cs_test                                                       # synthetic CS
python eval_wer.py ... --split babel_test --out outputs/eval_babel_test                                                    # real CS
python score_malay.py --dir outputs/eval_babel_test              # WER/CER/sub/del/ins under basic / Malay / Malay+numbers normalization
python eval_revolab.py --model_path <model> --runs <run>...      # Revolab benchmark, per domain
```
`eval_wer.py` writes `<system>_predictions.jsonl` (ref, hyp, per-clip errors) and merges into `summary.json`, so
systems can be added one at a time (`--skip_base`, `--only`).

**Quantize, serve, benchmark:**
```bash
python eval_quant.py --model_path <model> --adapter outputs/lora-1.7b-malay50h-babel/adapter --data <data> --split babel_test --lang Malay --configs bf16 q8 q4
cd deploy && cp .env.example .env && docker compose up -d --build     # vLLM 0.14, OpenAI-compatible API on :8100, FP8 weights
python deploy/bench_offline.py                                        # Part 2 baseline: transformers bf16, one clip at a time
python deploy/bench.py --levels 1 2 4 8 16 32 64 96 128 256           # served API: RTF per clip length and per concurrency level
python deploy/eval_api.py --data <data> --split babel_test --out outputs/eval_api && python score_malay.py --dir outputs/eval_api   # WER through the server
python plot_malay.py                                                  # regenerate every figure in docs/figures/ from results/malay/
```
The first `eval_quant.py` call also writes the merged checkpoint `outputs/<run>/merged` that the container serves.
Quantization on this model: int8 is free (babel_test 8.62 -> 8.65 WER), int4 costs ~1.4 WER, 3-bit collapses.
Serving on an RTX 5060 Ti 16 GB: transformers bf16 one clip at a time runs at RTF 0.39 with the GPU 17% busy; vLLM FP8
runs a single stream at RTF 0.043 and sustains 128 concurrent requests at p95 RTF 0.29 (685x real time, 54 req/s);
p95 RTF passes 0.5 at 256. Tables and the bottleneck analysis: `deploy/README.md`, `docs/malay.md`.

**Whole pipeline:** `scripts/run_malay.sh` runs all five data builds, the best run and the four evaluations; each
stage is skipped when its output exists.

## Where things live

- `results/malay/`: `train_log_*.jsonl`, `train_summary_*.json`, `test_*`, `revolab_*`, `cs_test_*`, `babel_test_*`
  predictions and scores for every system in the report.
- Best model on the Hub (public): adapter [`duclvQ/qwen3-asr-1.7b-malay-lora`](https://huggingface.co/duclvQ/qwen3-asr-1.7b-malay-lora),
  merged checkpoint [`duclvQ/qwen3-asr-1.7b-malay-merged`](https://huggingface.co/duclvQ/qwen3-asr-1.7b-malay-merged)
  (what `deploy/` serves; `MODEL_PATH=duclvQ/qwen3-asr-1.7b-malay-merged` works too). Other runs' adapters are on the
  training PC under `outputs/<run>/adapter` and in Google Drive `My_dataset/outputs/<run>/adapter`; the prepared data
  `malay_50h_babel` (10.6 GB) is in the same Drive folder.
- Not in git: `outputs/` (runs, evals, queue scripts), `models/`, `data/`, `.venv/`, `.env` (`HF_TOKEN`).

## Known behaviour and gotchas

- **Trainer is CPU-bound** at batch 4-8 (collator + dataloader). On Windows hybrid CPUs set `CPU_AFFINITY=0-11` to
  stay on P-cores (2x throughput); a Colab A100 was no faster than an RTX 5060 Ti for the same reason. Use a larger
  batch and more `--workers` on big GPUs (a resumed run must keep its batch size, so start a new run).
- `CUDA_MEM_FRACTION=0.8` on Windows prevents the allocator from silently spilling into shared system RAM.
- **Resume** restores the checkpoint's batch size; `train_qwen_asr.py` refuses a different `--batch_size`. Checkpoints
  written under numpy 1.x resume under numpy 2.x / torch >= 2.6 (the RNG file's numpy globals are allowlisted).
- Colab ships torchao 0.10, which breaks peft's LoRA injection: uninstall it (the notebook does).
- HF eval loss depends on eval batch size; compare runs by WER.
- Set `USE_TF=0` if TensorFlow is installed, or transformers' Trainer import breaks.
- One seed per run; differences of 1-2 WER on a 50-clip Revolab domain are within noise.

Tested with: Windows torch 2.8.0+cu128 / transformers 4.57.6 / peft 0.18.1 / numpy 1.26.4 / Python 3.13;
Colab torch 2.x / numpy 2.x; qwen-asr 0.0.6 everywhere.
