# Fine-tuning Qwen3-ASR-1.7B: Vietnamese and Malay

Code, data pipelines, results and serving for fine-tuning [Qwen3-ASR-1.7B](https://github.com/QwenLM/Qwen3-ASR)
with LoRA / QLoRA / full fine-tuning on two languages, plus weight-only quantization, vLLM serving and a realtime
(WebSocket + VAD) gateway. One trainer and one evaluator serve both tracks; everything else is data preparation,
scoring and reports.

| Track | Best system | Key result | Report |
|---|---|---|---|
| **Vietnamese** (ViMD, North + Central dialects) | full fine-tune, LoRA within noise | test WER 11.56 -> **8.59** (LoRA 8.72) | [docs/vietnamese.md](docs/vietnamese.md) |
| **Malay** + Malay-English code-switch | `lora-1.7b-malay50h-babel` | Revolab benchmark 15.01 -> **13.39**; real code-switch 17.9 -> **7.8** | [docs/malay.md](docs/malay.md) |

Public models (Vietnamese): [`duclvQ/qwen3-asr-1.7b-vimd-lora`](https://huggingface.co/duclvQ/qwen3-asr-1.7b-vimd-lora),
[`duclvQ/qwen3-asr-1.7b-vimd-qlora`](https://huggingface.co/duclvQ/qwen3-asr-1.7b-vimd-qlora),
[`duclvQ/qwen3-asr-1.7b-vimd-full`](https://huggingface.co/duclvQ/qwen3-asr-1.7b-vimd-full) (+ `q8/`, `q4/` HQQ).
Malay adapters are not published; see "Where things live" below.

## Repository layout

```
train_qwen_asr.py      LoRA / QLoRA / full trainer (--mode), per-step log, auto-resume from the last checkpoint
eval_wer.py            WER/CER of base + adapters + full models on a prepared split (shared by both tracks)
score_malay.py         Malay scoring (Revolab text normalizer + number spelling) of eval_wer.py predictions
eval_revolab.py        Revolab/ASR-Benchmark-Public (820 Malay clips, 12 domains)
vi_numbers.py          Vietnamese digit -> spoken-form normalization;  rescore_numbers.py re-scores with it
eval_quant.py          HQQ weight-only quantization sweep;  load_hqq_asr.py loads the q8/q4 checkpoints
tb_sync.py             mirrors outputs/*/train_log.jsonl into TensorBoard, live
smoke_test.py          loads the base model on the transformers or vllm backend and transcribes one clip
data_prep/             dataset builders -> the prepared format below (ViMD; Malay: build, subset, VAD, clean, mix, babel)
realtime/              WebSocket realtime gateway (Silero VAD) in front of vLLM + streaming / load-test clients
scripts/               setup.sh, env.sh (Linux venv + GPU), run_all.sh / run_full.sh (Vietnamese queues),
                       run_malay.sh (Malay data -> train -> eval), serve_vllm.sh
notebooks/             colab_train_malay_babel.ipynb: resume + evaluate a Malay run on a Colab A100 from Drive
docs/                  vietnamese.md, malay.md: full reports (tables, analysis, infrastructure notes, next steps)
results/               every run's args / summary / per-step train log and every test prediction (see below)
requirements.txt       pinned Python deps (torch installed separately)
```

Python scripts stay at the repository root on purpose: they import each other (`eval_wer` -> `train_qwen_asr`,
`score_malay` -> `eval_wer`, ...) and are run as `python <script>.py` from the root.

## Setup

**Linux (GPU box, everything incl. vLLM):**
```bash
scripts/setup.sh              # clones third_party/Qwen3-ASR, creates .venv with uv, installs qwen-asr[vllm]==0.0.6, downloads the model
source scripts/env.sh         # activates .venv and selects the GPU (edit CUDA_VISIBLE_DEVICES)
pip install bitsandbytes hqq silero-vad     # QLoRA, quantization, VAD, as needed
```

**Windows / conda (training + eval, no vLLM):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128    # match your CUDA
pip install -r requirements.txt
export USE_TF=0 CPU_AFFINITY=0-11 CUDA_MEM_FRACTION=0.8                  # see "Known behaviour"
```

**Colab:** open `notebooks/colab_train_malay_babel.ipynb` (needs a bf16 GPU: L4 / A100). It clones this repo, mounts
Drive, copies the prepared data to local disk and resumes training from the checkpoints kept on Drive.

Model weights: `Qwen/Qwen3-ASR-1.7B` from the Hub. Scripts default to `models/Qwen3-ASR-1.7B`; pass `--model_path`
(a local dir or the Hub id). The Revolab benchmark dataset is gated: `huggingface-cli login` first (the Colab notebook
reads an `HF_TOKEN` secret). `score_malay.py` and `eval_revolab.py` need a clone of
[revolab-asr-benchmark](https://github.com/revolab-sdn-bhd/revolab-asr-benchmark) for its text normalizer:
set `REVOLAB_DIR` to it.

## Data format

Every builder in `data_prep/` writes the same layout, which the trainer and evaluator memmap directly:

```
<data>/<split>/audio.bin    int16 mono 16 kHz samples, all clips concatenated
<data>/<split>/meta.jsonl   one line per clip: {"offset", "length", "text", ...}  (region/speaker/filename when known)
```

Splits are `train`, `valid`, `test`, plus `cs_test` and `babel_test` in the Malay builds. Clips are <= 30 s.
Reading `audio.bin` through a network or FUSE mount (Google Drive) is far too slow; keep it on local disk.

| Builder | Produces |
|---|---|
| `build_vimd_dataset.py` (+ `prepare_data.py`) | ViMD North + Central, 55h train / 7h valid / 7h test |
| `build_malay_dataset.py` | up to N hours of `mesolitica/malaya-speech-malay-stt`, fixed-seed valid/test |
| `subset_malay.py`, `vad_stats.py`, `clean_malay_subset.py` | 50h random subset; Silero VAD speech time per clip; drop truncated-label clips and top up |
| `build_malay_mix.py` | + Synth-Manglish code-switch + cs_pilot + 5h FLEURS English (tagged `language English`) |
| `build_malay_babel.py` | + BabelSpeech real conversational Malay, whole speakers held out as `babel_test` |

Default source paths in the Malay builders point at the training PC (`D:/hf_datasets/...`, `D:/data/...`); override
them with the flags, or set the variables at the top of `scripts/run_malay.sh`.

## Running

**Train** (auto-resumes if `--output_dir` has a checkpoint; a resume must keep the checkpoint's `--batch_size`):
```bash
python train_qwen_asr.py --mode lora  --model_path <model> --data <data> --lang Malay --output_dir outputs/<run> \
    --epochs 2 --lr 1e-4 --batch_size 4 --grad_acc 4 --eval_steps 200 --save_steps 200
python train_qwen_asr.py --mode qlora ...          # NF4 base, needs bitsandbytes
python train_qwen_asr.py --mode full  --lr 1e-5 ... # full fine-tune, saves outputs/<run>/final
python train_qwen_asr.py --mode lora --init_adapter outputs/<run>/adapter ...   # continue a finished adapter with a fresh LR schedule
python tb_sync.py & tensorboard --logdir outputs/tb                              # live curves
```
Outputs: `outputs/<run>/adapter` (or `final`), `checkpoint-*`, `train_log.jsonl` (every step: loss, LR, grad norm,
step time, GPU memory), `summary.json` (wall time, peak memory, final eval loss), `args.json`.

**Evaluate:**
```bash
python eval_wer.py --model_path <model> --data <data> [--split test|cs_test|babel_test] --lang Malay --runs <run>... --out outputs/eval_x
python score_malay.py --dir outputs/eval_x                       # Malay: WER/CER/sub/del/ins under 3 normalizations
python eval_revolab.py --model_path <model> --runs <run>...      # Malay: Revolab benchmark, per domain
python rescore_numbers.py                                        # Vietnamese: digits -> spoken numbers, re-score
python eval_quant.py --configs bf16 q8 q6 q4 --max_samples 400   # Vietnamese full model, HQQ sweep
```
`eval_wer.py` writes `<system>_predictions.jsonl` (ref, hyp, per-clip errors) and merges into `summary.json`, so
systems can be added one at a time (`--skip_base`, `--only`).

**Whole pipelines:** `scripts/run_all.sh` + `scripts/run_full.sh` (Vietnamese: LoRA, QLoRA, full, eval) and
`scripts/run_malay.sh` (Malay: all data builds, the best run, all four evaluations). Each stage is skipped when its
output exists.

**Serve and stream (Vietnamese full model):**
```bash
scripts/serve_vllm.sh                                               # OpenAI-compatible API on 127.0.0.1:8100
uvicorn gateway:app --app-dir realtime --port 8200 --workers 6      # realtime WebSocket gateway (Silero VAD)
python realtime/test_stream.py --clips 12                           # streams test clips at 1x, reports latency + WER
python realtime/load_test.py --users 16 32 56 --seconds 60          # simulated concurrent users
```
Model outputs start with `language Vietnamese<asr_text>`; strip `^\s*language\s+\S+?<asr_text>` client-side.
Details, capacity numbers and limitations: [docs/vietnamese.md](docs/vietnamese.md).

## Results summary

Vietnamese, ViMD test (1,368 clips, 939 unseen speakers), WER:

| zero-shot | LoRA r=16 | QLoRA | full FT |
|---|---|---|---|
| 11.56 | 8.72 | 9.45 | **8.59** |

About two thirds of that gain is recognition; the rest is label format (numbers spelled out, fillers). HQQ q8 is
free, q4 costs ~1.6 WER, q3/q2 collapse without calibration. vLLM serves ~16 req/s on a 24 GB GPU; the realtime
gateway keeps p95 final-transcript latency under 3 s up to ~28 concurrent users.

Malay, WER:

| System | in-domain test (1,994) | Revolab (820, 12 domains) | synthetic code-switch (221) | real code-switch, unseen speaker (377) |
|---|---|---|---|---|
| base | 52.19 | 15.01 | 25.69 | 17.94 |
| lora-mix (63h) | 37.35 | **13.38** | 8.69 | 14.76 |
| **lora-babel (83h)** | **37.03** | 13.39 | 8.69 | **7.83** |

The Malay-only LoRA (not shown) forgets English and loses to base out of domain; mixing in code-switched and English
speech fixes that. About a third of lora-babel's gain on the real code-switch set is label-convention matching
(spelling variants, spelled-out numbers); the rest is recognition. One test speaker only. Full tables, ablations and
caveats: [docs/malay.md](docs/malay.md).

## Where things live

- `results/<run>/` (Vietnamese) and `results/malay/` (Malay): `args.json`, `summary.json`, full per-step
  `train_log*.jsonl`, every test prediction (`*_predictions.jsonl`) and scores. These are the source for every number
  in the reports.
- Adapters / full models: Vietnamese on the Hub (links above). Malay adapters (`lora-1.7b-malay50h-mix`,
  `lora-1.7b-malay50h-babel`) are on the training PC under `outputs/<run>/adapter` and in Google Drive
  `My_dataset/outputs/<run>/adapter`; the prepared Malay data (`malay_50h_babel`, 10.6 GB) is in the same Drive folder.
- Working dirs that are not in git: `outputs/` (runs, evals, queue scripts), `models/`, `data/`, `.venv/`, `.env`
  (`HF_TOKEN`), `third_party/Qwen3-ASR` (upstream reference).

## Known behaviour and gotchas

- **Trainer is CPU-bound** at batch 4-8 (collator + dataloader). On Windows hybrid CPUs set `CPU_AFFINITY=0-11` to
  stay on P-cores (2x throughput); a Colab A100 was no faster than an RTX 5060 Ti for the same reason. Use a larger
  batch and more `--workers` on big GPUs.
- `CUDA_MEM_FRACTION=0.8` on Windows prevents the allocator from silently spilling into shared system RAM.
- **Resume** restores the checkpoint's batch size; `train_qwen_asr.py` refuses a different `--batch_size`. Checkpoints
  written under numpy 1.x resume under numpy 2.x / torch >= 2.6 (the RNG file's numpy globals are allowlisted).
- Colab ships torchao 0.10, which breaks peft's LoRA injection: uninstall it (the notebook does).
- HF eval loss depends on eval batch size; compare runs by WER.
- Set `USE_TF=0` if TensorFlow is installed in the environment, or transformers' Trainer import breaks.
- Full-FT checkpoints save weights only (`save_only_model`), so a resumed full run restarts the optimizer.
- Differences below ~0.5 WER (Vietnamese) or 1-2 WER on a 50-clip Revolab domain are within noise; one seed per run.

Tested with: Linux torch 2.9.1 / transformers 4.57.6 / peft 0.21.2 / vllm 0.14.0; Windows torch 2.8.0+cu128 /
transformers 4.57.6 / peft 0.18.1 / Python 3.13; qwen-asr 0.0.6 everywhere.
