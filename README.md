# Fine-tuning Qwen3-ASR-1.7B for Malay and Malay-English code-switch

LoRA fine-tuning of [Qwen3-ASR-1.7B](https://github.com/QwenLM/Qwen3-ASR) on Malay, with the data pipelines,
evaluation (in-domain, Revolab benchmark, synthetic and real code-switch), results and a Colab notebook for training
without a local GPU. Branches:

| Branch | Content |
|---|---|
| `main` | Malay track (this README, [docs/malay.md](docs/malay.md)) |
| `vi` | Vietnamese track: ViMD LoRA / QLoRA / full fine-tune, HQQ quantization, vLLM serving, realtime WebSocket gateway |

The trainer and evaluator are shared; `vi` has the older copies plus the Vietnamese-only tools.

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
train_qwen_asr.py      LoRA / QLoRA / full trainer (--mode), per-step log, auto-resume from the last checkpoint
eval_wer.py            WER/CER of base + adapters on a prepared split; writes per-clip predictions
score_malay.py         Malay scoring (Revolab text normalizer + number spelling) of eval_wer.py predictions
eval_revolab.py        Revolab/ASR-Benchmark-Public (820 clips, 12 domains), per-domain WER
tb_sync.py             mirrors outputs/*/train_log.jsonl into TensorBoard, live
smoke_test.py          loads the base model and transcribes one clip (transformers or vllm backend)
data_prep/             dataset builders: build, 50h subset, VAD stats, label cleaning, code-switch/English mix, BabelSpeech
scripts/               run_malay.sh (data -> train -> eval chain), setup.sh + env.sh (Linux venv + GPU)
notebooks/             colab_train_malay_babel.ipynb: resume training and evaluate on a Colab A100 from Drive
docs/malay.md          full report
results/malay/         every run's train log / summary and every test prediction (source of all numbers)
requirements.txt       pinned Python deps (torch installed separately)
```

Python scripts stay at the root on purpose: they import each other (`eval_wer` -> `train_qwen_asr`,
`score_malay` -> `eval_wer`) and are run as `python <script>.py` from the root.

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

**Whole pipeline:** `scripts/run_malay.sh` runs all five data builds, the best run and the four evaluations; each
stage is skipped when its output exists.

## Where things live

- `results/malay/`: `train_log_*.jsonl`, `train_summary_*.json`, `test_*`, `revolab_*`, `cs_test_*`, `babel_test_*`
  predictions and scores for every system in the report.
- Adapters are not in git or on the Hub: `lora-1.7b-malay50h-mix` and `lora-1.7b-malay50h-babel` are on the training
  PC under `outputs/<run>/adapter` and in Google Drive `My_dataset/outputs/<run>/adapter`. The prepared data
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
