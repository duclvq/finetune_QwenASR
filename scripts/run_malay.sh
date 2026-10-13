#!/usr/bin/env bash
# Malay pipeline: data -> LoRA training -> evaluation, with the paths and settings used for the reported runs.
# Resumable: every stage is skipped when its output exists, so rerun after an interruption.
#
#   scripts/run_malay.sh            # everything up to the best system (lora-1.7b-malay50h-babel)
#   STAGE=eval scripts/run_malay.sh # only the evaluations of the runs that exist
#
# Inputs you must have (see README "Malay track"):
#   MALAY_SRC   local copy of mesolitica/malaya-speech-malay-stt (parquet shards)
#   CS_PILOT    200 machine-labelled code-switch clips (train only)      [build_malay_mix.py]
#   FLEURS_EN   FLEURS en_us parquet glob                                [build_malay_mix.py]
#   BABEL       BabelSpeech Malay 50h (audio_info.json + wavs)           [build_malay_babel.py]
#   REVOLAB_DIR clone of https://github.com/revolab-sdn-bhd/revolab-asr-benchmark (text normalizer)
#   HF login for the gated dataset Revolab/ASR-Benchmark-Public (huggingface-cli login)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-models/Qwen3-ASR-1.7B}          # local dir or HF id Qwen/Qwen3-ASR-1.7B
DATA_ROOT=${DATA_ROOT:-D:/data}
MALAY_SRC=${MALAY_SRC:-D:/hf_datasets/malaya-speech-malay-stt/data}
CS_PILOT=${CS_PILOT:-D:/cs_mining/cs_pilot/train}
FLEURS_EN=${FLEURS_EN:-D:/data/fleurs_en/parquet-data/en_us/train-*.parquet}
BABEL=${BABEL:-D:/hf_datasets/babelspeech_malay50h}
export REVOLAB_DIR=${REVOLAB_DIR:-E:/revolab-asr-benchmark}
export USE_TF=0 PYTHONUTF8=1
# Windows + hybrid Intel CPU: keep the CPU-bound trainer on P-cores and cap CUDA memory (see docs/malay.md)
export CPU_AFFINITY=${CPU_AFFINITY:-} CUDA_MEM_FRACTION=${CUDA_MEM_FRACTION:-}

STAGE=${STAGE:-all}
TRAIN="--mode lora --model_path $MODEL --lang Malay --epochs 2 --lr 1e-4 --batch_size 4 --grad_acc 4 --eval_steps 200 --save_steps 200"

if [ "$STAGE" = all ]; then
  echo "== data"
  [ -d $DATA_ROOT/malay_prepared/train ]  || python data_prep/build_malay_dataset.py --src "$MALAY_SRC" --out $DATA_ROOT/malay_prepared --max_hours 500
  [ -d $DATA_ROOT/malay_50h/train ]       || python data_prep/subset_malay.py --src $DATA_ROOT/malay_prepared --out $DATA_ROOT/malay_50h --hours 50
  [ -f $DATA_ROOT/malay_50h/train/vad.jsonl ] || python data_prep/vad_stats.py $DATA_ROOT/malay_50h test train
  [ -d $DATA_ROOT/malay_50h_clean/train ] || python data_prep/clean_malay_subset.py --src $DATA_ROOT/malay_50h --pool $DATA_ROOT/malay_prepared --out $DATA_ROOT/malay_50h_clean
  [ -d $DATA_ROOT/malay_50h_mix/train ]   || python data_prep/build_malay_mix.py --base $DATA_ROOT/malay_50h_clean --out $DATA_ROOT/malay_50h_mix --cs_pilot "$CS_PILOT" --fleurs_en "$FLEURS_EN"
  [ -d $DATA_ROOT/malay_50h_babel/train ] || python data_prep/build_malay_babel.py --base $DATA_ROOT/malay_50h_mix --out $DATA_ROOT/malay_50h_babel --babel "$BABEL"

  echo "== train (best system; the earlier ablations used malay_50h, malay_50h_clean, malay_50h_mix with the same flags)"
  R=lora-1.7b-malay50h-babel
  [ -f outputs/$R/summary.json ] || python train_qwen_asr.py $TRAIN --data $DATA_ROOT/malay_50h_babel --output_dir outputs/$R
fi

echo "== eval: every outputs/lora-1.7b-malay50h*/adapter that exists, plus base"
RUNS=$(ls -d outputs/lora-1.7b-malay50h*/adapter 2>/dev/null | xargs -n1 dirname | xargs -n1 basename | tr '\n' ' ')
D=$DATA_ROOT/malay_50h_babel
python eval_wer.py     --model_path "$MODEL" --data $D --lang Malay --runs $RUNS --out outputs/eval_malay50h
python score_malay.py  --dir outputs/eval_malay50h
python eval_revolab.py --model_path "$MODEL" --runs $RUNS --out outputs/eval_revolab
python eval_wer.py     --model_path "$MODEL" --data $D --split cs_test    --lang Malay --runs $RUNS --out outputs/eval_cs_test
python eval_wer.py     --model_path "$MODEL" --data $D --split babel_test --lang Malay --runs $RUNS --out outputs/eval_babel_test
python score_malay.py  --dir outputs/eval_babel_test
echo MALAY_DONE
