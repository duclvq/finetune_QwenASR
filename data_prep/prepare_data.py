"""Convert ViMD parquet shards into 16kHz mono int16 audio (helper used by build_vimd_dataset.py).

For each split writes OUT/<split>/audio.bin (concatenated int16 samples) and
OUT/<split>/meta.jsonl (offset, length, text, region, province, speaker).
Clips longer than --max_sec are dropped (Qwen3-ASR training clips were kept <= 30s).

Usage: python prepare_data.py [--src data/vimd_nc] [--out data/prepared]
"""
import argparse
import glob
import io
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

SR = 16000


def clean_text(text):
    text = re.sub(r"\s*([,.;:!?])\s*", r"\1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def process_shard(path, max_sec):
    table = pq.read_table(path)
    rows = []
    for rec in table.to_pylist():
        audio, sr = sf.read(io.BytesIO(rec["audio"]["bytes"]), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
        if len(audio) > max_sec * SR or not rec["text"].strip():
            continue
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        meta = {
            "text": clean_text(rec["text"]),
            "region": rec["region"],
            "province": rec["province_name"],
            "speaker": rec["speakerID"],
            "filename": rec["filename"],
        }
        rows.append((pcm, meta))
    return rows, table.num_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/vimd_nc")
    ap.add_argument("--out", default="data/prepared")
    ap.add_argument("--max_sec", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    for split in ["train", "valid", "test"]:
        out_dir = os.path.join(args.out, split)
        if os.path.exists(os.path.join(out_dir, "meta.jsonl")):
            print(f"{split}: already prepared, skipping")
            continue
        os.makedirs(out_dir, exist_ok=True)
        shards = sorted(glob.glob(os.path.join(args.src, split, "*.parquet")))
        offset, kept, total = 0, 0, 0
        with open(os.path.join(out_dir, "audio.bin.part"), "wb") as fa, \
                open(os.path.join(out_dir, "meta.jsonl.part"), "w", encoding="utf-8") as fm, \
                ProcessPoolExecutor(args.workers) as ex:
            for i, (rows, n) in enumerate(ex.map(process_shard, shards, [args.max_sec] * len(shards)), 1):
                total += n
                for pcm, meta in rows:
                    fa.write(pcm.tobytes())
                    meta.update(offset=offset, length=len(pcm))
                    fm.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    offset += len(pcm)
                    kept += 1
                print(f"{split} [{i}/{len(shards)}] kept {kept}/{total}", flush=True)
        os.replace(os.path.join(out_dir, "audio.bin.part"), os.path.join(out_dir, "audio.bin"))
        os.replace(os.path.join(out_dir, "meta.jsonl.part"), os.path.join(out_dir, "meta.jsonl"))
        print(f"{split}: {kept}/{total} clips, {offset / SR / 3600:.1f} hours")


if __name__ == "__main__":
    main()
