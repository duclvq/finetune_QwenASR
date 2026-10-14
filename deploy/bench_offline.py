"""Baseline for Part 2: transformers backend, bf16, one clip at a time (no batching, no server).

    python deploy/bench_offline.py --model_path outputs/lora-1.7b-malay50h-babel/merged [--data ... --split babel_test]

Uses the same clip selection as bench.py (three duration buckets) and reports avg / p50 / p95 RTF per bucket and
overall, GPU memory and utilization. Writes --out/offline.json and offline.md.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import BUCKETS, GpuSampler, pick_clips, pctl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="outputs/lora-1.7b-malay50h-babel/merged")
    ap.add_argument("--data", default="D:/data/malay_50h_babel")
    ap.add_argument("--split", default="babel_test")
    ap.add_argument("--lang", default="Malay")
    ap.add_argument("--per_bucket", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default="outputs/bench_api")
    args = ap.parse_args()
    from qwen_asr import Qwen3ASRModel
    clips = pick_clips(args.data, args.split, args.per_bucket)
    w = Qwen3ASRModel.from_pretrained(args.model_path, dtype=torch.bfloat16, device_map={"": 0},
                                      max_inference_batch_size=1, max_new_tokens=400)
    w.model.eval()
    mem_load = torch.cuda.memory_allocated() / 2**30
    audio = np.memmap(os.path.join(args.data, args.split, "audio.bin"), dtype=np.int16, mode="r")  # noqa: F841
    import soundfile as sf, io
    wavs = [sf.read(io.BytesIO(c["wav"]), dtype="float32")[0] for c in clips]
    w.transcribe(audio=[(wavs[0], 16000)], language=args.lang)  # warm-up
    g = GpuSampler(); g.start()
    rows = []
    for _ in range(args.repeats):
        for c, x in zip(clips, wavs):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            w.transcribe(audio=[(x, 16000)], language=args.lang)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            rows.append({"bucket": c["bucket"], "seconds": c["seconds"], "wall": dt, "rtf": dt / c["seconds"]})
    g.stop.set(); g.join()
    res = {"model": args.model_path, "backend": "transformers bf16, batch 1", "gpu_gb_after_load": round(mem_load, 2),
           "gpu_gb_peak": round(torch.cuda.max_memory_allocated() / 2**30, 2), **g.summary(), "buckets": {}}
    for name, _, _ in BUCKETS + [("all", 0, 1e9)]:
        b = [r for r in rows if name == "all" or r["bucket"] == name]
        res["buckets"][name] = {"n": len(b), "rtf_avg": round(statistics.mean(r["rtf"] for r in b), 3),
                                "rtf_p50": pctl([r["rtf"] for r in b], 50), "rtf_p95": pctl([r["rtf"] for r in b], 95),
                                "latency_p50_s": pctl([r["wall"] for r in b], 50),
                                "throughput_x_realtime": round(sum(r["seconds"] for r in b) / sum(r["wall"] for r in b), 1)}
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, "offline.json"), "w"), indent=2)
    md = [f"Transformers bf16, batch 1, GPU after load {res['gpu_gb_after_load']} GB, peak {res['gpu_gb_peak']} GB, "
          f"util {res.get('gpu_util_mean', '?')}%, {res.get('gpu_power_w', '?')} W", "",
          "| Clip length | n | avg RTF | p50 RTF | p95 RTF | p50 latency | throughput (x real time) |", "|---|---|---|---|---|---|---|"]
    for name, b in res["buckets"].items():
        md.append(f"| {name} | {b['n']} | {b['rtf_avg']} | {b['rtf_p50']} | {b['rtf_p95']} | {b['latency_p50_s']} s | {b['throughput_x_realtime']} |")
    open(os.path.join(args.out, "offline.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
