"""Load-test the served /v1/audio/transcriptions endpoint: RTF per clip-length bucket and per concurrency level.

    python deploy/bench.py --data D:/data/malay_50h_babel --split babel_test --levels 1 2 4 8 16 32

Picks clips from three duration buckets (2-5 s, 5-15 s, 15-30 s), writes them as wav, then for each concurrency
level sends `requests_per_level` requests from a thread pool and reports avg / p50 / p95 RTF
(RTF = request wall time / audio seconds), throughput (audio seconds processed per wall second) and GPU
utilization / memory sampled with nvidia-smi during the run. Results go to --out as JSON and a markdown table.
"""
import argparse
import io
import json
import os
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import soundfile as sf

BUCKETS = [("2-5 s", 2, 5), ("5-15 s", 5, 15), ("15-30 s", 15, 30)]


def pick_clips(data, split, per_bucket, seed=7):
    meta = [json.loads(l) for l in open(os.path.join(data, split, "meta.jsonl"), encoding="utf-8")]
    audio = np.memmap(os.path.join(data, split, "audio.bin"), dtype=np.int16, mode="r")
    rng = np.random.default_rng(seed)
    clips = []
    for name, lo, hi in BUCKETS:
        idx = [i for i, m in enumerate(meta) if lo <= m["length"] / 16000 < hi]
        for i in rng.choice(idx, min(per_bucket, len(idx)), replace=False):
            m = meta[i]
            buf = io.BytesIO()
            sf.write(buf, audio[m["offset"]:m["offset"] + m["length"]], 16000, format="WAV")
            clips.append({"bucket": name, "seconds": m["length"] / 16000, "wav": buf.getvalue(), "ref": m["text"]})
    return clips


class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                                               "--format=csv,noheader,nounits"], text=True)
                u, m, p = [float(x) for x in out.strip().split(",")]
                self.samples.append((u, m, p))
            except Exception:
                pass
            self.stop.wait(0.5)

    def summary(self):
        if not self.samples:
            return {}
        u, m, p = zip(*self.samples)
        return {"gpu_util_mean": round(statistics.mean(u)), "gpu_util_p95": round(np.percentile(u, 95)),
                "gpu_mem_gb": round(max(m) / 1024, 2), "gpu_power_w": round(statistics.mean(p))}


def one(url, model, lang, clip, max_tokens=400):
    t0 = time.perf_counter()
    r = requests.post(url, files={"file": ("clip.wav", clip["wav"], "audio/wav")},
                      data={"model": model, "language": lang, "max_completion_tokens": max_tokens}, timeout=600)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return {"bucket": clip["bucket"], "seconds": clip["seconds"], "wall": dt, "rtf": dt / clip["seconds"],
            "text": r.json().get("text", "")}


def pctl(v, p):
    return round(float(np.percentile(v, p)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100/v1/audio/transcriptions")
    ap.add_argument("--model", default="qwen3-asr-malay")
    ap.add_argument("--language", default="ms")
    ap.add_argument("--data", default="D:/data/malay_50h_babel")
    ap.add_argument("--split", default="babel_test")
    ap.add_argument("--per_bucket", type=int, default=8)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--requests_per_level", type=int, default=48)
    ap.add_argument("--out", default="outputs/bench_api")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    clips = pick_clips(args.data, args.split, args.per_bucket)
    print(f"{len(clips)} clips, {sum(c['seconds'] for c in clips):.0f} s of audio", flush=True)
    one(args.url, args.model, args.language, clips[0])  # warm-up

    results = {"levels": [], "buckets": {}}
    for conc in args.levels:
        reqs = [clips[i % len(clips)] for i in range(max(args.requests_per_level, conc))]
        g = GpuSampler(); g.start()
        t0 = time.perf_counter()
        with ThreadPoolExecutor(conc) as ex:
            rows = list(ex.map(lambda c: one(args.url, args.model, args.language, c), reqs))
        wall = time.perf_counter() - t0
        g.stop.set(); g.join()
        rtf = [r["rtf"] for r in rows]
        audio_s = sum(r["seconds"] for r in rows)
        lvl = {"concurrency": conc, "requests": len(rows), "audio_s": round(audio_s), "wall_s": round(wall, 2),
               "rtf_avg": round(statistics.mean(rtf), 3), "rtf_p50": pctl(rtf, 50), "rtf_p95": pctl(rtf, 95),
               "latency_p50_s": pctl([r["wall"] for r in rows], 50), "latency_p95_s": pctl([r["wall"] for r in rows], 95),
               "throughput_x_realtime": round(audio_s / wall, 1), "req_per_s": round(len(rows) / wall, 2), **g.summary()}
        results["levels"].append(lvl)
        print(json.dumps(lvl), flush=True)
        if conc == 1:
            for name, _, _ in BUCKETS:
                b = [r for r in rows if r["bucket"] == name]
                results["buckets"][name] = {"clips": len(b), "rtf_avg": round(statistics.mean(r["rtf"] for r in b), 3),
                                            "rtf_p50": pctl([r["rtf"] for r in b], 50), "rtf_p95": pctl([r["rtf"] for r in b], 95),
                                            "latency_p50_s": pctl([r["wall"] for r in b], 50)}
    json.dump(results, open(os.path.join(args.out, "bench.json"), "w"), indent=2)

    md = ["| Clip length | n | avg RTF | p50 RTF | p95 RTF | p50 latency |", "|---|---|---|---|---|---|"]
    for name, b in results["buckets"].items():
        md.append(f"| {name} | {b['clips']} | {b['rtf_avg']} | {b['rtf_p50']} | {b['rtf_p95']} | {b['latency_p50_s']} s |")
    md += ["", "| Concurrent | avg RTF | p50 RTF | p95 RTF | p95 latency | throughput (x real time) | req/s | GPU util | VRAM | power |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for l in results["levels"]:
        md.append(f"| {l['concurrency']} | {l['rtf_avg']} | {l['rtf_p50']} | {l['rtf_p95']} | {l['latency_p95_s']} s | {l['throughput_x_realtime']} | "
                  f"{l['req_per_s']} | {l.get('gpu_util_mean', '')}% | {l.get('gpu_mem_gb', '')} GB | {l.get('gpu_power_w', '')} W |")
    open(os.path.join(args.out, "bench.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
