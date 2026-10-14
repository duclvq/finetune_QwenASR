"""Transcribe a whole prepared split through the served API and write eval_wer.py-format predictions,
so score_malay.py can score the deployed (FP8) model exactly like the offline bf16 one.

    python deploy/eval_api.py --data D:/data/malay_50h_babel --split babel_test --name vllm-fp8 --out outputs/eval_api
    python score_malay.py --dir outputs/eval_api
"""
import argparse
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval_wer import score  # noqa: E402

PREFIX = re.compile(r"^\s*language\s+\S+?<asr_text>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100/v1/audio/transcriptions")
    ap.add_argument("--model", default="qwen3-asr-malay")
    ap.add_argument("--language", default="ms")
    ap.add_argument("--data", default="D:/data/malay_50h_babel")
    ap.add_argument("--split", default="babel_test")
    ap.add_argument("--name", default="vllm-fp8")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=400,
                    help="vLLM's transcription endpoint otherwise decodes up to max-model-len; a clip that falls into "
                         "a repetition loop then emits ~1,200 words. 400 matches the offline evaluation.")
    ap.add_argument("--out", default="outputs/eval_api")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta = [json.loads(l) for l in open(os.path.join(args.data, args.split, "meta.jsonl"), encoding="utf-8")]
    audio = np.memmap(os.path.join(args.data, args.split, "audio.bin"), dtype=np.int16, mode="r")

    def one(m):
        buf = io.BytesIO()
        sf.write(buf, audio[m["offset"]:m["offset"] + m["length"]], 16000, format="WAV")
        r = requests.post(args.url, files={"file": ("clip.wav", buf.getvalue(), "audio/wav")},
                          data={"model": args.model, "language": args.language, "max_completion_tokens": args.max_tokens}, timeout=600)
        r.raise_for_status()
        return PREFIX.sub("", r.json().get("text", "")).strip()

    t0 = time.time()
    with ThreadPoolExecutor(args.concurrency) as ex:
        hyps = list(ex.map(one, meta))
    secs = time.time() - t0
    with open(os.path.join(args.out, f"{args.name}_predictions.jsonl"), "w", encoding="utf-8") as f:
        for m, h in zip(meta, hyps):
            f.write(json.dumps({"filename": m.get("filename", m.get("source")), "region": m.get("region"), "ref": m["text"], "hyp": h,
                                **score([m["text"]], [h])}, ensure_ascii=False) + "\n")
    res = {"infer_seconds": round(secs, 1), "All": {"clips": len(meta), **score([m["text"] for m in meta], hyps)}}
    path = os.path.join(args.out, "summary.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old[args.name] = res
    json.dump(old, open(path, "w"), indent=2)
    print(args.name, json.dumps(res))


if __name__ == "__main__":
    main()
