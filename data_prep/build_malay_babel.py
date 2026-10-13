"""Add BabelSpeech real code-switched Malay speech to the mix set.

BabelSpeech/50hours_Malay_Real-world_Colloquial_Conversation_and_Monologue_Speech_Dataset is ~35h of
human-checked YouTube speech (Malay with natural English code-switching) from only 10 speakers, one of which
(spk005, tech reviews) has ~23h. OUT = <base train> + BabelSpeech, with spk005-like dominance capped at
--max_spk_hours per speaker. Whole speakers in --test_speakers are held out as OUT/babel_test, a real
code-switch test set the model never hears. Transcripts are lowercased and stripped of punctuation to match
the Malay labels. valid/test/cs_test are copied from the base set unchanged.

Usage: python build_malay_babel.py --base D:/data/malay_50h_mix --out D:/data/malay_50h_babel
           --babel D:/hf_datasets/babelspeech_malay50h
"""
import argparse
import io
import json
import os
import re
import shutil
import zipfile

import numpy as np

from build_malay_mix import SR, clean_text, to_pcm, write_split


def speaker(filename):
    return re.search(r"spk\d+", filename).group(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="D:/data/malay_50h_mix")
    ap.add_argument("--out", default="D:/data/malay_50h_babel")
    ap.add_argument("--babel", default="D:/hf_datasets/babelspeech_malay50h")
    ap.add_argument("--test_speakers", default="spk001", help="comma-separated speakers held out as babel_test")
    ap.add_argument("--max_spk_hours", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    test_spk = set(args.test_speakers.split(","))

    info = json.load(open(os.path.join(args.babel, "audio_info.json"), encoding="utf-8"))
    train_rows, test_rows, spk_sec = [], [], {}
    for i in rng.permutation(len(info)):
        r = info[i]
        spk = speaker(r["filename"])
        if spk in test_spk:
            test_rows.append(r)
        elif spk_sec.get(spk, 0) < args.max_spk_hours * 3600:
            train_rows.append(r)
            spk_sec[spk] = spk_sec.get(spk, 0) + r["duration"]
    print("babel train hours per speaker:", {k: round(v / 3600, 1) for k, v in sorted(spk_sec.items())}, flush=True)

    zf = zipfile.ZipFile(os.path.join(args.babel, "wav.zip"))
    names = {os.path.basename(n): n for n in zf.namelist() if n.endswith(".wav")}

    def babel_clips(rows):
        for r in rows:
            text = clean_text(r["text"])
            if text and r["filename"] in names:
                yield to_pcm(zf.read(names[r["filename"]])), {"text": text, "source": f"babel:{r['filename']}"}

    base_meta = [json.loads(l) for l in open(os.path.join(args.base, "train", "meta.jsonl"), encoding="utf-8")]
    base_audio = np.memmap(os.path.join(args.base, "train", "audio.bin"), dtype=np.int16, mode="r")
    order = rng.permutation(len(base_meta) + len(train_rows))  # interleave so batches mix sources

    def train_clips():
        for i in order:
            if i < len(base_meta):
                m = base_meta[i]
                yield np.array(base_audio[m["offset"]: m["offset"] + m["length"]]), \
                    {k: v for k, v in m.items() if k not in ("offset", "length")}
            else:
                yield from babel_clips([train_rows[i - len(base_meta)]])

    n, h = write_split(os.path.join(args.out, "train"), train_clips())
    print(f"train: {n} clips, {h:.1f}h (base {len(base_meta)}, babel {len(train_rows)})", flush=True)
    n, h = write_split(os.path.join(args.out, "babel_test"), babel_clips(test_rows))
    print(f"babel_test ({args.test_speakers}): {n} clips, {h:.2f}h", flush=True)
    for split in ["valid", "test", "cs_test"]:
        shutil.copytree(os.path.join(args.base, split), os.path.join(args.out, split), dirs_exist_ok=True)


if __name__ == "__main__":
    main()
