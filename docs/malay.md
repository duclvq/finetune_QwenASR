# Malay results: Qwen3-ASR-1.7B LoRA on Malay and Malay-English code-switch (updated 2026-10-13)

**Bottom line.** A Malay-only LoRA wins in-domain but loses to the base model out of domain (truncated labels +
English forgetting). Mixing in code-switched and English speech fixes most of it (`lora-mix`), and adding 12 h of
real conversational Malay (`lora-babel`, BabelSpeech) halves the error on real code-switched speech from an unseen
speaker (WER 14.8 -> **7.8**) while leaving every other test unchanged. **`lora-1.7b-malay50h-babel` is the best
system.** Part of its gain on BabelSpeech is label-convention matching (see "How much of the babel gain is real").

The short version and the repository guide are in the top-level [README](../README.md).

## Systems

| System | Train data | Hours / clips | Steps | Trained on |
|---|---|---|---|---|
| base | — | — | — | — |
| lora | random 50h of `mesolitica/malaya-speech-malay-stt` | 50.0h / 29,966 | 3,746 | RTX 5060 Ti |
| lora-clean | same, minus truncated-label clips (VAD), topped up to 50h | 50.0h / 30,225 | 3,778 | RTX 5060 Ti |
| lora-mix | lora-clean + Synth-Manglish + cs_pilot (code-switch) + 5h FLEURS en_us (`language English`) | 63.0h / 34,425 | 4,304 | RTX 5060 Ti |
| lora-mix-cont | lora-mix trained 2 more epochs from its adapter (`--init_adapter`, lr 5e-5) | 63.0h | 4,304 | RTX 5060 Ti |
| **lora-babel** | lora-mix data + BabelSpeech Malay (12.0h, 9 speakers, spk005 capped at 10h; spk001 held out) | 83.0h / 40,771 | 5,098 | 800 steps RTX 5060 Ti, rest Colab A100 |

cs_pilot (200 train clips) comes from one podcast episode with machine-generated labels (Qwen3-ASR + Whisper merged
by an LLM, no human check), so it is used for training only and **no result is reported on it**.

All LoRA runs: r=16, alpha=32 on the LLM's attention + MLP (17.4M trainable, 0.85%), 2 epochs, lr 1e-4 cosine,
effective batch 16 (lora: 2 x 8; others: 4 x 4), bf16, gradient checkpointing, seed 42.
Train wall time: lora 4.1h, lora-clean 1.7h, lora-mix 2.3h on the RTX 5060 Ti 16GB (Windows, trainer pinned to
P-cores, see below); lora-babel 0.3h local + 3.0h on a Colab A100 (see "Colab").

## Test sets

| Set | Clips | What it is | Caveat |
|---|---|---|---|
| in-domain test | 1,994 | held-out `malaya-speech-malay-stt` clips | semi-supervised labels (Google STT), ~6% truncated |
| Revolab/ASR-Benchmark-Public | 820 | 12 domains, human labels | out of domain for every run |
| cs_test | 221 | held-out Synth-Manglish TTS clips | same voices/source as lora-mix training data: optimistic |
| babel_test | 377 (1.8h) | BabelSpeech speaker spk001, real colloquial code-switch, human labels | one speaker; same corpus/label conventions as lora-babel's extra train data |

Scoring: `score_malay.py` (Revolab's MalayTextNormalizer + Malay number spelling); Revolab numbers use the
benchmark's own scoring (best of `text`/`normalized_text` per row).

## Results

### In-domain test (1,994 clips)

"Clean" = the 1,860 clips whose label is not truncated.

| System | WER | Sub | Del | Ins | WER clean clips |
|---|---|---|---|---|---|
| base | 52.10 | 29.21 | 3.03 | 19.85 | 46.76 |
| lora | **37.07** | 23.04 | 6.60 | 7.42 | 34.92 |
| lora-clean | 37.32 | 23.25 | 5.59 | 8.49 | 34.69 |
| lora-mix | 37.35 | 23.29 | 5.35 | 8.71 | **34.63** |
| lora-mix-cont | 38.42 | | | | 35.34 |
| lora-babel | **37.03** | 22.79 | 5.70 | 8.54 | n/a |

### Revolab/ASR-Benchmark-Public (820 clips, 12 domains, out of domain)

| Domain | n | base | lora | lora-clean | lora-mix | lora-babel |
|---|---|---|---|---|---|---|
| **All** | 820 | 15.01 | 16.14 | 15.76 | **13.38** | 13.39 |
| news | 52 | 23.11 | 12.97 | 13.39 | **11.13** | 11.77 |
| short-inputs | 52 | 31.67 | 17.65 | 18.49 | **13.45** | **13.45** |
| drama | 52 | 26.91 | 26.80 | 26.14 | **21.26** | 22.26 |
| read-speech | 51 | 7.98 | 3.68 | 4.60 | 3.68 | **3.07** |
| commonvoice | 141 | 8.40 | 7.36 | 7.45 | 7.08 | **7.05** |
| fleurs | 153 | 7.34 | 7.38 | 7.47 | 6.93 | **6.83** |
| animation | 50 | 17.35 | 20.81 | 18.70 | **16.89** | 17.95 |
| parliament | 52 | 13.93 | 15.38 | 15.48 | 14.43 | **13.27** |
| street interview | 48 | **37.35** | 45.38 | 42.27 | 37.95 | 38.65 |
| podcast | 52 | **16.21** | 22.76 | 21.73 | 18.78 | 18.12 |
| singing | 51 | **18.72** | 36.76 | 29.45 | 23.06 | 25.11 |
| telephony | 66 | **14.14** | 33.47 | 38.37 | 19.11 | 16.29 |

lora-mix-cont: All 13.89. Domains have ~50 clips, so differences of 1-2 points between lora-mix and lora-babel are
within noise; the telephony gain (19.1 -> 16.3) is the one that looks real and is plausible (closest domain to
conversational speech).

### Synthetic code-switch test (cs_test)

221 held-out Synth-Manglish clips (the `malay_50h_babel` build; the earlier 203-clip version gave lora-mix 7.42).

| System | WER | CER |
|---|---|---|
| base | 25.69 | 9.77 |
| lora-mix | 8.69 | 4.01 |
| lora-babel | 8.69 | **3.95** |

TTS from the same source and voices as the training data, so in-distribution and optimistic; not evidence about
real code-switched speech. babel_test below is.

### Real code-switch test (babel_test: BabelSpeech speaker spk001, 377 clips, 20,175 words)

| System | WER | CER | Sub | Del | Ins | clips better / worse than lora-mix |
|---|---|---|---|---|---|---|
| base | 17.94 | 8.69 | 12.29 | 3.37 | 2.29 | |
| lora-mix | 14.76 | 6.96 | 8.98 | 2.74 | 3.05 | |
| lora-babel | **7.83** | **4.23** | 4.57 | 1.97 | 1.28 | 277 / 25 (75 equal) |

lora-mix never saw BabelSpeech, so its 17.9 -> 14.8 is a true out-of-domain gain on real code-switched speech.
lora-babel saw 12h of the same corpus from other speakers.

#### How much of the babel gain is real recognition?

lora-mix's single biggest error on this set is a spelling convention: it writes "kau orang" where BabelSpeech labels
write "korang" (287 clips), and similarly camera/kamera, battery/bateri, screen/skrin, "dia orang"/diorang, and
digits where the labels spell numbers out. lora-babel learned these conventions from the other BabelSpeech speakers.
Re-scoring with those spelling variants and number formats neutralised on both sides:

| System | WER (num-normalised) | WER with conventions neutralised |
|---|---|---|
| base | 17.71 | 17.06 |
| lora-mix | 14.41 | 11.16 |
| lora-babel | 7.80 | **7.57** |

So roughly a third of the lora-mix -> lora-babel gap is label convention; the remaining 3.6 points (32% relative)
is recognition. Remaining lora-babel errors are mostly discourse particles (dropped "kan", "lah", "dia"; inserted
"aku", "lah", "and") and diorang/dia, where labels themselves are inconsistent.

**Caveat:** one speaker, one recording setup. Treat 7.8 as an in-corpus number. The robust claims are: adding real
conversational data did not hurt any other test, and it helped telephony.

### Continued training (lora-mix-cont)

Eval loss had looked flat at the end of lora-mix only because the cosine LR had reached 0, so lora-mix was trained
2 more epochs from its final adapter (`--init_adapter`, fresh warmup + cosine, lr 5e-5, 4,304 steps).
Eval loss bottomed at 0.783 (step 1,800) and then rose to ~0.80 while train loss kept falling: overfitting on
epochs 3-4 of the same data. Worse everywhere (test 38.42, Revolab 13.89, cs_test 7.48). More epochs on the same
data do not help; new data does (lora-babel).

## Analysis

1. **In-domain gain is real and stable** (~37 WER vs 52) and does not depend on the data variant. Remaining errors
   are mostly short function words in fast conversational speech (di/dia, nak/anak, korang/orang, dah/dan).
2. **Truncated labels -> deletions.** The corpus is semi-supervised; Silero VAD finds ~6% of clips whose transcript is
   far too short for the speech and ~1% with no speech. Training on them taught the model to drop words. Removing
   them (lora-clean) lowers deletions, but the overall gain is small because the test labels are equally noisy.
3. **English / code-switch forgetting was the main out-of-domain problem.** Malay-only LoRAs map English speech onto
   Malay words ("that's all" -> "datang"), wrecking telephony (14 -> 33-38). Adding 7.3h code-switch + 5h English
   (lora-mix) fixes most of it; adding 12h real conversational Malay (lora-babel) adds nothing on Revolab overall but
   halves the error on real conversational code-switch.
4. Numbers and punctuation move WER by < 2 points for every system, except on BabelSpeech where spelled-out numbers
   and spelling conventions matter (see above).
5. Base WER is 18 on BabelSpeech but 52 on the in-domain test: most of the in-domain number is label noise, not
   model quality. Compare systems on clean-labelled sets where possible.

## Training-infrastructure findings

Windows, RTX 5060 Ti 16GB, i5-14400F:
- The trainer is single-thread CPU-bound at small batch. Windows schedules the hidden background process on E-cores,
  halving speed; pinning to P-cores (`CPU_AFFINITY=0-11`) took throughput from 4.4 to ~8.7 samples/s (GPU 30% -> 92%).
- CUDA "sysmem fallback" lets the caching allocator silently spill past VRAM into shared system RAM; a per-process
  cap (`CUDA_MEM_FRACTION=0.8`) prevents it.
- Trainer restores the checkpoint's `train_batch_size` on resume, silently ignoring a new `--batch_size`;
  `train_qwen_asr.py` refuses such a resume.
- HF eval loss is a mean of per-batch means, so it shifts with eval batch size; compare runs by WER, not eval loss.

Colab (`notebooks/colab_train_malay_babel.ipynb`, A100 40GB):
- Resuming the Windows checkpoint failed twice before training started: Colab's torchao 0.10 breaks peft's LoRA
  injection (uninstall it), and torch>=2.6 refuses the checkpoint's `rng_state.pth` pickled under numpy 1.x
  (`train_qwen_asr.py` now allowlists the numpy globals).
- The A100 was **not faster than the RTX 5060 Ti**: 2.55 s/step vs 1.9-2.3 s/step, peak 8.75GB of 40GB used. The
  dataloader/collator is the bottleneck and Colab's CPU is weak. For new Colab runs use batch 16, grad_acc 1 and more
  workers; a resumed run must keep the checkpoint's batch size.
- Data must be copied from Drive to local disk first (memmap through the Drive FUSE mount is far too slow); the
  output dir is a symlink into Drive so checkpoints survive disconnects.

## Next steps

1. A second held-out BabelSpeech speaker (spk009, 0.2h, costs nothing in training data) to get a multi-speaker real
   code-switch test; longer term, human-verified clips from a different source (e.g. the `D:/cs_mining` podcast
   pipeline once its labels are checked).
2. Add spelling-variant normalisation (korang/kau orang, kamera/camera, number spelling) to `score_malay.py` so
   systems are compared on recognition, and report both numbers.
3. Telephony and singing still trail base: narrowband / music augmentation or more such data.
4. Add data rather than epochs: more speakers and settings rather than more hours of spk005 (already 10 of the 12
   BabelSpeech hours).

## Reproduce

See `scripts/run_malay.sh` (the full data -> train -> eval chain with the paths used here) and the Colab notebook
for training/evaluating without a local GPU. Per-clip predictions, scores and train logs for all systems are in
`results/malay/` (`test_*`, `revolab_*`, `cs_test_*`, `babel_test_*`, `train_log_*`, `train_summary_*`). Adapters are
not in git: `lora-1.7b-malay50h-mix` and `lora-1.7b-malay50h-babel` are in the Drive folder
`My_dataset/outputs/<run>/adapter` and on the training PC under `outputs/<run>/adapter`.
