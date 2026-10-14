# Talk script: how the data was selected, and what the current training is for

Spoken narration for slides 3, 5 and 6 of `assessment_slides.pptx`. About 6 minutes at a normal pace.
Numbers in brackets are the ones on the slides; every figure comes from `results/malay/` or `results/` on the `vi` branch.

---

## 1. Starting point: what the model already does and what the brief asks for (slide 3, 40 s)

Qwen3-ASR-1.7B is a frozen audio encoder in front of a 1.7-billion-parameter Qwen3 decoder. It already transcribes
Malay reasonably well out of the box, so the question is never "can it learn the language" but "where does it
fail, and which data moves that".

The brief asks for a small labelled set, a clean train/eval split, at least one fine-tuning experiment, and a clear
explanation of why. It also says preference goes to English, Malay, Malaysian English and English-Malay
code-switching. So the track is Malay with real code-switching, and the data is messy in ways that make the data
decisions matter.

## 2. Method and preprocessing, fixed once (slide 3, 40 s)

Every run uses LoRA rank 16, alpha 32, on the decoder's attention and MLP projections: 17.4 million trainable
parameters, under one percent of the model, 70 MB per adapter. The audio encoder stays frozen in bf16. LoRA was
chosen because it makes one data experiment cost two to three hours on a 16 GB desktop GPU and because the adapter
merges back into the base for serving; the method was fixed first so that every later difference is a data difference.
Common settings: 2 epochs, learning rate 1e-4 cosine with 3 percent warmup, effective batch 16, bf16, gradient
checkpointing, seed 42, evaluation every 200 steps on 200 validation clips, checkpoints with optimizer state so a run
resumes after an interruption, including across machines.

Preprocessing is the same for every dataset in this repo: decode, resample to 16 kHz mono int16, drop clips over
30 seconds and empty labels, and write each split as one audio file plus a JSON-lines index that the trainer memory-maps.
No per-clip file I/O during training, which is what keeps the trainer simple and resumable.

## 3. Malay: five data decisions, each driven by a measured failure (slides 5 and 6, 3 min)

**Decision 1: the base corpus and a 50-hour subset.** The largest public Malay speech set is malaya-speech-malay-stt,
about 2,600 hours, but its labels are semi-supervised: Google speech-to-text output, lowercase, no punctuation.
We built a prepared pool and took a random 50-hour subset with a fixed seed, plus a 1,994-clip held-out test from the
same corpus. 50 hours is enough to move the model and small enough to iterate in two hours on a 16 GB desktop GPU.
Result: in-domain WER 52 to 37. But on the public Revolab benchmark, 820 human-labelled clips across 12 domains,
the Malay-only model was worse than the base model, 15.0 to 16.1, and telephony went from 14 to 33.

**Decision 2: clean the labels, don't trust them.** Error analysis showed deletions had doubled. We ran Silero VAD
over every clip and compared speech seconds to transcript length: about 6 percent of clips have labels far too
short for the speech, and 1 percent have no speech at all. Training on them teaches the model to drop words. We
removed those clips and topped the subset back up to 50 hours. Deletions fell, but the overall number barely moved,
because the test labels are just as noisy. The lesson we kept: compare systems on clean-labelled sets, which is why
Revolab and later BabelSpeech carry the conclusions.

**Decision 3: put English back.** Listening to the Revolab telephony errors showed the real problem: English words
inside Malay speech were being rewritten as Malay words, "that's all" became "datang". A Malay-only LoRA had
forgotten English. So we mixed in 7 hours of synthetic Manglish code-switch TTS, 200 machine-labelled real
code-switch clips used for training only, and 5 hours of FLEURS English tagged with the English language token.
That is lora-mix, 63 hours. Revolab dropped below the base model, 15.0 to 13.4, telephony came back from 38 to 19,
and the in-domain gain was untouched.

**Decision 4: real conversational code-switch, with speakers held out.** The synthetic code-switch test is TTS from
the same voices as training, so its 8.7 WER proves nothing about real speech. We added BabelSpeech, a human-labelled
corpus of colloquial Malay conversation and monologue, 12 hours from nine speakers, with one speaker capped at 10 hours
so he does not dominate, and one whole speaker, spk001, 377 clips, held out as the only real code-switch test. That
is the current run, lora-babel, 83 hours.

**Decision 5: more data, not more epochs.** Before adding BabelSpeech we tried the cheaper thing, two more epochs of
lora-mix on the same 63 hours. Eval loss went up, every test got worse. The flat loss curve had been the learning rate
reaching zero, not convergence. That settled the order: new data first.

## 4. What the current training is for (slide 5 and 6, 1 min 30)

lora-babel exists to answer one question: does real conversational Malay from other speakers improve recognition of
real code-switched speech from a speaker the model has never heard, without costing anything elsewhere?

The answer is yes on both halves. On the held-out speaker, WER goes from 17.9 for the base model and 14.8 for
lora-mix to 7.8. On every other test it is level with lora-mix: in-domain 37.0, Revolab 13.4, synthetic code-switch
8.7. Per clip, lora-babel is better on 277 of 377 clips and worse on 25.

We also checked what the gain is made of. About a third of lora-mix's gap is label convention, not acoustics: the
BabelSpeech labels write "korang" where lora-mix writes "kau orang", 287 times, plus Malay spellings like kamera
and bateri, and numbers spelled out. When we neutralise those conventions on both sides, lora-mix is 11.2 and
lora-babel 7.6, so 3.6 points of real recognition gain remain.

Two caveats we state on the slide: the real code-switch test is one speaker from the same corpus as the new training
data, so 7.8 is an in-corpus number; and one seed per run, so differences under a point on a 50-clip Revolab domain
are noise. Both are fixable with the next data step, a second held-out speaker and clips from a different source.

## 5. Why this is the right fine-tuning strategy for the brief (30 s)

Every change was a hypothesis, a data edit and a measurement on a clean test, not a hyperparameter search. LoRA
made each experiment cost two to three hours and a 70 MB adapter, which is what allowed five data variants in the time
available. The same trainer, the same evaluator and the same prepared format run on a Windows desktop, a Linux
server and Colab, so the whole chain is reproducible from `scripts/run_malay.sh`.

---

### One-line answers if asked

- Why 50 hours when the brief says 30-60 minutes? The pipeline runs unchanged on any size (`subset_malay.py --hours`);
  we used more because it was available and because the data questions only show up at scale.
- Why not clean the test labels too? We could not without re-labelling; that is why the human-labelled sets carry the conclusions.
- Why LoRA rank 16? Cost: one experiment in two to three hours on a 16 GB GPU, a 70 MB adapter that merges for
  serving; rank was not swept, and full fine-tuning was not run on Malay.
- What would you do next with data? A second held-out BabelSpeech speaker, human-verified clips from a different
  podcast source, and narrowband or music augmentation for telephony and singing, which still trail the base model.
