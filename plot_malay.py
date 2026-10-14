"""Draw every figure in docs/malay.md from results/malay/ (train logs, scores, per-clip predictions).

    python plot_malay.py [--results results/malay] [--out docs/figures]

Needs REVOLAB_DIR (clone of revolab-asr-benchmark) for the label-convention figure, which re-scores babel_test
with score_malay's normalizer; that figure is skipped with a warning if the import fails.
"""
import argparse
import collections
import difflib
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Systems in a fixed order with fixed colors (validated categorical palette; base is the neutral reference).
SYSTEMS = [
    ("base", "base", "#52514e"),
    ("lora-1.7b-malay50h", "lora", "#2a78d6"),
    ("lora-1.7b-malay50h-clean", "lora-clean", "#eb6834"),
    ("lora-1.7b-malay50h-mix", "lora-mix", "#1baf7a"),
    ("lora-1.7b-malay50h-mix-cont", "lora-mix-cont", "#eda100"),
    ("lora-1.7b-malay50h-babel", "lora-babel", "#6250d6"),
]
LABEL = {k: l for k, l, _ in SYSTEMS}
COLOR = {k: c for k, _, c in SYSTEMS}
LOG = {  # train_log file suffix per run
    "lora-1.7b-malay50h": "lora50h", "lora-1.7b-malay50h-clean": "lora50h-clean", "lora-1.7b-malay50h-mix": "lora50h-mix",
    "lora-1.7b-malay50h-mix-cont": "lora50h-mix-cont", "lora-1.7b-malay50h-babel": "lora50h-babel",
}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
ERR_COLORS = {"sub": "#2a78d6", "del": "#eb6834", "ins": "#1baf7a"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.edgecolor": GRID, "axes.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "grid.linestyle": "-", "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
})


def J(path):
    return json.load(open(path, encoding="utf-8"))


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def train_log(path):
    """Per-step rows, de-duplicated by step (a resumed run re-logs the steps after its last checkpoint)."""
    by_step, evals = {}, {}
    for r in rows(path):
        if "loss" in r:
            by_step[r["step"]] = r
        if "eval_loss" in r:
            evals[r["step"]] = r["eval_loss"]
    steps = sorted(by_step)
    return steps, by_step, sorted(evals.items())


def smooth(y, w=50):
    y = np.asarray(y, float)
    if len(y) < w:
        return y
    k = np.ones(w) / w
    out = np.convolve(y, k, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), out])


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", path)


def hbar(ax, labels, values, colors, fmt="{:.2f}", best=None):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.55, color=colors, edgecolor=SURFACE, linewidth=1.5)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    for yi, v, lab in zip(y, values, labels):
        ax.text(v + max(values) * 0.015, yi, fmt.format(v), va="center", ha="left", fontsize=8.5,
                color=INK, fontweight="bold" if lab == best else "normal")
    ax.set_xlim(0, max(values) * 1.18)
    ax.tick_params(axis="y", length=0)


# ----------------------------------------------------------------------------------------------- figures

def fig_train_curves(res, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    offset = {"lora-1.7b-malay50h-mix-cont": 4304}  # continuation of lora-mix: draw it after lora-mix's 4,304 steps
    for run, suffix in LOG.items():
        steps, by, evals = train_log(os.path.join(res, f"train_log_{suffix}.jsonl"))
        off = offset.get(run, 0)
        x = np.array(steps) + off
        if run != "lora-1.7b-malay50h":  # lora ran at batch 2 x 8: the Trainer logs its loss on a ~2x scale
            lab = LABEL[run] + ("  (continues lora-mix's adapter; drawn after step 4,304)" if off else "")
            a.plot(x, smooth([by[s]["loss"] for s in steps]), color=COLOR[run], label=lab)
        ex, ey = zip(*evals)
        b.plot(np.array(ex) + off, ey, color=COLOR[run], label=f"{LABEL[run]}  (final {ey[-1]:.3f})", marker="o",
               markersize=4, markeredgecolor=SURFACE, markeredgewidth=1)
    for ax in (a, b):
        ax.set_xlabel("optimizer step (effective batch 16)")
        ax.axvline(4304, color=GRID, linewidth=1)
    a.set_title("Train loss (per-step, 50-step rolling mean)")
    a.set_ylabel("loss")
    a.set_ylim(1.2, 3.2)
    b.set_title("Eval loss (200 valid clips, every 200 steps)")
    b.set_ylim(0.74, 1.0)
    a.text(60, 1.27, "lora (batch 2 x 8) omitted: its loss is logged on a different scale", fontsize=7.5, color=INK2)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    b.legend(loc="upper right")
    fig.suptitle("Training curves of the Malay LoRA runs", x=0.01, y=1.02, ha="left", fontweight="bold", fontsize=11)
    save(fig, out, "train_curves.png")


def fig_train_curves_babel(res, out):
    """Train and eval loss of the final run only (lora-babel), greys."""
    steps, by, evals = train_log(os.path.join(res, "train_log_lora50h-babel.jsonl"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
    a.plot(steps, smooth([by[s]["loss"] for s in steps]), color="#404040")
    a.set_title("Train loss, lora-babel (per-step, 50-step rolling mean)")
    a.set_ylabel("loss"); a.set_xlabel("optimizer step (effective batch 16)"); a.set_ylim(1.2, 3.2)
    ex, ey = zip(*evals)
    b.plot(ex, ey, color="#404040", marker="o", markersize=4, markeredgecolor=SURFACE, markeredgewidth=1)
    b.annotate(f"final {ey[-1]:.3f}", (ex[-1], ey[-1]), textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK2)
    b.set_title("Eval loss (200 valid clips, every 200 steps)")
    b.set_xlabel("optimizer step (effective batch 16)"); b.set_ylim(0.74, 1.0)
    a.axvline(800, color=GRID, linewidth=1); b.axvline(800, color=GRID, linewidth=1)
    a.text(830, 3.12, "resumed on Colab A100 from step 800", fontsize=7.5, color=INK2, va="top")
    save(fig, out, "train_curves_babel.png")


def fig_step_time(res, out):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for run, lab in [("lora-1.7b-malay50h-mix", "lora-mix, RTX 5060 Ti (Windows, P-core pinned)"),
                     ("lora-1.7b-malay50h-babel", "lora-babel: steps 1-800 RTX 5060 Ti, 801-5098 Colab A100")]:
        steps, by, _ = train_log(os.path.join(res, f"train_log_{LOG[run]}.jsonl"))
        pairs = [(s, by[s]["step_time_s"]) for s in steps if by[s]["step_time_s"] < 20]  # drop eval steps
        xs, st = zip(*pairs)
        ax.plot(xs, smooth(st, 101), color=COLOR[run], label=lab)
    ax.axvline(800, color=GRID, linewidth=1)
    ax.text(830, 3.3, "lora-babel resumes on Colab A100", fontsize=8, color=INK2, va="top")
    ax.set_ylim(0, 3.5)
    ax.set_ylabel("seconds per step (101-step rolling mean)")
    ax.set_xlabel("optimizer step")
    ax.set_title("Step time: the A100 is no faster than the local GPU (trainer is CPU-bound)")
    ax.legend(loc="lower right")
    save(fig, out, "step_time.png")


def fig_wer_overview(res, out):
    test = J(os.path.join(res, "test_scores.json"))
    rev = J(os.path.join(res, "revolab_summary.json"))
    cs = J(os.path.join(res, "cs_test_summary.json"))  # 221-clip build, basic normalization, all systems
    bab = J(os.path.join(res, "babel_test_scores.json"))
    panels = [
        ("In-domain test (1,994 clips)", {k: test[k]["malay"]["wer"] for k in test}),
        ("Revolab benchmark, all domains (820)", {k: rev[k]["All"]["wer"] for k in rev}),
        ("Synthetic code-switch, cs_test (221)", {k: cs[k]["All"]["wer"] for k in cs}),
        ("Real code-switch, babel_test (377)", {k: bab[k]["malay"]["wer"] for k in bab}),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (title, d) in zip(axes, panels):
        keys = [k for k, _, _ in SYSTEMS if k in d]
        vals = [d[k] for k in keys]
        best = LABEL[keys[int(np.argmin([v if k != "base" else 1e9 for k, v in zip(keys, vals)]))]]
        hbar(ax, [LABEL[k] for k in keys], vals, [COLOR[k] for k in keys], best=best)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("WER %")
    fig.suptitle("WER by system on the four test sets (lower is better; bold = best fine-tuned run)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, out, "wer_overview.png")


def fig_revolab_domains(res, out):
    rev = J(os.path.join(res, "revolab_summary.json"))
    runs = ["base", "lora-1.7b-malay50h-mix", "lora-1.7b-malay50h-babel"]
    domains = [d for d in rev["base"] if d not in ("infer_seconds", "All")]
    domains.sort(key=lambda d: rev["base"][d]["wer"])
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(domains))
    for i, d in enumerate(domains):
        vals = [rev[r][d]["wer"] for r in runs]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, linewidth=2, zorder=1)
    for r in runs:
        ax.scatter([rev[r][d]["wer"] for d in domains], y, s=60, color=COLOR[r], edgecolor=SURFACE, linewidth=1.5,
                   label=LABEL[r], zorder=3)
    ax.set_yticks(y, [f"{d} (n={rev['base'][d]['clips']})" for d in domains])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("WER % (benchmark scoring)")
    ax.set_title("Revolab benchmark by domain: base vs lora-mix vs lora-babel")
    ax.legend(loc="lower right")
    for d, i in zip(domains, y):  # direct-label the two domains where the fine-tunes still lose, and the biggest win
        if d in ("singing", "telephony", "short-inputs"):
            b, m = rev["base"][d]["wer"], rev["lora-1.7b-malay50h-babel"][d]["wer"]
            right = max(rev[r][d]["wer"] for r in runs)
            ax.annotate(f"base {b:.1f} -> lora-babel {m:.1f}", (right + 0.8, i), fontsize=7.5, color=INK2, va="center")
    save(fig, out, "revolab_domains.png")


def stacked_errors(ax, scores, keys, norm="malay"):
    y = np.arange(len(keys))[::-1]
    left = np.zeros(len(keys))
    for part in ("sub", "del", "ins"):
        v = np.array([scores[k][norm][part] for k in keys])
        ax.barh(y, v, left=left, height=0.55, color=ERR_COLORS[part], edgecolor=SURFACE, linewidth=1.5, label=part)
        left += v
    for yi, k, tot in zip(y, keys, left):
        ax.text(tot + 0.6, yi, f"{scores[k][norm]['wer']:.1f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [LABEL[k] for k in keys])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, left.max() * 1.15)
    ax.set_xlabel("WER % = substitutions + deletions + insertions")


def fig_error_breakdown(res, out):
    test = J(os.path.join(res, "test_scores.json"))
    bab = J(os.path.join(res, "babel_test_scores.json"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
    stacked_errors(a, test, [k for k, _, _ in SYSTEMS if k in test])
    a.set_title("In-domain test: base over-inserts, Malay-only LoRA over-deletes")
    stacked_errors(b, bab, [k for k, _, _ in SYSTEMS if k in bab])
    b.set_title("Real code-switch (babel_test)")
    b.legend(loc="lower right", title="error type", title_fontsize=8)
    fig.tight_layout()
    save(fig, out, "error_breakdown.png")


def fig_babel_per_clip(res, out):
    mix = rows(os.path.join(res, "babel_test_lora-1.7b-malay50h-mix_predictions.jsonl"))
    bab = rows(os.path.join(res, "babel_test_lora-1.7b-malay50h-babel_predictions.jsonl"))
    x = np.array([r["wer"] for r in mix]); y = np.array([r["wer"] for r in bab])
    better, worse, same = int((y < x).sum()), int((y > x).sum()), int((y == x).sum())
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=GRID, linewidth=1.5, zorder=1)
    ax.scatter(x, y, s=28, color=COLOR["lora-1.7b-malay50h-babel"], alpha=0.55, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("lora-mix WER % per clip")
    ax.set_ylabel("lora-babel WER % per clip")
    ax.set_title("babel_test, 377 clips: lora-babel vs lora-mix")
    ax.text(lim * 0.97, lim * 0.05, f"lora-babel better: {better} clips\nworse: {worse}\nequal: {same}",
            ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax.text(lim * 0.05, lim * 0.95, "above the line = lora-mix better", fontsize=7.5, color=INK2, va="top")
    ax.set_aspect("equal")
    save(fig, out, "babel_per_clip.png")


def tokens(s):
    return re.sub(r"[^\w' ]+", " ", s.lower()).split()


def top_substitutions(preds, n=10):
    sub = collections.Counter()
    for r in preds:
        a, b = tokens(r["ref"]), tokens(r["hyp"])
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if tag == "replace":
                for x, y in zip(a[i1:i2], b[j1:j2]):
                    sub[(x, y)] += 1
    return sub.most_common(n)


CONVENTIONS = [(r"\bkau orang\b", "korang"), (r"\bdia orang\b", "diorang"), (r"\bcamera\b", "kamera"),
               (r"\bbattery\b", "bateri"), (r"\bscreen\b", "skrin"), (r"\bcolou?r\b", "warna"), (r"\bokay\b", "ok"),
               (r"\bsmart phone\b", "smartphone"), (r"\bsahaja\b", "saja"), (r"\bdekat\b", "kat"), (r"\bitu\b", "tu"),
               (r"\bini\b", "ni"), (r"\btimes\b", "time"), (r"\bbagian\b", "bahagian")]


def fig_babel_conventions(res, out):
    runs = ["base", "lora-1.7b-malay50h-mix", "lora-1.7b-malay50h-babel"]
    preds = {r: rows(os.path.join(res, f"babel_test_{r}_predictions.jsonl")) for r in runs}
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9])
    for i, r in enumerate(runs[1:]):
        ax = fig.add_subplot(gs[0, i])
        top = top_substitutions(preds[r])
        labs = [f"{a} -> {b}" for (a, b), _ in top]
        hbar(ax, labs, [c for _, c in top], [COLOR[r]] * len(top), fmt="{:.0f}")
        ax.set_title(f"{LABEL[r]}: top substitutions (ref -> hyp)", fontsize=9)
        ax.set_xlabel("count over 377 clips")
    ax = fig.add_subplot(gs[0, 2])
    try:
        import jiwer
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from score_malay import NORMS
        num = NORMS["num"]

        def neutral(s):
            s = num(s)
            for a, b in CONVENTIONS:
                s = re.sub(a, b, s)
            return s
        raw = [jiwer.wer([num(r["ref"]) for r in preds[k]], [num(r["hyp"]) for r in preds[k]]) * 100 for k in runs]
        neu = [jiwer.wer([neutral(r["ref"]) for r in preds[k]], [neutral(r["hyp"]) for r in preds[k]]) * 100 for k in runs]
        y = np.arange(len(runs))[::-1]
        ax.barh(y + 0.17, raw, height=0.3, color="#2a78d6", edgecolor=SURFACE, linewidth=1.5, label="Malay + number normalization")
        ax.barh(y - 0.17, neu, height=0.3, color="#eb6834", edgecolor=SURFACE, linewidth=1.5, label="+ spelling conventions neutralized")
        for yi, a, b in zip(y, raw, neu):
            ax.text(a + 0.3, yi + 0.17, f"{a:.1f}", va="center", fontsize=8, color=INK)
            ax.text(b + 0.3, yi - 0.17, f"{b:.1f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y, [LABEL[k] for k in runs]); ax.tick_params(axis="y", length=0); ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(raw) * 1.25)
        ax.set_xlabel("WER % on babel_test")
        ax.set_title("A third of lora-mix's gap is label convention", fontsize=9)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7.5, ncol=1)
    except Exception as e:  # noqa: BLE001
        ax.axis("off")
        ax.text(0, 0.5, f"re-scoring skipped:\n{type(e).__name__}: {e}\n(set REVOLAB_DIR)", fontsize=8, color=INK2, wrap=True)
        print("warning: convention panel skipped:", e)
    fig.suptitle("Why lora-mix loses on babel_test: 'korang' is written 'kau orang', English spellings, digits",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, out, "babel_conventions.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/malay")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in (fig_train_curves, fig_train_curves_babel, fig_step_time, fig_wer_overview, fig_revolab_domains, fig_error_breakdown,
              fig_babel_per_clip, fig_babel_conventions):
        f(args.results, args.out)


if __name__ == "__main__":
    main()
