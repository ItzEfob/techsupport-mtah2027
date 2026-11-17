"""Choose the method's settings on IDSSE, and write them to params.json.

    python external/tune.py

For every candidate setting it runs the method on all prepared IDSSE matches, each degraded at
several ball-gap lengths, and scores the output with the organisers' own metric code. The setting
with the best average spellF1 wins. The competition match is never read here.
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, HERE)

import possession_metrics as metrics  # noqa: E402  the organisers' scorer
from degrade import degrade  # noqa: E402
from src.pipeline import predict  # noqa: E402

IDSSE_DIR = os.path.join(ROOT, "data", "idsse")
PARAMS_PATH = os.path.join(ROOT, "params.json")

# What we try. 25 frames = 1 second.
WINDOWS = (38, 50, 75, 100, 125)
# Dead-ball rule: None = off (version 1), otherwise how far beyond the line the ball must be.
MARGINS = (None, 0.0, 0.01, 0.03)
# How long the ball stays lost in the degraded data. Unknown for the real match, so we try three
# and want a setting that works across all of them.
BALL_GAPS = (10, 30, 75)

CODE = {name: index for index, name in enumerate(metrics.LABELS)}  # H A D X -> 0 1 2 3


def to_codes(labels, period):
    """Label letters -> the scorer's numbers, with one X frame between the two halves.

    The scorer treats a stretch of X in the ground truth as a break, and never joins a possession
    across one. IDSSE has no half-time frames, so we insert a single X to mark the break.
    """
    codes = np.array([CODE[label] for label in labels], dtype=np.int8)
    second_half = int(np.argmax(period == 2))
    return np.concatenate((codes[:second_half], [CODE["X"]], codes[second_half:]))


def score(truth, predicted, period):
    result = metrics.evaluate(to_codes(truth, period), to_codes(predicted, period))
    return result["spell_f1_mean"], result["macro_f1"]


def main():
    paths = sorted(glob.glob(os.path.join(IDSSE_DIR, "*.npz")))
    if not paths:
        raise SystemExit("no matches in {} -- run external/prepare_idsse.py first".format(IDSSE_DIR))

    # Every candidate setting is a (window, margin) pair. spell[c][g] collects one score per match.
    candidates = [(w, m) for m in MARGINS for w in WINDOWS]
    spell = {c: {g: [] for g in BALL_GAPS} for c in candidates}
    macro = {c: {g: [] for g in BALL_GAPS} for c in candidates}

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        clean = dict(np.load(path))
        print("scoring {} ...".format(name), flush=True)
        for gap in BALL_GAPS:
            noisy = degrade(clean, name=name, ball_mean_gap=gap)
            for window, margin in candidates:
                predicted = predict(noisy, {"smooth_window": window, "out_margin": margin})
                s, m = score(clean["labels"], predicted, clean["period"])
                spell[(window, margin)][gap].append(s)
                macro[(window, margin)][gap].append(m)

    print("\nmean spellF1 over {} matches (columns: mean ball gap in frames)".format(len(paths)))
    print("margin  window  " + "  ".join("gap{:<4}".format(g) for g in BALL_GAPS) + "   all    worst-match  macroF1")
    best, best_score = None, -1.0
    for c in candidates:
        window, margin = c
        per_gap = [float(np.mean(spell[c][g])) for g in BALL_GAPS]
        everything = [s for g in BALL_GAPS for s in spell[c][g]]
        overall = float(np.mean(everything))
        print("{:>6}  {:>6}  ".format("off" if margin is None else margin, window) + "  ".join("{:.4f} ".format(v) for v in per_gap)
              + "  {:.4f}  {:.4f}       {:.4f}".format(
                  overall, min(everything),
                  float(np.mean([m for g in BALL_GAPS for m in macro[c][g]]))))
        if overall > best_score:
            best, best_score = c, overall

    print("\nbest: smooth_window = {}, out_margin = {}  (mean spellF1 {:.4f})".format(
        best[0], best[1], best_score))
    with open(PARAMS_PATH, "w") as handle:
        json.dump({
            "params": {"smooth_window": best[0], "out_margin": best[1]},
            "source": "tuned by external/tune.py on IDSSE (7 Bundesliga matches, CC BY 4.0), "
                      "degraded by external/degrade.py; the competition match was not used",
            "idsse_mean_spell_f1": round(best_score, 4),
            "matches": [os.path.splitext(os.path.basename(p))[0] for p in paths],
        }, handle, indent=2)
        handle.write("\n")
    print("wrote params.json")


if __name__ == "__main__":
    main()
