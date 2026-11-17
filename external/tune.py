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
WINDOWS = (0, 12, 25, 38, 50, 75, 100, 125, 175, 250)
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

    # spell[w][g] and macro[w][g] collect one score per match.
    spell = {w: {g: [] for g in BALL_GAPS} for w in WINDOWS}
    macro = {w: {g: [] for g in BALL_GAPS} for w in WINDOWS}

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        clean = dict(np.load(path))
        print("scoring {} ...".format(name), flush=True)
        for gap in BALL_GAPS:
            noisy = degrade(clean, name=name, ball_mean_gap=gap)
            for window in WINDOWS:
                predicted = predict(noisy, {"smooth_window": window})
                s, m = score(clean["labels"], predicted, clean["period"])
                spell[window][gap].append(s)
                macro[window][gap].append(m)

    print("\nmean spellF1 over {} matches (columns: mean ball gap in frames)".format(len(paths)))
    print("window  " + "  ".join("gap{:<4}".format(g) for g in BALL_GAPS) + "   all    worst-match  macroF1")
    best_window, best_score = None, -1.0
    for window in WINDOWS:
        per_gap = [float(np.mean(spell[window][g])) for g in BALL_GAPS]
        everything = [s for g in BALL_GAPS for s in spell[window][g]]
        overall = float(np.mean(everything))
        print("{:>6}  ".format(window) + "  ".join("{:.4f} ".format(v) for v in per_gap)
              + "  {:.4f}  {:.4f}       {:.4f}".format(
                  overall, min(everything),
                  float(np.mean([m for g in BALL_GAPS for m in macro[window][g]]))))
        if overall > best_score:
            best_window, best_score = window, overall

    print("\nbest: smooth_window = {}  (mean spellF1 {:.4f})".format(best_window, best_score))
    with open(PARAMS_PATH, "w") as handle:
        json.dump({
            "params": {"smooth_window": best_window},
            "source": "tuned by external/tune.py on IDSSE (7 Bundesliga matches, CC BY 4.0), "
                      "degraded by external/degrade.py; the competition match was not used",
            "idsse_mean_spell_f1": round(best_score, 4),
            "matches": [os.path.splitext(os.path.basename(p))[0] for p in paths],
        }, handle, indent=2)
        handle.write("\n")
    print("wrote params.json")


if __name__ == "__main__":
    main()
