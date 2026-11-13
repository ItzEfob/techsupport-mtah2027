"""Make clean IDSSE tracking look like the automatic tracking we get for the competition match.

IDSSE sees the ball and all 22 players on every frame. The competition tracking does not: the ball
is missing on 41% of graded frames and about 20.5 of 22 players are tracked on average (both
figures are published in the organisers' DATA.md). A method tuned on clean data would fall apart
on the real match, so we hide parts of the IDSSE tracking before tuning on it.

    python external/degrade.py        # degrade every prepared match once and print what changed

Things go missing in stretches, not as isolated frames: a tracker that loses the ball loses it for
a while. So each gap has a random length around `mean_gap` frames.

The labels are never touched. Only the inputs get worse; the true answer stays the same.

Nothing is saved to disk: tune.py calls degrade() in memory, so it can try several severities.
"""

import glob
import os
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IDSSE_DIR = os.path.join(ROOT, "data", "idsse")

# Published by the organisers in DATA.md.
BALL_MISSING = 0.41            # share of graded frames without a ball position
PLAYER_MISSING = 1 - 20.5 / 22  # 20.5 tracked objects on average, out of 22

# Our own assumptions: how long a typical gap lasts, in frames (25 frames = 1 second). We do not
# know the real value, so tune.py should try several and keep settings that work across them.
BALL_MEAN_GAP = 30
PLAYER_MEAN_GAP = 50


def gap_mask(n_frames, missing_rate, mean_gap, rng):
    """Boolean array, True where the object is hidden.

    Built as alternating stretches: visible, hidden, visible, hidden... Each stretch length is
    drawn at random (a geometric distribution, the "keep going with fixed probability" one). The
    mean visible length is chosen so that the overall share of hidden frames is `missing_rate`.
    """
    if missing_rate <= 0:
        return np.zeros(n_frames, dtype=bool)
    mean_visible = mean_gap * (1 - missing_rate) / missing_rate

    mask = np.zeros(n_frames, dtype=bool)
    t = 0
    hidden = rng.random() < missing_rate  # start hidden or visible in the right proportion
    while t < n_frames:
        mean_length = mean_gap if hidden else mean_visible
        length = rng.geometric(1.0 / max(mean_length, 1.0))
        if hidden:
            mask[t:t + length] = True
        t += length
        hidden = not hidden
    return mask


def degrade(arrays, name="", seed=0, ball_missing=BALL_MISSING, ball_mean_gap=BALL_MEAN_GAP,
            player_missing=PLAYER_MISSING, player_mean_gap=PLAYER_MEAN_GAP):
    """Return a degraded copy of one match's arrays. The original is left unchanged.

    `name` and `seed` fix the random numbers: the same match with the same seed always loses the
    same frames, so every tuning run is repeatable.
    """
    rng = np.random.default_rng([zlib.crc32(name.encode()), seed])

    ball_xy = arrays["ball_xy"].copy()
    players_xy = arrays["players_xy"].copy()
    n_frames, n_players = players_xy.shape[:2]

    ball_xy[gap_mask(n_frames, ball_missing, ball_mean_gap, rng)] = np.nan

    # Each player loses their own stretches, independently of the others.
    for p in range(n_players):
        players_xy[gap_mask(n_frames, player_missing, player_mean_gap, rng), p] = np.nan

    out = dict(arrays)
    out["ball_xy"] = ball_xy
    out["players_xy"] = players_xy
    return out


def tracked_per_frame(players_xy):
    return float(np.mean(np.sum(~np.isnan(players_xy[:, :, 0]), axis=1)))


def main():
    paths = sorted(glob.glob(os.path.join(IDSSE_DIR, "*.npz")))
    if not paths:
        raise SystemExit("no matches in {} -- run external/prepare_idsse.py first".format(IDSSE_DIR))
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        clean = dict(np.load(path))
        noisy = degrade(clean, name=name)
        print(
            "{}  ball missing {:.3f} -> {:.3f}   players tracked {:.1f} -> {:.1f}".format(
                name,
                float(np.mean(np.isnan(clean["ball_xy"][:, 0]))),
                float(np.mean(np.isnan(noisy["ball_xy"][:, 0]))),
                tracked_per_frame(clean["players_xy"]),
                tracked_per_frame(noisy["players_xy"]),
            )
        )


if __name__ == "__main__":
    main()
