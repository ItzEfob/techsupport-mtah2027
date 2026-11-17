"""Turn noisy per-frame evidence into final labels.

Version 1 is a majority vote: each frame takes the team seen most often in the frames around it.
"""

import numpy as np

HOME, AWAY = 0, 1


def majority_vote(codes, window):
    """Labels for one stretch of play. `window` is the number of frames looked at on each side.

    D where nobody was seen in the whole window. Ties go to the home side.
    Uses running totals, so a wide window costs the same as a narrow one.
    """
    home_total = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_total = np.concatenate(([0], np.cumsum(codes == AWAY)))

    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_total[high] - home_total[low]
    away = away_total[high] - away_total[low]

    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def merge_short_blips(labels, min_len):
    """Remove possessions shorter than `min_len` frames that sit inside the other team's possession.

    "A A A H A A A" with a one-frame H becomes all A: a short spell of one team, with the other
    team on both sides, is far more likely to be a deflection or a tracking slip than a real change
    of possession. Only that case is merged. A short spell next to a dead ball is kept, because a
    quick possession right after a restart is common in real football.
    """
    if min_len <= 1:
        return labels
    labels = labels.copy()
    # Run-length encode: start index, length and label of each constant stretch.
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [len(labels)])))
    values = labels[starts]
    for i in range(1, len(starts) - 1):
        if (values[i] in ("H", "A") and lengths[i] < min_len
                and values[i - 1] == values[i + 1] and values[i - 1] in ("H", "A")
                and values[i - 1] != values[i]):
            labels[starts[i]:starts[i] + lengths[i]] = values[i - 1]
            values[i] = values[i - 1]  # so a chain of blips keeps merging into the same team
    return labels


def decode(codes, period, params):
    """Final label per frame: H, A, D, or X outside the two halves.

    Each half is decoded on its own, so nothing from the warm-up or half-time leaks into play.
    """
    labels = np.full(len(codes), "X", dtype="<U1")
    for half in (1, 2):
        inside = period == half
        if inside.any():
            voted = majority_vote(codes[inside], params["smooth_window"])
            labels[inside] = merge_short_blips(voted, params.get("min_possession", 0))
    return labels
