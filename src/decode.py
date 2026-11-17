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


def decode(codes, period, params):
    """Final label per frame: H, A, D, or X outside the two halves.

    Each half is decoded on its own, so nothing from the warm-up or half-time leaks into play.
    """
    labels = np.full(len(codes), "X", dtype="<U1")
    for half in (1, 2):
        inside = period == half
        if inside.any():
            labels[inside] = majority_vote(codes[inside], params["smooth_window"])
    return labels
