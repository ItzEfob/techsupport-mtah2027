"""Per-frame evidence for which team has the ball.

Version 1 is the simplest possible evidence: the team of the player nearest to the ball.
"""

import numpy as np

HOME, AWAY, NOBODY = 0, 1, -1

CHUNK = 20000  # frames handled at a time, to keep memory small


def nearest_team(ball_xy, players_xy, player_team, pitch):
    """Team (0 home, 1 away) of the player closest to the ball, or -1 when there is no answer.

    There is no answer when the ball is missing, or when no player is tracked on that frame.
    Positions are 0-1 on both axes but the pitch is not square, so they are scaled to metres
    first. Otherwise distances across the pitch would count for more than distances along it.
    """
    n_frames = len(ball_xy)
    out = np.full(n_frames, NOBODY, dtype=np.int8)
    scale = np.asarray(pitch, dtype=np.float32)

    for start in range(0, n_frames, CHUNK):
        stop = min(start + CHUNK, n_frames)
        ball = ball_xy[start:stop] * scale                   # (t, 2)
        players = players_xy[start:stop] * scale             # (t, N, 2)

        offset = players - ball[:, None, :]
        distance = np.sum(offset * offset, axis=2)           # (t, N), squared metres
        distance = np.where(np.isnan(distance), np.inf, distance)

        nearest = np.argmin(distance, axis=1)
        found = np.isfinite(distance[np.arange(stop - start), nearest])
        out[start:stop][found] = player_team[nearest[found]]
    return out
