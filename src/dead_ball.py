"""Dead-ball evidence from the tracking alone.

Version 1: the ball is out of play while its tracked position is outside the pitch. Positions are
0-1 on both axes, so "outside" means below 0 or above 1. `margin` (in the same 0-1 units) asks the
ball to be that far beyond the line first, so small tracking errors on the line do not count.
"""

import numpy as np


def ball_outside(ball_xy, margin):
    """True on frames where the ball is tracked and lies outside the pitch by more than `margin`.

    Frames with no ball position are False: a missing ball tells us nothing about being out.
    """
    x, y = ball_xy[:, 0], ball_xy[:, 1]
    with np.errstate(invalid="ignore"):  # comparisons with NaN are simply False
        return (x < -margin) | (x > 1 + margin) | (y < -margin) | (y > 1 + margin)
