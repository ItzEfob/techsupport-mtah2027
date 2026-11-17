"""The whole method in one function. tune.py and make_submission.py both call this, so the code
that is tuned on IDSSE is exactly the code that labels the competition match."""

from src.dead_ball import ball_outside
from src.decode import decode
from src.team_evidence import NOBODY, nearest_team


def predict(arrays, params):
    """arrays: the dictionary from load_atd.load() or prepare_idsse.py. Returns one label per frame.

    params:
        smooth_window   frames looked at on each side by the majority vote
        out_margin      dead-ball rule: ball beyond the pitch by this much (0-1 units) means D.
                        None switches the rule off.
    """
    codes = nearest_team(arrays["ball_xy"], arrays["players_xy"], arrays["player_team"],
                         arrays["pitch"])

    margin = params.get("out_margin")
    if margin is None:
        return decode(codes, arrays["period"], params)

    # A ball outside the pitch is not evidence for either team, so it does not vote...
    outside = ball_outside(arrays["ball_xy"], margin)
    codes = codes.copy()
    codes[outside] = NOBODY
    labels = decode(codes, arrays["period"], params)
    # ...and those frames are dead, whatever the vote around them says.
    labels[outside & (labels != "X")] = "D"
    return labels
