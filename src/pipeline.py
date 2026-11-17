"""The whole method in one function. tune.py and make_submission.py both call this, so the code
that is tuned on IDSSE is exactly the code that labels the competition match."""

from src.decode import decode
from src.team_evidence import nearest_team


def predict(arrays, params):
    """arrays: the dictionary from load_atd.load() or prepare_idsse.py. Returns one label per frame."""
    codes = nearest_team(arrays["ball_xy"], arrays["players_xy"], arrays["player_team"],
                         arrays["pitch"])
    return decode(codes, arrays["period"], params)
