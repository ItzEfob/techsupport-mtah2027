"""Convert IDSSE matches into the plain-array format our method works on.

IDSSE is the public Bundesliga dataset (7 matches, CC BY 4.0) that we tune on, because the rules
forbid tuning on the competition match. kloppy downloads it for us.

    python external/prepare_idsse.py              # all 7 matches
    python external/prepare_idsse.py J03WMX       # just one

Each match becomes one file, data/idsse/<match_id>.npz, holding:

    ball_xy      (T, 2)     ball position per frame, 0-1 pitch coordinates, NaN when missing
    players_xy   (T, N, 2)  position of each of the N squad players, NaN when not on the pitch
    player_team  (N,)       0 = home, 1 = away
    labels       (T,)       the true answer per frame: "H", "A" or "D"
    period       (T,)       1 = first half, 2 = second half
    pitch        (2,)       pitch length and width in metres

T is the number of frames. src/load_atd.py will produce the same arrays (minus `labels`) for the
competition match, so the method in src/ runs unchanged on both.
"""

import os
import sys

import numpy as np
from kloppy import sportec
from kloppy.domain import BallState, Ground

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "data", "idsse")

MATCH_IDS = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")

HOME, AWAY = 0, 1


def convert(dataset):
    """Turn a kloppy tracking dataset into our dictionary of arrays."""
    home_team = next(team for team in dataset.metadata.teams if team.ground == Ground.HOME)
    away_team = next(team for team in dataset.metadata.teams if team.ground == Ground.AWAY)

    # One fixed column per squad player, home first. A player who is on the bench (or was
    # substituted off) simply has NaN in that column for those frames.
    players = list(home_team.players) + list(away_team.players)
    column_of = {player.player_id: i for i, player in enumerate(players)}
    player_team = np.array(
        [HOME] * len(home_team.players) + [AWAY] * len(away_team.players), dtype=np.int8
    )

    frames = dataset.records
    n_frames, n_players = len(frames), len(players)

    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players_xy = np.full((n_frames, n_players, 2), np.nan, dtype=np.float32)
    labels = np.empty(n_frames, dtype="U1")
    period = np.empty(n_frames, dtype=np.int8)

    for t, frame in enumerate(frames):
        period[t] = frame.period.id

        if frame.ball_coordinates is not None:
            ball_xy[t] = (frame.ball_coordinates.x, frame.ball_coordinates.y)

        for player, data in frame.players_data.items():
            if data.coordinates is not None:
                players_xy[t, column_of[player.player_id]] = (data.coordinates.x, data.coordinates.y)

        # The label follows the same recipe the organisers describe for their ground truth:
        # dead ball -> D, otherwise the team the tracking names as owning the ball.
        if frame.ball_state == BallState.DEAD:
            labels[t] = "D"
        elif frame.ball_owning_team is home_team:
            labels[t] = "H"
        elif frame.ball_owning_team is away_team:
            labels[t] = "A"
        else:
            raise ValueError("frame {} is alive but has no owning team".format(frame.frame_id))

    dims = dataset.metadata.pitch_dimensions
    return {
        "ball_xy": ball_xy,
        "players_xy": players_xy,
        "player_team": player_team,
        "labels": labels,
        "period": period,
        "pitch": np.array([dims.pitch_length, dims.pitch_width], dtype=np.float32),
    }


def count_possessions(labels, period):
    """Number of possessions and their mean length in frames.

    A possession is an unbroken run of H or of A. A dead ball or the half-time break ends it.
    """
    lengths = []
    run = 0
    for t in range(len(labels)):
        same_run = (
            t > 0
            and labels[t] == labels[t - 1]
            and period[t] == period[t - 1]
        )
        if labels[t] == "D":
            if run:
                lengths.append(run)
            run = 0
        elif same_run:
            run += 1
        else:
            if run:
                lengths.append(run)
            run = 1
    if run:
        lengths.append(run)
    return len(lengths), (float(np.mean(lengths)) if lengths else 0.0)


def summarise(name, arrays, fps=25):
    labels, period = arrays["labels"], arrays["period"]
    n = len(labels)
    share = {key: float(np.mean(labels == key)) for key in ("H", "A", "D")}
    n_poss, mean_len = count_possessions(labels, period)
    on_pitch = np.mean(np.sum(~np.isnan(arrays["players_xy"][:, :, 0]), axis=1))
    print(
        "{}  frames {:6d}  H {:.3f}  A {:.3f}  D {:.3f}  possessions {:3d} of mean {:4.1f} s  "
        "ball missing {:.3f}  players on pitch {:.1f}".format(
            name, n, share["H"], share["A"], share["D"], n_poss, mean_len / fps,
            float(np.mean(np.isnan(arrays["ball_xy"][:, 0]))), on_pitch,
        )
    )


def main():
    match_ids = sys.argv[1:] or MATCH_IDS
    os.makedirs(OUT_DIR, exist_ok=True)
    for match_id in match_ids:
        print("loading {} ...".format(match_id), flush=True)
        dataset = sportec.load_open_tracking_data(match_id=match_id)
        if dataset.metadata.frame_rate != 25:
            raise ValueError("{}: expected 25 fps, got {}".format(match_id, dataset.metadata.frame_rate))
        arrays = convert(dataset)
        np.savez_compressed(os.path.join(OUT_DIR, match_id + ".npz"), **arrays)
        summarise(match_id, arrays)


if __name__ == "__main__":
    main()
