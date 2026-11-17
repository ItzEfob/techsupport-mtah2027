"""Read the competition tracking (ATD) into the same plain arrays prepare_idsse.py produces.

    ball_xy      (T, 2)     ball position per frame, 0-1 pitch coordinates, NaN when missing
    players_xy   (T, N, 2)  position of each of the N track slots, NaN when the slot is empty
    player_team  (N,)       0 = home, 1 = away
    period       (T,)       1 = first half, 2 = second half, 0 = not graded (warm-up, half-time...)
    pitch        (2,)       pitch length and width in metres

T is 199319, one entry per video frame, so entry i is row i of the submission.

The parsing follows the organisers' starter/baseline.py (MIT licence, Metrica Sports).
"""

import xml.etree.ElementTree as ET

import numpy as np

from games import GAME, GAMES, half_video_blocks, require, source_frames

HOME, AWAY = 0, 1


def team_of_each_slot(meta_path):
    """0/1 per track slot. The first team listed in the metadata is the home side."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if team_id == team_ids[0] else AWAY for team_id in slots], dtype=np.int8)


def load(game=GAME, data_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    player_team = team_of_each_slot(meta_path)
    n_frames, n_slots = source_frames("tactical", game), len(player_team)

    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players_xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)

    # One line per frame: "<frame_id>:<x,y,speed>;<x,y,speed>;...:<ball x,y,speed>".
    # A missing object is written as NaN, which float() reads as NaN, so no special case is needed.
    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            t = int(parts[0]) - 1  # the file counts frames from 1, a submission counts from 0

            ball = parts[2].split(",")
            ball_xy[t] = (float(ball[0]), float(ball[1]))

            groups = parts[1].split(";")
            if len(groups) != n_slots:
                raise ValueError("frame {}: {} player groups, expected {}".format(
                    parts[0], len(groups), n_slots))
            for slot, group in enumerate(groups):
                if group[0] != "N":  # skip empty slots quickly; most of the 46 are empty
                    values = group.split(",")
                    players_xy[t, slot] = (float(values[0]), float(values[1]))

    period = np.zeros(n_frames, dtype=np.int8)
    for half, first, last in half_video_blocks(game):
        period[first:last + 1] = half

    return {
        "ball_xy": ball_xy,
        "players_xy": players_xy,
        "player_team": player_team,
        "period": period,
        "pitch": np.array(GAMES[game]["pitch"], dtype=np.float32),
    }
