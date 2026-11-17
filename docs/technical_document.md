# More than a Hack 2027, Initial Challenge: Team TechSupport

## 1. Approach

Our submission labels every frame of BAR-ATH with one of `H`, `A`, `D` or `X` in three steps:

1. **Per-frame evidence.** When the tracking has a ball position, we find the nearest tracked
   player, measuring distance in metres so the pitch's length and width are weighted correctly.
   That player's team is the evidence for the frame. When the ball is missing or no player is
   tracked, the frame has no evidence.
2. **Temporal decoding.** Each half is decoded on its own. Every frame takes the team seen most
   often in a centred window of ±75 frames (±3 s). A frame whose whole window has no evidence is
   labelled `D`.
3. **Outside play.** Frames outside the two graded halves (before kickoff, half-time, after the
   final whistle) are labelled `X`.

We chose this method because it is simple, fully deterministic and easy to verify. Most of our
effort went into building a validation set that follows the rules (section 3), so that every
setting comes from external data.

## 2. Assets used

| Asset | Used | How |
|---|---|---|
| Tactical ATD (tracking) | Yes | The only input to the submission pipeline |
| Smart Tagging | No | Planned for dead-ball detection; not finished before the deadline |
| Homography | No | Not needed, since we work in pitch coordinates |
| Tactical / broadcast video | No | Not used |
| Broadcast ATD | No | Not aligned to the tactical timeline |

## 3. Training and tuning data

**Nothing in the method was trained, fitted or tuned on the competition match, its video or its
ground truth.** The single setting (the window width) was chosen on external data only.

- **Dataset:** IDSSE, *An integrated dataset of spatiotemporal and event data in elite soccer*
  (Bassek, Rein, Weber, Memmert, *Scientific Data* 12, 195, 2025,
  https://doi.org/10.1038/s41597-025-04505-y). Seven full German Bundesliga matches, CC BY 4.0,
  loaded with `kloppy.sportec.load_open_tracking_data`.
- **Labels:** taken directly from the dataset's own per-frame fields. `D` where `ball_state` is
  dead, otherwise `H` or `A` from `ball_owning_team`. This is the same recipe the organisers
  describe for their ground truth. **No labels were made by hand.**
- **Matching the competition data:** IDSSE tracking is complete, ours is not. Before tuning,
  `external/degrade.py` hides the ball on 41% of frames and leaves about 20.5 of 22 players
  tracked, which are the rates published in the organisers' DATA.md. Gaps come in stretches of
  random length. Because the true gap length is unknown, we tuned over three mean lengths (10, 30
  and 75 frames) and averaged the results.
- **Scoring:** the organisers' own `possession_metrics.py`, ranked on `spellF1`.

Mean `spellF1` over 7 matches × 3 gap lengths:

| Window (± frames) | 0 | 25 | 50 | **75** | 100 | 125 | 250 |
|---|---|---|---|---|---|---|---|
| spellF1 | 0.027 | 0.165 | 0.213 | **0.219** | 0.206 | 0.191 | 0.121 |

The window of 75 frames was selected. The choice is stable: 50 and 75 score almost the same.

## 4. Possession rules implemented

- A possession is the team the evidence favours over a ±3 s window. Brief touches, deflections and
  duels shorter than the window are absorbed into the surrounding possession, which is how the
  method follows the "controlled possession" rule.
- `D` is predicted only when the ball has been invisible for the entire window. We have no
  explicit detection of out-of-play balls or restarts.
- Halves are decoded separately, so no possession crosses half-time.

## 5. Limitations

- **Dead balls are the main weakness.** The method only predicts `D` during long tracking gaps.
  It misses most stoppages and over-predicts `D` during long occlusions in open play.
- **Hand-overs are blurred.** A majority vote moves each transition by up to the window size.
- **The degraded IDSSE data is only an approximation.** Our gaps are placed at random, while the
  real tracker loses the ball most often in crowded, contested moments. Our IDSSE scores are
  therefore probably optimistic.
- **The labelling conventions may differ.** IDSSE possessions average 5.5–9.2 s, against 14.2 s
  in the competition's public window. A window tuned on IDSSE may switch teams too readily on
  BAR-ATH.

**With more time we would add:** dead-ball detection from the ball leaving the pitch and from
Smart Tagging restarts; a switching model (HMM / Viterbi) with a cost for changing team in place
of the majority vote; and an explicit `X` during occlusions inside a possession.

## 6. Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MTAH_DATA_DIR=/path/to/data      # the folder that contains tactical/
python make_submission.py               # writes outputs/TechSupport.csv
python evaluation/validate_submission.py outputs/TechSupport.csv
```

The pipeline is deterministic and runs on CPU in about a minute. The committed `params.json`
holds the tuned setting. To re-run the tuning, which needs no competition data:

```bash
pip install -r requirements-external.txt
python external/prepare_idsse.py        # downloads and converts the 7 IDSSE matches
python external/tune.py                 # rewrites params.json
```

`games.py` and `evaluation/` are copied from the organisers' repository (MIT licence, Metrica
Sports). Their licence is kept in `evaluation/LICENSE-metrica`.
