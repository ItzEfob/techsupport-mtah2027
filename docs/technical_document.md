# More than a Hack 2027, Initial Challenge: Team TechSupport

## 1. Approach

Our submission labels every frame of BAR-ATH with one of `H`, `A`, `D` or `X` in four steps:

1. **Per-frame evidence.** When the tracking has a ball position, we find the nearest tracked
   player, measuring distance in metres so the pitch's length and width are weighted correctly.
   That player's team is the evidence for the frame. When the ball is missing or no player is
   tracked, the frame has no evidence.
2. **Dead-ball rule.** When the tracked ball lies outside the pitch (x or y below 0 or above 1 in
   normalised coordinates), the frame is a dead ball. Such frames give no team evidence and are
   labelled `D` whatever the decoding below says.
3. **Temporal decoding.** Each half is decoded on its own. Every frame takes the team seen most
   often in a centred window of ±50 frames (±2 s). A frame whose whole window has no evidence is
   labelled `D`.
4. **Outside play.** Frames outside the two graded halves (before kickoff, half-time, after the
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
ground truth.** Both settings (the window width and the dead-ball margin) were chosen on external
data only.

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

Mean over 7 matches × 3 gap lengths, best window for each dead-ball setting:

| Dead-ball rule | Best window | spellF1 | worst match | macroF1 |
|---|---|---|---|---|
| off | ±75 | 0.219 | 0.180 | 0.380 |
| **ball outside the pitch (margin 0)** | **±50** | **0.243** | **0.216** | **0.664** |
| outside by more than 0.01 | ±75 | 0.225 | 0.202 | 0.485 |
| outside by more than 0.03 | ±75 | 0.219 | 0.181 | 0.391 |

We selected the rule with margin 0 and a ±50 window. It beats the rule-off version at every gap
length and on the worst match, and it raises macroF1 from 0.38 to 0.66.

We also ran one automatic check on the provided tracking, which the organisers confirmed is
allowed: the BAR-ATH ball is outside the pitch on 3,949 of 158,857 graded frames, so the rule
has something to act on in the competition match.

## 4. Possession rules implemented

- A possession is the team the evidence favours over a ±3 s window. Brief touches, deflections and
  duels shorter than the window are absorbed into the surrounding possession, which is how the
  method follows the "controlled possession" rule.
- `D` is predicted when the tracked ball is outside the pitch, or when the ball has been
  invisible for the entire window. Restarts are not detected explicitly.
- Halves are decoded separately, so no possession crosses half-time.

## 5. Limitations

- **Dead balls are still the main weakness.** The out-of-pitch rule only fires when the tracker
  follows the ball past the line, which covers about 2.5% of graded frames. Fouls, offsides and
  other stoppages inside the pitch are missed.
- **The dead-ball rule is sensitive to its margin.** On IDSSE, a margin of 0 helps clearly while
  0.03 does nothing, so the gain depends on where each tracker places a ball just past the line.
- **Hand-overs are blurred.** A majority vote moves each transition by up to the window size.
- **The degraded IDSSE data is only an approximation.** Our gaps are placed at random, while the
  real tracker loses the ball most often in crowded, contested moments. Our IDSSE scores are
  therefore probably optimistic.
- **The labelling conventions may differ.** IDSSE possessions average 5.5–9.2 s, against 14.2 s
  in the competition's public window. A window tuned on IDSSE may switch teams too readily on
  BAR-ATH.

**With more time we would add:** dead-ball detection from Smart Tagging restarts and from all
players slowing down together; a switching model (HMM / Viterbi) with a cost for changing team in place
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
