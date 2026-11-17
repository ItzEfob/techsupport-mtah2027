# Team TechSupport: More than a Hack 2027, Initial Challenge

Per-frame ball possession (`H` / `A` / `D` / `X`) for FC Barcelona vs Athletic Club, built from
the provided automatic tracking only. The method and its limitations are described in
[docs/technical_document.md](docs/technical_document.md).

**Every setting was tuned on external data:** seven public Bundesliga matches (IDSSE, CC BY 4.0).
Nothing was trained, fitted or tuned on the competition match, its video or its ground truth.

## Reproduce the submission

Python 3.14 (tested with 3.14.10). The pipeline is deterministic and runs on CPU in about a
minute.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MTAH_DATA_DIR=/path/to/data      # the folder that contains tactical/
python make_submission.py               # writes outputs/TechSupport.csv
python evaluation/validate_submission.py outputs/TechSupport.csv
```

## Re-run the tuning (optional)

This step needs no competition data. It rewrites `params.json`.

```bash
pip install -r requirements-external.txt
python external/prepare_idsse.py        # downloads and converts the 7 IDSSE matches
python external/tune.py                 # scores each setting with the official metric
```

## Layout

| Path | Contents |
|---|---|
| `make_submission.py` | the one command: competition tracking in, submission CSV out |
| `src/` | the method: tracking loader, per-frame evidence, temporal decoding |
| `external/` | IDSSE preparation, degradation to match our tracking, tuning |
| `params.json` | the tuned settings, read by `make_submission.py` |
| `evaluation/`, `games.py` | copied from the organisers' repository (MIT, Metrica Sports) |
| `docs/` | technical document |

The match data and the submission file are not in this repository: `data/` and `outputs/` are
excluded by `.gitignore`.
