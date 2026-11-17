"""The one command: competition data in, submission CSV out.

    python make_submission.py

Reads the tracking from data/ (or MTAH_DATA_DIR), runs the method with the settings in
params.json, and writes outputs/TechSupport.csv. No randomness anywhere, so running it twice
gives the same file byte for byte.
"""

import json
import os

import numpy as np

from src.load_atd import load
from src.pipeline import predict

ROOT = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(ROOT, "params.json")
OUT_PATH = os.path.join(ROOT, "outputs", "TechSupport.csv")


def main():
    with open(PARAMS_PATH) as handle:
        params = json.load(handle)["params"]
    print("settings:", params)

    print("reading the tracking ...")
    arrays = load()
    labels = predict(arrays, params)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))

    counts = {key: int(np.sum(labels == key)) for key in ("H", "A", "D", "X")}
    print("wrote {}  ({} rows)  {}".format(os.path.relpath(OUT_PATH, ROOT), len(labels), counts))
    print("now check it:  python evaluation/validate_submission.py outputs/TechSupport.csv")


if __name__ == "__main__":
    main()
