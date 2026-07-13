"""
train/fit_thresholds.py - fit the detector thresholds LOCALLY (no Colab needed).

Same maths as train_drowsiness.ipynb (Youden's J on the class distributions),
packaged as a plain script so you can train on your own machine. Two steps:

    # 1) extract per-frame features (CPU-heavy; this is the slow part)
    python extract_features.py --videos ../../data/raw_videos --out features.csv

    # 2) fit + write learned_thresholds.json next to config.py (fast)
    python fit_thresholds.py --features features.csv

It fits three thresholds from the feature table:
    MAR_YAWN        Yawning mouth-open vs Singing/Alert/Distracted/Neutral   (greater)
    MMS_YAWN_LEVEL  Yawning lip-gap level vs Singing/Alert/Neutral           (greater)
    EAR_CLOSED      Sleeping/Drowsy eyes vs Alert eyes                       (less)

'Neutral' is the closed-mouth frames annotate_yawns.py freed from the Yawning
clips; using them as negatives pushes the yawn cuts UP (higher precision). Only a
threshold whose class separation J beats --min-j (default 0.5) is written, so a
weak fit can never make the live system worse than its shipped defaults. Delete
learned_thresholds.json to return to those defaults.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
CONFIG_DIR = THIS.parent.parent               # alertVSdrowsy/ (where config.py lives)
OUT_JSON = CONFIG_DIR / "learned_thresholds.json"
FIT_COLS = ("ear", "mar", "lip_gap_level")
DEFAULT_MIN_J = 0.5


def best_threshold(pos, neg, direction="greater"):
    """The cut that best separates two groups by Youden's J (= tpr - fpr).

    'greater' -> positives have LARGER values (MAR for yawning); 'less' -> SMALLER
    (EAR for closed eyes). Returns (None, -1.0) if either group is empty.
    """
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return None, -1.0
    cands = np.unique(np.concatenate([pos, neg]))
    best_t, best_j = None, -1.0
    for t in cands:
        if direction == "greater":
            tpr = (pos >= t).mean(); fpr = (neg >= t).mean()
        else:
            tpr = (pos <= t).mean(); fpr = (neg <= t).mean()
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t, best_j


def load_features(path):
    """Read the extractor CSV -> {col: {label: [values]}} for face-detected rows."""
    by = defaultdict(lambda: defaultdict(list))
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("face_found") != "1":
                continue
            label = row["label"]
            for col in FIT_COLS:
                try:
                    by[col][label].append(float(row[col]))
                except (KeyError, ValueError):
                    pass
    return by


def vals(by, col, labels):
    out = []
    for lab in labels:
        out.extend(by[col].get(lab, []))
    return out


def fit(by, min_j):
    """Return ({NAME: value} to write, [(name, t, j, kept) ...] for the report)."""
    specs = [
        ("MAR_YAWN", "mar", ["Yawning"],
         ["Singing", "Alert", "Distracted", "Neutral"], "greater"),
        ("MMS_YAWN_LEVEL", "lip_gap_level", ["Yawning"],
         ["Singing", "Alert", "Neutral"], "greater"),
        ("EAR_CLOSED", "ear", ["Sleeping", "Drowsy"],
         ["Alert"], "less"),
    ]
    learned, report = {}, []
    for name, col, pos_labels, neg_labels, direction in specs:
        t, j = best_threshold(vals(by, col, pos_labels),
                              vals(by, col, neg_labels), direction)
        kept = t is not None and j > min_j
        if kept:
            learned[name] = round(t, 3)
        report.append((name, t, j, kept))
    return learned, report


def main():
    ap = argparse.ArgumentParser(description="Fit detector thresholds from a feature CSV")
    ap.add_argument("--features", required=True, help="CSV from extract_features.py")
    ap.add_argument("--out", default=str(OUT_JSON),
                    help="where to write learned_thresholds.json (default: next to config.py)")
    ap.add_argument("--min-j", type=float, default=DEFAULT_MIN_J,
                    help="only write a threshold if its separation J beats this (default 0.5)")
    args = ap.parse_args()

    by = load_features(args.features)
    counts = {lab: len(v) for lab, v in by["mar"].items()}
    print("frames per class (face detected):")
    for lab in sorted(counts):
        print(f"  {lab:11} {counts[lab]}")
    if "Neutral" not in counts:
        print("  (no 'Neutral' class -> you haven't run annotate_yawns.py; fitting on "
              "folder labels only)")

    learned, report = fit(by, args.min_j)
    print("\nfitted thresholds:")
    for name, t, j, kept in report:
        if t is None:
            print(f"  {name:15} = (skipped - a class was empty)")
        else:
            note = "" if kept else f"  [DROPPED: J<={args.min_j}, keeping default]"
            print(f"  {name:15} = {t:.3f}  (J={j:.2f}){note}")

    Path(args.out).write_text(json.dumps(learned, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(json.dumps(learned, indent=2))
    if not learned:
        print("(nothing passed the J gate; the live system keeps its shipped defaults)")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    main()
