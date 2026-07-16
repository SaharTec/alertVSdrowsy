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
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))          # train/ itself, for subject_of

from build_mouth_dataset import subject_of  # noqa: E402 - after sys.path tweak

CONFIG_DIR = THIS.parent.parent               # alertVSdrowsy/ (where config.py lives)
OUT_JSON = CONFIG_DIR / "learned_thresholds.json"
FIT_COLS = ("ear", "mar", "lip_gap_level")
DEFAULT_MIN_J = 0.5
DEFAULT_VAL_FRAC = 0.2


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


def j_at(pos, neg, t, direction="greater"):
    """Youden's J of a GIVEN cut - used to score a fitted threshold on faces it
    was not fitted on."""
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0 or t is None:
        return -1.0
    if direction == "greater":
        return float((pos >= t).mean() - (neg >= t).mean())
    return float((pos <= t).mean() - (neg <= t).mean())


def load_features(path):
    """Read the extractor CSV -> {col: {label: [(value, subject)]}} for face rows.

    The subject rides along with every value so the fit can hold whole PEOPLE
    out. Without it, a threshold is chosen on the same faces it is then scored
    on, which reports how well the cut memorised this dataset rather than how it
    will behave on a driver it has never seen.
    """
    by = defaultdict(lambda: defaultdict(list))
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("face_found") != "1":
                continue
            label = row["label"]
            subj = subject_of(row.get("video", ""))
            for col in FIT_COLS:
                try:
                    by[col][label].append((float(row[col]), subj))
                except (KeyError, ValueError):
                    pass
    return by


def vals(by, col, labels, subjects=None):
    """Values for `labels`, optionally restricted to a set of subjects."""
    out = []
    for lab in labels:
        for v, s in by[col].get(lab, []):
            if subjects is None or s in subjects:
                out.append(v)
    return out


def all_subjects(by):
    return sorted({s for col in by for lab in by[col] for _, s in by[col][lab]})


def split_subjects(subjects, val_frac, seed):
    """Hold out val_frac of PEOPLE. Returns (train, held_out) as sets."""
    subs = sorted(subjects)
    random.Random(seed).shuffle(subs)
    cut = max(1, int(len(subs) * val_frac))
    return set(subs[cut:]), set(subs[:cut])


SPECS = [
    ("MAR_YAWN", "mar", ["Yawning"],
     ["Singing", "Alert", "Distracted", "Neutral"], "greater"),
    ("MMS_YAWN_LEVEL", "lip_gap_level", ["Yawning"],
     ["Singing", "Alert", "Neutral"], "greater"),
    ("EAR_CLOSED", "ear", ["Sleeping", "Drowsy"],
     ["Alert"], "less"),
]


def fit(by, min_j, train_subs=None, val_subs=None):
    """Fit each threshold on `train_subs`, score it on `val_subs`.

    Returns ({NAME: value} to write, [(name, t, j_train, j_val, kept) ...]).
    The --min-j gate is applied to the HELD-OUT J, not the fitted one: the fitted
    J always looks good on the faces that chose it, so gating on it would let a
    threshold that generalises badly overwrite a shipped default.
    """
    learned, report = {}, []
    for name, col, pos_labels, neg_labels, direction in SPECS:
        t, j_train = best_threshold(vals(by, col, pos_labels, train_subs),
                                    vals(by, col, neg_labels, train_subs), direction)
        if val_subs is None:
            j_val = j_train                     # no split: honest only for a smoke test
        else:
            j_val = j_at(vals(by, col, pos_labels, val_subs),
                         vals(by, col, neg_labels, val_subs), t, direction)
        kept = t is not None and j_val > min_j
        if kept:
            learned[name] = round(t, 3)
        report.append((name, t, j_train, j_val, kept))
    return learned, report


def main():
    ap = argparse.ArgumentParser(description="Fit detector thresholds from a feature CSV")
    ap.add_argument("--features", required=True, help="CSV from extract_features.py")
    ap.add_argument("--out", default=str(OUT_JSON),
                    help="where to write learned_thresholds.json (default: next to config.py)")
    ap.add_argument("--min-j", type=float, default=DEFAULT_MIN_J,
                    help="only write a threshold if its HELD-OUT separation J beats "
                         "this (default 0.5)")
    ap.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                    help="fraction of SUBJECTS held out to score the fit (default 0.2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-split", action="store_true",
                    help="fit on everything and report the fitted J (the old, "
                         "optimistic behaviour). Smoke tests only - the number it "
                         "prints is not a generalisation estimate.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the fit but do NOT write learned_thresholds.json. "
                         "config.py applies that file at import, so writing it "
                         "changes the live system: check the event-level harness "
                         "first.")
    args = ap.parse_args()

    by = load_features(args.features)
    counts = {lab: len(v) for lab, v in by["mar"].items()}
    print("frames per class (face detected):")
    for lab in sorted(counts):
        print(f"  {lab:11} {counts[lab]}")
    if "Neutral" not in counts:
        print("  (no 'Neutral' class -> you haven't run annotate_yawns.py; fitting on "
              "folder labels only)")

    subs = all_subjects(by)
    if args.no_split or len(subs) < 2:
        train_subs = val_subs = None
        print(f"\nsubjects: {len(subs)}  -> NO SPLIT: fitted J is optimistic, not a "
              f"generalisation estimate.")
    else:
        train_subs, val_subs = split_subjects(subs, args.val_frac, args.seed)
        print(f"\nsubjects: {len(subs)} total -> fit on {len(train_subs)}, "
              f"score on {len(val_subs)} held-out ({sorted(val_subs)})")

    learned, report = fit(by, args.min_j, train_subs, val_subs)
    print("\nfitted thresholds:")
    for name, t, j_train, j_val, kept in report:
        if t is None:
            print(f"  {name:15} = (skipped - a class was empty)")
        else:
            note = "" if kept else f"  [DROPPED: held-out J<={args.min_j}, keeping default]"
            print(f"  {name:15} = {t:.3f}  (J fit={j_train:.2f}, held-out={j_val:.2f})"
                  f"{note}")

    if args.dry_run:
        print("\n--dry-run: nothing written. Would have written:")
        print(json.dumps(learned, indent=2))
        return

    Path(args.out).write_text(json.dumps(learned, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(json.dumps(learned, indent=2))
    if not learned:
        print("(nothing passed the J gate; the live system keeps its shipped defaults)")
    else:
        print("\nNOTE: config.py applies this file at import, so the live system just "
              "changed. These cuts were fitted per-FRAME, but the yawn cue integrates "
              "over YAWN_MIN_SECONDS - confirm at the event level before trusting it:")
        print("  python eval_events.py --subjects val-b")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    main()
