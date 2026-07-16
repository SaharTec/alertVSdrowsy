"""
train/cuts.py - subgroup cuts over a saved eval_events.py report.

Answers "does the detector work equally well for everyone, and does it hold up
when tracking degrades?" - the edge-case half of the evaluation.

This is POST-HOC by design: eval_events.py already stores per-clip `subject`,
`class`, `face_frac` and the matched/GT event counts, so every cut below is a
regroup of numbers that already exist. No video is decoded, so a cut costs a
second rather than the ~10 minutes a re-run would - and, more importantly, the
cuts are guaranteed to describe the exact run they came from.

Both `aggregate` and `bootstrap_ci` are imported from eval_events rather than
reimplemented, so a subgroup's recall is computed the same way as the headline
recall and the CI stays a subject-cluster bootstrap.

Usage:
    python cuts.py --report reports/tuned_cnn083_valb.json
    python cuts.py --report reports/a.json --report reports/b.json   # compare
"""
import argparse
import json
from pathlib import Path

from eval_events import aggregate, bootstrap_ci

# A subgroup this small cannot support a claim. YawDD's val half holds ~3 clips
# per eyewear arm, so print the number but refuse to let it read as a finding.
MIN_SUBJECTS_FOR_CLAIM = 4


def eyewear_of(clip_name):
    """YawDD encodes eyewear in the filename: '11-MaleGlasses-Yawning.avi'.
    NoGlasses must be tested first - 'NoGlasses' contains 'Glasses'."""
    if "NoGlasses" in clip_name:
        return "NoGlasses"
    if "Glasses" in clip_name:
        return "Glasses"
    return "unknown"


def _recall(agg):
    return agg.get("yawn_event_recall")


def _fa_per_min(agg):
    return agg.get("false_yawn_events_per_min")


def describe(name, clips, n_boot):
    """One subgroup's row: size, recall (+CI), false alarms."""
    if not clips:
        return f"  {name:<22} (no clips)"
    agg = aggregate(clips)
    subs = {c["subject"] for c in clips}
    rec, fa = _recall(agg), _fa_per_min(agg)
    ci = bootstrap_ci(clips, _recall, n_boot=n_boot) if len(subs) >= 2 else None

    rec_s = "    n/a" if rec is None else f"{rec*100:5.1f}%"
    fa_s = "  n/a" if fa is None else f"{fa:5.3f}"

    # A cluster bootstrap over subjects who ALL scored identically resamples the
    # same value every time and collapses to a zero-width interval. "CI[100,100]"
    # then reads as near-certainty when it means the exact opposite: too few
    # subjects for any variation to show. Name it rather than print it.
    notes = []
    if ci and ci[0] == ci[1]:
        ci_s = "  CI degenerate"
        notes.append("no variation across these subjects - the CI is an artifact, not confidence")
    elif ci:
        ci_s = f" CI[{ci[0]*100:.0f},{ci[1]*100:.0f}]"
    else:
        ci_s = ""
    if len(subs) < MIN_SUBJECTS_FOR_CLAIM:
        notes.append("too small to claim")

    line = (f"  {name:<22} {len(clips):3d} clips / {len(subs):2d} subj  "
            f"recall {rec_s}{ci_s:<16}  false yawn/min {fa_s}"
            f"  gt={agg['n_gt_events']}")
    for n in notes:
        line += f"\n  {'':<22} ^ {n}"
    return line


def report_cuts(path, n_boot):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    clips = data["clips"]
    man = data.get("manifest", {})
    agg = data.get("aggregate", {})

    print(f"\n{'='*100}\n{path}")
    print(f"  scope={man.get('subjects_scope')}  use_cnn={man.get('use_cnn')}  "
          f"stride={man.get('stride')}  "
          f"YAWN_CNN_PROB={man.get('thresholds', {}).get('YAWN_CNN_PROB')}")
    print("=" * 100)

    # -- Face tracking: is there any dropout to worry about at all? ---------
    # Ask before cutting on it. A "worst-decile" recall over clips that all sit
    # at 100% face detection is theatre - it reports noise as an edge case.
    ffs = sorted(c["face_frac"] for c in clips)
    n_dec = max(1, len(ffs) // 10)
    print(f"\n-- Face tracking --")
    print(f"  face_frac  min={ffs[0]:.3f}  p10={ffs[n_dec]:.3f}  "
          f"median={ffs[len(ffs)//2]:.3f}  max={ffs[-1]:.3f}")
    by_class = {}
    for c in clips:
        by_class.setdefault(c["class"], []).append(c["face_frac"])
    for cls, v in sorted(by_class.items()):
        print(f"    {cls:<10} mean no-face {100*(1-sum(v)/len(v)):5.2f}%  "
              f"worst clip {100*(1-min(v)):5.2f}%")

    worst = sorted(clips, key=lambda c: c["face_frac"])[:n_dec]
    if ffs[n_dec] > 0.98:
        print(f"  Dropout is effectively absent (p10 = {ffs[n_dec]:.3f}): there is no")
        print(f"  degraded-tracking subgroup here to measure. Not a robustness")
        print(f"  result - this dataset simply never stresses face tracking.")
    else:
        print(describe("worst-dropout decile", worst, n_boot))

    # -- Eyewear: the subgroup most likely to break an EAR/mouth detector ---
    print(f"\n-- Eyewear subgroup --")
    groups = {}
    for c in clips:
        groups.setdefault(eyewear_of(c["clip"]), []).append(c)
    for name in ("NoGlasses", "Glasses", "unknown"):
        if name in groups:
            print(describe(name, groups[name], n_boot))

    print(f"\n-- Headline (for reference) --")
    print(f"  recall {agg.get('yawn_event_recall', 0)*100:.1f}%  "
          f"false yawn/min {agg.get('false_yawn_events_per_min', 0):.3f}  "
          f"p95 {agg.get('process_ms_p95', 0):.2f} ms")


def main():
    ap = argparse.ArgumentParser(description="Subgroup cuts over eval_events reports")
    ap.add_argument("--report", action="append", required=True,
                    help="report JSON from eval_events.py --out (repeatable)")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    for r in args.report:
        report_cuts(r, args.n_boot)
    print()


if __name__ == "__main__":
    main()
