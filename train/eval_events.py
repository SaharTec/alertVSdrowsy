"""
train/eval_events.py - measure what the live system actually DOES, per EVENT.

evaluate_yawn.py scores independent mouth crops, one frame at a time. The live
system never decides per frame: a yawn must be sustained YAWN_MIN_SECONDS before
the cue fires, and fusion re-decides once per ~1.5 s window. So a frame-level F1
says nothing about whether a yawn was caught, how late the alarm was, or how
often talking triggers a false one. This script measures those directly.

It drives the SAME VideoModel + FusionEngine main.py runs, on a video clock
(now = frame/fps), so a 2 s yawn in the file is measured as 2 s no matter how
fast the machine decodes. Ground truth is train/yawn_segments.json.

Two levels, because they answer different questions:

  Level A - the yawn CUE. Predicted events are maximal runs of the sustained
            `yawning` flag, matched to annotated segments by ANY overlap.
            Reports event recall, mean IoU, onset latency, and false yawn
            events/min on the negative classes.
  Level B - the FUSED state. A DROWSY verdict can come from eyes, nod OR yawn,
            so recall alone would let the eyes cue take credit for yawn
            detection. Every DROWSY window is attributed to its dominant cue via
            FusionResult.cue_fracs, and both numbers are reported: DROWSY from
            any cue (what the system really does) and DROWSY attributed to the
            yawn cue (what the yawn pathway does).

Negative classes (Singing = talking, Alert) need no annotation: they contain no
yawns, so every predicted yawn event there is a false alarm. That is what makes
the precision side cheap and honest.

AUDIO IS NOT EXERCISED. 0 of 117 Yawning clips have an audio track (Alert 10/114,
Singing 11/105), so audio_state is always "silence" here - stated rather than
hidden, because it bounds what this harness can claim.

  # baseline, MAR rule, held-out subjects
  python eval_events.py --subjects val-b
  # the CNN path at a candidate threshold
  python eval_events.py --subjects val-b --use-cnn --yawn-cnn-prob 0.35
  # smoke test (a few clips, no split)
  python eval_events.py --max-clips-per-class 3 --subjects all
  # no args beyond defaults -> full run on every annotated clip
"""
import argparse
import json
import random
import statistics as st
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

THIS = Path(__file__).resolve()
PARENT = THIS.parent.parent                  # alertVSdrowsy/ (runtime modules)
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(THIS.parent))         # train/ itself

from video_model import VideoModel  # noqa: E402 - after sys.path tweak
from fusion import FusionEngine  # noqa: E402
from config import (  # noqa: E402
    STATE_DROWSY, WEIGHT_EYES, WEIGHT_YAWN, WEIGHT_NOD,
    MAR_YAWN, YAWN_CNN_PROB, YAWN_MIN_SECONDS, EAR_CLOSED, EYE_CLOSED_SECONDS,
    MMS_YAWN_LEVEL, MMS_SPEECH_AMP_MAX, MMS_SPEECH_OSC_RATE_MIN, MMS_WINDOW_SECONDS,
    FUSION_INTERVAL_SECONDS,
)
from annotate_yawns import (  # noqa: E402 - the SAME label logic training used
    load_annotations, clip_key, yawn_ranges, ANN_PATH,
)
from build_mouth_dataset import subject_of  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
POS_CLASS = "Yawning"
DEFAULT_NEG_CLASSES = ["Singing", "Alert"]   # talking + normal driving: no yawns
DEF_VIDEOS = PARENT.parent / "data" / "raw_videos"


# ---------------------------------------------------------------------------
# Subject split - reuse the CNN's own split so we never report on a face the
# model trained on.
# ---------------------------------------------------------------------------
def _import_train_split():
    """Import train_yawn_cnn's split helpers past _tf_guard.

    train_yawn_cnn imports torch, and torch._dynamo probes find_spec("tensorflow")
    at import time, which _tf_guard's finder raises on (by design, to keep
    MediaPipe importable). MediaPipe is already loaded by now, so lift the guard
    for the import and restore it - the same dance yawn_cnn_infer.py does.
    """
    guards = [f for f in sys.meta_path if type(f).__name__ == "_BlockTensorFlow"]
    for g in guards:
        sys.meta_path.remove(g)
    try:
        from train_yawn_cnn import (subject_split, halve_val_subjects, load_manifest,
                                    DEF_MANIFEST, DEF_CROPS)
        return subject_split, halve_val_subjects, load_manifest, DEF_MANIFEST, DEF_CROPS
    finally:
        for g in guards:
            sys.meta_path.insert(0, g)


def held_out_subjects(val_frac=0.2, seed=0):
    """The subjects the yawn CNN never trained on, split into halves A and B.

    Reuses the CNN's own subject_split and halve_val_subjects, so the faces this
    harness reports on are exactly the ones the model never saw, and the A/B
    halves match evaluate_yawn.py's sweep. Returns (val_a, val_b) as sets.
    """
    subject_split, halve_val_subjects, load_manifest, DEF_MANIFEST, DEF_CROPS = \
        _import_train_split()
    rows = load_manifest(DEF_MANIFEST, DEF_CROPS)
    _, va = subject_split(rows, val_frac, seed)
    return halve_val_subjects(va, seed)


# ---------------------------------------------------------------------------
# Running one clip through the real pipeline
# ---------------------------------------------------------------------------
def run_clip(path, use_cnn=False, stride=1, max_frames=0):
    """Drive VideoModel + FusionEngine over one clip on a video clock.

    Mirrors main.py's per-frame call order. Returns a dict with the per-frame
    yawn-cue trace, the closed fusion windows, and per-frame process() timings.
    Fresh models per clip so streaks/baselines never leak across clips.
    """
    vm = VideoModel(use_cnn=use_cnn)
    fe = FusionEngine()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0

    yawn_flags, times, faces, timings = [], [], [], []
    windows = []                    # (t_start, t_end, state, cue_fracs)
    last_close = 0.0
    i = 0
    try:
        while True:
            ok = cap.grab()         # decode only the frames we actually process
            if not ok:
                break
            if max_frames and i >= max_frames:
                break
            if i % stride:
                i += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = i / fps             # video clock: duration rules measure clip seconds

            t0 = time.perf_counter()
            out = vm.process(frame, now=t)
            timings.append((time.perf_counter() - t0) * 1000.0)

            r = fe.update(out, audio_state="silence", now=t)
            if r.is_new:
                windows.append((last_close, t, r.state, dict(r.cue_fracs)))
                last_close = t

            yawn_flags.append(bool(out["yawning"]))
            faces.append(bool(out["face_found"]))
            times.append(t)
            i += 1
    finally:
        cap.release()
        vm.close()

    n = max(1, len(times))
    return {
        "fps": fps,
        "n_frames": i,
        "duration": (times[-1] + 1.0 / fps) if times else 0.0,
        "times": times,
        "yawn_flags": yawn_flags,
        "windows": windows,
        "face_frac": sum(faces) / n,
        "timings_ms": timings,
        "step": stride / fps,
    }


# ---------------------------------------------------------------------------
# Events + matching
# ---------------------------------------------------------------------------
def events_from_flags(flags, times, step):
    """Maximal runs of True -> [(t_start, t_end)] in seconds.

    t_end extends one sample past the last True frame: the cue was still up for
    that sample's duration, and a 1-frame event would otherwise have zero length.
    """
    events, start = [], None
    for k, f in enumerate(flags):
        if f and start is None:
            start = times[k]
        elif not f and start is not None:
            events.append((start, times[k - 1] + step))
            start = None
    if start is not None:
        events.append((start, times[-1] + step))
    return events


def gt_segments(ann, fps):
    """Annotated yawn spans as [(t_start, t_end)] seconds. Frames are inclusive."""
    return [(s / fps, (e + 1) / fps) for s, e in yawn_ranges(ann)]


def _overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _iou(a, b):
    inter = _overlap(a, b)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match_events(gt, pred):
    """Match on ANY overlap; return (n_matched, ious, latencies).

    Any-overlap rather than IoU>=0.5 on purpose: the cue only fires after
    YAWN_MIN_SECONDS, so a correct detection structurally starts ~1.2 s late and
    can never reach IoU 0.5 on a short yawn. Requiring it would measure the
    duration gate, not the detector. IoU is reported alongside instead, and the
    lateness is the latency number.
    """
    matched, ious, lats = 0, [], []
    for g in gt:
        hits = [p for p in pred if _overlap(g, p) > 0]
        if not hits:
            continue
        matched += 1
        best = max(hits, key=lambda p: _overlap(g, p))
        ious.append(_iou(g, best))
        lats.append(min(h[0] for h in hits) - g[0])    # first cue onset vs truth
    return matched, ious, lats


def drowsy_windows(windows, attributed_to=None):
    """DROWSY windows, optionally only those whose dominant drowsy cue is `X`.

    A DROWSY verdict can come from eyes, nod or yawn. Weighting each cue fraction
    by the same WEIGHT_* fusion uses recovers which one actually carried the
    window - that is what cue_fracs was added for.
    """
    out = []
    for (t0, t1, state, cf) in windows:
        if state != STATE_DROWSY:
            continue
        if attributed_to is not None and dominant_cue(cf) != attributed_to:
            continue
        out.append((t0, t1))
    return out


def dominant_cue(cf):
    """Which drowsy cue carried this window, or None if none contributed."""
    contrib = {
        "eyes": WEIGHT_EYES * cf.get("eyes", 0.0),
        "yawn": WEIGHT_YAWN * cf.get("yawn", 0.0),
        "nod": WEIGHT_NOD * cf.get("nod", 0.0),
    }
    top = max(contrib, key=contrib.get)
    return top if contrib[top] > 0 else None


# ---------------------------------------------------------------------------
# Scoring a set of clips
# ---------------------------------------------------------------------------
def score_clip(label, path, ann, use_cnn, stride, max_frames):
    """Run one clip and reduce it to the per-clip numbers the report needs."""
    res = run_clip(path, use_cnn=use_cnn, stride=stride, max_frames=max_frames)
    pred = events_from_flags(res["yawn_flags"], res["times"], res["step"])
    gt = gt_segments(ann, res["fps"]) if ann else []

    matched, ious, lats = match_events(gt, pred)
    d_any = drowsy_windows(res["windows"])
    d_yawn = drowsy_windows(res["windows"], attributed_to="yawn")

    return {
        "clip": path.name,
        "class": label,
        "subject": subject_of(path.name),
        "fps": round(res["fps"], 2),
        "duration": round(res["duration"], 2),
        "face_frac": round(res["face_frac"], 3),
        "n_gt": len(gt),
        "n_pred": len(pred),
        "matched": matched,
        "ious": ious,
        "latencies": lats,
        "gt": gt,
        "drowsy_any": [t for t in d_any],
        "drowsy_yawn": [t for t in d_yawn],
        "drowsy_by_cue": Counter(dominant_cue(cf) for (_, _, s, cf) in res["windows"]
                                 if s == STATE_DROWSY),
        "timings_ms": res["timings_ms"],
    }


def aggregate(clips):
    """Roll per-clip results into the headline metrics."""
    pos = [c for c in clips if c["class"] == POS_CLASS]
    neg = [c for c in clips if c["class"] != POS_CLASS]

    n_gt = sum(c["n_gt"] for c in pos)
    n_matched = sum(c["matched"] for c in pos)
    ious = [x for c in pos for x in c["ious"]]
    lats = [x for c in pos for x in c["latencies"]]

    neg_secs = sum(c["duration"] for c in neg)
    neg_events = sum(c["n_pred"] for c in neg)

    # Level B: a GT yawn counts as detected if any DROWSY window overlaps it.
    def _drowsy_recall(key):
        hit = 0
        for c in pos:
            for g in c["gt"]:
                if any(_overlap(g, w) > 0 for w in c[key]):
                    hit += 1
        return hit

    by_cue = Counter()
    for c in neg:
        by_cue.update(c["drowsy_by_cue"])
    # .get: per-frame timings are dropped when a report is written (they dwarf
    # everything else in the JSON), so a clip reloaded from a saved report has
    # none. Every other metric survives the round-trip, which is what lets
    # cuts.py regroup a finished run instead of re-decoding the videos.
    timings = [t for c in clips for t in c.get("timings_ms", [])]

    return {
        "n_clips": len(clips),
        "n_pos_clips": len(pos),
        "n_neg_clips": len(neg),
        "n_subjects": len({c["subject"] for c in clips}),
        "yawn_event_recall": n_matched / n_gt if n_gt else None,
        "n_gt_events": n_gt,
        "n_matched": n_matched,
        "mean_iou": round(st.mean(ious), 3) if ious else None,
        "latency_median": round(st.median(lats), 2) if lats else None,
        "latency_iqr": ([round(x, 2) for x in _iqr(lats)] if len(lats) > 3 else None),
        "false_yawn_events_per_min": (neg_events / (neg_secs / 60.0)) if neg_secs else None,
        "neg_minutes": round(neg_secs / 60.0, 2),
        "drowsy_recall_any_cue": _drowsy_recall("drowsy_any") / n_gt if n_gt else None,
        "drowsy_recall_yawn_cue": _drowsy_recall("drowsy_yawn") / n_gt if n_gt else None,
        "neg_drowsy_per_min_by_cue": {
            (k or "none"): round(v / (neg_secs / 60.0), 3)
            for k, v in by_cue.items()} if neg_secs else {},
        "face_frac_mean": round(st.mean([c["face_frac"] for c in clips]), 3) if clips else None,
        "process_ms_p50": round(st.median(timings), 2) if timings else None,
        "process_ms_p95": round(_pct(timings, 95), 2) if timings else None,
    }


def _pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[k]


def _iqr(xs):
    return (_pct(xs, 25), _pct(xs, 75))


def bootstrap_ci(clips, metric_fn, n_boot=1000, seed=0, alpha=0.05):
    """Cluster-bootstrap over SUBJECTS, not events.

    A subject's yawns are correlated (same face, lighting, habit), so resampling
    events would pretend they are independent and return a dishonestly tight
    interval. Resampling whole subjects keeps the correlation intact.
    """
    by_subj = defaultdict(list)
    for c in clips:
        by_subj[c["subject"]].append(c)
    subs = sorted(by_subj)
    if len(subs) < 2:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        pick = [rng.choice(subs) for _ in subs]
        sample = [c for s in pick for c in by_subj[s]]
        v = metric_fn(aggregate(sample))
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return (round(_pct(vals, 100 * alpha / 2), 4),
            round(_pct(vals, 100 * (1 - alpha / 2)), 4))


# ---------------------------------------------------------------------------
# Clip discovery
# ---------------------------------------------------------------------------
def iter_clips(videos_dir, classes, subjects, max_per_class):
    root = Path(videos_dir)
    for cls in classes:
        folder = root / cls
        if not folder.is_dir():
            print(f"[eval] (skip) no folder {folder}")
            continue
        clips = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if subjects is not None:
            clips = [p for p in clips if subject_of(p.name) in subjects]
        if max_per_class:
            clips = clips[:max_per_class]
        for p in clips:
            yield cls, p


def run_manifest(args, subjects):
    """Everything needed to reproduce this run, recorded next to its numbers."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=str(PARENT), capture_output=True, text=True,
                                timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - provenance is nice-to-have, never fatal
        commit = "unknown"
    learned = PARENT / "learned_thresholds.json"
    return {
        "git_commit": commit,
        "use_cnn": args.use_cnn,
        "stride": args.stride,
        "subjects_scope": args.subjects,
        "n_subjects_selected": (len(subjects) if subjects is not None else None),
        "audio": "silence (Yawning clips have no audio track)",
        "learned_thresholds_present": learned.exists(),
        "thresholds": {
            "MAR_YAWN": MAR_YAWN, "YAWN_CNN_PROB": YAWN_CNN_PROB,
            "YAWN_MIN_SECONDS": YAWN_MIN_SECONDS, "EAR_CLOSED": EAR_CLOSED,
            "EYE_CLOSED_SECONDS": EYE_CLOSED_SECONDS,
            "MMS_YAWN_LEVEL": MMS_YAWN_LEVEL,
            "MMS_SPEECH_AMP_MAX": MMS_SPEECH_AMP_MAX,
            "MMS_SPEECH_OSC_RATE_MIN": MMS_SPEECH_OSC_RATE_MIN,
            "MMS_WINDOW_SECONDS": MMS_WINDOW_SECONDS,
            "FUSION_INTERVAL_SECONDS": FUSION_INTERVAL_SECONDS,
        },
    }


def parity_check(clip, tol=0.005):
    """Prove this harness decides what main.py decides, on the same clip.

    The harness is a second code path (main.py's loop is entangled with the HUD,
    the writer and the alerter), so "we measure the deployed system" is an
    assumption until something checks it. This runs the real main.py headless and
    compares the fused verdict sequence against run_clip's.

    Times are compared RELATIVE to each run's first window: main.py's clock is
    `time.time() + frame/fps` while the harness starts at 0, so the absolute
    epochs differ by a constant that carries no meaning.
    """
    import csv as _csv
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="dms_parity_"))
    cmd = [sys.executable, str(PARENT / "main.py"), "--source", str(clip),
           "--no-audio", "--no-display", "--log-dir", str(tmp)]
    proc = subprocess.run(cmd, cwd=str(PARENT), capture_output=True, text=True,
                          timeout=900)
    csv_path = tmp / "events.csv"
    if not csv_path.exists():
        print(f"  [FAIL] main.py wrote no events.csv\n{proc.stdout}\n{proc.stderr}")
        return False
    rows = list(_csv.DictReader(csv_path.open(encoding="utf-8")))
    main_seq = [(float(r["epoch"]), r["state"]) for r in rows]

    res = run_clip(clip)
    harn_seq = [(t1, state) for (_t0, t1, state, _cf) in res["windows"]]

    ok = True
    if len(main_seq) != len(harn_seq):
        print(f"  [FAIL] {clip.name}: {len(main_seq)} main windows vs "
              f"{len(harn_seq)} harness windows")
        return False
    if not main_seq:
        print(f"  [FAIL] {clip.name}: no fusion windows closed")
        return False
    m0, h0 = main_seq[0][0], harn_seq[0][0]
    for k, ((me, ms), (he, hs)) in enumerate(zip(main_seq, harn_seq)):
        if ms != hs:
            print(f"  [FAIL] {clip.name} window {k}: main={ms} harness={hs}")
            ok = False
        if abs((me - m0) - (he - h0)) > tol:
            print(f"  [FAIL] {clip.name} window {k}: t main={me-m0:.3f} "
                  f"harness={he-h0:.3f}")
            ok = False
    if ok:
        states = Counter(s for _, s in harn_seq)
        print(f"  [OK] {clip.name}: {len(harn_seq)} windows identical  {dict(states)}")
    return ok


def print_report(agg, man, ci=None):
    def pct(x):
        return "n/a" if x is None else f"{x*100:.1f}%"
    print("\n=== run ===")
    print(f"  commit={man['git_commit']}  cnn={man['use_cnn']}  stride={man['stride']}  "
          f"subjects={man['subjects_scope']} (n={man['n_subjects_selected']})")
    print(f"  thresholds: MAR_YAWN={man['thresholds']['MAR_YAWN']} "
          f"YAWN_CNN_PROB={man['thresholds']['YAWN_CNN_PROB']} "
          f"YAWN_MIN_SECONDS={man['thresholds']['YAWN_MIN_SECONDS']}  "
          f"learned_thresholds={man['learned_thresholds_present']}")
    print(f"  audio: {man['audio']}")

    print("\n=== Level A - yawn cue events ===")
    print(f"  clips={agg['n_clips']} (pos={agg['n_pos_clips']} neg={agg['n_neg_clips']}) "
          f"subjects={agg['n_subjects']}")
    r = agg["yawn_event_recall"]
    line = f"  event recall            {pct(r)}  ({agg['n_matched']}/{agg['n_gt_events']})"
    if ci:
        line += f"   95% CI [{pct(ci[0])}, {pct(ci[1])}]"
    print(line)
    print(f"  mean IoU                {agg['mean_iou']}")
    print(f"  onset latency (s)       median={agg['latency_median']}  IQR={agg['latency_iqr']}")
    fa = agg["false_yawn_events_per_min"]
    print(f"  false yawn events/min   {'n/a' if fa is None else f'{fa:.3f}'}  "
          f"(over {agg['neg_minutes']} min of Singing+Alert)")

    print("\n=== Level B - fused DROWSY ===")
    print(f"  recall, any cue         {pct(agg['drowsy_recall_any_cue'])}")
    print(f"  recall, yawn cue only   {pct(agg['drowsy_recall_yawn_cue'])}")
    print(f"  neg DROWSY/min by cue   {agg['neg_drowsy_per_min_by_cue'] or 'none'}")

    print("\n=== health / speed (guard, not a goal) ===")
    print(f"  face found (mean)       {pct(agg['face_frac_mean'])}")
    print(f"  process() ms p50/p95    {agg['process_ms_p50']} / {agg['process_ms_p95']}")


def main():
    ap = argparse.ArgumentParser(
        description="Event-level evaluation of the live drowsiness pipeline")
    ap.add_argument("--videos", default=str(DEF_VIDEOS))
    ap.add_argument("--classes", default=",".join([POS_CLASS] + DEFAULT_NEG_CLASSES),
                    help="folders to score; the first is the positive class")
    ap.add_argument("--use-cnn", action="store_true",
                    help="use the trained yawn CNN instead of the MAR rule")
    ap.add_argument("--yawn-cnn-prob", type=float, default=None,
                    help="override config's YAWN_CNN_PROB (for the event-level sweep)")
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth frame; simulates a slower machine")
    ap.add_argument("--subjects", default="val-b",
                    choices=["all", "val", "val-a", "val-b"],
                    help="which subjects to score. val-* are the CNN's held-out "
                         "subjects, halved: sweep on val-a, REPORT on val-b")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="must match train_yawn_cnn.py's value")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-clips-per-class", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-ci", action="store_true", help="skip the bootstrap CI")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=None, help="write the full report JSON here")
    ap.add_argument("--parity", type=int, default=0, metavar="N",
                    help="instead of scoring: run main.py headless on N clips per "
                         "class and assert it produces the harness's exact fused "
                         "verdict sequence, then exit")
    args = ap.parse_args()

    # The CNN threshold lives as a module global in video_model; rebinding it is
    # how we sweep an operating point without touching config.py on disk.
    if args.yawn_cnn_prob is not None:
        import video_model
        video_model.YAWN_CNN_PROB = args.yawn_cnn_prob
        if not args.use_cnn:
            print("[eval] note: --yawn-cnn-prob has no effect without --use-cnn")

    if args.subjects == "all":
        subjects = None
    else:
        va, vb = held_out_subjects(args.val_frac, args.seed)
        subjects = {"val": va | vb, "val-a": va, "val-b": vb}[args.subjects]
        print(f"[eval] {args.subjects}: {len(subjects)} subjects -> {sorted(subjects)}")

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    anns = load_annotations(ANN_PATH)

    if args.parity:
        print("=== parity gate: main.py vs this harness ===")
        every = list(iter_clips(args.videos, classes, subjects, args.parity))
        good = all([parity_check(p) for _cls, p in every])
        print("parity " + ("OK - the harness measures the deployed system"
                           if good else "FAILED - the harness is NOT main.py"))
        return 0 if good else 1

    clips = []
    todo = list(iter_clips(args.videos, classes, subjects, args.max_clips_per_class))
    if not todo:
        print("[eval] no clips selected - check --videos / --subjects.")
        return 1
    for k, (cls, path) in enumerate(todo, 1):
        ann = anns.get(clip_key(cls, path))
        if cls == POS_CLASS and not ann:
            print(f"[eval] (skip) {path.name}: no annotation")
            continue
        print(f"[eval] {k}/{len(todo)} {cls}/{path.name}", flush=True)
        clips.append(score_clip(cls, path, ann, args.use_cnn, args.stride,
                                args.max_frames))

    agg = aggregate(clips)
    man = run_manifest(args, subjects)
    ci = None
    if not args.no_ci and agg["n_gt_events"]:
        ci = bootstrap_ci(clips, lambda a: a["yawn_event_recall"],
                          n_boot=args.n_boot, seed=args.seed)
    print_report(agg, man, ci)

    if args.out:
        payload = {"manifest": man, "aggregate": agg, "recall_ci95": ci,
                   "clips": [{k: v for k, v in c.items() if k != "timings_ms"}
                             for c in clips]}
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str),
                                  encoding="utf-8")
        print(f"\n[eval] wrote {args.out}")
    return 0


# ---------------------------------------------------------------------------
# Self-test: the pure event logic, on synthetic traces. No video, no models.
# Run: python eval_events.py --self-test
# ---------------------------------------------------------------------------
def _self_test():
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'OK' if good else 'FAIL'}] {name}: got={got} want={want}")

    def rounded(events, nd=3):
        """Compare event times at 3 decimals - these are seconds off a frame
        clock, and exact float equality would test IEEE754, not the logic."""
        return [(round(a, nd), round(b, nd)) for a, b in events]

    times = [i * 0.1 for i in range(10)]
    flags = [False, True, True, False, False, True, False, False, False, False]
    check("two runs -> two events",
          rounded(events_from_flags(flags, times, 0.1)),
          [(0.1, 0.3), (0.5, 0.6)])
    check("all False -> none", events_from_flags([False] * 3, times[:3], 0.1), [])
    check("trailing run closes", rounded(events_from_flags([False, True], times[:2], 0.1)),
          [(0.1, 0.2)])

    # inclusive frames -> seconds
    check("gt spans", rounded(gt_segments({"yawns": [[0, 29]]}, 30.0)), [(0.0, 1.0)])

    # any-overlap matching + latency
    m, ious, lats = match_events([(1.0, 3.0)], [(2.0, 4.0)])
    check("overlap matches", m, 1)
    check("latency is lateness", round(lats[0], 2), 1.0)
    check("iou", round(ious[0], 2), 0.33)
    m, _, _ = match_events([(1.0, 3.0)], [(5.0, 6.0)])
    check("disjoint -> miss", m, 0)

    # attribution: the cue with the most weighted evidence carries the window
    check("yawn dominates", dominant_cue({"eyes": 0.1, "yawn": 1.0, "nod": 0.0}), "yawn")
    check("eyes dominates", dominant_cue({"eyes": 1.0, "yawn": 0.1, "nod": 0.0}), "eyes")
    check("no cue -> None", dominant_cue({"eyes": 0.0, "yawn": 0.0, "nod": 0.0}), None)

    print("event-logic self-test " + ("OK" if ok else "FAILED"))
    return ok


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    if "--self-test" in sys.argv:
        sys.exit(0 if _self_test() else 1)
    sys.exit(main())
