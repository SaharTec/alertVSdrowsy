"""
train/build_mouth_dataset.py - turn clips into a labelled MOUTH-CROP dataset for
the transfer-learning yawn CNN (section 5 of the project).

This is the data-prep step for `train_yawn_cnn.py`. For each frame it:
  1. runs the SAME MediaPipe FaceMesh the live system uses (via VideoModel), so
     the mouth is located exactly as at runtime (no train/serve skew);
  2. crops a square box around the mouth (lips + corners, padded for jaw context)
     and resizes it to a fixed size (default 112x112);
  3. gives the crop a binary label:
       * 1 = yawn      : YAWN frames from Yawning clips (per yawn_segments.json)
       * 0 = not-yawn  : NEUTRAL frames from Yawning clips + every frame of the
                         "hard negative" classes (talking/singing: mouth open but
                         NOT a yawn - the exact confusers precision cares about)
     IGNORE frames (ambiguous) are skipped entirely.
  4. parses the SUBJECT from the filename so training can split by person
     (no face in both train and test -> honest evaluation, section 6).

Outputs:
  <out-dir>/{yawn,notyawn}/<video>__f<frame>.jpg   (default out-dir: train/mouth_crops)
  <manifest>  CSV: path,label,subject,source_class,video,frame,mar   (path relative to out-dir;
              mar is the per-frame mouth-aspect-ratio, used by evaluate_yawn.py's baseline)

This is CPU-heavy (decodes frames + FaceMesh); run it yourself. Use --stride to
subsample frames and the --max-* flags for a quick smoke test.

  # smoke (tiny)
  python build_mouth_dataset.py --videos ../../data/raw_videos \
      --max-clips-per-class 2 --max-frames 60
  # full
  python build_mouth_dataset.py --videos ../../data/raw_videos
  # no args -> run the (headless) helper self-test and exit
  python build_mouth_dataset.py
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

THIS = Path(__file__).resolve()
PARENT = THIS.parent.parent                  # alertVSdrowsy/ (runtime modules)
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(THIS.parent))         # train/ itself, for annotate_yawns

from video_model import VideoModel  # noqa: E402 - after sys.path tweak
from annotate_yawns import (  # noqa: E402
    resolve_label, load_annotations, clip_key, ANN_PATH,
)
from config import (  # noqa: E402
    MOUTH_INDICES, INNER_LIP_TOP, INNER_LIP_BOTTOM,
)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
POS_CLASS = "Yawning"                         # source of positive (yawn) frames
DEFAULT_NEG_CLASSES = ["Singing", "Alert"]    # hard-negative source folders
# Mouth corners (61, 291) come from config's HEAD_POSE_LANDMARKS; combined with the
# lip points they bound the whole mouth for a stable crop box.
MOUTH_BOX_INDICES = sorted(set(MOUTH_INDICES + [61, 291,
                                               INNER_LIP_TOP, INNER_LIP_BOTTOM]))
DEFAULT_SIZE = 112
DEFAULT_MARGIN = 1.9                          # box = mouth-span * this (adds jaw/cheek)
_FIELDS = ["path", "label", "subject", "source_class", "video", "frame", "mar"]


# ---------------------------------------------------------------------------
# Pure helpers (no video/FaceMesh) - unit-tested in the self-test below.
# ---------------------------------------------------------------------------
def subject_of(filename):
    """Group a person's clips under one id so subject-grouped CV can hold a whole
    person out. YawDD names like '11-MaleGlasses-Yawning.avi' -> '11-MaleGlasses'
    (same person appears across Yawning/Talking); '_' names -> first token; else
    the bare stem."""
    stem = Path(filename).stem
    if "-" in stem:
        parts = stem.split("-")
        return "-".join(parts[:2]) if len(parts) >= 2 else parts[0]
    if "_" in stem:
        return stem.split("_")[0]
    return stem


def mouth_box(points, w, h, margin=DEFAULT_MARGIN):
    """Square crop box (x0, y0, x1, y1) around mouth pixel points, clamped to the
    frame. `points` is an Nx2 array of pixel coords. Returns None if degenerate."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return None
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    half = max(x_max - x_min, y_max - y_min) / 2.0 * margin
    if half < 2.0:
        return None
    x0 = int(max(0, round(cx - half))); y0 = int(max(0, round(cy - half)))
    x1 = int(min(w, round(cx + half))); y1 = int(min(h, round(cy + half)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Real work.
# ---------------------------------------------------------------------------
def iter_sources(videos_dir, neg_classes, max_per_class):
    """Yield (source_class, is_pos, path). Positives come from Yawning/, negatives
    from each folder in neg_classes."""
    root = Path(videos_dir)
    wanted = [(POS_CLASS, True)] + [(c, False) for c in neg_classes]
    for cls, is_pos in wanted:
        folder = root / cls
        if not folder.is_dir():
            print(f"[build] (skip) no folder {folder}")
            continue
        clips = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if max_per_class:
            clips = clips[:max_per_class]
        for clip in clips:
            yield cls, is_pos, clip


def _lm_points(landmarks, indices, w, h):
    return [[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
            for i in indices]


def crop_clip(vm, source_class, is_pos, path, ann, out_dir, size, margin,
              stride, max_frames):
    """Process one clip; save crops; yield manifest row dicts."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    subject = subject_of(path.name)
    i = 0
    kept = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and i >= max_frames:
                break
            if i % stride != 0:
                i += 1
                continue
            out = vm.process(frame, now=i / fps)
            frame_i = i
            i += 1
            if not out["face_found"] or out["landmarks"] is None:
                continue

            # Decide the label for this frame.
            if is_pos:                                   # a Yawning clip
                if ann is not None:
                    resolved = resolve_label(frame_i, ann)
                    if resolved == "IGNORE":
                        continue
                    label = 1 if resolved == "YAWN" else 0
                else:
                    label = 1                            # unannotated yawn clip: assume yawn
            else:
                label = 0                                # hard-negative class

            h, w = frame.shape[:2]
            pts = _lm_points(out["landmarks"], MOUTH_BOX_INDICES, w, h)
            box = mouth_box(pts, w, h, margin)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            crop = cv2.resize(frame[y0:y1, x0:x1], (size, size))

            sub = "yawn" if label == 1 else "notyawn"
            crop_dir = out_dir / sub
            crop_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{path.stem}__f{frame_i:06d}.jpg"
            cv2.imwrite(str(crop_dir / fname), crop)
            kept += 1
            yield {
                "path": f"{sub}/{fname}",            # relative to the crops dir
                "label": label, "subject": subject,
                "source_class": source_class, "video": path.name, "frame": frame_i,
                "mar": round(float(out["mar"]), 5),  # for the rule baseline in evaluate_yawn.py
            }
    finally:
        cap.release()
    print(f"      -> {kept} crops  (subject={subject})")


def main():
    ap = argparse.ArgumentParser(description="Build a labelled mouth-crop dataset")
    ap.add_argument("--videos", help="dataset root with <Label>/ subfolders")
    ap.add_argument("--out-dir", default=str(THIS.parent / "mouth_crops"),
                    help="where crop images go (default train/mouth_crops)")
    ap.add_argument("--manifest", default=str(THIS.parent / "mouth_manifest.csv"),
                    help="output manifest CSV (default train/mouth_manifest.csv)")
    ap.add_argument("--annotations", default=str(ANN_PATH),
                    help="yawn_segments.json (labels YAWN/NEUTRAL/IGNORE in Yawning clips)")
    ap.add_argument("--neg-classes", default=",".join(DEFAULT_NEG_CLASSES),
                    help="comma-separated hard-negative folders (default Singing,Alert)")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE, help="crop size px")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                    help="crop box = mouth-span * margin")
    ap.add_argument("--stride", type=int, default=3,
                    help="keep every Nth frame (speed/size; default 3)")
    ap.add_argument("--max-clips-per-class", type=int, default=0, help="0 = all")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all (per clip)")
    args = ap.parse_args()

    if not args.videos:
        sys.exit(0 if _run_selftest() else 1)

    neg_classes = [c.strip() for c in args.neg_classes.split(",") if c.strip()]
    annotations = load_annotations(args.annotations)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if annotations:
        print(f"[build] {len(annotations)} yawn annotation(s) loaded")
    print(f"[build] positives: {POS_CLASS}   negatives: {neg_classes}")

    vm = VideoModel()                              # one FaceMesh, reused across clips
    n_clips = 0
    counts = {0: 0, 1: 0}
    subjects = set()
    manifest = Path(args.manifest)
    try:
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            writer.writeheader()
            for source_class, is_pos, clip in iter_sources(
                    args.videos, neg_classes, args.max_clips_per_class):
                n_clips += 1
                ann = annotations.get(clip_key(source_class, clip)) if is_pos else None
                print(f"[{n_clips}] {source_class:9} {clip.name}")
                for row in crop_clip(vm, source_class, is_pos, clip, ann, out_dir,
                                     args.size, args.margin, args.stride,
                                     args.max_frames):
                    writer.writerow(row)
                    counts[row["label"]] += 1
                    subjects.add(row["subject"])
    finally:
        vm.close()

    print(f"\nDone. {n_clips} clips -> {counts[1]} yawn + {counts[0]} not-yawn crops "
          f"across {len(subjects)} subjects.")
    print(f"  crops:    {out_dir}")
    print(f"  manifest: {manifest}")
    if counts[1] == 0 or counts[0] == 0:
        print("  WARNING: one class is empty - check --videos / annotations.")


def _run_selftest():
    ok = True
    # subject_of
    cases = {
        "11-MaleGlasses-Yawning.avi": "11-MaleGlasses",
        "1-FemaleNoGlasses-Talking.avi": "1-FemaleNoGlasses",
        "Yawning001_yawn1.mp4": "Yawning001",
        "VID_20260527_163416.mp4": "VID",
        "single.mp4": "single",
    }
    for name, want in cases.items():
        got = subject_of(name)
        flag = "" if got == want else "  <-- MISMATCH"
        ok = ok and got == want
        print(f"  subject_of({name!r}) = {got!r} (expect {want!r}){flag}")

    # The two properties every subject-grouped split silently depends on. If
    # either breaks, the split still runs and still prints a subject count - it
    # just stops meaning what it says, and every held-out number becomes a lie.
    #
    # (a) One person's clips MUST collapse to one id across classes. YawDD films
    #     the same driver yawning and talking; if these split, the same face
    #     lands in train and val and every reported score is optimistic.
    same = subject_of("11-MaleGlasses-Yawning.avi") == subject_of("11-MaleGlasses-Talking.avi")
    ok = ok and same
    print(f"  [{'OK' if same else 'FAIL'}] same person across classes -> one subject")
    # (b) Different people MUST NOT collapse. YawDD reuses the leading number:
    #     10-Female... and 10-Male... are two drivers. Keying on the number alone
    #     would merge them and over-restrict the split.
    distinct = subject_of("10-FemaleNoGlasses-Normal.avi") != subject_of("10-MaleNoGlasses-Normal.avi")
    ok = ok and distinct
    print(f"  [{'OK' if distinct else 'FAIL'}] same number, different people -> distinct subjects")
    # mouth_box: a 100x40 mouth centred at (300,250) in a 640x480 frame.
    pts = [[250, 230], [350, 230], [250, 270], [350, 270]]
    box = mouth_box(pts, 640, 480, margin=2.0)
    x0, y0, x1, y1 = box
    sq = abs((x1 - x0) - (y1 - y0)) <= 1              # square
    inside = 0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 480
    contains = x0 <= 250 and x1 >= 350 and y0 <= 230 and y1 >= 270
    ok = ok and sq and inside and contains
    print(f"  mouth_box -> {box}  square={sq} inside={inside} contains_mouth={contains}")
    # clamping at the border shouldn't crash or go negative.
    edge = mouth_box([[5, 5], [15, 15]], 640, 480, margin=6.0)
    ok = ok and edge is not None and edge[0] >= 0 and edge[1] >= 0
    print(f"  mouth_box(edge) -> {edge}")
    print("build_mouth_dataset self-test", "OK" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    main()
