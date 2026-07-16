"""
train/extract_features.py - turn labelled clips into a per-frame feature table.

It walks a dataset folder laid out as:

    <videos_dir>/
        Yawning/   clip1.mp4  clip2.avi ...
        Singing/   ...
        Alert/     ...
        Drowsy/    ...
        Sleeping/  ...
        Distracted/...

and writes one CSV row per processed frame with the SAME features the live
system uses at runtime - because it drives the very same VideoModel.process()
from the parent folder. That train/runtime parity is the whole point: whatever
thresholds you fit on this table will mean exactly the same thing live.

Output columns:
    video, label, frame, face_found, ear, mar, yaw, pitch, roll,
    lip_gap_level, mms_amplitude, mms_oscillation, mouth_state

Usage (local smoke test - tiny, just to prove the path):
    python extract_features.py --videos ../../data/raw_videos --out features.csv \
        --max-clips-per-class 2 --max-frames 120

Usage (full run - do this in Colab, see train/README.md):
    python extract_features.py --videos /content/drive/MyDrive/dms_data --out features.csv

NOTE: extracting features from every clip is CPU-heavy; run the full version in
Colab, not on the local machine. The flags above keep a local check fast.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2

# Make the parent folder (the runtime modules) importable so we reuse the exact
# same feature maths. _tf_guard inside video_model keeps MediaPipe importable.
THIS = Path(__file__).resolve()
PARENT = THIS.parent.parent
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(THIS.parent))   # train/ itself, so we can import annotate_yawns

from video_model import VideoModel  # noqa: E402 - after sys.path tweak
from annotate_yawns import (  # noqa: E402 - same-folder helper, after sys.path tweak
    resolve_label, load_annotations, clip_key, ANN_PATH,
)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_FIELDS = ["video", "label", "frame", "face_found", "ear", "mar", "yaw",
           "pitch", "roll", "lip_gap_level", "mms_amplitude",
           "mms_oscillation", "mouth_state"]


def iter_clips(videos_dir, max_per_class):
    """Yield (label, path) for each clip under <videos_dir>/<label>/."""
    root = Path(videos_dir)
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = class_dir.name
        clips = sorted(p for p in class_dir.iterdir()
                       if p.suffix.lower() in VIDEO_EXTS)
        if max_per_class:
            clips = clips[:max_per_class]
        for clip in clips:
            yield label, clip


def extract_clip(label, path, max_frames, ann=None):
    """Run VideoModel over one clip; yield a row dict per frame.

    If `ann` (this clip's entry from yawn_segments.json) is given, the per-frame
    label comes from resolve_label() instead of the folder name:
      * YAWN frames stay "Yawning" (the positive class - name unchanged so the
        notebook's `vals(['Yawning'], ...)` query is untouched);
      * NEUTRAL frames become "Neutral" (a training NEGATIVE - exactly the
        closed-mouth frames that used to pollute the positive class);
      * IGNORE frames are skipped entirely (no row written).
    Every frame is still fed to VideoModel so its streak/baseline/MMS state
    advances correctly - only the WRITING of ignored rows is suppressed.
    """
    vm = VideoModel()                       # fresh state per clip (streaks/baseline)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    i = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and i >= max_frames:
                break
            out = vm.process(frame, now=i / fps)   # video-time clock; runs on EVERY frame
            # lip_gap_level isn't returned directly; ask the model for the window
            # max it just used (max gap == the MMS "level") so this stays a
            # faithful record without reaching into its buffer layout.
            level = vm.lip_gap_level()
            frame_i = i
            i += 1

            row_label = label
            if ann is not None:
                resolved = resolve_label(frame_i, ann)
                if resolved == "IGNORE":
                    continue                       # drop ambiguous frame from training
                row_label = "Yawning" if resolved == "YAWN" else "Neutral"

            yield {
                "video": path.name, "label": row_label, "frame": frame_i,
                "face_found": int(out["face_found"]),
                "ear": round(out["ear"], 5), "mar": round(out["mar"], 5),
                "yaw": round(out["yaw"], 5), "pitch": round(out["pitch"], 5),
                "roll": round(out["roll"], 5),
                "lip_gap_level": round(float(level), 5),
                "mms_amplitude": round(out["mms_amplitude"], 5),
                "mms_oscillation": out["mms_oscillation"],
                "mouth_state": out["mouth_state"],
            }
    finally:
        cap.release()
        vm.close()


def main():
    ap = argparse.ArgumentParser(description="Extract per-frame features from labelled clips")
    ap.add_argument("--videos", required=True, help="root dir with <label>/ subfolders")
    ap.add_argument("--out", default="features.csv", help="output CSV path")
    ap.add_argument("--max-clips-per-class", type=int, default=0,
                    help="limit clips per class (0 = all; use a small number for a smoke test)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="limit frames per clip (0 = all)")
    ap.add_argument("--annotations", default=str(ANN_PATH),
                    help="yawn_segments.json from annotate_yawns.py: per-frame "
                         "YAWN/NEUTRAL/IGNORE relabel for annotated clips. A "
                         "missing file is a silent no-op (folder labels as before).")
    args = ap.parse_args()

    annotations = load_annotations(args.annotations)
    if annotations:
        print(f"[extract] applying {len(annotations)} clip annotation(s) "
              f"from {args.annotations}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_clips = n_rows = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for label, clip in iter_clips(args.videos, args.max_clips_per_class):
            n_clips += 1
            ann = annotations.get(clip_key(label, clip))
            tag = "  [annotated]" if ann is not None else ""
            print(f"[{n_clips}] {label:11} {clip.name}{tag}")
            for row in extract_clip(label, clip, args.max_frames, ann):
                writer.writerow(row)
                n_rows += 1

    print(f"\nDone. {n_clips} clips -> {n_rows} frames written to {out_path}")


if __name__ == "__main__":
    main()
