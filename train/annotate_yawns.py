"""
train/annotate_yawns.py - hand-label the TRUE yawn span inside each yawn clip.

Why this exists
---------------
A clip in `Yawning/` is mostly NOT a yawn: there's a closed-mouth lead-in and
trail-out around the one real yawn. `extract_features.py` used to label every
frame of the clip `Yawning`, so those closed-mouth frames polluted the positive
class and dragged the fitted MAR / MMS thresholds DOWN -> false "DROWSY" on
ordinary talking. This tool lets you mark, per clip, three kinds of frame:

  * YAWN    - inside any yawn segment you mark (you can mark SEVERAL per video,
              e.g. a driver who yawns twice with a calm gap between).
  * NEUTRAL - everything else (mouth closed, normal posture). A training NEGATIVE.
  * IGNORE  - ambiguous transition frames (already tired: heavy eyes / slight
              head droop, but not yet yawning). DROPPED from training entirely,
              so they can't teach the model the wrong thing.

It writes `train/yawn_segments.json`; `extract_features.py` reads it and relabels
frames accordingly (IGNORE rows are simply not written). Clips with no entry keep
the old whole-clip labelling, so this is fully opt-in.

Run
---
  # GUI: annotate every clip in <root>/Yawning/  (needs a display)
  python annotate_yawns.py --videos ../../data/raw_videos

  # GUI: a single clip
  python annotate_yawns.py --clip ../../data/raw_videos/Yawning/clip1.mp4

  # no args -> run the (headless) resolve_label self-test and exit
  python annotate_yawns.py

Mark as MANY yawn segments per video as you need: g..h adds one segment, repeat
for the next yawn. Frames between segments stay NEUTRAL.

Keys (also drawn on screen):
  Space play/pause   a / d  step -1 / +1   [ / ]  jump -10 / +10
  g YAWN start       h YAWN end (adds the segment)   x delete last yawn segment
  i IGNORE start     o IGNORE end (adds the range)   c delete last ignore range
  r reset this clip  n save + next clip   p previous clip   q/ESC save + quit
"""
import argparse
import json
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_CLASS = "Yawning"
ANN_PATH = Path(__file__).resolve().parent / "yawn_segments.json"
MAX_PRELOAD = 6000          # safety cap on frames held in memory per clip
DISPLAY_MAX_W = 1100        # shrink very large frames to fit the screen

# Colours (BGR) for the three resolved labels.
COLORS = {"YAWN": (0, 200, 0), "NEUTRAL": (150, 150, 150), "IGNORE": (0, 0, 220)}


# ---------------------------------------------------------------------------
# Pure logic (no OpenCV) - resolve one frame's label from a clip's annotation.
# IGNORE takes precedence over YAWN where ranges overlap (exclude the ambiguous
# frame rather than train on it). This is the function extract_features.py reuses.
# ---------------------------------------------------------------------------
def yawn_ranges(ann):
    """The clip's yawn segments as a list of [start, end].

    Accepts the plural 'yawns' (a list of segments) and, for back-compat, a
    single 'yawn': [s, e]. Returns [] when nothing is marked.
    """
    ys = ann.get("yawns")
    if ys:
        return ys
    y = ann.get("yawn")
    return [y] if y else []


def resolve_label(frame_i, ann):
    """Return 'YAWN' | 'NEUTRAL' | 'IGNORE' for frame_i given a clip annotation.

    ann = {"yawns": [[s, e], ...], "ignore": [[s, e], ...]}
    IGNORE wins over YAWN where they overlap (exclude the ambiguous frame).
    """
    for s, e in ann.get("ignore", []) or []:
        if s <= frame_i <= e:
            return "IGNORE"
    for s, e in yawn_ranges(ann):
        if s <= frame_i <= e:
            return "YAWN"
    return "NEUTRAL"


def load_annotations(path=ANN_PATH):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a corrupt file shouldn't crash the tool
        print(f"[annotate] could not read {p} ({exc}); starting empty.")
        return {}


def save_annotations(data, path=ANN_PATH):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def clip_key(class_label, clip_path):
    """The dataset-relative key extract_features.py also builds: '<Label>/<file>'."""
    return f"{class_label}/{Path(clip_path).name}"


def iter_clips(videos_dir, class_label):
    folder = Path(videos_dir) / class_label
    if not folder.is_dir():
        raise SystemExit(f"[annotate] no such folder: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)


# ---------------------------------------------------------------------------
# Editing state for the clip currently on screen. Yawns and ignores are both
# LISTS of committed [start, end] ranges, each with its own pending start marker
# so we can show a live preview while you pick the end. g..h appends a yawn, i..o
# appends an ignore; final_ann() commits only fully-closed ranges.
# ---------------------------------------------------------------------------
class ClipEdit:
    def __init__(self, saved):
        saved = saved or {}
        self.yawns = [list(r) for r in yawn_ranges(saved)]
        self.ignore = [list(r) for r in (saved.get("ignore") or [])]
        self.yawn_pending = None
        self.ig_pending = None

    def display_ann(self, cur):
        """Annotation including the live preview of any open range (uses cursor)."""
        yawns = [list(r) for r in self.yawns]
        if self.yawn_pending is not None:
            yawns.append(sorted((self.yawn_pending, cur)))
        ignore = [list(r) for r in self.ignore]
        if self.ig_pending is not None:
            ignore.append(sorted((self.ig_pending, cur)))
        return {"yawns": yawns, "ignore": ignore}

    def final_ann(self):
        """Only fully-closed ranges; pending half-marks are dropped."""
        return {"yawns": [sorted(r) for r in self.yawns],
                "ignore": [sorted(r) for r in self.ignore]}


def _read_frames(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    frames = []
    while len(frames) < MAX_PRELOAD:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def _draw(frames, idx, edit, fps, clip_name, clip_i, n_clips):
    import cv2
    frame = frames[idx].copy()
    h, w = frame.shape[:2]
    if w > DISPLAY_MAX_W:
        scale = DISPLAY_MAX_W / w
        frame = cv2.resize(frame, (DISPLAY_MAX_W, int(h * scale)))
        h, w = frame.shape[:2]

    ann = edit.display_ann(idx)
    label = resolve_label(idx, ann)
    color = COLORS[label]

    # Coloured border = this frame's resolved label.
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)

    lines = [
        f"clip {clip_i + 1}/{n_clips}  {clip_name}",
        f"frame {idx + 1}/{len(frames)}   t={idx / fps:5.2f}s   fps={fps:.1f}",
        f"LABEL: {label}",
        f"yawns({len(ann['yawns'])}): {ann['yawns']}",
        f"ignore({len(ann['ignore'])}): {ann['ignore']}",
        "Space play | a/d step | [ ] jump10 | g/h add-yawn | x del-yawn",
        "i/o add-ignore | c del-ignore | r reset | n next | p prev | q quit",
    ]
    y = 24
    for i, text in enumerate(lines):
        tc = color if i == 2 else (255, 255, 255)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tc, 1)
        y += 26

    # Timeline strip: one column per frame, coloured by label, with a cursor.
    strip_h = 16
    n = len(frames)
    bar = frame[h - strip_h:h, :, :]
    for x in range(w):
        fi = int(x / max(w - 1, 1) * (n - 1))
        bar[:, x] = COLORS[resolve_label(fi, ann)]
    cx = int(idx / max(n - 1, 1) * (w - 1))
    cv2.line(frame, (cx, h - strip_h), (cx, h), (255, 255, 255), 1)
    return frame


def annotate(clips, class_label, ann_path):
    import cv2
    data = load_annotations(ann_path)
    ci = 0
    win = "annotate yawns"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    while 0 <= ci < len(clips):
        clip = clips[ci]
        key = clip_key(class_label, clip)
        frames, fps = _read_frames(clip)
        if not frames:
            print(f"[annotate] {clip.name}: no frames, skipping.")
            ci += 1
            continue
        edit = ClipEdit(data.get(key))
        idx, playing = 0, False
        n = len(frames)

        def commit():
            ann = edit.final_ann()
            ann["fps"] = round(fps, 3)
            ann["n_frames"] = n
            if not ann["yawns"]:
                print(f"[annotate] WARNING {key}: no yawn marked -> whole clip NEUTRAL.")
            data[key] = ann
            save_annotations(data, ann_path)

        nav = None  # 'next' | 'prev' | 'quit'
        while nav is None:
            img = _draw(frames, idx, edit, fps, clip.name, ci, len(clips))
            cv2.imshow(win, img)
            k = cv2.waitKeyEx(30 if playing else 0)
            ch = (k & 0xFF) if k != -1 else -1

            if playing and k == -1:
                idx = min(idx + 1, n - 1)
                if idx == n - 1:
                    playing = False
                continue

            if ch in (ord('q'), 27):           # q / ESC
                nav = "quit"
            elif ch == ord('n'):
                nav = "next"
            elif ch == ord('p'):
                nav = "prev"
            elif ch == ord(' '):
                playing = not playing
            elif ch in (ord('d'),) or k == 2555904:   # right
                idx = min(idx + 1, n - 1)
            elif ch in (ord('a'),) or k == 2424832:    # left
                idx = max(idx - 1, 0)
            elif ch == ord(']'):
                idx = min(idx + 10, n - 1)
            elif ch == ord('['):
                idx = max(idx - 10, 0)
            elif ch == ord('g'):
                edit.yawn_pending = idx
            elif ch == ord('h'):
                if edit.yawn_pending is not None:
                    edit.yawns.append(sorted((edit.yawn_pending, idx)))
                    edit.yawn_pending = None
            elif ch == ord('x'):
                if edit.yawns:
                    edit.yawns.pop()
            elif ch == ord('i'):
                edit.ig_pending = idx
            elif ch == ord('o'):
                if edit.ig_pending is not None:
                    edit.ignore.append(sorted((edit.ig_pending, idx)))
                    edit.ig_pending = None
            elif ch == ord('c'):
                if edit.ignore:
                    edit.ignore.pop()
            elif ch == ord('r'):
                edit = ClipEdit(None)

        commit()
        if nav == "quit":
            break
        ci += 1 if nav == "next" else -1
        ci = max(ci, 0)

    cv2.destroyAllWindows()
    print(f"[annotate] saved {len(data)} clip annotation(s) -> {ann_path}")


def _run_selftest():
    # Two yawn segments in one video, with ignore bands (one overlapping a yawn).
    ann = {"yawns": [[10, 20], [30, 40]], "ignore": [[5, 8], [15, 16]]}
    cases = {0: "NEUTRAL", 6: "IGNORE", 9: "NEUTRAL", 10: "YAWN",
             14: "YAWN", 15: "IGNORE", 16: "IGNORE", 17: "YAWN", 20: "YAWN",
             25: "NEUTRAL", 30: "YAWN", 40: "YAWN", 41: "NEUTRAL"}
    ok = all(resolve_label(f, ann) == want for f, want in cases.items())
    # back-compat: a single 'yawn': [s, e] is still understood
    ok = ok and resolve_label(12, {"yawn": [10, 20]}) == "YAWN"
    # nothing marked -> all NEUTRAL
    ok = ok and resolve_label(3, {"yawns": [], "ignore": []}) == "NEUTRAL"
    for f, want in cases.items():
        got = resolve_label(f, ann)
        flag = "" if got == want else "  <-- MISMATCH"
        print(f"  frame {f:2d}: {got:7} (expect {want}){flag}")
    print("annotate_yawns self-test", "OK" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Hand-label yawn segments in clips")
    ap.add_argument("--videos", help="dataset root containing the class subfolder")
    ap.add_argument("--clip", help="annotate a single video file")
    ap.add_argument("--class", dest="cls", default=DEFAULT_CLASS,
                    help=f"class subfolder to annotate (default {DEFAULT_CLASS})")
    ap.add_argument("--out", default=str(ANN_PATH), help="annotations JSON path")
    args = ap.parse_args()

    if not args.videos and not args.clip:
        sys.exit(0 if _run_selftest() else 1)

    if args.clip:
        clip = Path(args.clip)
        # key off the clip's parent folder so it matches the extractor's '<Label>/<file>'
        class_label = clip.parent.name
        clips = [clip]
    else:
        class_label = args.cls
        clips = iter_clips(args.videos, class_label)
        if not clips:
            raise SystemExit(f"[annotate] no clips under {args.videos}/{class_label}")

    annotate(clips, class_label, Path(args.out))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    main()
