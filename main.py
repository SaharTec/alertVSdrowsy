"""
main.py - the entry point that runs the whole drowsiness detector.

It wires the modules together and runs the real-time loop:

    frame -> video_model.process() ----.
                                         > fusion.update() -> alert.update()
    mic   -> audio_model.get_state() --'                 -> logger.log()
                                                          -> draw HUD

Run it:
    python main.py                      # live webcam + microphone
    python main.py --no-audio           # webcam only (no mic / no PANNs)
    python main.py --source clip.mp4    # run on a recorded video instead
    python main.py --source clip.mp4 --no-display   # headless (no window)

Keys (when a window is shown):
    q / ESC  - quit
    m        - toggle the full face mesh overlay

Two clock modes keep timing correct:
  * Webcam  -> wall-clock time (real seconds).
  * --source-> a "video clock" (frame_index / fps), so a 2-second yawn in the
    file is measured as 2 seconds even if the file is processed faster than
    real time. Without this, sustained-duration rules wouldn't fire on a clip.
"""
import argparse
import os
import sys
import time

# _tf_guard must load before mediapipe; importing video_model pulls it in, but
# we also import it here explicitly so main is robust if imports get reordered.
import _tf_guard  # noqa: F401
import cv2
import mediapipe as mp

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console: print safely
except AttributeError:
    pass

from config import (DRAW_INDICES, CAMERA_INDEX,
                    STATE_ALERT, STATE_DROWSY, STATE_NEUTRAL)
from video_model import VideoModel
from fusion import FusionEngine
from alert import AlertManager
from logger import EventLogger
import debug

# Audio is optional - guard the import so a missing dep never blocks the demo.
try:
    from audio_model import AudioModel
    _AUDIO_AVAILABLE = True
except Exception:  # noqa: BLE001
    _AUDIO_AVAILABLE = False

_STATE_COLORS = {                  # BGR
    STATE_ALERT: (0, 200, 0),
    STATE_DROWSY: (0, 0, 255),
    STATE_NEUTRAL: (180, 180, 180),
}


def _draw_hud(frame, vid, fused, audio, banner, show_mesh):
    """Overlay the live state, the raw signals and the audio scores."""
    h, w = frame.shape[:2]
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    mp_face = mp.solutions.face_mesh

    lm = vid.get("landmarks")
    if lm is not None:
        if show_mesh:
            mp_draw.draw_landmarks(
                image=frame, landmark_list=lm,
                connections=mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles
                .get_default_face_mesh_tesselation_style())
        for idx in DRAW_INDICES:
            p = lm.landmark[idx]
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2, (0, 255, 0), -1)
    else:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Big state + confidence box (top-left).
    color = _STATE_COLORS.get(fused.state, (200, 200, 200))
    cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"{fused.state}  ({fused.confidence*100:.0f}%)",
                (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    # Reasons (just under the state box).
    cv2.putText(frame, ", ".join(fused.reasons)[:70], (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # Raw signals (bottom-left).
    lines = [
        f"EAR {vid['ear']:.3f}   MAR {vid['mar']:.3f}",
        f"MMS amp {vid['mms_amplitude']:.3f}  osc {vid['mms_oscillation']}  "
        f"mouth:{vid['mouth_state']}",
        f"pitch {vid['pitch']:+.2f}  eyes:{int(vid['eyes_closed'])} "
        f"yawn:{int(vid['yawning'])} nod:{int(vid['nodding'])}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (20, h - 70 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    # Audio HUD (top-right).
    if audio is not None:
        a = audio.get()
        voice = max(a.get("speech", 0.0), a.get("singing", 0.0))
        cv2.putText(frame, f"audio: {audio.status}", (w - 360, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(frame, f"yawn {a.get('yawn', 0.0):.2f}  voice {voice:.2f}  "
                    f"-> {fused.audio_state}", (w - 360, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    # Alert banner (red bar) when the alarm is speaking.
    if banner:
        cv2.rectangle(frame, (0, h - 150), (w, h - 110), (0, 0, 255), -1)
        cv2.putText(frame, banner, (20, h - 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser(description="Real-time driver drowsiness detector")
    ap.add_argument("--source", default=None,
                    help="video file to run on instead of the webcam")
    ap.add_argument("--camera", type=int, default=CAMERA_INDEX,
                    help="webcam index (default from config)")
    ap.add_argument("--no-audio", action="store_true",
                    help="disable the microphone / PANNs audio thread")
    ap.add_argument("--audio-file", default=None,
                    help="run the audio sense on THIS file's soundtrack instead of "
                         "the microphone. Pair it with --source (same file) to "
                         "reproduce the live audio+video verdict on a recorded clip, "
                         "e.g. on Colab where there is no mic.")
    ap.add_argument("--require-audio", action="store_true",
                    help="OPT-IN: abort if the file-audio model can't initialize "
                         "(no audio track / ffmpeg / PANNs) instead of silently "
                         "continuing video-only. Default OFF = today's behavior.")
    ap.add_argument("--no-display", action="store_true",
                    help="don't open a window (headless; useful on a clip)")
    ap.add_argument("--save-video", default=None,
                    help="write an annotated MP4 (the same HUD burned into every "
                         "frame) to this path. Works even with --no-display, so you "
                         "can review a clip's detection where there is no window "
                         "(e.g. Google Colab).")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N frames (0 = unlimited)")
    ap.add_argument("--use-cnn", action="store_true",
                    help="use the trained yawn CNN (train/yawn_cnn.pt) for the yawn "
                         "cue instead of the MAR threshold")
    ap.add_argument("--require-cnn", action="store_true",
                    help="OPT-IN: with --use-cnn, abort if the yawn CNN can't load "
                         "instead of silently falling back to the MAR rule. Default "
                         "OFF = today's graceful fallback is preserved.")
    ap.add_argument("--log-dir", default=None,
                    help="OPT-IN: write events.csv / events.jsonl / snapshots under "
                         "THIS directory instead of the default logs/. Lets a second "
                         "run keep its own outputs for side-by-side comparison. "
                         "Default None = the usual logs/ (unchanged).")
    ap.add_argument("--debug", action="store_true",
                    help="print a real-time debug trace (speech detected, yawns, "
                         "drowsy events, and every state transition)")
    args = ap.parse_args()

    # One switch turns on all the dbg(...) traces across the modules.
    debug.set_enabled(args.debug)

    # Opt-in guards can only add strictness; they never fire on the default path.
    if args.require_audio and args.no_audio:
        ap.error("--require-audio conflicts with --no-audio")
    if args.require_cnn and not args.use_cnn:
        ap.error("--require-cnn has no effect without --use-cnn")

    use_file = args.source is not None
    
    if args.audio_file and not use_file:
        print("Warning: --audio-file without --source uses wall-clock timing and will "
            "drift out of sync. Pair it with --source (the same file).")

    # A file run with neither --audio-file nor --no-audio opens the MICROPHONE and
    # tags the room while "analyzing" the clip, mixing room audio into what reads
    # as a video-only result. Error rather than auto-disabling: an auto-fallback
    # changes behavior invisibly, one explicit flag does not.
    if use_file and not args.audio_file and not args.no_audio:
        ap.error("--source without --audio-file would use the MICROPHONE for the audio "
                 "sense, tagging the room instead of the clip. Pass --no-audio for a "
                 "video-only run, or --audio-file <same file> to use the clip's own "
                 "soundtrack.")

    cap = cv2.VideoCapture(args.source if use_file else args.camera)
    if not cap.isOpened():
        raise IOError(f"Could not open {'file ' + args.source if use_file else 'camera ' + str(args.camera)}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0

    # Build the pipeline.
    video = VideoModel(use_cnn=args.use_cnn, require_cnn=args.require_cnn)
    fusion = FusionEngine()
    alerter = AlertManager()
    # Default (no --log-dir): the usual logs/ paths from config, unchanged. When a
    # --log-dir is given, a second run keeps its own events.csv/jsonl/snapshots so
    # the two runs don't overwrite each other.
    src_id = args.source if use_file else f"camera:{args.camera}"
    if args.log_dir:
        from pathlib import Path
        _ld = Path(args.log_dir)
        logger = EventLogger(csv_path=_ld / "events.csv",
                             jsonl_path=_ld / "events.jsonl",
                             snapshot_dir=_ld / "snapshots",
                             source=src_id)
    else:
        logger = EventLogger(source=src_id)

    audio = None
    if not args.no_audio:
        if args.audio_file:
            # Recorded-clip audio: tag the file's own soundtrack with the SAME PANNs
            # model the mic uses, so a Colab/offline run matches the live verdict.
            print(f"Reading audio from file: {args.audio_file} "
                  "(first run downloads PANNs ~300 MB)...")
            from file_audio_model import FileAudioModel
            audio = FileAudioModel(args.audio_file)
            audio.start()
            print(f"  audio: {audio.status}")
            # OPT-IN only: turn the (default) silent video-only fallback into a
            # loud abort, so a run flagged --require-audio provably used the audio
            # model. Without the flag this block is skipped and behavior is today's.
            if args.require_audio and not audio.status.startswith("ready"):
                raise SystemExit(
                    f"--require-audio: audio model did not initialize "
                    f"({audio.status}). The clip needs an audio track and "
                    f"ffmpeg + panns-inference must be available.")
        elif not _AUDIO_AVAILABLE:
            print("Audio unavailable (audio_model import failed) - video only.")
        else:
            print("Starting audio thread (first run downloads PANNs ~300 MB)...")
            audio = AudioModel()
            audio.start()

    show_mesh = True
    # Optional annotated-video output. Created lazily on the first frame because we
    # need that frame's exact size for the writer. A saved run still needs the HUD
    # drawn, even when no window is shown.
    writer = None
    save_path = args.save_video

    start = time.time()
    frame_idx = 0
    print("Running. Press q/ESC to quit, m to toggle mesh." if not args.no_display
          else "Running headless...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream." if use_file else "Camera read failed.")
                break
            if not use_file:
                frame = cv2.flip(frame, 1)   # mirror the webcam, feels natural

            # Clock: video-time for a file, wall-clock for the webcam.
            now = (start + frame_idx / fps) if use_file else time.time()

            vid = video.process(frame, now=now)
            # Pass the clock: the mic model ignores `now`; the file model uses it to
            # return the audio window aligned with this frame's video time.
            audio_state = audio.get_audio_state(now) if audio is not None else "silence"
            fused = fusion.update(vid, audio_state, now=now)
            banner = alerter.update(fused.state, fused.reasons, now=now)

            if fused.is_new:
                logger.log(fused, vid, frame=frame, now=now)

            # Draw the HUD when we either show a window OR save a video.
            if not args.no_display or save_path is not None:
                _draw_hud(frame, vid, fused, audio, banner, show_mesh)

            if save_path is not None:
                if writer is None:                       # first frame -> open writer
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                    if not writer.isOpened():
                        raise IOError(f"Could not open VideoWriter for {save_path}")
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("Driver drowsiness - q quit, m mesh", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("m"):
                    show_mesh = not show_mesh

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                print(f"Reached --max-frames {args.max_frames}.")
                break
    finally:
        cap.release()
        video.close()
        if writer is not None:
            writer.release()
        if audio is not None:
            audio.stop()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Done. Processed {frame_idx} frames. "
          f"Alarms fired: {alerter.fire_count}. Log: {logger.csv_path}")
    if writer is not None:
        print(f"Annotated video written to: {save_path}")
    elif save_path is not None:
        print(f"No annotated video written to {save_path} (no frames were processed).")



if __name__ == "__main__":
    main()
