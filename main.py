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
    ap.add_argument("--no-display", action="store_true",
                    help="don't open a window (headless; useful on a clip)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N frames (0 = unlimited)")
    ap.add_argument("--use-cnn", action="store_true",
                    help="use the trained yawn CNN (train/yawn_cnn.pt) for the yawn "
                         "cue instead of the MAR threshold")
    ap.add_argument("--debug", action="store_true",
                    help="print a real-time debug trace (speech detected, yawns, "
                         "drowsy events, and every state transition)")
    args = ap.parse_args()

    # One switch turns on all the dbg(...) traces across the modules.
    debug.set_enabled(args.debug)

    use_file = args.source is not None
    cap = cv2.VideoCapture(args.source if use_file else args.camera)
    if not cap.isOpened():
        raise IOError(f"Could not open {'file ' + args.source if use_file else 'camera ' + str(args.camera)}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0

    # Build the pipeline.
    video = VideoModel(use_cnn=args.use_cnn)
    fusion = FusionEngine()
    alerter = AlertManager()
    logger = EventLogger()

    audio = None
    if not args.no_audio:
        if not _AUDIO_AVAILABLE:
            print("Audio unavailable (audio_model import failed) - video only.")
        else:
            print("Starting audio thread (first run downloads PANNs ~300 MB)...")
            audio = AudioModel()
            audio.start()

    show_mesh = True
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
            audio_state = audio.get_audio_state() if audio is not None else "silence"
            fused = fusion.update(vid, audio_state, now=now)
            banner = alerter.update(fused.state, fused.reasons, now=now)

            if fused.is_new:
                logger.log(fused, vid, frame=frame, now=now)

            if not args.no_display:
                _draw_hud(frame, vid, fused, audio, banner, show_mesh)
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
        if audio is not None:
            audio.stop()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Done. Processed {frame_idx} frames. "
          f"Alarms fired: {alerter.fire_count}. Log: {logger.csv_path}")


if __name__ == "__main__":
    main()
