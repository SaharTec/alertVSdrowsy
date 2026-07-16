"""
video_model.py - the camera "sense".

Wraps a single MediaPipe Face Mesh and turns each webcam frame into a compact
per-frame reading of the driver:

    {ear, mar, yaw, pitch, roll,            # raw geometric features
     mms_amplitude, mms_oscillation,        # the Mouth Movement Score signal
     mouth_state,                           # 'speech' | 'yawn' | 'still'
     eyes_closed, yawning, nodding,         # boolean drowsiness cues
     state,                                 # ALERT | DROWSY | NEUTRAL (this frame)
     face_found, landmarks}                 # for the HUD

WHY a class and not a function: the speech-vs-yawn decision and the nod
detection both need MEMORY across frames (a yawn is a slow opening *held over
time*; a nod is a deviation *sustained*; MMS is motion *over a window*). The
class keeps that rolling state between calls to process().

The EAR / MAR / head-pose maths is the same proven logic the parent project
trained on (so a future trained model stays compatible); the MMS logic and the
rule-based per-frame state are new here.
"""
import time
from collections import deque
from datetime import datetime

import _tf_guard  # noqa: F401 - MUST precede mediapipe; blocks TF so it imports
import cv2
import numpy as np
import mediapipe as mp

from debug import dbg

from config import (
    RIGHT_EYE_INDICES, LEFT_EYE_INDICES, MOUTH_INDICES,
    HEAD_POSE_LANDMARKS, HEAD_POSE_3D_MODEL,
    INNER_LIP_TOP, INNER_LIP_BOTTOM, INTEROCULAR_LEFT, INTEROCULAR_RIGHT,
    NOSE_TIP, CHIN,
    EAR_CLOSED, EYE_CLOSED_SECONDS,
    MAR_YAWN, YAWN_MIN_SECONDS, YAWN_CNN_PROB,
    NOD_METRIC_DELTA, NOD_METRIC_MAX, NOD_MIN_SECONDS, NOD_BASELINE_ALPHA,
    NOD_SMOOTH_FRAMES, YAW_FACING_ROAD,
    MMS_WINDOW_SECONDS, MMS_YAWN_LEVEL, MMS_SPEECH_AMP_MAX, MMS_SPEECH_OSC_RATE_MIN,
    MMS_MOTION_EPS,
    STATE_ALERT, STATE_DROWSY, STATE_NEUTRAL,
)


# ---------------------------------------------------------------------------
# Small helper: how long has a condition been continuously true?
# A short gap (<= grace) does not break the streak, which absorbs the
# frame-to-frame jitter typical of landmark detection.
# ---------------------------------------------------------------------------
class _Streak:
    def __init__(self, grace=0.3):
        self.grace = grace
        self.start = None
        self.last_active = None

    def update(self, active, now):
        if active:
            if self.start is None:
                self.start = now
            self.last_active = now
        elif self.last_active is not None and now - self.last_active > self.grace:
            self.start = None
            self.last_active = None

    def duration(self, now):
        return 0.0 if self.start is None else now - self.start


# ---------------------------------------------------------------------------
# Geometry helpers (no per-frame state) - same maths as the parent project.
# ---------------------------------------------------------------------------
def _coords(landmarks, indices, w, h):
    return np.array(
        [[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
         for i in indices],
        dtype=np.float64,
    )


def _aspect_ratio(points):
    """Generic 6-point aspect ratio: (|p1-p5| + |p2-p4|) / (2*|p0-p3|)."""
    v1 = np.linalg.norm(points[1] - points[5])
    v2 = np.linalg.norm(points[2] - points[4])
    h = np.linalg.norm(points[0] - points[3])
    return 0.0 if h == 0 else (v1 + v2) / (2.0 * h)


def _head_pose(landmarks, w, h):
    """Return (yaw, pitch, roll) normalized to ~[-1, 1] via solvePnP."""
    image_points = _coords(landmarks, HEAD_POSE_LANDMARKS, w, h)
    focal = float(w)
    cam = np.array([[focal, 0, w / 2.0],
                    [0, focal, h / 2.0],
                    [0, 0, 1.0]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(HEAD_POSE_3D_MODEL, image_points, cam, dist,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy < 1e-6:  # gimbal lock
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1]); yaw = np.arctan2(-rmat[2, 0], sy); roll = 0.0
    else:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2]); yaw = np.arctan2(-rmat[2, 0], sy); roll = np.arctan2(rmat[1, 0], rmat[0, 0])
    return yaw / np.pi, pitch / np.pi, roll / np.pi


def _inner_lip_gap(landmarks, w, h):
    """Inner-lip opening, normalized by the eye-to-eye distance.

    This is the RAW MMS signal: how far apart the inner lips are, scaled so the
    number means the same whether the face is near or far from the camera.
    """
    top = np.array([landmarks.landmark[INNER_LIP_TOP].x * w,
                    landmarks.landmark[INNER_LIP_TOP].y * h])
    bot = np.array([landmarks.landmark[INNER_LIP_BOTTOM].x * w,
                    landmarks.landmark[INNER_LIP_BOTTOM].y * h])
    le = np.array([landmarks.landmark[INTEROCULAR_LEFT].x * w,
                   landmarks.landmark[INTEROCULAR_LEFT].y * h])
    re = np.array([landmarks.landmark[INTEROCULAR_RIGHT].x * w,
                   landmarks.landmark[INTEROCULAR_RIGHT].y * h])
    iod = np.linalg.norm(le - re)
    if iod == 0:
        return 0.0
    return float(np.linalg.norm(top - bot) / iod)


def _nod_metric(landmarks, w, h):
    """Where the nose tip sits within the eyes->chin span (0..1-ish).

    Drooping the head down tucks the chin and lowers this ratio. Scale- and
    translation-invariant, and free of the +-pi wrap that makes solvePnP pitch
    unusable for a threshold. Returns None if the face geometry is degenerate.
    """
    eyes_y = (landmarks.landmark[INTEROCULAR_LEFT].y +
              landmarks.landmark[INTEROCULAR_RIGHT].y) / 2.0 * h
    nose_y = landmarks.landmark[NOSE_TIP].y * h
    chin_y = landmarks.landmark[CHIN].y * h
    face_h = chin_y - eyes_y
    if face_h <= 1.0:
        return None
    return float((nose_y - eyes_y) / face_h)


class VideoModel:
    """Per-frame driver reader. Create once, call process(frame) every frame."""

    def __init__(self, use_cnn=False, require_cnn=False):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Optional trained yawn CNN. Opt-in and defensive: ANY failure (torch
        # missing, no checkpoint) disables it and the proven MAR rule is used, so
        # enabling the model can never break the pipeline (mirrors the audio thread).
        self._yawn_cnn = None
        if use_cnn:
            try:
                from yawn_cnn_infer import YawnCNN
                self._yawn_cnn = YawnCNN()
                print(f"[video] yawn CNN enabled (val F1={self._yawn_cnn.best_val_f1:.3f}); "
                      f"P(yawn) >= {YAWN_CNN_PROB} replaces MAR >= {MAR_YAWN}.")
            except Exception as exc:  # noqa: BLE001 - fall back to the rule path
                # OPT-IN strictness: only when the caller demanded the CNN do we
                # refuse to fall back. Default (require_cnn=False) keeps the exact
                # graceful MAR fallback below.
                if require_cnn:
                    raise RuntimeError(
                        f"--require-cnn: yawn CNN failed to load ({exc}). Ensure "
                        f"torch+torchvision are installed and train/yawn_cnn.pt exists."
                    ) from exc
                print(f"[video] yawn CNN unavailable ({exc}); using the MAR rule.")
                self._yawn_cnn = None
        # Rolling window of the normalized inner-lip gap -> MMS amplitude/oscillation.
        self._lip_window = deque()      # (timestamp, gap); trimmed by time in process()
        # Streak timers for the sustained-condition cues.
        self._eye_streak = _Streak()
        self._yawn_streak = _Streak()
        self._nod_streak = _Streak()
        # Rising-edge tracker so the yawn debug log fires ONCE per yawn, not
        # every frame the yawn stays detected.
        self._was_yawning = False
        # Slowly-learned resting nod-ratio, so nod detection auto-calibrates to
        # the camera mounting. None until the first valid metric is seen.
        self._nod_baseline = None
        # Median-filter buffer to remove single-frame spikes in the nod metric.
        self._nod_smooth = deque(maxlen=NOD_SMOOTH_FRAMES)
        # Forward-fill on missed detections (mirrors the parent preprocessing).
        # There is deliberately no _last_gap here: the lip window takes measured
        # gaps only, so a stale one would have no reader.
        self._last_valid = [0.0, 0.0, 0.0, 0.0, 0.0]   # ear, mar, yaw, pitch, roll
        self._last_nod_metric = 0.5

    # -- MMS: classify mouth motion over the rolling window ----------------
    def lip_gap_level(self):
        """How wide the mouth got over the window - the MMS 'level'.

        Public because the level is the reliable yawn cue and callers (e.g.
        train/extract_features.py) want to record it; reaching into _lip_window
        from outside would break every time its layout changes.
        """
        return max((g for _, g in self._lip_window), default=0.0)

    def _mouth_state(self):
        """Return ('speech' | 'yawn' | 'still', amplitude, oscillation, osc_rate).

        Calibrated from real clips (see scratchpad/mms_probe). The reliable yawn
        cue is the LEVEL the mouth reached (how wide it opened), not the travel
        within a half-second window - because a yawn is often held wide open, so
        its in-window travel is small even though the mouth is gaping. So we
        check "wide open" first; only a low, busy signal is called speech. This
        guarantees a wide opening is never mislabeled "speech" (the bug that
        suppressed yawn detection). Audio remains the true speech tie-breaker in
        the fusion layer.

        The speech test uses the sign-change RATE, not the raw count: the window
        holds however many samples the frame rate delivered, so a count would
        mean different things at 25 and 30 fps. The count is still returned for
        the logs, but nothing decides on it.
        """
        if len(self._lip_window) < 3:
            return "still", 0.0, 0, 0.0
        g = np.asarray([v for _, v in self._lip_window], dtype=np.float32)
        span = self._lip_window[-1][0] - self._lip_window[0][0]
        level = float(g.max())                  # how wide the mouth got
        amplitude = float(g.max() - g.min())    # how far it travelled

        # Oscillation = how many times the lip-gap velocity changes sign,
        # counting only motion above MOTION_EPS so resting jitter isn't counted.
        vel = np.diff(g)
        signs = np.sign(np.where(np.abs(vel) < MMS_MOTION_EPS, 0.0, vel))
        signs = signs[signs != 0]
        oscillation = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
        osc_rate = (oscillation / span) if span > 0 else 0.0

        # Wide opening -> yawn shape (held-open yawns included).
        if level >= MMS_YAWN_LEVEL:
            return "yawn", amplitude, oscillation, osc_rate
        # Small, busy motion -> talking.
        if amplitude <= MMS_SPEECH_AMP_MAX and osc_rate >= MMS_SPEECH_OSC_RATE_MIN:
            return "speech", amplitude, oscillation, osc_rate
        return "still", amplitude, oscillation, osc_rate

    def process(self, frame, now=None):
        """Read one BGR frame. Returns the per-frame dict described in the module docstring."""
        if now is None:
            now = time.time()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        face_found = bool(results.multi_face_landmarks)
        landmarks = results.multi_face_landmarks[0] if face_found else None

        if face_found:
            right = _coords(landmarks, RIGHT_EYE_INDICES, w, h)
            left = _coords(landmarks, LEFT_EYE_INDICES, w, h)
            mouth = _coords(landmarks, MOUTH_INDICES, w, h)
            ear = (_aspect_ratio(right) + _aspect_ratio(left)) / 2.0
            mar = _aspect_ratio(mouth)
            yaw, pitch, roll = _head_pose(landmarks, w, h)
            gap = _inner_lip_gap(landmarks, w, h)
            nm = _nod_metric(landmarks, w, h)
            nod_metric = nm if nm is not None else self._last_nod_metric
            self._last_valid = [ear, mar, yaw, pitch, roll]
            self._last_nod_metric = nod_metric
        else:
            # Forward-fill so the streak timing stays in real time. `gap` is not
            # forward-filled: see the lip-window append below.
            ear, mar, yaw, pitch, roll = self._last_valid
            gap = None
            nod_metric = self._last_nod_metric

        # ---- MMS over the rolling lip-gap window -------------------------
        # Trimmed by TIME, so the window spans MMS_WINDOW_SECONDS whatever the
        # frame rate - a fixed maxlen made the span a function of the hardware.
        # Only a MEASURED gap enters the window: the forward-filled value above
        # is a guess about a mouth we cannot see, and feeding it here let the
        # mouth cue keep reporting through a dropout (_nod_smooth below already
        # gates on face_found - this cue was the inconsistent one). Trimming
        # still runs, so a dropout drains the window by time and _mouth_state
        # falls back to "still": no face, no claim about the mouth.
        if face_found:
            self._lip_window.append((now, gap))
        cutoff = now - MMS_WINDOW_SECONDS
        while self._lip_window and self._lip_window[0][0] < cutoff:
            self._lip_window.popleft()
        mouth_state, mms_amp, mms_osc, mms_osc_rate = self._mouth_state()

        # ---- Nod: the nose-in-eyes->chin ratio drops below its resting value -
        facing_road = abs(yaw) <= YAW_FACING_ROAD
        # Smooth (median) to remove single-frame spikes.
        if face_found:
            self._nod_smooth.append(nod_metric)
        nm_s = float(np.median(self._nod_smooth)) if self._nod_smooth else nod_metric
        if self._nod_baseline is None:
            self._nod_baseline = nm_s
        # Drooping DOWN lowers the ratio, so the drowsy direction is baseline - nm.
        droop = self._nod_baseline - nm_s
        # A real droop is moderate and sustained; a drop beyond NOD_METRIC_MAX is
        # a tracking glitch, not a nod (and must not freeze the baseline).
        nod_now = face_found and NOD_METRIC_DELTA <= droop <= NOD_METRIC_MAX
        # Track the resting ratio whenever we're not in a genuine droop, so the
        # baseline self-calibrates and absorbs glitches.
        if face_found and not nod_now:
            self._nod_baseline += NOD_BASELINE_ALPHA * (nm_s - self._nod_baseline)
        self._nod_streak.update(nod_now, now)
        nodding = self._nod_streak.duration(now) >= NOD_MIN_SECONDS

        # ---- Eyes closed (sustained) -------------------------------------
        self._eye_streak.update(face_found and ear < EAR_CLOSED, now)
        eyes_closed = self._eye_streak.duration(now) >= EYE_CLOSED_SECONDS

        # ---- Yawn: mouth wide open, sustained, and NOT busy talking ------
        # "Mouth wide" is judged by the trained CNN when enabled, otherwise by the
        # MAR threshold. The sustained-duration timer (YAWN_MIN_SECONDS) and the
        # MMS speech-gate are unchanged - only the per-frame test differs.
        if self._yawn_cnn is not None and face_found:
            yawn_prob = self._yawn_cnn.yawn_prob(frame, landmarks)
            mouth_wide = yawn_prob >= YAWN_CNN_PROB
        else:
            yawn_prob = 0.0
            mouth_wide = face_found and mar >= MAR_YAWN
        wide_open = mouth_wide and mouth_state != "speech"
        self._yawn_streak.update(wide_open, now)
        yawning = self._yawn_streak.duration(now) >= YAWN_MIN_SECONDS

        # Debug: log the instant a new yawn is confirmed (rising edge), with its
        # timestamp. `now` is wall-clock on the webcam and a video-clock on a file.
        if yawning and not self._was_yawning:
            dbg(f"Yawn detected at {datetime.fromtimestamp(now).strftime('%H:%M:%S.%f')[:-3]} "
                f"(t={now:.2f})")
        self._was_yawning = yawning

        # ---- Per-frame state ---------------------------------------------
        # DROWSY if any sustained drowsiness cue is present.
        # ALERT if clearly engaged: eyes open, talking, facing the road.
        # Otherwise NEUTRAL (quiet / unclear - fusion + audio refine this).
        if eyes_closed or yawning or nodding:
            state = STATE_DROWSY
        elif (face_found and ear >= EAR_CLOSED and mouth_state == "speech"
              and facing_road):
            state = STATE_ALERT
        else:
            state = STATE_NEUTRAL

        return {
            "ear": ear, "mar": mar, "yaw": yaw, "pitch": pitch, "roll": roll,
            "nod_metric": round(nm_s, 4),
            "mms_amplitude": mms_amp, "mms_oscillation": mms_osc,
            "mms_osc_rate": round(mms_osc_rate, 3),   # the value the speech test uses
            "yawn_prob": round(yawn_prob, 3),   # CNN P(yawn); 0.0 when CNN disabled
            "mouth_state": mouth_state,
            "eyes_closed": eyes_closed, "yawning": yawning, "nodding": nodding,
            "facing_road": facing_road,
            "state": state,
            "face_found": face_found,
            "landmarks": landmarks,
        }

    def close(self):
        self.face_mesh.close()


# ---------------------------------------------------------------------------
# Self-test. No real face is available here, so the tests that need landmark
# geometry drive _mouth_state by filling _lip_window directly - legitimate for a
# same-module unit test, and the only way to exercise the MMS maths offline.
# Run: python video_model.py   (exits non-zero on failure)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    _passed = True

    def check(name, ok, detail=""):
        global _passed
        _passed &= bool(ok)
        print(f"  [{'OK' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    print("video_model self-test:")

    # -- 1. process() survives a frame with no face ------------------------
    vm = VideoModel()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    out = vm.process(blank)
    check("blank frame -> no face", out["face_found"] is False)
    check("blank frame -> NEUTRAL", out["state"] == STATE_NEUTRAL,
          f"got {out['state']}")
    check("blank frame -> mouth 'still'", out["mouth_state"] == "still",
          f"got {out['mouth_state']}")

    # -- 2. _Streak: onset, grace, and reset -------------------------------
    s = _Streak(grace=0.3)
    check("streak: idle -> 0s", s.duration(0.0) == 0.0)
    for t in (0.0, 0.5, 1.0):
        s.update(True, t)
    check("streak: 1s of activity -> 1s", s.duration(1.0) == 1.0)
    s.update(False, 1.2)          # a 0.2s gap is inside the 0.3s grace
    check("streak: blink inside grace survives", s.duration(1.2) == 1.2,
          f"got {s.duration(1.2)}")
    s.update(False, 1.4)          # 0.4s since last active: past grace
    check("streak: gap past grace resets", s.duration(1.4) == 0.0,
          f"got {s.duration(1.4)}")

    # -- 3. MMS is frame-rate invariant (the Phase 3.1 regression) ---------
    # A 6 Hz lip oscillation is talking at any frame rate. Before the window
    # became time-based and oscillation became a RATE, the same mouth read
    # differently at 25 and 30 fps - a hardware-dependent verdict.
    def mms_at(fps, signal):
        m = VideoModel.__new__(VideoModel)      # no FaceMesh needed for _mouth_state
        m._lip_window = deque()
        n = int(MMS_WINDOW_SECONDS * fps)
        for i in range(n):
            t = i / fps
            m._lip_window.append((t, signal(t)))
        return m._mouth_state()[0]

    talking = lambda t: 0.05 + 0.03 * np.sin(2 * np.pi * 6.0 * t)
    at25, at30 = mms_at(25.0, talking), mms_at(30.0, talking)
    check("MMS: talking reads 'speech' at 25 fps", at25 == "speech", f"got {at25}")
    check("MMS: talking reads 'speech' at 30 fps", at30 == "speech", f"got {at30}")
    check("MMS: 25 fps and 30 fps agree", at25 == at30, f"{at25} vs {at30}")

    wide = lambda t: 0.40                        # a held-open yawn: no travel
    check("MMS: held-open mouth reads 'yawn' at both rates",
          mms_at(25.0, wide) == "yawn" == mms_at(30.0, wide))

    # -- 4. Face dropout drains the lip window (the Phase 3.3 regression) --
    # The forward-filled gap used to enter the window, so the mouth cue kept
    # reporting a mouth nobody could see: a yawn frozen at the moment tracking
    # was lost would read "yawn" forever. Now only measured gaps enter, so the
    # window ages out and the cue goes quiet.
    for i in range(15):                          # seed a wide-open mouth
        vm._lip_window.append((i / 30.0, 0.40))
    check("dropout: window starts on a yawn", vm._mouth_state()[0] == "yawn")
    t = 0.5
    for _ in range(30):                          # 1s of no-face frames
        out = vm.process(blank, now=t)
        t += 1 / 30.0
    check("dropout: stale gaps do not persist", len(vm._lip_window) == 0,
          f"{len(vm._lip_window)} samples left")
    check("dropout: mouth cue falls silent", out["mouth_state"] == "still",
          f"got {out['mouth_state']}")
    vm.close()

    print("video_model self-test", "OK" if _passed else "had FAILURES")
    sys.exit(0 if _passed else 1)
