"""
config.py - single source of truth for the drowsiness detector.

Everything you might want to tune lives here: MediaPipe landmark indices, the
EAR / MAR / head-pose / MMS thresholds, the fusion timing, the alert messages
and the log paths. The other modules import from here and never hard-code a
magic number, so tuning the system means editing ONE file.

If a `learned_thresholds.json` file (produced by `train/`) sits next to this
file, its values override the defaults below at import time - so you can fit
thresholds to data without touching code. See `_apply_learned_overrides()`.

Units note: head pose (yaw/pitch/roll) is normalized to roughly [-1, 1] by the
head-pose math (radians / pi), so all the pose thresholds here are on that
scale, NOT in degrees.
"""
from pathlib import Path

import numpy as np

# ===========================================================================
# Paths
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_CSV = LOG_DIR / "events.csv"
LOG_JSONL = LOG_DIR / "events.jsonl"
SNAPSHOT_DIR = LOG_DIR / "snapshots"
LEARNED_THRESHOLDS_FILE = BASE_DIR / "learned_thresholds.json"

# ===========================================================================
# Unified output states (what fusion.py emits and alert.py reacts to)
# ===========================================================================
STATE_ALERT = "ALERT"      # awake & engaged: talking, eyes open, facing road
STATE_DROWSY = "DROWSY"    # yawning / eyes closing / head nodding
STATE_NEUTRAL = "NEUTRAL"  # not clearly either (e.g. quiet, looking around)

# ===========================================================================
# MediaPipe FaceMesh landmark indices
# ---------------------------------------------------------------------------
# Eye / mouth / head-pose sets are reused from the parent project (proven to
# give a clean EAR/MAR/pose). The MMS lip points (13/14) and the outer-eye
# normalization corners (33/263) are added for the Mouth Movement Score.
# ===========================================================================
# Eyes - 6 points each, ordered for the generic aspect-ratio formula
# (|p1-p5| + |p2-p4|) / (2*|p0-p3|).
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Mouth - 6 points, same aspect-ratio formula -> MAR (rises when mouth opens).
MOUTH_INDICES = [78, 82, 312, 308, 317, 87]

# Head pose - nose tip, chin, eye outer corners, mouth corners. Matched against
# HEAD_POSE_3D_MODEL via cv2.solvePnP to recover yaw/pitch/roll.
HEAD_POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]
HEAD_POSE_3D_MODEL = np.array([
    (0.0,    0.0,    0.0),     # nose tip
    (0.0,   -330.0, -65.0),    # chin
    (-225.0, 170.0, -135.0),   # left eye outer corner
    (225.0,  170.0, -135.0),   # right eye outer corner
    (-150.0, -150.0, -125.0),  # left mouth corner
    (150.0,  -150.0, -125.0),  # right mouth corner
], dtype=np.float64)

# --- Nod-detection landmarks (2D geometry, not solvePnP) -------------------
# Nose tip and chin, combined with the eye corners above, give a stable "how far
# down is the head" ratio (see HEAD-NOD section). solvePnP pitch proved unusable.
NOSE_TIP = 1
CHIN = 152

# --- MMS landmarks (the novel part of this project) ------------------------
# Inner-lip gap: upper inner lip (13) to lower inner lip (14). Its size over
# time is the raw MMS signal. We normalize it by the inter-ocular distance
# (outer eye corners 33 <-> 263) so the score is invariant to how close the
# driver's face is to the dashboard camera.
INNER_LIP_TOP = 13
INNER_LIP_BOTTOM = 14
INTEROCULAR_LEFT = 33
INTEROCULAR_RIGHT = 263

# All landmark dots we draw on the HUD (for the mesh overlay).
DRAW_INDICES = sorted(set(
    RIGHT_EYE_INDICES + LEFT_EYE_INDICES + MOUTH_INDICES +
    [INNER_LIP_TOP, INNER_LIP_BOTTOM]
))

# ===========================================================================
# Eye-closure (drowsiness) thresholds
# ===========================================================================
EAR_CLOSED = 0.20              # EAR below this = eyes considered closed
EYE_CLOSED_SECONDS = 1.0       # closed continuously this long -> drowsy signal

# ===========================================================================
# Yawn thresholds (mouth wide open, sustained - NOT brief talking)
# ===========================================================================
MAR_YAWN = 0.60                # MAR above this = mouth wide open
YAWN_MIN_SECONDS = 1.2 
YAWN_COUNT_REQUIRED = 3
YAWN_WINDOW_SECONDS = 30.0
# open this long -> counts as a yawn (vs a word)
# Optional trained-CNN yawn cue (opt-in via VideoModel(use_cnn=True) / --use-cnn).
# When enabled, the per-frame "mouth is yawning" test becomes P(yawn) >= this
# (from train/yawn_cnn.pt) instead of MAR >= MAR_YAWN. The sustained-duration
# rule (YAWN_MIN_SECONDS) and everything downstream are unchanged.
YAWN_CNN_PROB = 0.50           # CNN P(yawn) above this = mouth wide open (yawn shape)

# ===========================================================================
# Head-nod (drooping) detection
# ---------------------------------------------------------------------------
# IMPORTANT: solvePnP pitch turned out to be bistable/unreliable on real webcams
# (it sits near the +-pi wrap and flips between two solutions), which produced
# constant false "nodding". So the nod cue is a pure 2D-geometry metric instead:
#
#     nod_metric = (nose_tip_y - eye_line_y) / (chin_y - eye_line_y)
#
# i.e. where the nose sits within the eyes->chin span. Drooping the head DOWN
# tucks the chin and LOWERS this ratio. It is scale- and translation-invariant
# and has no wraparound, so it's stable frame to frame (measured std ~0.03 on
# real clips). video_model tracks a slow baseline and flags a nod when the ratio
# drops below it by NOD_METRIC_DELTA for NOD_MIN_SECONDS - self-calibrating to
# wherever the camera sits.
# ===========================================================================
NOD_METRIC_DELTA = 0.06        # drop below the resting ratio that = drooping
NOD_METRIC_MAX = 0.30          # a drop beyond this = tracking glitch -> ignored
NOD_MIN_SECONDS = 0.8          # sustained this long -> nodding signal
NOD_BASELINE_ALPHA = 0.02      # EMA rate for the slow resting-ratio baseline
NOD_SMOOTH_FRAMES = 7          # median-filter window that removes single-frame spikes

# Head turned away from the road (yaw) - used so we don't call a sideways
# glance "alert facing the road". Not a drowsiness signal by itself, and only
# affects the (non-alarm) ALERT label, so its solvePnP noise is harmless here.
YAW_FACING_ROAD = 0.35         # |yaw| below this = looking ahead

# ===========================================================================
# Mouth Movement Score (MMS) - speech vs yawn discriminator
# ---------------------------------------------------------------------------
# Over a short rolling window of the normalized inner-lip gap we measure:
#   amplitude     = max - min over the window (how far the mouth travels)
#   oscillation   = number of velocity sign-changes (how "busy" the motion is)
# Speech  -> small amplitude + high oscillation (rhythmic chatter).
# Yawning -> large amplitude + low oscillation (one slow wide opening).
# ===========================================================================
MMS_WINDOW_FRAMES = 15         # ~0.5 s at 30 fps - one window of lip motion
# Calibrated from real YawDD clips (scratchpad/mms_probe): a yawn's inner-lip
# gap reaches a LEVEL of ~0.5-0.8, singing ~0.42, talking ~0.02. The "level"
# (how wide the mouth got) is the reliable yawn cue; the sign-change count is a
# weak speech cue on this data, so it only labels low-amplitude busy motion.
MMS_YAWN_LEVEL = 0.30          # window-max gap above this = a wide (yawn) opening
MMS_SPEECH_AMP_MAX = 0.18      # speech stays below this travel amplitude
MMS_SPEECH_OSC_MIN = 2         # speech has at least this many sign-changes/window
MMS_YAWN_AMP_MIN = 0.25        # (kept for reference) large travel also = yawn
MMS_MOTION_EPS = 0.02          # |velocity| below this is jitter, not motion

# ===========================================================================
# Fusion (combine video + audio into one state every ~1-2 s)
# ===========================================================================
FUSION_INTERVAL_SECONDS = 1.5  # emit a unified state this often
FUSION_MIN_EVIDENCE = 0.08     # below this total evidence -> stay NEUTRAL (avoids
                               # declaring ALERT/DROWSY on a negligible signal)
AUDIO_SPEECH_THRESH = 0.20     # audio "speech/singing" score above this = voice
AUDIO_YAWN_THRESH = 0.20       # audio "yawn" score above this = audible yawn
# How strongly each agreeing signal counts toward DROWSY confidence (weighted
# fusion). Tuned so any single strong visual cue can carry the decision but
# agreement across cues pushes confidence higher.
WEIGHT_EYES = 0.45             # eye-closure cue -> DROWSY evidence
WEIGHT_YAWN = 0.35             # yawn cue        -> DROWSY evidence
WEIGHT_NOD = 0.30              # head-nod cue    -> DROWSY evidence
WEIGHT_SPEECH = 0.45           # visual speech   -> ALERT evidence (mirror of eyes)
WEIGHT_AUDIO = 0.20            # audio agreement -> nudges whichever side it supports

# ===========================================================================
# Alerts
# ===========================================================================
DROWSY_HOLD_SECONDS = 1.5      # fused state must stay DROWSY this long -> speak
ALERT_COOLDOWN_SECONDS = 8.0   # min gap between repeated spoken warnings
RECOVERY_SECONDS = 1.5         # back to ALERT this long -> re-arm the alarm
ALERT_MESSAGE = "Warning. You appear drowsy. Please pull over and take a rest."
# Cause-specific variant: spoken when the DROWSY state was driven by yawning
# (Rule 5), so the warning matches what happened. Falls back to ALERT_MESSAGE
# for eye-closure / head-nod drowsiness.
YAWN_ALERT_MESSAGE = "You keep yawning and may be getting drowsy. Please take a break."
BEEP_FREQ_HZ = 1000            # winsound.Beep fallback tone
BEEP_MS = 600

# ===========================================================================
# Misc runtime
# ===========================================================================
CAMERA_INDEX = 0               # default webcam
SNAPSHOT_ON_DROWSY = True      # save a frame when we enter DROWSY


def _apply_learned_overrides():
    """Override any constant above from learned_thresholds.json, if present.

    The JSON is a flat {"NAME": value} map (e.g. {"EAR_CLOSED": 0.18}). Only
    keys that already exist as module-level UPPER_CASE constants are applied,
    so a stray key can never inject something unexpected. Silent no-op if the
    file is missing or unreadable - the defaults above always work on their own.
    """
    try:
        import json
        if not LEARNED_THRESHOLDS_FILE.exists():
            return
        data = json.loads(LEARNED_THRESHOLDS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - never let bad JSON break startup
        print(f"[config] ignoring learned_thresholds.json ({exc})")
        return

    g = globals()
    applied = []
    for key, value in data.items():
        if key.isupper() and key in g and isinstance(g[key], (int, float)):
            g[key] = value
            applied.append(key)
    if applied:
        print(f"[config] applied learned thresholds: {', '.join(applied)}")


_apply_learned_overrides()
