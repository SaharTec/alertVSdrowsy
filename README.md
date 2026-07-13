# Driver Drowsiness & Alertness Detection System

A real-time, in-vehicle system that watches the driver through a dashboard
webcam (and listens through the microphone) and decides whether they are
**ALERT** (awake, talking, looking at the road) or **DROWSY** (yawning, eyes
closing, head nodding). When drowsiness persists it speaks a warning and logs
the event with a snapshot.

It is a **dual-model pipeline with rule-based fusion**, built around a novel
**Mouth Movement Score (MMS)** that tells *talking* apart from *yawning* — the
one case a camera alone cannot resolve (an open mouth looks the same either way).

> A plain-English, step-by-step walkthrough of every module lives in
> [`explaine.txt`](explaine.txt).

---

## How it works (at a glance)

```
            ┌───────────────┐
  webcam ─► │ video_model.py│ EAR / MAR / head-pose / MMS ─► per-frame state ┐
            └───────────────┘                                                │
                                                                             ▼
            ┌───────────────┐                                        ┌──────────────┐
  mic   ──► │ audio_model.py│ speech / yawn / silence ─────────────► │  fusion.py   │
            └───────────────┘  (PANNs Cnn14, background thread)      │ 1 verdict /  │
                                                                     │  ~1.5 s      │
                                                                     └──────┬───────┘
                                                          ALERT/DROWSY/NEUTRAL
                                              ┌───────────────────┼───────────────────┐
                                              ▼                   ▼                   ▼
                                        ┌──────────┐        ┌──────────┐        ┌──────────┐
                                        │ alert.py │ speak  │logger.py │ CSV+   │  HUD on  │
                                        │ + cooldown│ warn   │+snapshot │ JSONL  │  screen  │
                                        └──────────┘        └──────────┘        └──────────┘
```

- **Video** (`video_model.py`): MediaPipe Face Mesh → eye-closure (EAR), mouth
  opening (MAR), head nod (self-calibrating pitch), and the **MMS** (how wide /
  how busy the lips move) → a per-frame ALERT/DROWSY/NEUTRAL.
- **Audio** (`audio_model.py`): PANNs Cnn14 sound tagger on a background thread
  → `speech` / `yawn` / `silence`. Fully optional and self-disabling.
- **Fusion** (`fusion.py`): averages ~1.5 s of both senses into one explainable
  verdict with a confidence and reasons. Audio breaks the talk-vs-yawn tie.
- **Alert** (`alert.py`): speaks a warning (pyttsx3, background thread; beep
  fallback) once DROWSY is held, repeats on a cooldown, re-arms on recovery.
- **Logger** (`logger.py`): every verdict → `logs/events.csv` + `events.jsonl`;
  a snapshot is saved on each transition into DROWSY.
- **Config** (`config.py`): every threshold in one place; optionally overridden
  by `learned_thresholds.json` produced by `train/`.

---

## Install

```bash
pip install -r requirements.txt
```

### TensorFlow note (important on this machine)
MediaPipe and TensorFlow clash in this environment — if TensorFlow is installed,
`import mediapipe` can crash. Two options:

1. **Do nothing** — the bundled [`_tf_guard.py`](_tf_guard.py) blocks the TF
   import automatically so MediaPipe loads cleanly. This keeps the folder
   self-contained.
2. **Remove TF globally** (cleaner if you don't need it elsewhere):
   ```bash
   pip uninstall -y tensorflow tensorflow-hub
   ```

The system never uses TensorFlow itself (the audio model runs on PyTorch).

---

## Run

```bash
python main.py                     # live webcam + microphone
python main.py --no-audio          # webcam only (skip the mic / PANNs)
python main.py --source clip.mp4   # run on a recorded video
python main.py --source clip.mp4 --no-display   # headless (no window)
```

Keys (when a window is shown): **q**/**ESC** quit, **m** toggle the face mesh.

Each module also self-tests on its own, e.g. `python video_model.py`,
`python fusion.py`, `python alert.py`, `python logger.py`.

---

## Tuning

All thresholds live in [`config.py`](config.py) — EAR/MAR/MMS levels, the
sustained-duration requirements, fusion weights, alert cooldown, etc. Edit there;
nothing is hard-coded elsewhere. The current values were calibrated against real
YawDD clips (a yawn reaches MAR > 1.0 and a lip-gap "level" of 0.5–0.8; singing
only ~0.57 / 0.42; talking ~0.02).

If you fit thresholds to your own data via `train/`, drop the resulting
`learned_thresholds.json` next to `config.py` and it overrides the defaults at
startup — no code change needed.

---

## Collecting your own data (and augmentation)

For best results, record short (5–15 s) dashboard-angle clips of **yourself and
a few different people**, labelled by behaviour, and mix them with public sets
(YawDD, NTHU-DDD). Aim for variety where it matters:

**What to collect**
- **Yawning**: real yawns, mouth wide and held open (the key drowsy cue).
- **Drowsy / nodding**: head slowly drooping down, eyes heavy/half-closed.
- **Sleeping**: eyes fully closed for several seconds.
- **Alert / talking**: speaking normally while facing the road (the main
  not-drowsy case the MMS must separate from yawning).
- **Singing / wide talking**: mouth opens wide but rhythmically — the hardest
  confuser for yawning; collect several.
- Vary: **glasses / no glasses**, **lighting** (bright day, dusk, night with
  dash light), **camera angle/height**, **beards/face shapes**, and **head
  turns** (so "looking away" isn't mistaken for drowsy).

**Augmentation** (apply in training to multiply each clip cheaply)
- **Brightness / contrast / gamma** jitter → robustness to day vs night.
- **Small rotations (±10°), crops, and translations** → robustness to where the
  camera is mounted.
- **Horizontal flip** → driver position / mirrored cameras.
- **Mild Gaussian noise / blur** → cheap cameras and motion.
- Keep augmentation **mild and realistic** — a dashboard camera never sees a
  driver upside-down, so don't augment in ways that can't happen in the car.

---

## Training (Google Colab)

See [`train/README.md`](train/README.md). In short: the `train/` folder extracts
the same features used at runtime from your labelled clips, fits thresholds (or a
small model) in a Colab notebook with the data mounted from Google Drive, and
exports `learned_thresholds.json` back here. Heavy training is meant to be run by
you in Colab, not on the local CPU.

---

## Files

| File | Role |
|------|------|
| `config.py` | every tunable threshold + landmark indices |
| `video_model.py` | camera sense: EAR/MAR/pose/MMS → per-frame state |
| `audio_model.py` | mic sense: PANNs Cnn14 → speech/yawn/silence |
| `fusion.py` | combine both into one verdict every ~1.5 s |
| `alert.py` | spoken warning + cooldown |
| `logger.py` | CSV/JSONL event log + drowsy snapshots |
| `main.py` | entry point: wires it all + HUD |
| `_tf_guard.py` | keeps MediaPipe importable despite TensorFlow |
| `explaine.txt` | plain-English walkthrough of every step |
| `train/` | Colab feature extraction + threshold fitting |
```
