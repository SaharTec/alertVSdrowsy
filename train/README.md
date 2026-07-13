# `train/` — fit the detector to your data (Google Colab)

The live system ships with thresholds calibrated on public YawDD clips. This
folder lets you **re-fit those thresholds to your own drivers / camera / lighting**
and export a `learned_thresholds.json` that the runtime picks up automatically.

It deliberately reuses the **exact same feature maths** as the live system
(`extract_features.py` drives the runtime `VideoModel`), so any threshold you fit
means the same thing at runtime — no train/serve skew.

> Heavy extraction over a full dataset is CPU-bound. Run it in **Colab**, not on
> the local machine. The local smoke commands below are only to prove the code
> path.

---

## Files
- **`annotate_yawns.py`** — *(optional but recommended)* a manual OpenCV tool to
  mark, per `Yawning/` clip, the **true yawn span(s)**. Writes
  `yawn_segments.json`. See **"Why annotate yawns"** below.
- **`extract_features.py`** — walks `<videos>/<Label>/*.mp4` and writes one CSV
  row per frame: `ear, mar, yaw, pitch, roll, lip_gap_level, mms_amplitude,
  mms_oscillation, mouth_state`. If `yawn_segments.json` is present it relabels
  Yawning clips per frame (see below); otherwise it labels every frame by folder.
- **`train_drowsiness.ipynb`** — the Colab notebook: mount Drive → extract →
  inspect each class → fit `MAR_YAWN`, `MMS_YAWN_LEVEL`, `EAR_CLOSED` (Youden's J
  on the class distributions) → write `learned_thresholds.json` back to the
  project folder.

---

## Why annotate yawns (precision fix)
A clip in `Yawning/` is mostly **not** a yawn — there's a closed-mouth lead-in and
trail-out around the one real yawn. Labelling every frame `Yawning` poisons the
positive class with closed-mouth frames, which drags `MAR_YAWN`/`MMS_YAWN_LEVEL`
**down** and makes the live system fire on ordinary talking. `annotate_yawns.py`
fixes this by giving each frame one of three labels:

| label | what it is | used as |
|---|---|---|
| **YAWN** | inside a marked yawn span | the positive class (`Yawning`) |
| **NEUTRAL** | closed mouth / normal posture, outside the span | a **negative** |
| **IGNORE** | ambiguous transition (already tired, not yet yawning) | **dropped** |

You can mark **several yawn spans per clip** (`g`/`h` repeatedly) and overlay
IGNORE bands (`i`/`o`). IGNORE wins over YAWN where they overlap. Output:
```json
{ "Yawning/clip.mp4": { "yawns": [[34,183]], "ignore": [], "fps": 29.97, "n_frames": 387 } }
```

**Run the annotator (needs a display, so run it locally — it's lightweight):**
```bash
cd train
python annotate_yawns.py --videos ../../data/raw_videos       # whole Yawning/ folder
python annotate_yawns.py --clip ../../data/raw_videos/Yawning/clip.mp4   # one clip
python annotate_yawns.py                                       # no args: run self-test
```
Keys are drawn on screen: `Space` play · `a`/`d` step · `[`/`]` jump10 ·
`g`/`h` add-yawn · `x` del-yawn · `i`/`o` add-ignore · `c` del-ignore ·
`r` reset · `n`/`p` next/prev clip · `q`/ESC save+quit. It resumes from an
existing `yawn_segments.json`. The notebook (cell 6) adds the new `Neutral` rows
to the yawn-threshold negatives automatically.

---

## Dataset layout (in Google Drive)
```
MyDrive/
  alertVSdrowsy/         <- the whole project folder (so the notebook can import it)
  dms_data/
    Yawning/   *.mp4
    Singing/   *.mp4      <- the key confuser for yawning; collect several
    Alert/     *.mp4      <- talking / awake, facing the road
    Drowsy/    *.mp4      <- head nodding, heavy eyes
    Sleeping/  *.mp4      <- eyes closed for seconds
    Distracted/*.mp4      <- looking away (not drowsy)
```
You can combine **your own dashboard-angle clips** with public sets
(**YawDD**, **NTHU-DDD**). See the data-collection & augmentation guidance in the
top-level [`../README.md`](../README.md).

---

## Run it

**Step 0 — annotate (local, optional but recommended):** run
`annotate_yawns.py` over your `Yawning/` clips to produce `train/yawn_segments.json`.
Because the notebook imports the whole project folder from Drive, **upload that
JSON to `MyDrive/alertVSdrowsy/train/`** so the Colab extraction picks it up. Skip
this and everything still works the old way (whole-clip `Yawning` labels).

**In Colab** (the real run): open `train_drowsiness.ipynb` and run the cells top
to bottom. It mounts Drive, extracts features from `dms_data/`, fits the
thresholds, and saves `MyDrive/alertVSdrowsy/learned_thresholds.json`.

**Local smoke test** (tiny, just to verify the path works):
```bash
cd train
python extract_features.py --videos ../../data/raw_videos --out features.csv \
    --max-clips-per-class 2 --max-frames 120
```

**Train fully on your own machine (no Colab)** — same maths, two commands:
```bash
cd train
# 1) extract features from ALL clips (CPU-heavy; this is the slow part)
python extract_features.py --videos ../../data/raw_videos --out features.csv
# 2) fit + write learned_thresholds.json next to config.py (fast)
python fit_thresholds.py --features features.csv
```
`fit_thresholds.py` is the script form of notebook cells 4-7 (Youden's J), so the
result is identical to the Colab run. It prints the per-class frame counts and the
fitted `MAR_YAWN` / `MMS_YAWN_LEVEL` / `EAR_CLOSED` (with each cut's separation J),
and only writes a threshold whose `J > 0.5`. Then just `python main.py` — the live
system loads the JSON automatically.

---

## How the fitted file is used
At startup, `config.py` looks for `learned_thresholds.json` next to itself. If
present, it overrides only the matching numeric thresholds and prints which ones
it applied. Delete the file to return to the shipped defaults. Because only keys
with a **good class separation (Youden's J > 0.5)** are written, a weak fit
can't quietly make the live system worse than its calibrated defaults.

---

## Deep-learning option: a fine-tuned yawn CNN (transfer learning)
Alongside the rule/threshold path, this folder has a self-contained **transfer-
learning** pipeline that fine-tunes a pretrained **MobileNetV2** to detect yawns
from **mouth crops** — a proper trained model with a loss function, plus an honest
evaluation against the rule baseline. It is **additive and opt-in**: the live
system is untouched and keeps running on rules.

Three steps (needs `torch` + `torchvision`, already installed):
```bash
cd train
# 1) build the mouth-crop dataset (CPU-heavy). Labels come from yawn_segments.json:
#    YAWN->yawn, NEUTRAL->not-yawn, IGNORE dropped; Singing/Alert are hard negatives.
python build_mouth_dataset.py --videos ../../data/raw_videos
# 2) fine-tune MobileNetV2 (fast on GPU). Freezes the backbone, trains the head,
#    then unfreezes; class-weighted cross-entropy; SUBJECT-GROUPED hold-out split.
python train_yawn_cnn.py --manifest mouth_manifest.csv --epochs 10
# 3) score the CNN vs the rule baseline (mar >= MAR_YAWN) on the held-out people:
python evaluate_yawn.py --model yawn_cnn.pt --manifest mouth_manifest.csv
```
Notes:
- **Subject-grouped split** (`GroupShuffleSplit`) keeps each person entirely in
  train *or* validation — no face leakage, so the metrics are trustworthy.
- Reported metrics are **Precision / Recall / F1 on the yawn class + confusion
  matrix** (accuracy alone lies under class imbalance).
- Every script has a tiny smoke mode (`--max-*`, `--subset`, `--epochs 1`,
  `--no-pretrained`) to verify the path before a full run.
- This does **not** change the live detector. Wiring the CNN into `main.py` would
  be a separate, optional step (same opt-in idea as `learned_thresholds.json`).

## Want an LSTM instead?
The **parent project** trains a sequence model on the per-frame feature rows with
subject-grouped cross-validation (`train.py`, `cross_validate.py`,
`src/dataset.py`) — point those at the same clips if you'd rather have a learned
LSTM than the CNN or the rule thresholds. All three paths are complementary.
