"""
yawn_cnn_infer.py - run the trained yawn CNN on a live frame (opt-in).

This is the runtime bridge for the model produced by train/train_yawn_cnn.py
(a fine-tuned MobileNetV2 that classifies a MOUTH CROP as yawn / not-yawn). It is
ADDITIVE and OPT-IN, exactly like the audio thread and learned_thresholds.json:
the live system only touches it when VideoModel is built with use_cnn=True, so a
missing torch install or a bad checkpoint can never break the default rule path.

Given a full BGR frame + MediaPipe landmarks it:
  1. crops a square box around the mouth - the SAME box train/build_mouth_dataset.py
     used to make the training data (same indices, same margin), so there is no
     train/serve skew;
  2. resizes + normalizes exactly as the val transform did (img_size + ImageNet
     mean/std, both read from the checkpoint - not hard-coded);
  3. returns P(yawn) in [0, 1].

The crop maths (MOUTH_BOX_INDICES / _lm_points / mouth_box) is deliberately COPIED
from train/build_mouth_dataset.py rather than imported: that module imports
VideoModel at load time, so importing it here (and here from video_model) would be
a circular import. Keeping the ~15 lines local makes this runtime module
self-contained and torch-only when actually used. If you change the crop in
build_mouth_dataset.py, mirror it here (and retrain).

torch / torchvision are imported LAZILY inside __init__ so importing this module
stays cheap and the rule path never pays for them.
"""
from pathlib import Path

import cv2
import numpy as np

from config import MOUTH_INDICES, INNER_LIP_TOP, INNER_LIP_BOTTOM

# Mouth corners (61, 291) + lip points bound the whole mouth for a stable crop.
# MUST match build_mouth_dataset.MOUTH_BOX_INDICES.
MOUTH_BOX_INDICES = sorted(set(MOUTH_INDICES + [61, 291,
                                               INNER_LIP_TOP, INNER_LIP_BOTTOM]))
DEFAULT_MODEL = Path(__file__).resolve().parent / "train" / "yawn_cnn.pt"
DEFAULT_MARGIN = 1.9      # box = mouth-span * this; MUST match build_mouth_dataset


# --- crop helpers: copied verbatim from train/build_mouth_dataset.py ---------
def _lm_points(landmarks, indices, w, h):
    return [[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
            for i in indices]


def mouth_box(points, w, h, margin=DEFAULT_MARGIN):
    """Square crop box (x0, y0, x1, y1) around mouth pixel points, clamped to the
    frame. Returns None if degenerate."""
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


class YawnCNN:
    """Loads yawn_cnn.pt once; yawn_prob(frame, landmarks) -> P(yawn) in [0, 1].

    Raises on construction if torch / the checkpoint is unavailable, so the caller
    (VideoModel) can catch it and fall back to the rule path with a clear message.
    """

    def __init__(self, model_path=DEFAULT_MODEL, device=None):
        # torch._dynamo.trace_rules PROBES importlib.util.find_spec("tensorflow")
        # at import time. _tf_guard's finder RAISES there (by design, to keep
        # MediaPipe importable), which aborts torch's import. MediaPipe is already
        # loaded by the time we get here, so we lift the guard just for the torch
        # import and restore it immediately - TF is only probed, never imported.
        import sys
        guards = [f for f in sys.meta_path if type(f).__name__ == "_BlockTensorFlow"]
        for g in guards:
            sys.meta_path.remove(g)
        try:
            import torch  # lazy: the rule path must not require torch
            from torch import nn
            from torchvision import transforms
            from torchvision.models import mobilenet_v2
        finally:
            for g in guards:
                sys.meta_path.insert(0, g)   # restore: TF stays blocked for MediaPipe

        self._torch = torch
        ck = torch.load(str(model_path), map_location="cpu")
        self.classes = ck.get("classes", ["notyawn", "yawn"])
        self.size = int(ck.get("img_size", 112))
        self.yawn_index = self.classes.index("yawn")

        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features,
                                        len(self.classes))
        model.load_state_dict(ck["state_dict"])
        model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Same normalization the val transform used (mean/std from the checkpoint).
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(ck.get("mean", [0.485, 0.456, 0.406]),
                                 ck.get("std", [0.229, 0.224, 0.225])),
        ])
        self.best_val_f1 = ck.get("best_val_f1")

    def yawn_prob(self, frame, landmarks):
        """P(yawn) for the mouth in this BGR frame. 0.0 if no usable crop."""
        if landmarks is None:
            return 0.0
        h, w = frame.shape[:2]
        box = mouth_box(_lm_points(landmarks, MOUTH_BOX_INDICES, w, h), w, h)
        if box is None:
            return 0.0
        x0, y0, x1, y1 = box
        crop = cv2.resize(frame[y0:y1, x0:x1], (self.size, self.size))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)   # training saw RGB crops
        t = self.tf(rgb).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            probs = self._torch.softmax(self.model(t), dim=1)[0]
        return float(probs[self.yawn_index].item())


# ---------------------------------------------------------------------------
# Self-test: load the shipped checkpoint and run one inference on a synthetic
# frame with fake landmarks, proving the module + crop + model path work
# end-to-end without a webcam. Run: python yawn_cnn_infer.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    class _FakeLandmark:
        def __init__(self, x, y):
            self.x = x; self.y = y

    class _FakeLandmarks:
        """Enough of the MediaPipe interface for _lm_points: .landmark[i].x/.y in
        [0,1]. We place every needed index near the frame centre so the crop is valid."""
        def __init__(self):
            self._pts = {i: _FakeLandmark(0.5 + 0.02 * (i % 5 - 2),
                                          0.6 + 0.02 * (i % 3 - 1))
                         for i in MOUTH_BOX_INDICES}

        @property
        def landmark(self):
            return self._pts

    try:
        cnn = YawnCNN()
    except Exception as exc:  # noqa: BLE001
        print(f"yawn_cnn_infer self-test SKIPPED (model/torch unavailable: {exc})")
        sys.exit(0)

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    p = cnn.yawn_prob(frame, _FakeLandmarks())
    print("yawn_cnn_infer self-test OK")
    print(f"  classes      = {cnn.classes}  img_size={cnn.size}")
    print(f"  best_val_f1  = {cnn.best_val_f1}")
    print(f"  P(yawn) on a random crop = {p:.3f}  (just proves the path runs)")
    assert 0.0 <= p <= 1.0, "probability out of range"
