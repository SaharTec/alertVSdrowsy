"""
train/evaluate_yawn.py - score the fine-tuned yawn CNN and compare it to the
rule-based baseline on the SAME held-out subjects (section 6 of the project).

It reproduces train_yawn_cnn.py's subject-grouped split (pass the same --val-frac
and --seed), so both models are judged on people they never saw in training.
For every held-out mouth crop it produces two predictions:

  * CNN      : the fine-tuned MobileNetV2 (yawn_cnn.pt);
  * baseline : the rule  mar >= MAR_YAWN  (config.py) - the per-frame form of the
               live system's yawn cue, read from the manifest's 'mar' column.

and reports, for each, the metrics that matter under class imbalance:
Precision / Recall / F1 on the yawn class, plus the confusion matrix.

  python evaluate_yawn.py --model yawn_cnn.pt --manifest mouth_manifest.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                             accuracy_score)

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))         # alertVSdrowsy/ for config
sys.path.insert(0, str(THIS.parent))                # train/ for the shared modules

from config import MAR_YAWN  # noqa: E402
from train_yawn_cnn import (  # noqa: E402 - reuse the SAME data + split + model code
    DEF_MANIFEST, DEF_CROPS, DEF_OUT, CLASSES,
    MouthCrops, load_manifest, subject_split, make_transforms, build_model,
)


def report(name, y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", pos_label=1, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n=== {name} ===")
    print(f"  accuracy={acc:.3f}   yawn  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
    print("  confusion (rows=true, cols=pred)   notyawn  yawn")
    print(f"    true notyawn                       {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"    true yawn                          {cm[1,0]:5d}  {cm[1,1]:5d}")
    return {"precision": p, "recall": r, "f1": f1, "accuracy": acc}


@torch.no_grad()
def cnn_predict(model, rows, crops_dir, img_size, batch_size, device):
    _, val_tf = make_transforms(img_size)
    dl = DataLoader(MouthCrops(rows, crops_dir, val_tf),
                    batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    for x, _ in dl:
        preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
    return preds


def main():
    ap = argparse.ArgumentParser(description="Evaluate the yawn CNN vs the rule baseline")
    ap.add_argument("--model", default=str(DEF_OUT), help="yawn_cnn.pt from train_yawn_cnn.py")
    ap.add_argument("--manifest", default=str(DEF_MANIFEST))
    ap.add_argument("--crops-dir", default=str(DEF_CROPS))
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="MUST match the value used in training (default 0.2)")
    ap.add_argument("--seed", type=int, default=0, help="MUST match training (default 0)")
    ap.add_argument("--split", choices=["val", "all"], default="val",
                    help="evaluate on the held-out subjects (val) or everything")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    rows = load_manifest(args.manifest, args.crops_dir)
    if not rows:
        raise SystemExit(f"[eval] no usable rows in {args.manifest}")
    if "mar" not in rows[0]:
        raise SystemExit("[eval] manifest has no 'mar' column - rebuild it with the "
                         "current build_mouth_dataset.py so the baseline can be scored.")

    _, val_rows = subject_split(rows, args.val_frac, args.seed)
    eval_rows = rows if args.split == "all" else val_rows
    subj = sorted(set(r["subject"] for r in eval_rows))
    y_true = [int(r["label"]) for r in eval_rows]
    print(f"[eval] split={args.split}  frames={len(eval_rows)}  "
          f"subjects={len(subj)} {subj}")
    print(f"[eval] class balance: notyawn={y_true.count(0)}  yawn={y_true.count(1)}")

    # --- baseline: per-frame MAR threshold (the live system's yawn cue) ---
    base_pred = [1 if float(r["mar"]) >= MAR_YAWN else 0 for r in eval_rows]
    base = report(f"BASELINE  rule: mar >= MAR_YAWN ({MAR_YAWN})", y_true, base_pred)

    # --- CNN ---
    ckpt = torch.load(args.model, map_location=device)
    model = build_model(pretrained=False)               # weights come from the checkpoint
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    cnn_pred = cnn_predict(model, eval_rows, args.crops_dir,
                           ckpt.get("img_size", 112), args.batch_size, device)
    cnn = report("CNN  fine-tuned MobileNetV2", y_true, cnn_pred)

    # --- headline comparison ---
    print("\n--- summary (yawn class) -------------------------------")
    print(f"  {'metric':10} {'baseline':>10} {'CNN':>10} {'delta':>10}")
    for k in ("precision", "recall", "f1"):
        d = cnn[k] - base[k]
        print(f"  {k:10} {base[k]:10.3f} {cnn[k]:10.3f} {d:+10.3f}")
    winner = "CNN" if cnn["f1"] > base["f1"] else ("baseline" if base["f1"] > cnn["f1"] else "tie")
    print(f"  -> higher yawn-F1: {winner}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    main()
