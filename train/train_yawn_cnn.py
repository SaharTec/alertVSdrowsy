"""
train/train_yawn_cnn.py - fine-tune a pretrained MobileNetV2 to detect yawns from
mouth crops (the transfer-learning model, section 5 of the project).

Pipeline:
  * read the manifest from build_mouth_dataset.py (crop path, label, subject);
  * SUBJECT-GROUPED hold-out split (a fraction of *people* go to validation, so no
    face is in both train and val -> honest numbers, section 6);
  * load MobileNetV2 pretrained on ImageNet, swap the head to 2 classes;
  * TRANSFER LEARNING in two phases:
      phase 1 - freeze the backbone, train only the new head;
      phase 2 - unfreeze and fine-tune the whole net at a lower learning rate;
  * CLASS-WEIGHTED cross-entropy (yawn frames are the minority);
  * save the best-by-val-F1 model to yawn_cnn.pt.

This trains fast on the GPU you have. Run it yourself:

  # smoke (tiny; random-init to avoid the weight download)
  python train_yawn_cnn.py --subset 400 --epochs 1 --no-pretrained

  # real run
  python train_yawn_cnn.py --manifest mouth_manifest.csv --epochs 10

Then evaluate with evaluate_yawn.py.
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

THIS = Path(__file__).resolve()
DEF_MANIFEST = THIS.parent / "mouth_manifest.csv"
DEF_CROPS = THIS.parent / "mouth_crops"
DEF_OUT = THIS.parent / "yawn_cnn.pt"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASSES = ["notyawn", "yawn"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class MouthCrops(Dataset):
    def __init__(self, rows, crops_dir, tf):
        self.rows = rows
        self.crops_dir = Path(crops_dir)
        self.tf = tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.crops_dir / r["path"]).convert("RGB")
        return self.tf(img), int(r["label"])


def load_manifest(path, crops_dir, subset=0, seed=0):
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)]
    # keep only rows whose crop file actually exists
    crops_dir = Path(crops_dir)
    rows = [r for r in rows if (crops_dir / r["path"]).exists()]
    if subset and len(rows) > subset:
        random.Random(seed).shuffle(rows)
        rows = rows[:subset]
    return rows


def subject_split(rows, val_frac, seed):
    """Hold out val_frac of SUBJECTS (not rows) for validation."""
    groups = [r["subject"] for r in rows]
    n_subj = len(set(groups))
    if n_subj < 2:
        print(f"  WARNING: only {n_subj} subject(s) -> falling back to a random "
              f"row split (fine for a smoke test, NOT for real evaluation).")
        idx = list(range(len(rows)))
        random.Random(seed).shuffle(idx)
        cut = max(1, int(len(idx) * val_frac))
        va, tr = set(idx[:cut]), set(idx[cut:])
        return [rows[i] for i in tr], [rows[i] for i in va]
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(gss.split(rows, groups=groups))
    return [rows[i] for i in tr_idx], [rows[i] for i in va_idx]


def halve_val_subjects(val_rows, seed):
    """Split the held-out SUBJECTS into halves A and B.

    Choosing an operating point (a probability threshold) and then reporting the
    score at that point on the same data inflates the result - the threshold was
    fitted to those very faces. So: sweep on A, report on B. Neither half may
    come from the training subjects, which the model has already memorized.
    Returns (subjects_a, subjects_b) as sets. Defined here, next to
    subject_split, so every script that needs this split gets the same one.
    """
    subs = sorted({r["subject"] for r in val_rows})
    random.Random(seed).shuffle(subs)
    cut = len(subs) // 2
    return set(subs[:cut]), set(subs[cut:])


def make_transforms(size):
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),   # day/night robustness
        transforms.RandomRotation(8),                           # camera-angle robustness
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(pretrained):
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    m = mobilenet_v2(weights=weights)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, len(CLASSES))    # swap head: 1000 -> 2
    return m


def set_backbone_trainable(model, flag):
    for p in model.features.parameters():
        p.requires_grad = flag


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, n = 0.0, 0
    all_y, all_p = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_y.extend(y.cpu().tolist())
        all_p.extend(logits.argmax(1).cpu().tolist())
    p, r, f1, _ = precision_recall_fscore_support(
        all_y, all_p, labels=[1], average="binary", pos_label=1, zero_division=0)
    return total_loss / max(n, 1), p, r, f1, (all_y, all_p)


def main():
    ap = argparse.ArgumentParser(description="Fine-tune MobileNetV2 for yawn detection")
    ap.add_argument("--manifest", default=str(DEF_MANIFEST))
    ap.add_argument("--crops-dir", default=str(DEF_CROPS))
    ap.add_argument("--out", default=str(DEF_OUT))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--freeze-epochs", type=int, default=3,
                    help="epochs with the backbone frozen before fine-tuning")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="head LR (phase 1)")
    ap.add_argument("--fine-tune-lr", type=float, default=1e-4, help="LR after unfreezing")
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of SUBJECTS held out")
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", type=int, default=0, help="cap total rows (smoke test)")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random init instead of ImageNet weights (smoke only)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"[train] device={device}  pretrained={not args.no_pretrained}")

    rows = load_manifest(args.manifest, args.crops_dir, args.subset, args.seed)
    if not rows:
        raise SystemExit(f"[train] no usable rows in {args.manifest} "
                         f"(did build_mouth_dataset.py run?)")
    train_rows, val_rows = subject_split(rows, args.val_frac, args.seed)

    def cls_counts(rr):
        c = {0: 0, 1: 0}
        for r in rr:
            c[int(r["label"])] += 1
        return c
    tc, vc = cls_counts(train_rows), cls_counts(val_rows)
    print(f"[train] train={len(train_rows)} {tc}   val={len(val_rows)} {vc}   "
          f"subjects: {len(set(r['subject'] for r in train_rows))} train / "
          f"{len(set(r['subject'] for r in val_rows))} val")
    if vc[0] == 0 or vc[1] == 0:
        print("  WARNING: validation is missing a class; F1 will be unreliable "
              "(use more subjects / a larger dataset).")

    train_tf, val_tf = make_transforms(args.img_size)
    train_dl = DataLoader(MouthCrops(train_rows, args.crops_dir, train_tf),
                          batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers)
    val_dl = DataLoader(MouthCrops(val_rows, args.crops_dir, val_tf),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    model = build_model(not args.no_pretrained).to(device)

    # Class-weighted CE from TRAIN counts (yawn is the minority).
    n = len(train_rows)
    w = torch.tensor([n / (2 * max(tc[0], 1)), n / (2 * max(tc[1], 1))],
                     dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    print(f"[train] class weights (notyawn, yawn) = {w.tolist()}")

    # Phase 1: freeze backbone, train the head only.
    set_backbone_trainable(model, False)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:                 # enter phase 2
            set_backbone_trainable(model, True)
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.fine_tune_lr)
            print(f"[train] --- unfroze backbone; fine-tuning at lr={args.fine_tune_lr} ---")

        tr_loss, *_ = run_epoch(model, train_dl, criterion, device, optimizer)
        va_loss, vp, vr, vf1, _ = run_epoch(model, val_dl, criterion, device)
        phase = "head " if epoch <= args.freeze_epochs else "fine "
        print(f"  epoch {epoch:2d} [{phase}] train_loss={tr_loss:.3f}  "
              f"val_loss={va_loss:.3f}  val P/R/F1(yawn)={vp:.2f}/{vr:.2f}/{vf1:.2f}")
        if vf1 >= best_f1:
            best_f1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({
        "state_dict": best_state, "arch": "mobilenet_v2",
        "img_size": args.img_size, "classes": CLASSES,
        "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "best_val_f1": best_f1,
    }, args.out)
    print(f"\n[train] saved best model (val F1={best_f1:.3f}) -> {args.out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    main()
