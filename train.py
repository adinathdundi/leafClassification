"""
train.py

Trains the RGB model and the silhouette-mask model on IDENTICAL splits
(same images, same train/val/test assignment, same seed), so the only
variable between the two runs is the input representation.

Both modes read exclusively from PlantVillage's `segmented/` folder —
`color/` and `grayscale/` are not used. See dataset.py for why this works
(segmentation only zeroes the background; leaf pixels are untouched).

Usage:
    python train.py --data_root /path/to/plantvillage_dataset --mode rgb   --epochs 15
    python train.py --data_root /path/to/plantvillage_dataset --mode mask  --epochs 15

Each run saves:
    runs/<mode>/best_model.pt
    runs/<mode>/metrics.json      (per-epoch train/val loss+acc, final test acc,
                                    confusion matrix, per-class report)
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from dataset import LeafDataset, build_file_list
from model import ResNetV1

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_samples(samples, val_frac=0.15, test_frac=0.15, seed=SEED):
    """Stratified-ish split: shuffle once with a fixed seed so RGB and mask
    runs get IDENTICAL splits when called with the same seed."""
    rng = random.Random(seed)
    samples = samples.copy()
    rng.shuffle(samples)
    n = len(samples)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val = samples[:n_val]
    test = samples[n_val:n_val + n_test]
    train = samples[n_val + n_test:]
    return train, val, test


def run_epoch(model, loader, criterion, optimizer, device, train=True, desc=""):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
    with context:
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
            # running loss/acc shown live in the progress bar
            pbar.set_postfix(loss=f"{total_loss / total:.4f}", acc=f"{correct / total:.4f}")
    return total_loss / total, correct / total


def evaluate_full(model, loader, device, idx_to_class, desc="test"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc, leave=False, unit="batch"):
            x = x.to(device)
            out = model(x)
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())
    cm = confusion_matrix(all_labels, all_preds).tolist()
    report = classification_report(
        all_labels, all_preds,
        target_names=[idx_to_class[i] for i in range(len(idx_to_class))],
        output_dict=True, zero_division=0,
    )
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return acc, cm, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                         help="Path containing the segmented/ subfolder")
    parser.add_argument("--mode", choices=["rgb", "mask"], required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--depth", type=int, default=20,
                         help="ResNet depth, must be 6n+2 (e.g. 20, 32, 44, 56, 110). "
                              "20 is the fastest/shallowest option, a good default "
                              "to start with before going deeper.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                         help="L2 weight decay, mirroring the original Keras "
                              "kernel_regularizer=l2(1e-4). Applied to all "
                              "parameters (PyTorch has no easy per-layer-only "
                              "equivalent), unlike Keras which applied it only "
                              "to conv kernels, not biases/BN.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", default="runs")
    parser.add_argument("--limit_samples", type=int, default=None,
                         help="If set, use only this many images total (before "
                              "the train/val/test split) — for a quick smoke "
                              "test of the pipeline, e.g. --limit_samples 500 "
                              "--epochs 1, before committing to a full run.")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    segmented_root = os.path.join(args.data_root, "segmented")

    samples, class_to_idx = build_file_list(segmented_root)
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    print(f"Found {len(samples)} images across {len(class_to_idx)} classes")

    if args.limit_samples is not None:
        rng = random.Random(SEED)
        rng.shuffle(samples)
        samples = samples[:args.limit_samples]
        print(f"--limit_samples set: using only {len(samples)} images for a quick run")

    train_s, val_s, test_s = split_samples(samples)
    print(f"Split sizes -> train {len(train_s)} | val {len(val_s)} | test {len(test_s)}")

    train_ds = LeafDataset(train_s, class_to_idx, segmented_root,
                            mode=args.mode, img_size=args.img_size, augment=True)
    val_ds = LeafDataset(val_s, class_to_idx, segmented_root,
                          mode=args.mode, img_size=args.img_size, augment=False)
    test_ds = LeafDataset(test_s, class_to_idx, segmented_root,
                           mode=args.mode, img_size=args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    in_channels = 3 if args.mode == "rgb" else 1
    model = ResNetV1(in_channels=in_channels, depth=args.depth,
                      num_classes=len(class_to_idx)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ResNetV1 depth={args.depth} in_channels={in_channels} -> {n_params:,} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                            factor=0.5, patience=2)

    out_dir = Path(args.out_dir) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_acc = 0.0
    epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"[{args.mode}] epochs", unit="epoch")
    for epoch in epoch_bar:
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True,
            desc=f"[{args.mode}] epoch {epoch}/{args.epochs} train",
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False,
            desc=f"[{args.mode}] epoch {epoch}/{args.epochs} val",
        )
        scheduler.step(val_acc)

        # keep a one-line running summary on the outer epoch bar
        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", train_acc=f"{train_acc:.4f}",
                               val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}")
        tqdm.write(f"[{args.mode}] epoch {epoch}/{args.epochs} "
                   f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                   f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    # Final test-set evaluation using best checkpoint
    model.load_state_dict(torch.load(out_dir / "best_model.pt"))
    test_acc, cm, report = evaluate_full(model, test_loader, device, idx_to_class,
                                          desc=f"[{args.mode}] final test eval")
    print(f"[{args.mode}] FINAL TEST ACCURACY: {test_acc:.4f}")

    metrics = {
        "mode": args.mode,
        "in_channels": in_channels,
        "depth": args.depth,
        "history": history,
        "test_accuracy": test_acc,
        "confusion_matrix": cm,
        "classification_report": report,
        "class_to_idx": class_to_idx,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
