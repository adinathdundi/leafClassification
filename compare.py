"""
compare.py

Run after training both modes:
    python train.py --data_root ... --mode rgb
    python train.py --data_root ... --mode mask
    python compare.py --runs_dir runs

Produces:
    - printed summary table (overall test accuracy + gap)
    - per-class accuracy comparison (which disease classes rely most on
      shape vs. color/texture — the most scientifically interesting part)
    - side-by-side training curves and confusion matrices saved as PNGs
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(runs_dir, mode):
    path = Path(runs_dir) / mode / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — did you train mode='{mode}' yet?")
    with open(path) as f:
        return json.load(f)


def per_class_accuracy(metrics):
    cm = np.array(metrics["confusion_matrix"])
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.diag(cm) / cm.sum(axis=1)
    acc = np.nan_to_num(acc)
    idx_to_class = {v: k for k, v in metrics["class_to_idx"].items()}
    return {idx_to_class[i]: acc[i] for i in range(len(acc))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--out_dir", default="comparison_outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    rgb = load_metrics(args.runs_dir, "rgb")
    mask = load_metrics(args.runs_dir, "mask")

    print("=" * 60)
    print("OVERALL TEST ACCURACY")
    print("=" * 60)
    print(f"  RGB (color + texture + shape) : {rgb['test_accuracy']:.4f}")
    print(f"  Silhouette mask (shape only)  : {mask['test_accuracy']:.4f}")
    gap = rgb["test_accuracy"] - mask["test_accuracy"]
    print(f"  Gap (RGB - mask)              : {gap:.4f}")
    print()
    print("Interpretation: the gap is the portion of classification accuracy")
    print("that comes from color/texture cues rather than pure leaf-blade shape.")
    print("A SMALL gap for a given disease means shape alone nearly separates it")
    print("from healthy leaves; a LARGE gap means the disease signal is mostly")
    print("color/texture (e.g. lesion coloration) with little effect on silhouette.")

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for m, label in [(rgb, "RGB"), (mask, "Mask")]:
        epochs = [h["epoch"] for h in m["history"]]
        val_acc = [h["val_acc"] for h in m["history"]]
        axes[0].plot(epochs, val_acc, marker="o", label=label)
    axes[0].set_title("Validation accuracy per epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("val accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    labels = ["RGB", "Mask"]
    values = [rgb["test_accuracy"], mask["test_accuracy"]]
    axes[1].bar(labels, values, color=["#4C72B0", "#55A868"])
    axes[1].set_title("Final test accuracy")
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(values):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_comparison.png", dpi=150)
    print(f"\nSaved {out_dir / 'accuracy_comparison.png'}")

    # Per-class accuracy comparison -> most interesting scientific output
    rgb_pc = per_class_accuracy(rgb)
    mask_pc = per_class_accuracy(mask)
    classes = sorted(rgb_pc.keys())
    diffs = [(c, rgb_pc[c] - mask_pc.get(c, 0), rgb_pc[c], mask_pc.get(c, 0)) for c in classes]
    diffs.sort(key=lambda t: t[1], reverse=True)

    print("\n" + "=" * 60)
    print("PER-CLASS GAP (RGB_acc - mask_acc), sorted largest gap first")
    print("Large gap  -> disease relies on color/texture, not shape")
    print("Small/neg. -> disease is (also) visible in leaf silhouette")
    print("=" * 60)
    for c, d, r, mm in diffs:
        print(f"  {c:45s} rgb={r:.3f}  mask={mm:.3f}  gap={d:+.3f}")

    fig2, ax2 = plt.subplots(figsize=(10, max(6, 0.28 * len(classes))))
    y = np.arange(len(classes))
    names = [d[0] for d in diffs]
    gaps = [d[1] for d in diffs]
    ax2.barh(y, gaps, color=["#c44e52" if g > 0 else "#55A868" for g in gaps])
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("RGB accuracy - Mask accuracy")
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_title("Per-class reliance on color/texture vs. shape")
    plt.tight_layout()
    plt.savefig(out_dir / "per_class_gap.png", dpi=150)
    print(f"Saved {out_dir / 'per_class_gap.png'}")


if __name__ == "__main__":
    main()
