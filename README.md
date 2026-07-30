# Leaf geometry vs. color/texture: how much does shape matter for disease classification?

How much is the structure of the leaf explained by the structure alone? On average(over all leaves), I got a 60% accuracy in a classification task in an ablation after considering only a binary silhouette of a leaf. 
This project uses the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) (54,000+ images, 38 plant/disease classes) for a 38 class classification task trained on a ResNet18 model[ResNet Architecture](https://arxiv.org/abs/1512.03385)

## Results

| Input | Test accuracy |
|---|---|
| Full color image (color + texture + shape) | **94.4%** |
| Binary Silhouette only (shape alone, no color/texture) | **60.2%** |

60% from shape alone! — across 38 fine-grained categories, versus a ~3% random-chance baseline — means leaf geometry carries real diagnostic signal on its own. However, interestingly, different leaves and different studies yielded different accuracy levels

![Overall accuracy comparison](accuracy_comparison.png)



![Per-class accuracy gap](per_class_gap.png)

This makes biological sense: diseases that visibly warp, curl, or fold the blade itself (viral leaf curl, certain blights) show up strongly in shape; diseases that are essentially surface blemishes (bacterial spot, rust) leave the blade's overall geometry untouched and only show up in color/texture.



## How to reproduce

```bash
pip install torch torchvision opencv-python scikit-learn matplotlib numpy tqdm
```

Download the PlantVillage dataset from Kaggle and extract it. You only need the `segmented/` folder — `color/` and `grayscale/` can be ignored:

```
plantvillage_dataset/
    segmented/
        Apple___Apple_scab/
        Apple___Black_rot/
        ... (38 class folders)
```

Train both models (same seed and split, so results are directly comparable):

```bash
python train.py --data_root /path/to/plantvillage_dataset --mode rgb  --epochs 15
python train.py --data_root /path/to/plantvillage_dataset --mode mask --epochs 15
```

Useful flags: `--depth` (ResNet depth, must be `6n+2`, default 20), `--img_size`, `--batch_size`, `--limit_samples N` (quick smoke test on a subset before committing to a full run — see [Notes](#notes--things-you-may-want-to-tune)).

Compare the two runs:

```bash
python compare.py --runs_dir runs
```

This prints the overall and per-class accuracy gap, and saves `accuracy_comparison.png` / `per_class_gap.png` (the charts embedded above).

## Files

| File | Purpose |
|---|---|
| `segmentation.py` | Derives clean binary masks from the `segmented/` images |
| `dataset.py` | PyTorch `Dataset`, reads only from `segmented/`, same split, either `mode="rgb"` or `mode="mask"` |
| `model.py` | ResNet18 (CIFAR-style, depth=6n+2), parameterized by `in_channels` (3 or 1) and `depth` |
| `train.py` | Training loop + evaluation, saves `metrics.json` |
| `compare.py` | Loads both `metrics.json`, prints/plots the comparison |
