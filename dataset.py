"""
dataset.py

Works entirely off the PlantVillage `segmented/` folder — `color/` and
`grayscale/` are not needed at all:

    plantvillage_dataset/
        segmented/
            Apple___Apple_scab/
                image1.JPG      <- leaf's real color/texture, black background
                ...
            Apple___Black_rot/
                ...
            ... (38 classes total)


That means we can build BOTH inputs from this one folder:
    mode="rgb"  -> load the segmented image as-is (leaf color+texture,
                   black background) and normalize it
    mode="mask" -> load the *same* file and threshold it down to a binary
                   silhouette (via segmentation.mask_from_segmented_image)

This is a cleaner ablation than using color/ for RGB: the only difference
between the two modes is now "keep color+texture" vs. "collapse to binary
shape" — both start from identical pixels and an identical (black) backdrop,
rather than the RGB run also having to deal with real photo-background
clutter that the mask run never sees.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from segmentation import mask_from_segmented_image


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def build_file_list(segmented_root: str):
    """Walk segmented/ and return (relative_path, class_name) pairs + class->idx map."""
    segmented_root = Path(segmented_root)
    classes = sorted([d.name for d in segmented_root.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    samples = []
    for c in classes:
        class_dir = segmented_root / c
        for f in sorted(class_dir.iterdir()):
            if f.suffix in IMG_EXTENSIONS:
                samples.append((f"{c}/{f.name}", c))
    return samples, class_to_idx


class LeafDataset(Dataset):
    def __init__(self, samples, class_to_idx, segmented_root,
                 mode="rgb", img_size=224, augment=False):
        """
        Parameters
        ----------
        samples : list of (relative_path, class_name)
            As returned by build_file_list. Pass a pre-split subset for train/val/test.
        class_to_idx : dict
        segmented_root : str
            Path to the segmented/ folder. Used for BOTH modes.
        mode : "rgb" or "mask"
        img_size : int
            Images are resized to (img_size, img_size).
        augment : bool
            If True, applies light augmentation (flip/rotation) consistently
            regardless of mode — important so RGB and mask models see
            comparable augmentation strength.
        """
        assert mode in ("rgb", "mask")
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.segmented_root = Path(segmented_root)
        self.mode = mode
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _read_segmented(self, rel_path):
        img = cv2.imread(str(self.segmented_root / rel_path))
        if img is None:
            raise FileNotFoundError(
                f"Could not read {rel_path} from segmented/. "
                "The file may be missing or corrupted."
            )
        return img

    def _load_rgb(self, rel_path):
        img = self._read_segmented(rel_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        # ImageNet-style normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return img

    def _load_mask(self, rel_path):
        seg_img = self._read_segmented(rel_path)
        mask = mask_from_segmented_image(seg_img)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)
        mask = mask[:, :, None]  # (H, W, 1) — single channel, no color/texture info
        return mask

    def _augment(self, arr):
        # Same random flip/rotation logic applied identically whether arr is
        # 3-channel RGB or 1-channel mask, so augmentation strength matches.
        if np.random.rand() < 0.5:
            arr = np.fliplr(arr).copy()
        k = np.random.choice([0, 1, 2, 3])
        if k:
            arr = np.rot90(arr, k).copy()
        return arr

    def __getitem__(self, idx):
        rel_path, class_name = self.samples[idx]
        if self.mode == "rgb":
            arr = self._load_rgb(rel_path)
        else:
            arr = self._load_mask(rel_path)

        if self.augment:
            arr = self._augment(arr)

        arr = torch.from_numpy(arr.transpose(2, 0, 1))  # HWC -> CHW
        label = self.class_to_idx[class_name]
        return arr, label
