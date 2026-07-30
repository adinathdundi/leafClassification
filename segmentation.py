"""
segmentation.py

PlantVillage ships three parallel versions of every image:
    color/       - original RGB leaf photo on a plain background
    grayscale/   - grayscale version of the same
    segmented/   - background already removed; leaf on pure black background

Because `segmented/` already has the background stripped out, we don't need
to build a background-removal model ourselves. We just need to turn the
segmented RGB image into a clean binary silhouette mask:
    foreground = "not black" (with a small tolerance for JPEG compression noise)

This module also includes a fallback Otsu-based segmenter in case you want
to segment `color/` images directly (e.g. if you're using a different
dataset later, or want to sanity check that the two approaches agree).
"""

import cv2
import numpy as np


def mask_from_segmented_image(segmented_img: np.ndarray, black_thresh: int = 15) -> np.ndarray:
    """
    Derive a clean binary silhouette mask from a PlantVillage `segmented/` image.

    Parameters
    ----------
    segmented_img : np.ndarray
        BGR or RGB image as loaded by cv2.imread (background already black).
    black_thresh : int
        Pixels with all channels below this value are treated as background.
        JPEG compression can leave near-black (not pure 0,0,0) background pixels,
        so a small tolerance is needed rather than checking for exact 0.

    Returns
    -------
    mask : np.ndarray, dtype=uint8, values {0, 255}
        Same height/width as input. 255 = leaf, 0 = background.
    """
    gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY)
    mask = _clean_mask(mask)
    return mask


def mask_from_color_image_otsu(color_img: np.ndarray) -> np.ndarray:
    """
    Fallback: segment a plain-background color leaf photo directly using
    Otsu thresholding on saturation (leaves are saturated green/yellow/brown;
    PlantVillage backgrounds are typically a plain gray/white card).

    Use this only if you don't have access to the `segmented/` folder.
    """
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = _clean_mask(mask)
    return mask


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Morphological cleanup + keep only the largest connected component.

    This removes speckle noise and stray background pixels, and guards
    against multiple disconnected blobs (e.g. a stem fragment or shadow)
    being counted as part of the leaf.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask  # nothing found; return as-is

    # label 0 is background; pick the largest non-background component
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return clean


def mask_to_3channel(mask: np.ndarray) -> np.ndarray:
    """Replicate a 1-channel mask to 3 channels (0/1 float), useful if you
    want to reuse a pretrained 3-channel backbone (e.g. ImageNet ResNet)
    on the silhouette input instead of training a 1-channel model from scratch.
    """
    m = (mask > 0).astype(np.float32)
    return np.stack([m, m, m], axis=-1)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python segmentation.py <path_to_segmented_image>")
        sys.exit(0)

    img = cv2.imread(sys.argv[1])
    if img is None:
        raise FileNotFoundError(sys.argv[1])
    mask = mask_from_segmented_image(img)
    out_path = "mask_preview.png"
    cv2.imwrite(out_path, mask)
    leaf_pixel_fraction = (mask > 0).mean()
    print(f"Saved {out_path} | leaf covers {leaf_pixel_fraction:.1%} of frame")
