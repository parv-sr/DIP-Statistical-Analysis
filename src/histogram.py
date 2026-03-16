import cv2
import numpy as np
from typing import List

def extract_histograms_rgb(image_rgb: np.ndarray, bins: int = 256):
    """
    Compute normalized per-channel histograms for an RGB image.
    Returns list: [R_hist, G_hist, B_hist], each sums to 1.
    """
    hists = []
    for i in range(3):  # R, G, B
        hist, _ = np.histogram(image_rgb[:, :, i].flatten(), bins=bins, range=(0, 256), density=False)
        hist = hist.astype(float)
        if hist.sum() > 0:
            hist = hist / hist.sum()
        else:
            hist = np.zeros_like(hist, dtype=float)
        hists.append(hist)
    return hists

def extract_histograms_hsv(image_hsv: np.ndarray, H_bins: int = 36, S_bins: int = 10, V_bins: int = 10):
    """
    Compute normalized per-channel histograms for an HSV image.
    Hue range in OpenCV: 0..179. We'll treat that as 0..360 by mapping bin centers later if needed.
    Returns list: [H_hist, S_hist, V_hist], each sums to 1.
    """
    h = image_hsv[:, :, 0].flatten()  
    s = image_hsv[:, :, 1].flatten()  
    v = image_hsv[:, :, 2].flatten()  

    # 1. Extract raw counts
    H_hist, _ = np.histogram(h, bins=H_bins, range=(0, 180))
    S_hist, _ = np.histogram(s, bins=S_bins, range=(0, 256))
    V_hist, _ = np.histogram(v, bins=V_bins, range=(0, 256))

    processed_hists = []
    for hist in (H_hist, S_hist, V_hist):
        # 2. Convert to float BEFORE division to avoid integer truncation
        f_hist = hist.astype(float)
        total = f_hist.sum()
        if total > 0:
            f_hist /= total
        processed_hists.append(f_hist)

    return processed_hists

def flatten_hist_list(hist_list):
    """
    Flatten list-of-channel histograms into a single 1D vector:
    [chan1_bins..., chan2_bins..., chan3_bins...]
    """
    return np.concatenate([h.flatten() for h in hist_list])
