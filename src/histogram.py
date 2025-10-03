import cv2
import numpy as np

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
    h = image_hsv[:, :, 0].flatten()  # 0..179
    s = image_hsv[:, :, 1].flatten()  # 0..255
    v = image_hsv[:, :, 2].flatten()  # 0..255

    H_hist, _ = np.histogram(h, bins=H_bins, range=(0, 180), density=False)
    S_hist, _ = np.histogram(s, bins=S_bins, range=(0, 256), density=False)
    V_hist, _ = np.histogram(v, bins=V_bins, range=(0, 256), density=False)

    # normalize to sum=1
    for hist in (H_hist, S_hist, V_hist):
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist[:] = hist.astype(float) / hist_sum
        else:
            hist[:] = 0.0

    return [H_hist.astype(float), S_hist.astype(float), V_hist.astype(float)]

def flatten_hist_list(hist_list):
    """
    Flatten list-of-channel histograms into a single 1D vector:
    [chan1_bins..., chan2_bins..., chan3_bins...]
    """
    return np.concatenate([h.flatten() for h in hist_list])
