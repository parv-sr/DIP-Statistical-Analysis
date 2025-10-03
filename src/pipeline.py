import os
import glob
import numpy as np

from histogram import extract_histograms_rgb, extract_histograms_hsv, flatten_hist_list
from preprocessing import preprocess_image
from similarity import combined_similarity, hue_hist_to_unit_vectors
from visualisation import plot_histogram_overlay, plot_hue_overlay

# --- Configuration ---
DATA_DIR = 'data'
RESULTS_DIR = 'results'
BOLLYWOOD_DIR = os.path.join(DATA_DIR, 'bollywood')
FIELD_DIR = os.path.join(DATA_DIR, 'field')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

def process_dataset(image_paths: list, dataset_name: str):
    """
    Processes images and returns:
      - avg_hist_rgb: list of 3 arrays (R,G,B) normalized
      - avg_hist_hsv: list of 3 arrays (H,S,V) normalized
      - hue_degs_list: concatenated hue degrees (0..360) for KDE plotting
      - pixel_arrays_rgb: list of arrays per channel for RGB KDE
    """
    print(f"--- Processing {dataset_name} dataset ---")
    all_hists_rgb = []
    all_hists_hsv = []
    hue_angles = []  # degrees
    pixel_arrays_rgb = [[], [], []]

    output_dir = os.path.join(PROCESSED_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        img_rgb, img_hsv = preprocess_image(img_path, output_dir)
        if img_rgb is None:
            continue

        hists_rgb = extract_histograms_rgb(img_rgb, bins=256)
        hists_hsv = extract_histograms_hsv(img_hsv, H_bins=36, S_bins=10, V_bins=10)

        all_hists_rgb.append(hists_rgb)
        all_hists_hsv.append(hists_hsv)

        # collect hue degrees for KDE plotting (OpenCV hue H in 0..179 -> degrees = H * 2)
        h_flat = img_hsv[:, :, 0].flatten().astype(float)
        hue_deg = h_flat * 2.0  # 0..358
        hue_angles.extend(hue_deg.tolist())

        # collect RGB pixels for KDE
        for i in range(3):
            pixel_arrays_rgb[i].extend(img_rgb[:, :, i].flatten().tolist())

    # average histograms across images
    if all_hists_rgb:
        avg_hist_rgb = [np.mean([img_hist[ch] for img_hist in all_hists_rgb], axis=0) for ch in range(3)]
    else:
        avg_hist_rgb = [np.zeros(256), np.zeros(256), np.zeros(256)]

    if all_hists_hsv:
        avg_hist_hsv = [np.mean([img_hist[ch] for img_hist in all_hists_hsv], axis=0) for ch in range(3)]
    else:
        avg_hist_hsv = [np.zeros(36), np.zeros(10), np.zeros(10)]

    # convert pixel arrays to numpy arrays
    pixel_arrays_rgb = [np.array(arr) if len(arr) else np.array([]) for arr in pixel_arrays_rgb]
    hue_angles = np.array(hue_angles) if len(hue_angles) else np.array([])

    return avg_hist_rgb, avg_hist_hsv, hue_angles, pixel_arrays_rgb

def summarize_results(metrics, avg_h_bwood, avg_h_field, out_path):
    """
    Write human-readable summary with per-channel metrics and combined score.
    metrics: dict returned from combined_similarity (see similarity.py)
    avg_h_bwood/avg_h_field: average hue histograms (for reporting mean angles)
    """
    # compute circular means for readable labels
    def circular_mean_deg(h_hist):
        N = len(h_hist)
        if h_hist.sum() == 0:
            return None
        bin_centers_deg = (np.arange(N) + 0.5) * (360.0 / N)
        angles_rad = np.deg2rad(bin_centers_deg)
        x = np.sum(h_hist * np.cos(angles_rad))
        y = np.sum(h_hist * np.sin(angles_rad))
        mean_angle = (np.rad2deg(np.arctan2(y, x))) % 360
        R = np.sqrt(x**2 + y**2)
        return mean_angle, R

    mean_bwood = circular_mean_deg(avg_h_bwood)
    mean_field = circular_mean_deg(avg_h_field)

    lines = []
    lines.append("=== Quantitative Color Analysis Summary ===\n")
    lines.append(f"Combined similarity score (0..1; higher = more similar): {metrics['combined_score']:.4f}\n")
    lines.append("Per-channel metrics:\n")
    lines.append(f" - Hue circular cosine similarity (0..1): {metrics['hue_sim_circular_cosine_0_1']:.4f}\n")
    lines.append(f" - Hue circular EMD normalized (0..1; smaller=more similar): {metrics['hue_emd_norm_0_1']:.4f}\n")
    lines.append(f" - Saturation cosine: {metrics['sat_cos']:.4f}, sat JSD: {metrics['sat_jsd']:.4f}, sat combined sim: {metrics['sat_sim_combined']:.4f}\n")
    lines.append(f" - Value cosine: {metrics['val_cos']:.4f}, val JSD: {metrics['val_jsd']:.4f}, val combined sim: {metrics['val_sim_combined']:.4f}\n\n")

    if mean_bwood is not None and mean_field is not None:
        lines.append(f"Dominant hue (Bollywood): {mean_bwood[0]:.1f}° (concentration R={mean_bwood[1]:.3f})\n")
        lines.append(f"Dominant hue (Field):     {mean_field[0]:.1f}° (concentration R={mean_field[1]:.3f})\n")

    # interpretative sentence (brief)
    lines.append("\nInterpretation:\n")
    lines.append(" - A high combined score indicates the two palettes share a common warm/cool tendency.\n")
    lines.append(" - Hue metrics capture 'color family' closeness (e.g., reds/oranges vs magentas). Saturation and Value capture vividness and brightness differences.\n")
    lines.append("\nEnd of summary.\n")

    with open(out_path, 'w') as f:
        f.writelines(lines)
    print(f"Saved textual summary to {out_path}")
    return "\n".join(lines)

def main():
    print("Starting Wedding Color Analysis Pipeline...")

    bollywood_images = glob.glob(os.path.join(BOLLYWOOD_DIR, '*.*'))
    field_images = glob.glob(os.path.join(FIELD_DIR, '*.*'))

    if not bollywood_images or not field_images:
        print("Error: Ensure there are images in both 'data/bollywood' and 'data/field' directories.")
        return

    # Process datasets
    avg_rgb_bwood, avg_hsv_bwood, hue_deg_bwood, pixels_rgb_bwood = process_dataset(bollywood_images, 'bollywood')
    avg_rgb_field, avg_hsv_field, hue_deg_field, pixels_rgb_field = process_dataset(field_images, 'field')

    # Compute combined similarity on HSV average histograms
    # Unpack H,S,V histograms
    h1, s1, v1 = avg_hsv_bwood
    h2, s2, v2 = avg_hsv_field

    combined_score, metrics = combined_similarity(h1, s1, v1, h2, s2, v2, weights=(0.6, 0.2, 0.2))

    # --- Visualizations ---
    os.makedirs(os.path.join(RESULTS_DIR, 'histograms'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'kde'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'summary'), exist_ok=True)

    # Histogram overlays (RGB)
    hist_path = os.path.join(RESULTS_DIR, 'histograms', 'rgb_histogram_overlay.png')
    plot_histogram_overlay(avg_rgb_bwood, avg_rgb_field, channel_names=['R','G','B'],
                           output_path=hist_path, x_limits=(0,255),
                           title='RGB Histogram Comparison: Bollywood vs Field')

    # Hue KDE overlay (with wrap)
    hue_kde_path = os.path.join(RESULTS_DIR, 'kde', 'hue_kde_overlay.png')
    plot_hue_overlay(hue_deg_bwood, hue_deg_field, hue_kde_path, title='Hue KDE (wrap) - Bollywood vs Field')

    # Save textual summary
    summary_path = os.path.join(RESULTS_DIR, 'summary', 'analysis_summary.txt')
    summary_text = summarize_results(metrics, h1, h2, summary_path)
    print("\n" + summary_text)

if __name__ == '__main__':
    main()
