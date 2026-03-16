import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.histogram import extract_histograms_hsv, extract_histograms_rgb
from src.preprocessing import preprocess_image
from src.similarity import combined_similarity
from src.visualisation import plot_histogram_overlay, plot_hue_overlay


@dataclass(frozen=True)
class ImagePipelineConfig:
    data_dir: str = "data"
    results_dir: str = "results"
    bollywood_folder: str = "bollywood"
    field_folder: str = "field"
    processed_folder: str = "processed"
    weights: Tuple[float, float, float] = (0.6, 0.2, 0.2)


@dataclass
class DatasetArtifacts:
    avg_rgb_hist: List[np.ndarray]
    avg_hsv_hist: List[np.ndarray]
    hue_degrees: np.ndarray
    rgb_pixels: List[np.ndarray]


class ImageDatasetProcessor:
    """Sequential processor that takes images from disk to statistical representations."""

    def __init__(self, config: ImagePipelineConfig):
        self.config = config

    def load_image_paths(self, dataset_name: str) -> List[str]:
        folder = os.path.join(self.config.data_dir, dataset_name)
        # Use recursive globbing to find files in all subdirectories
        paths = glob.glob(os.path.join(folder, "**", "*.*"), recursive=True)
        # Filter out any directories that might accidentally match the *.* pattern
        return sorted([p for p in paths if os.path.isfile(p)])

    def process_dataset(self, image_paths: List[str], dataset_name: str) -> DatasetArtifacts:
        print(f"--- Processing {dataset_name} dataset ---")

        all_hists_rgb: List[List[np.ndarray]] = []
        all_hists_hsv: List[List[np.ndarray]] = []
        hue_angles: List[float] = []
        pixel_arrays_rgb: List[List[float]] = [[], [], []]

        output_dir = os.path.join(self.config.data_dir, self.config.processed_folder, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        for img_path in image_paths:
            img_rgb, img_hsv = preprocess_image(img_path, output_dir)
            if img_rgb is None or img_hsv is None:
                continue

            hists_rgb = extract_histograms_rgb(img_rgb, bins=256)
            hists_hsv = extract_histograms_hsv(img_hsv, H_bins=36, S_bins=10, V_bins=10)

            all_hists_rgb.append(hists_rgb)
            all_hists_hsv.append(hists_hsv)

            h_flat = img_hsv[:, :, 0].flatten().astype(float)
            hue_angles.extend((h_flat * 2.0).tolist())

            for channel in range(3):
                pixel_arrays_rgb[channel].extend(img_rgb[:, :, channel].flatten().tolist())

        avg_rgb = self._average_histograms(all_hists_rgb, default_sizes=[256, 256, 256])
        avg_hsv = self._average_histograms(all_hists_hsv, default_sizes=[36, 10, 10])

        return DatasetArtifacts(
            avg_rgb_hist=avg_rgb,
            avg_hsv_hist=avg_hsv,
            hue_degrees=np.asarray(hue_angles, dtype=float) if hue_angles else np.array([]),
            rgb_pixels=[np.asarray(arr, dtype=float) if arr else np.array([]) for arr in pixel_arrays_rgb],
        )

    @staticmethod
    def _average_histograms(histograms: List[List[np.ndarray]], default_sizes: List[int]) -> List[np.ndarray]:
        if not histograms:
            return [np.zeros(size, dtype=float) for size in default_sizes]
        return [np.mean([h[channel] for h in histograms], axis=0) for channel in range(3)]


class SimilarityReporter:
    @staticmethod
    def summarize(metrics: Dict[str, float], avg_h_bwood: np.ndarray, avg_h_field: np.ndarray, out_path: str) -> str:
        def circular_mean_deg(h_hist: np.ndarray):
            if np.sum(h_hist) <= 0:
                return None
            n_bins = len(h_hist)
            bin_centers_deg = (np.arange(n_bins) + 0.5) * (360.0 / n_bins)
            angles_rad = np.deg2rad(bin_centers_deg)
            x = np.sum(h_hist * np.cos(angles_rad))
            y = np.sum(h_hist * np.sin(angles_rad))
            mean_angle = float(np.rad2deg(np.arctan2(y, x)) % 360)
            concentration = float(np.sqrt(x**2 + y**2))
            return mean_angle, concentration

        mean_bwood = circular_mean_deg(avg_h_bwood)
        mean_field = circular_mean_deg(avg_h_field)

        lines = [
            "=== Quantitative Color Analysis Summary ===\n",
            f"Combined similarity score (0..1; higher = more similar): {metrics['combined_score']:.4f}\n",
            "Per-channel metrics:\n",
            f" - Hue circular cosine similarity (0..1): {metrics['hue_sim_circular_cosine_0_1']:.4f}\n",
            f" - Hue circular EMD normalized (0..1; smaller=more similar): {metrics['hue_emd_norm_0_1']:.4f}\n",
            f" - Saturation cosine: {metrics['sat_cos']:.4f}, sat JSD: {metrics['sat_jsd']:.4f}, sat combined sim: {metrics['sat_sim_combined']:.4f}\n",
            f" - Value cosine: {metrics['val_cos']:.4f}, val JSD: {metrics['val_jsd']:.4f}, val combined sim: {metrics['val_sim_combined']:.4f}\n\n",
        ]

        if mean_bwood and mean_field:
            lines.append(
                f"Dominant hue (Bollywood): {mean_bwood[0]:.1f}° (concentration R={mean_bwood[1]:.3f})\n"
            )
            lines.append(
                f"Dominant hue (Field):     {mean_field[0]:.1f}° (concentration R={mean_field[1]:.3f})\n"
            )

        lines.extend(
            [
                "\nInterpretation:\n",
                " - A high combined score indicates the two palettes share a common warm/cool tendency.\n",
                " - Hue metrics capture color-family closeness. Saturation and Value capture vividness and brightness differences.\n",
                "\nEnd of summary.\n",
            ]
        )

        with open(out_path, "w", encoding="utf-8") as file:
            file.writelines(lines)
        print(f"Saved textual summary to {out_path}")
        return "".join(lines)


class ImageSimilarityPipeline:
    def __init__(self, config: ImagePipelineConfig):
        self.config = config
        self.processor = ImageDatasetProcessor(config)
        self.reporter = SimilarityReporter()

    def run(self) -> str:
        print("Starting Wedding Color Analysis Pipeline...")

        bollywood_paths = self.processor.load_image_paths(self.config.bollywood_folder)
        field_paths = self.processor.load_image_paths(self.config.field_folder)

        if not bollywood_paths or not field_paths:
            raise RuntimeError("Ensure both datasets contain images under data/bollywood and data/field.")

        print(f"Found {len(bollywood_paths)} files in {self.config.bollywood_folder}")
        print(f"Found {len(field_paths)} files in {self.config.field_folder}")

        bollywood = self.processor.process_dataset(bollywood_paths, self.config.bollywood_folder)
        field = self.processor.process_dataset(field_paths, self.config.field_folder)

        if not self._has_valid_hsv_data(bollywood.avg_hsv_hist, field.avg_hsv_hist):
            raise RuntimeError("One dataset has no valid HSV histogram data after preprocessing.")

        h1, s1, v1 = bollywood.avg_hsv_hist
        h2, s2, v2 = field.avg_hsv_hist
        _, metrics = combined_similarity(h1, s1, v1, h2, s2, v2, weights=self.config.weights)

        hist_dir = os.path.join(self.config.results_dir, "histograms")
        kde_dir = os.path.join(self.config.results_dir, "kde")
        summary_dir = os.path.join(self.config.results_dir, "summary")
        os.makedirs(hist_dir, exist_ok=True)
        os.makedirs(kde_dir, exist_ok=True)
        os.makedirs(summary_dir, exist_ok=True)

        plot_histogram_overlay(
            bollywood.avg_rgb_hist,
            field.avg_rgb_hist,
            channel_names=["R", "G", "B"],
            output_path=os.path.join(hist_dir, "rgb_histogram_overlay.png"),
            x_limits=(0, 255),
            title="RGB Histogram Comparison: Bollywood vs Field",
        )

        plot_hue_overlay(
            bollywood.hue_degrees,
            field.hue_degrees,
            os.path.join(kde_dir, "hue_kde_overlay.png"),
            title="Hue KDE (wrap) - Bollywood vs Field",
        )

        summary_text = self.reporter.summarize(
            metrics,
            h1,
            h2,
            os.path.join(summary_dir, "analysis_summary.txt"),
        )
        print("\n" + summary_text)
        return summary_text

    @staticmethod
    def _has_valid_hsv_data(*datasets: List[np.ndarray]) -> bool:
        for channels in datasets:
            if any(np.sum(channel) <= 0 for channel in channels):
                return False
        return True


def main() -> None:
    pipeline = ImageSimilarityPipeline(ImagePipelineConfig())
    pipeline.run()


if __name__ == "__main__":
    main()