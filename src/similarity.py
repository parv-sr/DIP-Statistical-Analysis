import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

EPS = 1e-12


def _as_probability_vector(hist: np.ndarray, fallback_uniform: bool = True) -> np.ndarray:
    """Return a finite probability vector and prevent division-by-zero edge cases."""
    arr = np.asarray(hist, dtype=float).flatten()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0.0] = 0.0

    total = arr.sum()
    if not np.isfinite(total) or total <= EPS:
        if fallback_uniform and arr.size > 0:
            return np.ones_like(arr, dtype=float) / arr.size
        return np.zeros_like(arr, dtype=float)

    return arr / total


# ---- Hue helpers (circular) ----
def hue_hist_to_unit_vectors(h_hist):
    """Project hue histogram to a weighted 2D unit-circle vector."""
    p = _as_probability_vector(h_hist)
    n_bins = len(p)
    if n_bins == 0:
        return np.array([0.0, 0.0])

    bin_centers_deg = (np.arange(n_bins) + 0.5) * (360.0 / n_bins)
    angles_rad = np.deg2rad(bin_centers_deg)
    x = np.sum(p * np.cos(angles_rad))
    y = np.sum(p * np.sin(angles_rad))
    return np.array([x, y], dtype=float)


def circular_cosine_similarity(h1, h2):
    """Cosine similarity of circular hue vectors mapped from [-1,1] to [0,1]."""
    v1 = hue_hist_to_unit_vectors(h1)
    v2 = hue_hist_to_unit_vectors(h2)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 <= EPS or n2 <= EPS:
        return 0.0

    sim = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))


def circular_wasserstein(h1, h2):
    """Rotation-aware normalized EMD on circular hue histograms."""
    p = _as_probability_vector(h1)
    q = _as_probability_vector(h2)
    n_bins = len(p)

    if n_bins == 0 or n_bins != len(q):
        return 1.0

    positions = np.arange(n_bins)
    min_d = float("inf")

    for shift in range(n_bins):
        q_roll = np.roll(q, shift)
        d = wasserstein_distance(positions, positions, u_weights=p, v_weights=q_roll)
        if d < min_d:
            min_d = d

    return float(np.clip(min_d / max(n_bins / 2.0, EPS), 0.0, 1.0))


# ---- Saturation & Value comparison ----
def jensen_shannon_channel(p, q):
    p_norm = _as_probability_vector(p)
    q_norm = _as_probability_vector(q)

    if p_norm.size == 0 or q_norm.size == 0 or p_norm.size != q_norm.size:
        return 1.0

    result = jensenshannon(p_norm, q_norm)
    if np.isnan(result):
        return 1.0
    return float(np.clip(result, 0.0, 1.0))


def cosine_similarity_flat(p, q):
    """Cosine similarity for flattened nonnegative vectors in [0,1]."""
    p_vec = np.asarray(p, dtype=float).flatten()
    q_vec = np.asarray(q, dtype=float).flatten()

    if p_vec.size == 0 or q_vec.size == 0 or p_vec.size != q_vec.size:
        return 0.0

    denom = np.linalg.norm(p_vec) * np.linalg.norm(q_vec)
    if denom <= EPS:
        return 0.0

    sim = float(np.dot(p_vec, q_vec) / denom)
    return float(np.clip(sim, 0.0, 1.0))


# ---- Combined similarity ----
def combined_similarity(h1, s1, v1, h2, s2, v2, weights=(0.6, 0.2, 0.2)):
    """
    Combine hue/saturation/value similarities into a weighted final score in [0,1].
    """
    w_h, w_s, w_v = np.asarray(weights, dtype=float)
    w_total = w_h + w_s + w_v
    if w_total <= EPS or not np.isfinite(w_total):
        raise ValueError("weights must contain finite positive values")

    w_h, w_s, w_v = w_h / w_total, w_s / w_total, w_v / w_total

    hue_sim = circular_cosine_similarity(h1, h2)
    hue_emd = circular_wasserstein(h1, h2)

    sat_cos = cosine_similarity_flat(s1, s2)
    sat_jsd = jensen_shannon_channel(s1, s2)

    val_cos = cosine_similarity_flat(v1, v2)
    val_jsd = jensen_shannon_channel(v1, v2)

    sat_sim = 0.5 * sat_cos + 0.5 * (1.0 - sat_jsd)
    val_sim = 0.5 * val_cos + 0.5 * (1.0 - val_jsd)

    combined = float(np.clip(w_h * hue_sim + w_s * sat_sim + w_v * val_sim, 0.0, 1.0))

    metrics = {
        "hue_sim_circular_cosine_0_1": hue_sim,
        "hue_emd_norm_0_1": hue_emd,
        "sat_cos": sat_cos,
        "sat_jsd": sat_jsd,
        "sat_sim_combined": sat_sim,
        "val_cos": val_cos,
        "val_jsd": val_jsd,
        "val_sim_combined": val_sim,
        "combined_score": combined,
        "weights_used": (w_h, w_s, w_v),
    }
    return combined, metrics