import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

# ---- Hue helpers (circular) ----
def hue_hist_to_unit_vectors(h_hist):
    """
    Convert a hue histogram (bins covering 0..180 OpenCV) to a weighted sum of unit vectors.
    h_hist: 1D numpy array normalized to sum=1, length = N
    Returns: 2D vector (x, y) representing the combined unit-vector projection.
    """
    N = len(h_hist)
    # bin centers in degrees: OpenCV hue 0..179 corresponds to 0..360 degrees (multiply by 2).
    bin_centers_deg = (np.arange(N) + 0.5) * (360.0 / N)
    angles_rad = np.deg2rad(bin_centers_deg)
    x = np.sum(h_hist * np.cos(angles_rad))
    y = np.sum(h_hist * np.sin(angles_rad))
    # resultant vector length R = sqrt(x^2 + y^2). We can use (x,y) as is, or normalize to unit length if needed.
    return np.array([x, y])

def circular_cosine_similarity(h1, h2):
    """
    Compute cosine similarity between hue distributions using their vector projections.
    Returns value in [-1,1]. For our use, both will be non-negative in many cases; we can map to [0,1].
    """
    v1 = hue_hist_to_unit_vectors(h1)
    v2 = hue_hist_to_unit_vectors(h2)
    # If either vector is zero, handle gracefully
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    sim = np.dot(v1, v2) / (n1 * n2)
    # map -1..1 to 0..1 if you prefer nonnegative similarity:
    sim01 = (sim + 1.0) / 2.0
    return sim01

def circular_wasserstein(h1, h2):
    """
    Compute a rotation-aware 1D Wasserstein (EMD) distance on hue histograms by circularly aligning bins.
    Returns minimal Wasserstein distance across all circular shifts of h2.
    Smaller = more similar. This is in units of bin-index; you may normalize if you like.
    """
    N = len(h1)
    positions = np.arange(N)
    min_d = float('inf')
    # Make sure input histograms are probability distributions
    p = h1.astype(float)
    q = h2.astype(float)
    if p.sum() > 0:
        p = p / p.sum()
    if q.sum() > 0:
        q = q / q.sum()
    for r in range(N):
        q_roll = np.roll(q, r)
        d = wasserstein_distance(positions, positions, u_weights=p, v_weights=q_roll)
        if d < min_d:
            min_d = d
    # Normalize by maximum possible distance (N/2) -> map to 0..1
    return min_d / (N / 2.0)

# ---- Saturation & Value comparison ----
def jensen_shannon_channel(p, q):
    # p, q are probability vectors (sum to 1)
    if p.sum() > 0:
        p = p / p.sum()
    if q.sum() > 0:
        q = q / q.sum()
    return jensenshannon(p, q)  # returns a distance in [0, 1]

def cosine_similarity_flat(p, q):
    """
    Cosine similarity for nonnegative vectors; returns value in [0,1] for probability vectors.
    """
    p = p.flatten().astype(float)
    q = q.flatten().astype(float)
    denom = (np.linalg.norm(p) * np.linalg.norm(q))
    if denom == 0:
        return 0.0
    sim = np.dot(p, q) / denom
    # Clip numeric noise
    return float(np.clip(sim, -1.0, 1.0))

# ---- Combined similarity ----
def combined_similarity(h1, s1, v1, h2, s2, v2, weights=(0.6, 0.2, 0.2)):
    """
    h1,s1,v1 and h2,s2,v2 are channel histograms (probability vectors).
    weights: (w_hue, w_sat, w_val) that sum to 1.0
    
    Returns:
        combined_score in [0..1] (higher means more similar),
        metrics dict with individual channel metrics for inspection.
    """
    w_h, w_s, w_v = weights

    # Hue: circular cosine similarity (map to 0..1)
    hue_sim = circular_cosine_similarity(h1, h2)  # 0..1

    # Also compute circular-Wasserstein distance normalized to 0..1 (smaller is better)
    hue_emd = circular_wasserstein(h1, h2)  # 0..1 (0 = identical)

    # S & V: use cosine similarity and JSD for both (we'll keep both metrics)
    sat_cos = cosine_similarity_flat(s1, s2)
    sat_jsd = jensen_shannon_channel(s1, s2)  # 0..1 (0 identical)

    val_cos = cosine_similarity_flat(v1, v2)
    val_jsd = jensen_shannon_channel(v1, v2)

    # Convert JSD (distance) into similarity token via (1 - jsd)
    sat_sim_from_jsd = 1.0 - sat_jsd
    val_sim_from_jsd = 1.0 - val_jsd

    # Combine S and V similarity by averaging cosine and (1 - JSD)
    sat_sim = 0.5 * sat_cos + 0.5 * sat_sim_from_jsd
    val_sim = 0.5 * val_cos + 0.5 * val_sim_from_jsd

    # Final combined score: weighted sum (give hue most weight)
    combined = w_h * hue_sim + w_s * sat_sim + w_v * val_sim

    metrics = {
        'hue_sim_circular_cosine_0_1': hue_sim,
        'hue_emd_norm_0_1': hue_emd,
        'sat_cos': sat_cos,
        'sat_jsd': sat_jsd,
        'sat_sim_combined': sat_sim,
        'val_cos': val_cos,
        'val_jsd': val_jsd,
        'val_sim_combined': val_sim,
        'combined_score': combined
    }
    return combined, metrics
