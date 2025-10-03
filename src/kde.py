import numpy as np
from scipy.stats import gaussian_kde

def apply_kde(pixel_data: np.ndarray):
    """
    Applies Kernel Density Estimation to a set of pixel values.

    Args:
        pixel_data (np.ndarray): A 1D array of pixel values for a single channel.

    Returns:
        A tuple (xs, density) where xs is the evaluation points and density is the KDE values.
    """
    if len(pixel_data) == 0:
        return np.array([]), np.array([])
        
    kde = gaussian_kde(pixel_data)
    # Evaluate KDE on a fine grid
    xs = np.linspace(min(pixel_data), max(pixel_data), 256)
    density = kde(xs)
    return xs, density