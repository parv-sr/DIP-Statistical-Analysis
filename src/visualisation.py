import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_theme(style="whitegrid")

def plot_histogram_overlay(hist1, hist2, channel_names, output_path, x_limits=None, title=None):
    fig, axes = plt.subplots(1, len(channel_names), figsize=(6 * len(channel_names), 5), sharey=True)
    if title:
        fig.suptitle(title, fontsize=16)

    colors = ['#1f77b4', '#ff7f0e']  # blue, orange
    for i, (ax, name) in enumerate(zip(axes, channel_names)):
        a = hist1[i]
        b = hist2[i]
        ax.plot(a, color=colors[0], label='Bollywood')
        ax.fill_between(np.arange(len(a)), a, color=colors[0], alpha=0.25)
        ax.plot(b, color=colors[1], label='Field')
        ax.fill_between(np.arange(len(b)), b, color=colors[1], alpha=0.25)
        ax.set_title(name)
        if x_limits is not None:
            ax.set_xlim(x_limits)
        ax.set_xlabel('Bin index')
    axes[0].set_ylabel('Normalized frequency')
    axes[0].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved histogram overlay to {output_path}")

def plot_hue_kde_with_wrap(hue_array_deg_0_360, label, ax=None):
    """
    hue_array_deg_0_360: 1D array of hue angles in degrees [0..360)
    We'll replicate the data at +/-360 to ensure wrap-around for KDE.
    Returns axis with seaborn kde plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    # replicate for wrap
    data = np.array(hue_array_deg_0_360)
    data_wrap = np.concatenate([data - 360.0, data, data + 360.0])
    sns.kdeplot(data_wrap, ax=ax, label=label, fill=True)
    ax.set_xlim(0, 360)
    ax.set_xlabel('Hue (degrees)')
    ax.set_xticks(np.arange(0, 361, 60))
    return ax

def plot_hue_overlay(hue_deg_bwood, hue_deg_field, output_path, title=None):
    fig, ax = plt.subplots(figsize=(10,4))
    if title:
        fig.suptitle(title, fontsize=14)
    plot_hue_kde_with_wrap(hue_deg_bwood, label='Bollywood', ax=ax)
    plot_hue_kde_with_wrap(hue_deg_field, label='Field', ax=ax)
    ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved hue KDE overlay to {output_path}")
