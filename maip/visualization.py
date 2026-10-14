"""Four-frequency images, shared raw-amplitude scale and optimization trace."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(result_file, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(result_file, allow_pickle=False) as data:
        images = data["images"]
        mask = data["display_mask"]
        labels = data["frequency_labels"]
        losses = data["loss"]
        attention = data["attention_scores"]
        channels = data["channel_weights"]
    valid = images[:, mask]
    vmin = min(0.0, float(valid.min()))
    vmax = float(valid.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), layout="constrained")
    palette = plt.get_cmap("turbo").copy()
    palette.set_bad("white")
    for image, label, axis in zip(images, labels, axes):
        artist = axis.imshow(np.ma.array(image, mask=~mask), cmap=palette, vmin=vmin, vmax=vmax, interpolation="nearest")
        axis.set_title(str(label), pad=9)
        axis.set_axis_off()
    bar = fig.colorbar(artist, ax=axes, shrink=0.85, pad=0.025)
    bar.set_label("Conductivity change (model units)")
    fig.suptitle("MAIP · four-frequency reconstruction", fontsize=13)
    fig.savefig(output / "reconstruction.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(5.8, 3.3), layout="constrained")
    axis.semilogy(np.arange(1, len(losses) + 1), np.maximum(losses, np.finfo(float).tiny), color="#316b9d", linewidth=1.6)
    axis.set(xlabel="Iteration", ylabel="Mean absolute voltage error", title="Measurement fitting")
    axis.grid(alpha=0.18)
    fig.savefig(output / "loss.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), layout="constrained", gridspec_kw={"width_ratios": [1.2, 1]})
    artist = axes[0].imshow(attention, vmin=0, vmax=1, cmap="Blues")
    axes[0].set(xticks=range(4), yticks=range(4), xticklabels=range(1, 5), yticklabels=range(1, 5), xlabel="Input frequency channel", ylabel="Output frequency channel", title="Learned mixing weights")
    fig.colorbar(artist, ax=axes[0], shrink=0.8)
    axes[1].bar(range(1, 5), channels, color="#7196ba", width=0.55)
    axes[1].set(xticks=range(1, 5), xlabel="Frequency channel", ylabel="Scale", title="Learned channel scales")
    fig.savefig(output / "attention.png", dpi=180)
    plt.close(fig)
