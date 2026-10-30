"""Shared, restrained chart style for every figure in the report."""
import matplotlib.pyplot as plt

COLORS = {
    "primary": "#1F6F5C",   # tea green
    "accent": "#C26A2E",    # copper / Ceylon tea liquor
    "negative": "#B23A48",
    "muted": "#9AA3A8",
    "ink": "#22292F",
    "grid": "#E6E8EA",
}
ELEV_COLORS = {"high": "#1F6F5C", "medium": "#C26A2E", "low": "#4A5A9A"}


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.edgecolor": COLORS["muted"],
        "axes.labelcolor": COLORS["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "legend.fontsize": 8.5,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "axes.axisbelow": True,
    })
