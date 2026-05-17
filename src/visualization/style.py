import matplotlib.pyplot as plt


CLASS_PALETTE = [
    "#2F80ED",
    "#27AE60",
    "#F2994A",
    "#EB5757",
    "#9B51E0",
    "#00A6A6",
    "#6FCF97",
    "#4F4F4F",
]


def set_thesis_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D7DE",
            "axes.labelcolor": "#1F2933",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "font.size": 10,
            "grid.color": "#E5E7EB",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def polish_axes(ax, xlabel=None, ylabel=None, title=None):
    if title:
        ax.set_title(title, pad=14)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax
