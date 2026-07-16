"""Plot functions"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np


RCPARAMS_DICT: dict = {
    "figure.figsize": (12, 3),
    "figure.dpi": 250,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.grid": True,
    "axes.axisbelow": True,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "DejaVu Sans"],
    # "text.usetex": True,
}


def set_plot_style(rcParams_dict: dict | None = None, append_to_structural_tools_defaults: bool = True) -> None:
    if rcParams_dict:
        if append_to_structural_tools_defaults:
            plt.rcParams.update(RCPARAMS_DICT)

        plt.rcParams.update(rcParams_dict)
    else:
        plt.rcParams.update(RCPARAMS_DICT)


def plot_internal_diagram(
    x: np.ndarray,
    ys: list[np.ndarray] | np.ndarray,
    labels: list[str] | str,
    xlabel: str = "Location",
    ylabel: str = "Value",
    title: str = "Diagram",
    figsize: tuple[float, float] = (12, 3),
    envelope: bool = False,
    show_member: bool = True,
    show_plot: bool = True,
    save_png: str | None = None,
) -> tuple[Figure, Axes]:

    if not show_plot:
        plt.ioff

    fig, ax = plt.subplots(figsize=figsize)

    if not isinstance(ys, list):
        ys = [ys]
    if not isinstance(labels, list):
        labels = [labels]

    assert len(ys) == len(labels)

    for i, y_value_set in enumerate(ys):
        ax.plot(x, y_value_set, label=labels[i])

    if envelope:
        y_values = []
        for line in ax.lines:
            y_value = np.array(line.get_ydata())
            y_values.append(y_value)

        y_values = np.vstack(y_values)
        env_max = np.max(y_values, axis=0)
        env_min = np.min(y_values, axis=0)
        ax.fill_between(x, env_min, env_max, color="gray", alpha=0.25, zorder=1, label="_nolegend_")

    if show_member:
        ax.plot(
            [x[0], x[-1]],
            [0, 0],
            color="black",
            linewidth=3,
            solid_capstyle="butt",
            zorder=0,
            label="_nolegend_",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.legend()
    plt.tight_layout()
    if save_png:
        plt.savefig(save_png)
    if show_plot:
        plt.ion()
        plt.show()
    else:
        plt.close()
        plt.ion()
    return fig, ax
