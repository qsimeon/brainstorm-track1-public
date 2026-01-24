from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from matplotlib.figure import Figure

# set matplotlib style to fivethirtyeight
plt.style.use("fivethirtyeight")


def dot_plot(
    data: np.ndarray,
    channels_coords: np.ndarray,
    marker_size: int = 20,
    cmap="inferno",
    ax: Axes | None = None,
    cmin: float | None = None,
    cmax: float | None = None,
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # plot the data as a dot plot
    ax.scatter(
        channels_coords[:, 0],
        channels_coords[:, 1],
        c=data,
        s=marker_size,
        cmap=cmap,
        vmin=cmin,
        vmax=cmax,
        edgecolors=(0.8, 0.8, 0.8),
        linewidths=0.1,
    )
    ax.set(xticks=[1, 32], yticks=[1, 30])

    # set black background
    ax.set_facecolor("#111111")

    fig = ax.get_figure()
    fig.set_facecolor("#111111")  # type: ignore
    ax.grid(False)

    # hide axes
    ax.set(
        xticks=[],
        yticks=[],
        xticklabels=[],
        yticklabels=[],
        frame_on=False,
    )
    fig.tight_layout()  # type: ignore

    return fig, ax  # type: ignore
