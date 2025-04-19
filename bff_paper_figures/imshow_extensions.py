import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray


def imshow_with_extents(xaxis, yaxis, data: NDArray, aspect_ratio: float = 1, vmin=None, vmax=None):
    if data.shape != (len(yaxis), len(xaxis)):
        raise ValueError("x-axis, y-axis shape doesn't match data")
    if len(xaxis) <= 1 or len(yaxis) <= 1:
        raise ValueError("x-axis, y-axis must have more than one data point")
    xstep = xaxis[1] - xaxis[0]
    ystep = yaxis[1] - yaxis[0]
    extents = [
        min(xaxis) - xstep / 2,
        max(xaxis) + xstep / 2,
        min(yaxis) - ystep / 2,
        max(yaxis) + ystep / 2,
    ]
    intrinsic_aspect_ratio = (max(yaxis) - min(yaxis) + ystep) / (max(xaxis) - min(xaxis) + xstep)
    return plt.imshow(
        data,
        extent=extents,
        aspect=aspect_ratio / intrinsic_aspect_ratio,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        cmap="inferno",
    )


def imshow_with_extents_and_crop(
    xaxis: NDArray,
    yaxis: NDArray,
    data: NDArray,
    aspect_ratio: float = 1,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    vmin=None,
    vmax=None,
):
    small_quantity_to_ensure_no_cropping = 1
    if xmin is None:
        xmin = min(xaxis) - small_quantity_to_ensure_no_cropping
    if xmax is None:
        xmax = max(xaxis) + small_quantity_to_ensure_no_cropping
    if ymin is None:
        ymin = min(yaxis) - small_quantity_to_ensure_no_cropping
    if ymax is None:
        ymax = max(yaxis) + small_quantity_to_ensure_no_cropping
    x_mask = np.all(np.array([(xmin <= xaxis), xaxis <= xmax]), axis=0)
    y_mask = np.all(np.array([(ymin <= yaxis), yaxis <= ymax]), axis=0)
    imshow_with_extents(
        xaxis[x_mask], yaxis[y_mask], data[y_mask, :][:, x_mask], vmin=vmin, vmax=vmax, aspect_ratio=aspect_ratio
    )
