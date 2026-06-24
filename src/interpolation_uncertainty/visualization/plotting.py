import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Union
from copy import deepcopy


def show_depth(dataset_dict: dict,
               title: Union[str, None] = None,
               cmap: str = 'terrain') -> None:
    """Plot bathymetric depth data from either a raster or point-cloud dataset.

    Parameters
    ----------
    dataset_dict : dict
        Dataset dictionary with ``'filetype'`` of either ``'raster'`` or
        ``'points'``, as returned by :func:`read_file` or :func:`raster_to_points`.
    title : str or None, optional
        Custom plot title. If ``None``, a title is generated from the filename
        and resolution. Default is ``None``.
    cmap : str, optional
        Matplotlib colourmap name. Default is ``'terrain'``.

    Returns
    -------
    None
    """
    fig, ax1 = plt.subplots(figsize=(16, 10))
    points_dataset = deepcopy(dataset_dict)
    res = points_dataset["metadata"]['resolution']
    ndv = points_dataset["metadata"]['ndv_value']
    points_dataset["data"] = np.where(points_dataset["data"] == ndv, np.nan, points_dataset["data"])
    if points_dataset["filetype"] == 'raster':
        im = ax1.imshow(points_dataset["data"], cmap=cmap, aspect='equal')
        shape_0 = points_dataset["data"].shape[0]
        shape_1 = points_dataset["data"].shape[1]
        fig.colorbar(im, label='Depth (m)', shrink=0.5)
        locs = ax1.get_xticks()
        ax1.set_xticks(locs)
        ax1.set_xticklabels([str(int(x * res)) for x in locs])
        locs = ax1.get_yticks()
        ax1.set_yticks(locs)
        ax1.set_yticklabels([str(int(y * res)) for y in locs])
        ax1.tick_params(axis='x', labelrotation=90)
        ax1.set_xlim(left=0, right=shape_1)
        ax1.set_ylim(top=0, bottom=shape_0)
    elif points_dataset["filetype"] == 'points':
        x = points_dataset["data"][:, 0]
        y = points_dataset["data"][:, 1]
        depth = points_dataset["data"][:, 2]
        sc = ax1.scatter(x, y, c=depth, cmap=cmap, marker='.', s=1)
        fig.colorbar(sc, label='Depth (m)', shrink=0.5)
        shape_0 = int((np.max(y) - np.min(y)) / res)
        shape_1 = int((np.max(x) - np.min(x)) / res)
    else:
        raise ValueError(f"Unrecognized file type for plotting: {points_dataset['filetype']}")
    ax1.set_xlabel("West-East (m)")
    ax1.set_ylabel("North-South (m)")
    if title is None:
        fn = Path(points_dataset["filename"]).name
        title = f"{fn} at {res}m resolution"
    ax1.set_title(f"""
                Surface:{title} at {res}m resolution
                Dimensions: {shape_0 * res / 1000}km by {shape_1 * res / 1000}km
                    """)
