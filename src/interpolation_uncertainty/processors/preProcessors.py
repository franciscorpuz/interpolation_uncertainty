from typing import Union
import numpy as np
from interpolation_uncertainty.readers.bathymetryDataset import (RasterDataset,
                                                       CSVDataset,
                                                       BPSDataset)



def remove_edge_ndv(raster_data: RasterDataset,
                    max_iterations: Union[int, None]) -> RasterDataset:
        """
        Iteratively removes edge rows and columns containing no-depth values
        in the raster depth

        Parameters
        ----------
        raster_data : RasterDataset
            A RasterDataset object representing surface elevation or bathymetry depth.
            Expected to be numeric (e.g., float, int).

        max_iterations : int, optional
            Maximum number of iterations to perform. Defaults to half the depth width

        Returns
        -------
        RasterDataset
            A RasterDataset object with ndv_values rows and column edges removed, dimensions possibly cropped

        """

        # Type check for depth
        if not isinstance(raster_data, RasterDataset):
            raise TypeError(f"Input 'depth' must be a RasterDataset object. type:{type(raster_data)}")
        # Handle initial empty array or None input
        if raster_data is None or raster_data.size == 0:
            raise ValueError("Input 'depth' array cannot be None or empty.")

        # Create a working copy to avoid modifying the original array
        elev = raster_data.copy()
        original_shape = raster_data.shape

        # Set up value for max_iteration if none declared
        if max_iterations is None:
            max_dimensions = np.max(original_shape)
            max_iterations = int(np.max(max_dimensions) / 2)

        # Extract no_data_value
        ndv = raster_data.metadata["ndv_value"]

        if ndv == np.nan:
            def is_ndv(data_array):
                return np.any(np.isnan(data_array))
        else:
            def is_ndv(data_array):
                return np.any(data_array == ndv)

        shrink_idx = 0
        have_ndv = True
        # remove edges that have ndv elements
        # continue until all edges are ndv-free or exceeded 100 iterations
        # assumes that all inner elements are non NaN
        while have_ndv:
            tmp = elev[0, :]
            if is_ndv(tmp):
                elev = elev[1:, :]
            tmp = elev[:, 0]
            if is_ndv(tmp):
                elev = elev[:, 1:]
            tmp = elev[-1, :]
            if is_ndv(tmp):
                elev = elev[:-1, :]
            tmp = elev[:, -1]
            if is_ndv(tmp):
                elev = elev[:, :-1]
            shrink_idx += 1
            if not np.any(is_ndv(elev)):
                have_ndv = False
            if shrink_idx > max_iterations:
                break

        if is_ndv(elev):
            print("Warning: Processed Depth depth still contains NDV values.")

        elev = raster_data.wrap(elev)

        return elev


def raster_to_points(raster_data: RasterDataset,
                     remove_ndv: bool = True) -> RasterDataset:
    """
    Transforms raster 2d data into array of points (x, y, depth)

    Parameters
    ----------
    raster_data : RasterDataset
        A RasterDataset object representing surface elevation or bathymetry depth.
        Expected to be numeric (e.g., float, int).
    remove_ndv : bool, optional
        If True, removes points with no-data values (NDV) from the output. Defaults to True.

    Returns
    -------
    RasterDataset
        A RasterDataset object with 3 columns (x, y, depth) and rows corresponding to 
        the number of valid points in the original raster.

    """

    # Type check for depth
    if not isinstance(raster_data, RasterDataset):
        raise TypeError(f"Input 'depth' must be a RasterDataset object. type:{type(raster_data)}")
    # Handle initial empty array or None input
    if raster_data is None or raster_data.size == 0:
        raise ValueError("Input 'depth' array cannot be None or empty.")

    elev = raster_data.copy()
    depth_gt = raster_data.metadata["geotransform"]
    resolution = raster_data.metadata["resolution"]

    # Get raster dimensions
    rows, cols = elev.shape

    # Create coordinate arrays based on geotransform
    x_coords = depth_gt[0] + (np.arange(cols) * depth_gt[1]) + (resolution / 2)
    y_coords = depth_gt[3] + (np.arange(rows) * depth_gt[5]) + (resolution / 2)

    # Create meshgrid of coordinates and flatten it along with the depth values
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
    points = np.column_stack((x_mesh.flatten(), y_mesh.flatten(), elev.flatten()))

    if remove_ndv:
        ndv = raster_data.metadata["ndv_value"]
        if ndv == np.nan:
            valid_mask = ~np.isnan(points[:, 2])
        else:
            valid_mask = points[:, 2] != ndv
        points = points[valid_mask]

    points_dataset = raster_data.wrap(points)
    points_dataset.filetype = "points"

    return points_dataset