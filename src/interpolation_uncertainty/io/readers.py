import numpy as np
from osgeo import gdal
from pathlib import Path
from typing import Union

gdal.UseExceptions()


def read_file(filename: Union[str, Path], verbose: bool = False) -> dict:
    """Read a raster or BAG file and extract depth data and metadata.

    Parameters
    ----------
    filename : str or Path
        Path to the raster file (.tif, .tiff, or .bag).
    verbose : bool, optional
        If ``True``, prints additional information during file reading. Default is ``False``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'data'`` : np.ndarray — 2D array of raw depth values (NDV not masked).
        - ``'filename'`` : str — base name of the file.
        - ``'filetype'`` : str — always ``'raster'``.
        - ``'metadata'`` : dict containing ``'ndv_value'``, ``'resolution'``,
          ``'full_path'``, and ``'geotransform'``.
    """
    with gdal.Open(str(filename)) as ds:
        if not ds:
            raise RuntimeError(
                f"GDAL failed to open file: '{filename}'")

        depth_band = ds.GetRasterBand(1)
        if not depth_band:
            raise RuntimeError(
                f"Error retrieving depth information from {filename}.")

        ndv_value = depth_band.GetNoDataValue()
        raw_depth_data = depth_band.ReadAsArray()
        depth_gt = ds.GetGeoTransform()
        resolution = depth_gt[1]
        if resolution < 1:
            print(f"WARNING: detected resolution value is <= 1"
                  f"\n Setting resolution value to 1")
            resolution = 1

        data_dict = {'data': raw_depth_data,
                'filename': Path(filename).name,
                'filetype': 'raster',
                'metadata': {'ndv_value': ndv_value,
                             'resolution': resolution,
                             'full_path': filename,
                             'geotransform': depth_gt}
                }

        if verbose:
            print(f"File '{filename}' read successfully.")
            row_count, col_count = raw_depth_data.shape
            print(f"Data shape: {row_count} rows x {col_count} columns")
            print(f"No-data value: {ndv_value}")
            print(f"Resolution: {resolution} m/pixel")
            print(f"Geotransform values--> ")
            gt_keys = ['Origin x-coordinate:',
                       'Pixel width:',
                       'Row rotation:',
                       'Origin y-coordinate:',
                       'Column rotation:',
                       'Pixel height:']
            print("\n".join(f"{k} {v}" for k, v in zip(gt_keys, depth_gt)), end="\n\n")

        return data_dict
