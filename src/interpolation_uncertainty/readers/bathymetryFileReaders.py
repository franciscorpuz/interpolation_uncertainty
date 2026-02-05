from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from interpolation_uncertainty.readers.bathymetryDataset import (RasterDataset,
                                                       CSVDataset,
                                                       BPSDataset)

from osgeo import gdal
gdal.UseExceptions()


class BathymetryFileReader(ABC):
    """
    Base Class for various file readers (TIFF, CSV, BPS, etc.)
    """

    @abstractmethod
    def read_file(self, filename: str | Path):
        """ Return a BathymetryDataset with properties:
            filename, type, depth_data, metadata """
        pass


class RasterReader(BathymetryFileReader):
    """
    Subclass for Raster readers
    """

    def read_file(self,
                  filename: str | Path,
                  remove_ndv: bool = True) -> RasterDataset:
        """
        Implements the following operations:
        1. Read input TIFF file using gdal
        2. Saves other TFF information in the "metadata" dictionary (ndv_value, resolution)
        3. Removes NDV values from the depth array (can be controlled in the config file under RASTER)

        Parameters
        ----------
        filename: str or Path object
        remove_ndv : bool
                     option to remove no-data-value in the resulting data array
                     (default: true)
        Returns
        -------
        RasterBathymetryDataset : dict
                          : properties (filename : str,
                                        type: 'raster',
                                       depth_data : np.array,
                                       metadata : dict)

        """

        with gdal.Open(str(filename)) as ds:
            if not ds:
                raise RuntimeError(
                    f"GDAL failed to open TIFF file: '{filename}'")

            # Read raster data from TIFF
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

            if remove_ndv:
                # Clean up ndv_values
                depth_data = RasterReader.remove_edge_ndv(raster_data=raw_depth_data,
                                                          ndv_value=ndv_value)
            else:
                depth_data = raw_depth_data

            return RasterDataset(depth_data,
                                 filename=Path(filename).name,
                                 filetype='raster',
                                 metadata={'ndv_value': ndv_value,
                                               'resolution': resolution,
                                               'full_path': filename})

    @classmethod
    def remove_edge_ndv(cls,
                        raster_data: np.ndarray,
                        ndv_value: np.number,
                        max_iterations: int = None) -> np.ndarray:
        """
        Iteratively removes edge rows and columns containing no-depth values
        in the raster depth

        Parameters
        ----------
        raster_data : np.ndarray
            A 2D NumPy array representing surface elevation or bathymetry depth.
            Expected to be numeric (e.g., float, int).

        ndv_value : np.number
            Data representing the NoDataValue in the depth array

        max_iterations : int, optional
            Maximum number of iterations to perform. Defaults to half the depth width

        Returns
        -------
        np.ndarray
            A 2D NumPy array with ndv_values removed, possibly cropped

        """

        # Type check for depth
        if not isinstance(raster_data, np.ndarray):
            raise TypeError(f"Input 'depth' must be a NumPy array (np.ndarray). type:{type(raster_data)}")
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
        ndv = ndv_value

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

        return elev


class CSVReader(BathymetryFileReader):
    """
    Subclass for CSV file readers (future implementation)
    """

    def read_file(self, filename: str | Path) -> CSVDataset:
        """
        Parameters
        ----------
        filename: str or Path

        Returns
        -------
        CSVDataset
        """



class BPSReader(BathymetryFileReader):
    """
    Subclass for BPS file readers (future implementation)
    """
    def read_file(self, filename: str | Path) -> BPSDataset:
        """
        Parameters
        ----------
        filename: str or Path

        Returns
        -------
        BPSDataset
        """
        pass




