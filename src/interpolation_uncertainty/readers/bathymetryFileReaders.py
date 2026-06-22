from abc import ABC, abstractmethod
from typing import Union
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
                  filename: Union[str, Path],
                  as_points: bool = False) -> RasterDataset:
        """
        Implements the following operations:
        1. Read input TIFF file using gdal
        2. Saves other TFF information in the "metadata" dictionary (ndv_value, resolution)
        3. Transform raster data into array of points (x, y, depth) if as_points == True

        Parameters
        ----------
        filename: str or Path object
        as_points: bool, optional
            If True, transforms the raster 2d data into a list of points (x, y, depth)

        
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

            depth_data = raw_depth_data

            return RasterDataset(depth_data,
                                 filename=Path(filename).name,
                                 filetype='raster',
                                 metadata={'ndv_value': ndv_value,
                                            'resolution': resolution,
                                            'full_path': filename,
                                            'geotransform': depth_gt}
                                )                  




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




