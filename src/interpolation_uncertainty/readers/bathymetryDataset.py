from typing import Dict, Union
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class BathymetryDataset(np.ndarray):
    """
    Subclassed the Numpy Array class to include bathymetry related metadata to the input depth data

    Attributes
    ----------
    filename : str
        Name of the source file.
    filetype : str
        Type of the dataset (e.g. 'raster', 'bag').
    metadata : Dict
        Dictionary containing metadata such as resolution, bounds, etc.
    """

    def __new__(cls, depth_data:np.ndarray,
                filename: str,
                filetype: str,
                metadata: dict):
        # 1. Convert the input to an array of our class type
        obj = np.asanyarray(depth_data).view(cls)

        # 2. Attach the initial metadata
        obj.filename = filename
        obj.filetype = filetype
        obj.metadata = metadata
        obj.orig_shape = obj.shape
        return obj

    def __array_finalize__(self, obj):
        """
        Preserve metadata during numpy operations
        """

        if obj is None: return
        # Copy metadata from the parent object to the new slice/instance
        self.filename = getattr(obj, 'filename', " ")
        self.filetype = getattr(obj, 'filetype', " ")
        self.metadata = getattr(obj, 'metadata', {})
        self.orig_shape = getattr(obj, 'orig_shape', None)

    def wrap(self, raw_data):
        """
        Create new RasterDataset with same metadata as the current object
        """
        # Cast the raw data to the current class
        new_obj = np.asanyarray(raw_data).view(self.__class__)

        # 2. Sync the metadata dictionary
        new_obj.__dict__.update(self.__dict__)

        return new_obj


    def __repr__(self):
        # Custom print to show bathymetry with associated metadata
        return (f"BathymetryDataset(data={super().__repr__()}, "
                f"\nfilename='{self.filename}', "
                f"\nfiletype='{self.filetype}, "
                f"\nmetadata='{self.metadata}')")
    
    def show_depth(self, title: Union[str, None] = None):
        """
        Plots the depth for visualization
        Parameters
        ----------
        title: str
            Custom plot title
            Default is filename with linespacing and resolution information

        Returns
        -------
        none

        """
        return NotImplementedError("show_depth method not implemented in base BathymetryDataset class.")




class RasterDataset(BathymetryDataset):
    """
    Subclass for Raster-type bathymetry with additional helper functions
    """

    def __new__(cls, depth_data, **kwargs):
        # create BathymetryDataset datatype
        obj = super().__new__(cls, depth_data, **kwargs)
        return obj


    def __array_finalize__(self, obj):
        """
        Preserve metadata during numpy operations
        """
        super().__array_finalize__(obj)
        if obj is None: return
        # Copy base metadata
        self.filename = getattr(obj, 'filename', " ")
        self.filetype = getattr(obj, 'filetype', "raster")
        self.metadata = getattr(obj, 'metadata', {})
        self.orig_shape = getattr(obj, 'orig_shape', None)

    @property
    def resolution(self):
        """Returns data resolution extracted from the raster metadata"""
        return self.metadata['resolution']

    @property
    def min_val(self):
        """Returns minimum value in the depth array, excluding NaNs"""
        return np.nanmin(self.flatten())

    @property
    def max_val(self):
        """Returns maximum value in the depth array, excluding NaNs"""
        return np.nanmax(self.flatten())

    @property
    def ndv_value(self):
        """Returns no-data-value extracted from the raster metadata"""
        return self.metadata['ndv_value']


    def show_depth(self, title: Union[str, None] = None):
        """
        Plots the depth for visualization
        Parameters
        ----------
        title: str
            Custom plot title
            Default is filename with linespacing and resolution information

        Returns
        -------
        none

        """
        fig, ax1 = plt.subplots()
        res = self.metadata['resolution']
        if self.filetype == 'raster':
            im = ax1.imshow(self, cmap='terrain', aspect='equal')
            shape_0 = self.shape[0]
            shape_1 = self.shape[1]
            fig.colorbar(im, label='Depth (m)')
            locs = ax1.get_xticks()
            ax1.set_xticks(locs)
            ax1.set_xticklabels([str(int(x * res)) for x in locs])
            locs = ax1.get_yticks()
            ax1.set_yticks(locs)
            ax1.set_yticklabels([str(int(y * res)) for y in locs])
            ax1.tick_params(axis='x', labelrotation=90)
            ax1.set_xlim(left=0, right=shape_1)
            ax1.set_ylim(top=0, bottom=shape_0)
        elif self.filetype == 'points':
            x = self[:, 0]
            y = self[:, 1]
            depth = self[:, 2]
            sc = ax1.scatter(x, y, c=depth, cmap='terrain', marker='.', s=1)
            fig.colorbar(sc, label='Depth (m)')
            shape_0 = int((np.max(y) - np.min(y)) / res)
            shape_1 = int((np.max(x) - np.min(x)) / res)
        else:
            raise ValueError(f"Unrecognized file type for plotting: {self.filetype}")

        ax1.set_xlabel("West-East (m)")
        ax1.set_ylabel("North-South (m)")
        if title is None:
            fn = Path(self.filename).name
            title = f"{fn} at {res}m resolution"
        ax1.set_title(f"""
                    Surface:{title} at {res}m resolution
                    Dimensions: {shape_0 * res / 1000}km by {shape_1 * res / 1000}km
                        """)
        


    def __repr__(self):
        """
        Convenience function for printing raster-specific information

        Returns
        -------
        none
        """

        return (f"RasterDataset(data={super().__repr__()}, "
                f"\nshape: {self.shape},"
                f"\nfilename='{self.filename}', "
                f"\nfull path: {self.metadata['full_path']},"
                f"\nresolution: {self.resolution},"
                f"\nndv_value: {self.ndv_value},"
                f"\nfiletype='{self.filetype}, "
                f"\nmetadata='{self.metadata}'")


class CSVDataset(BathymetryDataset):
    """
    Subclass for Raster-type bathymetry with additional helper functions
    """

    def __new__(cls, depth_data, **kwargs):
        # create BathymetryDataset datatype
        obj = super().__new__(cls, depth_data, **kwargs)
        return obj


    def __array_finalize__(self, obj):
        """
        Preserve metadata during numpy operations
        """
        super().__array_finalize__(obj)
        if obj is None: return
        # Copy base metadata
        self.filename = getattr(obj, 'filename', " ")
        self.filetype = getattr(obj, 'filetype', "csv")
        self.metadata = getattr(obj, 'metadata', {})
        self.orig_shape = getattr(obj, 'orig_shape', None)


class BPSDataset(BathymetryDataset):
    """
    Subclass for Raster-type bathymetry with additional helper functions
    """

    def __new__(cls, depth_data, **kwargs):
        # create BathymetryDataset datatype
        obj = super().__new__(cls, depth_data, **kwargs)
        return obj


    def __array_finalize__(self, obj):
        """
        Preserve metadata during numpy operations
        """
        super().__array_finalize__(obj)
        if obj is None: return
        # Copy base metadata
        self.filename = getattr(obj, 'filename', " ")
        self.filetype = getattr(obj, 'filetype', " bps")
        self.metadata = getattr(obj, 'metadata', {})
        self.orig_shape = getattr(obj, 'orig_shape', None)

