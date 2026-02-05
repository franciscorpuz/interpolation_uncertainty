import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from interpolation_uncertainty.readers.bathymetryDataset import RasterDataset
from interpolation_uncertainty.methods.rasterSpectralMethods import (GlenPSD,
                                                           GlenAmplitude,
                                                           EliasUncertainty)
from interpolation_uncertainty.methods.rasterSpatialMethods import (RasterSpatialStd,
                                                          RasterSpatialDiff,
                                                          RasterSpatialGaussian)

from functools import partial

raster_methods = {'amp_v1': GlenAmplitude,
                  'psd_v1': GlenPSD,
                  'amp_v2': partial(EliasUncertainty, method='amplitude'),
                  'psd_n': partial(EliasUncertainty, method='psd_n'),
                  'psd_lf': partial(EliasUncertainty, method='psd_lf'),
                  'psd_df': partial(EliasUncertainty, method='psd_df'),
                  'spectrum': partial(EliasUncertainty, method='spectrum'),
                  'spatial_std': partial(RasterSpatialStd, method='spatial_std'),
                  'spatial_diff': partial(RasterSpatialDiff, method='spatial_diff'),
                  'spatial_gaussian': partial(RasterSpatialGaussian, method='spatial_gaussian')
                  }

spectral_methods = ['amp_v1', 'psd_v1', 'amp_v2', 'psd_n', 'psd_lf', 'psd_df', 'spectrum']
spatial_methods = ['spatial_std', 'spatial_diff', 'spatial_gaussian']


class RasterProcessor:
    """
    Base class for methods that uses BathymetryDataset as input
    """
    def __init__(self, linespacing_meters: int,
                        multiple: int,
                        max_multiple: int):

        # define parameters needed for most of the methods
        self.linespacing_meters = linespacing_meters
        self.multiple = multiple
        self.max_multiple = max_multiple


    def get_column_indices(self, rasterDataset: RasterDataset) -> np.ndarray:

        """
        Determine indices of the columns to be used as simulated sampling lines
        Indices are rounded to the nearest even integer for convenience

        Parameters:
        -------------
        array_len : int
                    Length of the array containing bathymetry depth

        Returns:
        -----------
        col_indxs : np.array
                    array indices corresponding to sample points/lines
                    :rtype: np.ndarray

        """
        array_len: int = rasterDataset.shape[1]
        resolution: int = rasterDataset.metadata["resolution"]
        linespacing_meters: int = self.linespacing_meters
        max_multiple: int = self.max_multiple

        # Round the desired linespacing to the nearest even integer
        # for computational convenience
        linespacing_in_pixels  = np.round(linespacing_meters / resolution)
        if (linespacing_in_pixels % 2) != 0:
            linespacing_in_pixels = linespacing_in_pixels - 1

        if array_len < linespacing_in_pixels:
            raise ValueError(
                f"""
                Desired linespacing should be less than the Spatial coverage.
                Entered Linespacing: {linespacing_meters}m
                Bathymetric coverage: {array_len * resolution}m"""
            )

        # Valid sampling columns is determined by the window length
        # Window lengths are multiples of the linespacing
        window_size_pixels = linespacing_in_pixels * max_multiple
        start_col = int(
            (window_size_pixels // 2) -
            (linespacing_in_pixels // 2)
        )

        last_col = int(
            array_len
            - (window_size_pixels // 2)
            + (linespacing_in_pixels // 2)
            - 1
        )

        # actual sampling indices will be determined by the desired linespacing
        column_indices = np.arange(start_col, last_col, (linespacing_in_pixels + 1)).astype(int)

        return column_indices

    def matrix2strip(self, depth: RasterDataset) -> RasterDataset:
        """
        Transform depth matrix into a single vertical strip with
        width equal to the linespacing

        Parameters
        ----------
        depth : rasterDataset
                1d vector or 2d array of bathymetric values

        Returns
        --------
        strip : np.ndarray
                bathymetric depth transformed into a single strip of
                 width equal to the specified linespacing for
                 further raster processing


        """

        column_indices = self.get_column_indices(depth)

        # if depth is a vector, convert to matrix
        if len(depth.shape) < 1:
            new_depth = depth.wrap(np.expand_dims(depth, axis=0))
            depth = new_depth

        current_multiple = self.multiple
        start, end = column_indices[0], column_indices[1]
        linespacing = end - start - 1
        window_size = linespacing * current_multiple
        midpoint = start + (linespacing // 2) + 1

        # Determine column boundaries for window segment
        # -1 / +1 will include sampling columns at the edges
        start_col = int(midpoint - (window_size // 2)) - 1
        end_col = int(column_indices[-1] + (window_size // 2) + 1)

        # Get sliding window view of depth using window_size
        # +2 will compensate for the additional pixels on the edges
        window_views = sliding_window_view(depth[:, start_col:end_col],
                                           window_shape=(depth.shape[0],
                                                         window_size + 2))

        # remove extra dimension and only retain views of the window size
        stride = linespacing + 1
        window_views = window_views.squeeze()[::stride]

        # reshape to 2d matrix
        strips = window_views.reshape(-1, window_size + 2)

        # cast back to rasterDataset
        output = depth.wrap(strips)

        return output

    @staticmethod
    def strip2matrix(data_strip: RasterDataset,
                    column_indices: np.ndarray) -> RasterDataset:

        """
        Reverses the matrix2strip function, reverts the strip back
        to the original dimensions of the bathymetric depth

        Parameters
        ----------
        data_strip : RasterDataset
                    processed depth in strip form
        column_indices : np.array
                         column indices/location of the sampling lines

        Returns
        --------
        unstripped : np.array
                    values in their original spatial location


        """

        # placeholder for reconstructed matrix
        output = np.full(data_strip.orig_shape, np.nan)


        # "Cut" the long strip into vertical segments with length (rows)
        # equal to the original depth matrix
        num_rows, num_cols = data_strip.orig_shape[0], data_strip.shape[1]
        window_views = sliding_window_view(data_strip,
                                           window_shape=(num_rows,
                                                         num_cols))

        # remove extra dimension and only retain views of the window size
        stride = num_rows
        segment_strips = window_views.squeeze()[::stride]

        # print(f"num_rows, num_cols: {num_rows, num_cols}")
        # print(f"segment strips shape: {segment_strips.shape}")
        # print(f"segment strips: {segment_strips}")

        # Start with the first slice of the segment_strips
        strip_0 = segment_strips[0, :, :]

        # Remove the first column of the succeeding slides as they overlap
        # with the last column of the previous slice
        strip_rest = segment_strips[1:, :, 1:]

        # concatenate succeeding slices in the 2nd dimension
        strip_rest = np.transpose(strip_rest, (0, 2, 1))
        strip_rest = strip_rest.reshape(-1, strip_rest.shape[2]).T

        # concatenate first slice with the rest of the segments
        unstripped = np.concatenate((strip_0, strip_rest), axis=1)

        # Place reconstructed array into proper columns
        end_col = int(column_indices[0] + unstripped.shape[1])
        output[:, column_indices[0]:end_col] = unstripped
        # output[:, column_indices[0]:end_col + 1] = unstripped
        # output[:, column_indices[0]:column_indices[-1] + 1] = unstripped

        # Crop out columns with np.nan pixels
        cols_with_nan = np.isnan(output).any(axis=0)
        output = output[:, ~cols_with_nan]

        # cast to rasterDataset
        output = data_strip.wrap(output)

        return output

    def compute_interpolation(self, data: RasterDataset) -> RasterDataset:
        interpolated_strip = np.linspace(start=data[:, 0],
                                         stop=data[:, -1],
                                         num=data.shape[1]).T

        # cast to RasterDataset
        output = data.wrap(interpolated_strip)
        return output

    def compute_residual(self, bathy_data: RasterDataset, take_abs:bool = True) -> RasterDataset:
        interpolation = self.compute_interpolation(bathy_data)
        residual = bathy_data - interpolation
        if take_abs:
            residual = np.abs(residual)
            residual = bathy_data.wrap(residual)
        return residual


    def compute_interpolated_surface(self, rasterDataset: RasterDataset) -> RasterDataset:
        column_indices = self.get_column_indices(rasterDataset)

        # transform into strip of width equal to linespacing
        depthdata_as_strip = self.matrix2strip(rasterDataset)

        # execute the interpolation
        interpolated_strip = self.compute_interpolation(depthdata_as_strip)

        # transform back to original dimensions
        return self.strip2matrix(data_strip=interpolated_strip,
                                 column_indices=column_indices)

    def compute_residual_surface(self, bathy_data: RasterDataset) -> RasterDataset:
        interpolated_surface = self.compute_interpolated_surface(bathy_data)

        # number of columns may decrease due to subsampling in the interpolation
        new_cols = interpolated_surface.shape[1]

        # residual is original bathy values minus the computed interpolation
        # use only the number of columns present after the interpolation process
        residual_surface = bathy_data[:, :new_cols] - interpolated_surface
        residual_surface = bathy_data.wrap(residual_surface)
        return np.abs(residual_surface)


    def post_process(self, uncertainty_strip:RasterDataset) -> RasterDataset:
        """
        Create mirror image of uncertainty_strip to regain original dimension of the strip
        as output of spatial or spectral methods reduce the number of columns by half

        Parameters
        ----------
        uncertainty_strip

        Returns
        -------
        uncertainty_strip with double the width

        """
        # Compute original width
        linespacing_width = int((uncertainty_strip.shape[1] - 2) / self.multiple)

        # Include a column of zeroes on each end for the output strip
        output = np.zeros(shape=(uncertainty_strip.shape[0], linespacing_width + 2))
        num_cols = output.shape[1]

        # Create mirror image
        selected_data = uncertainty_strip[:, :int(num_cols / 2)]
        output[:, :int(num_cols / 2)] = selected_data
        output[:, int(num_cols / 2):] = np.fliplr(selected_data)

        output = uncertainty_strip.wrap(output)

        return output


    def estimate_uncertainty(self, method: str,
                             residual_data: RasterDataset,
                             is_strip:bool = False,
                             **kwargs):
        """ Pipeline for computing the uncertainty estimate from the residual"""

        if is_strip:
            residual_strip = residual_data
        else:
            residual_strip = self.matrix2strip(residual_data)

        # Extract selected method and apply to residual
        estimation_method = raster_methods[method]
        output_strip = estimation_method(data_strip=residual_strip,
                                 linespacing_meters=self.linespacing_meters,
                                 current_multiple=self.multiple,
                                 max_multiple=self.max_multiple,
                                 **kwargs).estimate_uncertainty()


        column_indices = self.get_column_indices(residual_data)

        if method in spectral_methods:
            # spectral methods just need to be mirrored
            output_strip = residual_data.wrap(output_strip)
            output = self.post_process(output_strip)
            if not is_strip:
                return self.strip2matrix(data_strip=output, column_indices=column_indices)
            else:
                return output

        elif method in spatial_methods:
            # output of spatial methods is a dictionary of various combinations
            # (e.g, mean, std, envelope1, envelope2, etc)
            # loop over objects in the dictionary for post processing

            for key in output_strip.keys():
                current_output = output_strip[key]
                current_output = residual_data.wrap(current_output)
                post_processed = self.post_process(current_output)
                if not is_strip:
                    output_strip[key] = self.strip2matrix(data_strip=post_processed, column_indices=column_indices)
                else:
                    output_strip[key] = post_processed
            return output_strip
        else:
            raise ValueError(f"Unexpected method: {method}")












