from interpolation_uncertainty.methods.rasterMethods import RasterMethods
import numpy as np
from scipy.stats import genextreme
from numpy.lib.stride_tricks import sliding_window_view
from interpolation_uncertainty.readers.bathymetryDataset import RasterDataset

class RasterSpatialProcessor(RasterMethods):
    """
    Class implementation for Spatial Methods of estimating Uncertainty
    """

    def __init__(self, data_strip: RasterDataset,
                 method: str,
                 current_multiple: int,
                 min_window: int = 2, **kwargs):
        self.min_window = min_window
        self.data_strip = data_strip
        self.current_multiple = current_multiple
        self.num_rows, self.num_cols = data_strip.shape
        self.interpolation_cell_distance = ((self.num_cols - 2) // self.current_multiple) + 2
        self.shape = (self.num_rows, self.interpolation_cell_distance)
        self.operator = method

    def estimate_uncertainty(self):
        """
        Calculate the variance of the provided depth array in parts.
        """

        # containers
        mins = []
        maxs = []
        stds = []


        for win_len in range(self.min_window, self.interpolation_cell_distance // 2 + 1):
            windows = sliding_window_view(self.data_strip, window_shape=win_len, axis=-1)
            mins.append(np.min(windows, axis=-1))
            maxs.append(np.max(windows, axis=-1))
            stds.append(np.std(windows, axis=-1))

        payload = {"mins": mins,
                   "maxs": maxs,
                   "stds": stds}
        # results = {}

        if self.operator == 'spatial_std':
            results = computeSpatialStd(payload)
        elif self.operator == 'spatial_diff':
            results = computeSpatialDiff(payload)
        elif self.operator == 'spatial_gaussian':
            results = computeSpatialGaussian(payload)
        else:
            raise ValueError(f"Unexpected spatial operator: {self.operator}")

        return results


def computeSpatialStd(payload: dict, min_window: int = 2) -> dict:

    stds = payload["stds"]

    num_rows = stds[0].shape[0]
    num_cols = len(stds) + 1

    std_mean = np.zeros(shape=(num_rows, num_cols))
    std_max = np.zeros(shape=(num_rows, num_cols))
    std_std = np.zeros(shape=(num_rows, num_cols))
    std_envelope1 = np.zeros(shape=(num_rows, num_cols))
    std_envelope2 = np.zeros(shape=(num_rows, num_cols))
    std_envelope3 = np.zeros(shape=(num_rows, num_cols))

    for win_len in range(min_window, num_cols + 1):
        std_mean[:, win_len-1] = np.mean(stds[win_len-min_window], axis=-1)
        std_max[:, win_len-1] = np.max(stds[win_len-min_window], axis=-1)
        std_std[:, win_len-1] = np.std(stds[win_len-min_window], axis=-1)
        std_envelope1[:, win_len-1] = std_mean[:, win_len-1] + std_std[:, win_len-1]
        std_envelope2[:, win_len-1] = std_mean[:, win_len-1] + (std_std[:, win_len-1] * 2)
        std_envelope3[:, win_len-1] = std_mean[:, win_len-1] + (std_std[:, win_len-1] * 3)

    results = {'std_mean': std_mean,
               'std_max': std_max,
               'std_std': std_std,
               'std_envelope1': std_envelope1,
               'std_envelope2': std_envelope2,
               'std_envelope3': std_envelope3}

    return results

def computeSpatialDiff(payload: dict, min_window: int = 2) -> dict:
    mins = payload["mins"]
    maxs = payload["maxs"]

    num_rows = mins[0].shape[0]
    num_cols = len(mins) + 1

    difference_mean = np.zeros(shape=(num_rows, num_cols))
    difference_std = np.zeros(shape=(num_rows, num_cols))
    difference_max = np.zeros(shape=(num_rows, num_cols))
    difference_envelope1 = np.zeros(shape=(num_rows, num_cols))
    difference_envelope2 = np.zeros(shape=(num_rows, num_cols))
    difference_envelope3 = np.zeros(shape=(num_rows, num_cols))

    for win_len in range(min_window, num_cols + 1):
        differences = maxs[win_len-min_window] - mins[win_len-min_window]
        difference_mean[:, win_len-1] = np.mean(differences, axis=-1)
        difference_std[:, win_len-1] = np.std(differences, axis=-1)
        difference_max[:, win_len-1] = np.max(differences, axis=-1)
        difference_envelope1[:, win_len-1] = difference_mean[:, win_len-1] + difference_std[:, win_len-1]
        difference_envelope2[:, win_len-1] = difference_mean[:, win_len-1] + (difference_std[:, win_len-1] * 2)
        difference_envelope3[:, win_len-1] = difference_mean[:, win_len-1] + (difference_std[:, win_len-1] * 3)

    result = {'difference_mean': difference_mean,
              'difference_std': difference_std,
              'difference_max': difference_max,
              'difference_envelope1': difference_envelope1,
              'difference_envelope2': difference_envelope2,
              'difference_envelope3': difference_envelope3}

    return result

def computeSpatialGaussian(payload: dict, min_window: int = 2) -> dict:

    mins = payload["mins"]
    maxs = payload["maxs"]
    num_rows = mins[0].shape[0]
    num_cols = len(mins) + 1

    gaussian_mean = np.zeros(shape=(num_rows, num_cols))
    gaussian_p95 = np.zeros(shape=(num_rows, num_cols))
    gaussian_p99 = np.zeros(shape=(num_rows, num_cols))

    for win_len in range(min_window, num_cols + 1):
        differences = maxs[win_len-min_window] - mins[win_len-min_window]
        gaussian_mean[:, win_len-1] = np.mean(differences, axis=-1)
        gaussian_p95[:, win_len-1] = np.percentile(differences, 95, axis=-1)
        gaussian_p99[:, win_len-1] = np.percentile(differences, 99, axis=-1)

    results = {'gaussian_mean': gaussian_mean,
               'gaussian_p95': gaussian_p95,
               'gaussian_p99': gaussian_p99}

    return results

class RasterSpatialStd(RasterSpatialProcessor):

    def estimate_uncertainty(self):
        """
        Calculate the variance of the provided depth array in parts.
        """

        # containers
        std_mean = np.zeros(self.shape)
        std_max = np.zeros(self.shape)
        std_envelope1 = np.zeros(self.shape)
        std_envelope2 = np.zeros(self.shape)
        std_envelope3 = np.zeros(self.shape)
        win_len = 0
        results = {}

        for win_len in range(self.min_window, self.interpolation_cell_distance // 2 + 1):
            num_convolutions = self.num_cols - win_len + 1
            differences = np.full((self.num_rows, num_convolutions), 0.0)
            mins = np.full((self.num_rows, num_convolutions), 0.0)
            maxs = np.full((self.num_rows, num_convolutions), 0.0)
            stds = np.full((self.num_rows, num_convolutions), 0.0)
            for step in range(num_convolutions):
                mins[:, step] = np.min(self.data_strip[:, step:step + win_len], axis=-1)
                maxs[:, step] = np.max(self.data_strip[:, step:step + win_len], axis=-1)
                stds[:, step] = np.std(self.data_strip[:, step:step + win_len], axis=-1)
                # differences = maxs - mins

            std_mean[:, win_len - 1] = np.mean(stds , axis=-1)
            std_max[:, win_len - 1] = np.max(stds, axis=-1)
            std_std = np.std(stds, axis=-1)
            std_envelope1[:, win_len - 1] = std_mean[:, win_len - 1] + std_std
            std_envelope2[:, win_len - 1] = std_mean[:, win_len - 1] + 2 * std_std
            std_envelope3[:, win_len - 1] = std_mean[:, win_len - 1] + 3 * std_std


        results = {'std_mean': std_mean,
                   'std_max': std_max,
                   'std_envelope1': std_envelope1,
                   'std_envelope2': std_envelope2,
                   'std_envelope3': std_envelope3}

        # for key in results.keys():
        #     results[key] = self.post_process(results[key])

        return results

class RasterSpatialDiff(RasterSpatialProcessor):

    def estimate_uncertainty(self):

        """
        Calculate the variance of the provided depth array in parts.
        """

        # containers
        difference_mean = np.zeros(self.shape)
        difference_max = np.zeros(self.shape)
        diff_envelope1 = np.zeros(self.shape)
        diff_envelope2 = np.zeros(self.shape)
        diff_envelope3 = np.zeros(self.shape)
        win_len = 0

        for win_len in range(self.min_window, self.interpolation_cell_distance  // 2 + 1):
            num_convolutions = self.num_cols - win_len + 1
            differences = np.full((self.num_rows, num_convolutions), 0.0)
            mins = np.full((self.num_rows, num_convolutions), 0.0)
            maxs = np.full((self.num_rows, num_convolutions), 0.0)
            for step in range(num_convolutions):
                mins[:, step] = np.min(self.data_strip[:, step:step + win_len], axis=-1)
                maxs[:, step] = np.max(self.data_strip[:, step:step + win_len], axis=-1)
                differences[:, step] = maxs[:, step] - mins[:, step]

            diff_mean = np.mean(differences, axis=-1)
            diff_std = np.std(differences, axis=-1)
            diff_max = np.max(differences, axis=-1)
            difference_mean[:, win_len - 1] = diff_mean
            difference_max[:, win_len - 1] = diff_max
            diff_envelope1[:, win_len - 1] = diff_mean + diff_std
            diff_envelope2[:, win_len - 1] = diff_mean + 2 * diff_std
            diff_envelope3[:, win_len - 1] = diff_mean + 3 * diff_std

        results = {'difference_mean': difference_mean,
                   'difference_max': difference_max,
                   'difference_envelope1': diff_envelope1,
                   'difference_envelope2': diff_envelope2,
                   'difference_envelope3': diff_envelope3,
                   }

        # for key in results.keys():
        #     results[key] = self.post_process(results[key])

        return results

class RasterSpatialGEV(RasterSpatialProcessor):

    def estimate_uncertainty(self):

        """
        Calculate the variance of the provided depth array in parts.
        """

        # containers
        gev_mean = np.zeros(self.shape)
        gev_p95_stats = np.zeros(self.shape)
        gev_p99_stats = np.zeros(self.shape)
        win_len = 0


        for win_len in range(self.min_window, self.interpolation_cell_distance  // 2 + 1):
            num_convolutions = self.num_cols - win_len + 1
            differences = np.full((self.num_rows, num_convolutions), 0.0)
            for step in range(num_convolutions):
                mins = np.min(self.data_strip[:, step:step + win_len], axis=-1)
                maxs = np.max(self.data_strip[:, step:step + win_len], axis=-1)
                differences[:, step] = maxs - mins

                for i in range(self.num_rows):
                    shape_, loc_, scale_ = genextreme.fit(differences[i])
                    gev_mean[i, win_len - 1] = loc_
                    gev_p95_stats[i, win_len - 1] = genextreme.ppf(0.95, shape_, loc_, scale_)
                    gev_p99_stats[i, win_len - 1] = genextreme.ppf(0.99, shape_, loc_, scale_)


        results = {'gev_mean': gev_mean,
                   'gev_p95_stats': gev_p95_stats,
                   'gev_p99_stats': gev_p99_stats
                   }

        return results

class RasterSpatialGaussian(RasterSpatialProcessor):

    # def __init__(self, bathy_file: RasterBathymetry) -> None:
    #     super().__init__(bathy_file)

    def estimate_uncertainty(self):

        """
        Calculate the variance of the provided depth array in parts.
        """

        # containers
        gaussian_mean = np.zeros(self.shape)
        gaussian_p95_stats = np.zeros(self.shape)
        gaussian_p99_stats = np.zeros(self.shape)
        win_len = 0


        for win_len in range(self.min_window, self.interpolation_cell_distance  // 2 + 1):
            num_convolutions = self.num_cols - win_len + 1
            differences = np.full((self.num_rows, num_convolutions), 0.0)
            for step in range(num_convolutions):
                mins = np.min(self.data_strip[:, step:step + win_len], axis=-1)
                maxs = np.max(self.data_strip[:, step:step + win_len], axis=-1)
                # stds = np.std(self.data_strip[:, step:step + win_len], axis=-1)
                differences[:, step] = maxs - mins

                gaussian_mean[:, win_len - 1] = np.mean(differences, axis=-1)
                gaussian_p95_stats[:, win_len - 1] = np.percentile(differences, 95, axis=-1)
                gaussian_p99_stats[:, win_len - 1] = np.percentile(differences, 99, axis=-1)


        results = {'gaussian_mean': gaussian_mean,
                   'gaussian_p95': gaussian_p95_stats,
                   'gaussian_p99': gaussian_p99_stats
                   }

        return results