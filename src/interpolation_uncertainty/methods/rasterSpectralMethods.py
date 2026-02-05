from interpolation_uncertainty.methods.rasterMethods import RasterMethods
from interpolation_uncertainty.readers.bathymetryDataset import RasterDataset
import numpy as np
from scipy import signal


def create_spatial_signal(resolution: int, max_cell_number: int):
    """
    Create the distance and frequency dependent scaling factors.
    """
    frequencies = np.fft.rfftfreq(max_cell_number, resolution)
    if len(frequencies) % 2 == 0:
        frequencies = frequencies[:-1]
    distances = np.arange(max_cell_number) * resolution
    distances_2d, freq_2d = np.meshgrid(distances, frequencies)
    spatial_scale = distances_2d * freq_2d
    spatial_scale = np.where(spatial_scale < 0.25, spatial_scale, 0.25)
    spatial_signal = np.sin(spatial_scale * 2 * np.pi)

    return spatial_signal


class RasterSpectralProcessor(RasterMethods):
    """
    Class implementation for Spectral Methods of estimating Uncertainty
    """

    def __init__(self, data_strip: RasterDataset,
                 window_type: str = 'hann', **kwargs):
        self.window_type = window_type
        self.data_strip = data_strip
        self.window_vector = signal.windows.get_window(window=self.window_type,
                                                       Nx=self.data_strip.shape[1],
                                                       fftbins=False)
        self.windowed_input = self.data_strip * self.window_vector
        res = self.data_strip.metadata['resolution']
        self.rfft_values = np.abs(np.fft.rfft(self.windowed_input, axis=1))
        _, self.rfft_cols = self.rfft_values.shape
        self.rfft_frequencies = np.fft.rfftfreq(self.windowed_input.shape[1], d=res)




class GlenAmplitude(RasterSpectralProcessor):

    def estimate_uncertainty(self):
        scale_factor = np.sum(self.window_vector)
        if self.rfft_cols % 2 == 0:
            self.rfft_values[:, 1:-1] = self.rfft_values[:, 1:-1] * 2
            self.rfft_values = self.rfft_values[:, :-1]
            self.rfft_frequencies = self.rfft_frequencies[:-1]
        else:
            self.rfft_values[:, 1:] = self.rfft_values[:, 1:] * 2

        energy = self.rfft_values / np.sum(scale_factor)

        # compute contribution per frequency
        spatial_signal = create_spatial_signal(self.data_strip.metadata["resolution"],
                                               self.windowed_input.shape[1])
        uncertainty = energy @ spatial_signal
        return uncertainty

    # @property
    # def uncertainty_surface(self):
    #     uncertainty = self.estimate_uncertainty()
    #     return self.strip2matrix(uncertainty,
    #                              self.bathydataset.depth_data.shape,
    #                              self.column_indices)


class GlenPSD(RasterSpectralProcessor):

    def estimate_uncertainty(self):
        # scale_factor = np.sum(self.window_vector ** 2) / self.bathydataset.metadata['resolution']
        self.rfft_values = self.rfft_values ** 2
        if self.rfft_cols % 2 == 0:
            self.rfft_values[:, 1:-1] = self.rfft_values[:, 1:-1] * 2
        else:
            self.rfft_values[:, 1:] = self.rfft_values[:, 1:] * 2
        self.rfft_values = self.rfft_values[:, :-1]
        self.rfft_frequencies = self.rfft_frequencies[:-1]

        energy = self.rfft_values
        energy_freqs = self.rfft_frequencies

        # compute contribution per frequency
        spatial_signal = create_spatial_signal(self.data_strip.metadata["resolution"],
                                               self.windowed_input.shape[1])
        variance = energy @ spatial_signal

        # normalize energy (convert m^2 to meters)
        variance = variance / len(energy_freqs)
        uncertainty = np.sqrt(variance)

        return uncertainty

def compute_energy_elias(data: np.ndarray,
                         resolution: int,
                         method: str,
                         window_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT energy using 'method' process

    Parameters
    ----------
    data : np.array
           Input depth
    resolution : int
                 Spatial resolution of the array
    method : str
             FFT Method used to estimate signal energy
    window_values : np.array
                    scaling window

    Returns
    -------
    np.array
            Spectral energy in the signal
    """

    rfft_values = np.abs(np.fft.rfft(data, axis=1))
    _, num_cols = rfft_values.shape
    rfft_frequencies = np.fft.rfftfreq(data.shape[1], d=resolution)

    if method == "amplitude":  # ASD in Scipy doc

        cden = np.sqrt(np.sum(window_values ** 2))
        energy = np.sqrt(resolution) * rfft_values / cden

        if num_cols % 2 == 0:
            energy[:, 1:-1] = energy[:, 1:-1] * 2
        else:
            energy[:, 1:] = energy[:, 1:] * 2

        energy = energy[:, :-1]
        rfft_frequencies = rfft_frequencies[:-1]

    elif method == "psd":
        cden = np.sqrt(np.sum(window_values ** 2))
        energy = resolution * (np.abs(rfft_values / cden) ** 2)

        if num_cols % 2 == 0:  # even length → Nyquist bin exists
            energy[:, 1:-1] *= 2
        else:  # odd length → no Nyquist bin
            energy[:, 1:] *= 2

        energy = energy[:, :-1]
        rfft_frequencies = rfft_frequencies[:-1]

    elif method == "psd_n":
        cden = np.sqrt(np.sum(window_values ** 2))
        energy = resolution * (np.abs(rfft_values / cden) ** 2)

        if num_cols % 2 == 0:  # even length → Nyquist bin exists
            energy[:, 1:-1] *= 2
        else:  # odd length → no Nyquist bin
            energy[:, 1:] *= 2

        energy = energy[:, :-1]
        rfft_frequencies = rfft_frequencies[:-1]

    elif method == "psd_lf":
        cden = np.sqrt(np.sum(window_values ** 2))
        energy = resolution * (np.abs(rfft_values / cden) ** 2)

        if num_cols % 2 == 0:  # even length → Nyquist bin exists
            energy[:, 1:-1] *= 2
        else:  # odd length → no Nyquist bin
            energy[:, 1:] *= 2

        energy = energy[:, :-1]
        rfft_frequencies = rfft_frequencies[:-1]

    elif method == "psd_df":
        cden = np.sqrt(np.sum(window_values ** 2))
        energy = resolution * (np.abs(rfft_values / cden) ** 2)

        if num_cols % 2 == 0:  # even length → Nyquist bin exists
            energy[:, 1:-1] *= 2
        else:  # odd length → no Nyquist bin
            energy[:, 1:] *= 2

        energy = energy[:, :-1]
        rfft_frequencies = rfft_frequencies[:-1]

    elif method == "spectrum":  # amplitude spectrum
        # camp = np.abs(np.sum(window_values))
        #
        # energy = rfft_values / camp
        #
        # if num_cols % 2 == 0:  # even length → Nyquist bin exists
        #     energy[:, 1:-1] *= 2
        # else:  # odd length → no Nyquist bin
        #     energy[:, 1:] *= 2

        cden = np.sqrt(np.sum(window_values ** 2))
        energy = resolution * (np.abs(rfft_values / cden) ** 2)

        if num_cols % 2 == 0:  # even length → Nyquist bin exists
            energy[:, 1:-1] *= 2
        else:  # odd length → no Nyquist bin
            energy[:, 1:] *= 2

        energy = energy[:, :-1]
        rfft_frequencies = rfft_frequencies[:-1]

    else:
        raise ValueError(
            f"""Unknown FFT Method: {method}
                FFT options: {'amplitude', 'psd_n', 'psd_lf', 'spectrum'}
                """)

    # energy = rfft_values / scale_factor

    return energy, rfft_frequencies


class EliasUncertainty(RasterSpectralProcessor):

    def __init__(self, method:str, **kwargs):
        self.energy_freqs = None
        self.energy = None
        self.method = method
        super().__init__(**kwargs)

    def estimate_uncertainty(self):
        preprocessed_signal = self.windowed_input
        res = self.data_strip.metadata['resolution']
        self.energy, self.energy_freqs = compute_energy_elias(preprocessed_signal,
                                                            res,
                                                            self.method,
                                                            self.window_vector)

        df = 1.0 / (preprocessed_signal.shape[1] * res)

        # compute contribution per frequency
        spatial_signal = create_spatial_signal(res, preprocessed_signal.shape[1])

        variance = None
        if self.method == "amplitude":
            variance = ((self.energy ** 2) / preprocessed_signal.shape[1]) @ spatial_signal
        elif self.method == "psd":
            variance = (self.energy / preprocessed_signal.shape[1]) @ spatial_signal
        elif self.method == "psd_n":
            variance = (self.energy / preprocessed_signal.shape[1]) @ spatial_signal
        elif self.method == "psd_lf":
            variance = (self.energy / len(self.energy_freqs)) @ spatial_signal
        elif self.method == "psd_df":
            variance = (self.energy * df) @ spatial_signal
        elif self.method == "spectrum":
            variance = (self.energy / len(self.energy_freqs)) @ spatial_signal
        else:
            raise ValueError(f"Method not found: {self.method}")

        if variance is None:
            raise ValueError('Variance values is None')
        uncertainty = np.sqrt(variance)

        return uncertainty