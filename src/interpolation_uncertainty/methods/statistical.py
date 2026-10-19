
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def statistical_estimator(
    data: np.ndarray,
    resolution: int,
    line_spacing: int,
    min_window: int = 2,
    multiple: int = 1,
    method: str = "SAR",
):
    """
    Efficiently compute uncertainty estimates statistically using sliding window

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (num_lines, num_samples).
    min_window : int
        Minimum window length to start sliding over.
    multiple : int
        Multiplier controlling the spacing of the windows.
    method : str
        One of ['diff', 'std', 'gev', 'gaussian'].

    Returns
    -------
    Tuple of np.ndarray
        Depending on `method`, returns corresponding uncertainty statistics.
    """

    num_lines, num_samples = data.shape
    interpolation_cell_distance = ((num_samples - 2) // multiple) + 2
    half = interpolation_cell_distance // 2

    # Preallocate arrays
    shape = (num_lines, interpolation_cell_distance)
    SAR = np.zeros(shape)
    SMR = np.zeros(shape)
    SER = np.zeros(shape)

    # --- Main computation loop ---
    if interpolation_cell_distance % 2:
        max_len = interpolation_cell_distance // 2 + 1
    else:
        max_len = interpolation_cell_distance // 2
    for win_len in range(min_window, max_len + 1):
        windows = sliding_window_view(data, window_shape=win_len, axis=-1)
        # windows shape: (num_lines, num_convolutions, win_len)
        mins = np.min(windows, axis=-1)
        maxs = np.max(windows, axis=-1)
        ranges = maxs - mins

        range_mean = np.mean(ranges, axis=-1)
        range_std = np.std(ranges, axis=-1)
        range_max = np.max(ranges, axis=-1)
        SAR[:, win_len-1] = range_mean
        SER[:, win_len - 1] = range_mean + range_std
        SMR[:, win_len-1] = range_max

    # --- Mirror the ends for symmetry (using half instead of win_len) ---
    width = interpolation_cell_distance
    is_odd = width % 2

    if is_odd:
        SAR[:, half + 1:] = np.fliplr(SAR[:, :half])
        SER[:, half + 1:] = np.fliplr(SER[:, :half])
        SMR[:, half + 1:] = np.fliplr(SMR[:, :half])
    else:
        SAR[:, half:] = np.fliplr(SAR[:, :half])
        SER[:, half:] = np.fliplr(SER[:, :half])
        SMR[:, half:] = np.fliplr(SMR[:, :half])

    if method == 'SAR':
        return SAR
    elif method == 'SER':
        return SER
    elif method == 'SMR':
        return SMR
    elif method == 'ALL':
        return SAR, SER, SMR