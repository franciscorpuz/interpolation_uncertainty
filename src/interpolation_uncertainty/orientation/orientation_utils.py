
import numpy as np
from scipy.signal import find_peaks

### orientation utility functions

def find_temp_offset(heights: list[float] | np.ndarray) -> float:
    """If data has near vertical or horizontal features, this function returns an offset computed from
    the widest valley using inverted peak detection.

    This is useful for determining the appropriate rotation angle to align features in the data.

    Parameters
    ----------
    heights : list of float or np.ndarray
        1D array of histogram bin heights.

    Returns
    -------
    angle_of_valley : float
        The angle corresponding to the widest valley in the histogram.
        If no valleys are found, returns 0 indicating no shift is needed.
    """
    from scipy.signal import peak_widths

    y = np.asarray(heights, dtype=float)
    inv_y = -y  # Invert so troughs become peaks

    # 1. Identify inverted peaks and compute their topological prominence
    troughs, properties = find_peaks(inv_y, prominence=0)


    if len(troughs) == 0:
        # No valleys found; return 0 indicating no shift is needed
        return 0

    # 2. Compute widths at the chosen prominence level
    widths, width_heights, left_ips, right_ips = peak_widths(
        inv_y,
        troughs
    )


    # 3. Find the index corresponding to the widest valley
    widest_idx = np.argmax(widths)
    angle_of_valley = int((left_ips[widest_idx] + right_ips[widest_idx]) / 2)

    return angle_of_valley