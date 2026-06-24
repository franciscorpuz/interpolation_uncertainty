import numpy as np

from interpolation_uncertainty.methods.spectral import compute_uncertainty


def process_line_pair(
    line_pair: tuple[int, int],
    restored_mask: np.ndarray,
    current_depth_raster: np.ndarray,
    resolution: float,
    method: str = 'psd',
) -> dict | None:
    """Compute the RMS interpolation uncertainty for one pair of adjacent tracklines.

    For the two tracklines at columns ``line_pair[0]`` and ``line_pair[1]``,
    this function:

    1. Extracts the valid row extent for each line from ``restored_mask``.
    2. Interpolates any depth gaps along each line with ``np.interp``.
    3. Clips both profiles to their shared row overlap.
    4. Subtracts the mean (DC removal) so the spectrum captures seafloor
       roughness rather than absolute depth.
    5. Computes the one-sided PSD via :func:`compute_uncertainty` using the
       inter-line distance as the segment length.
    6. Averages the two PSDs and collapses each segment to a scalar RMS
       uncertainty: ``sqrt(sum(PSD * delta_f))`` in metres.

    Parameters
    ----------
    line_pair : tuple of int
        ``(col_left, col_right)`` — column indices of two adjacent tracklines
        in ``restored_mask``.
    restored_mask : np.ndarray of bool
        Clean line mask in the rotated frame, used to identify valid row
        extents for each line.
    current_depth_raster : np.ndarray of float
        Depth values at trackline pixels (NaN elsewhere), as returned by
        :func:`assign_depths_to_lines`.
    resolution : float
        Spatial sampling interval in metres per pixel, forwarded to
        :func:`compute_uncertainty` to convert FFT bins to physical
        frequencies (cycles/m).

    Returns
    -------
    dict or None
        ``None`` if the two lines have no shared row overlap or if the shared
        overlap is shorter than ``linespan`` (not enough samples for one
        segment).  Otherwise a dict with keys:

        - ``'uncertainty_rms'`` : np.ndarray, shape ``(n_segments,)`` —
          RMS uncertainty per overlapping segment in metres.
        - ``'row_start'`` : int — first shared row (inclusive).
        - ``'row_end'``   : int — last shared row (exclusive).
        - ``'col_start'`` : int — first gap column (``line_pair[0] + 1``).
        - ``'col_end'``   : int — last gap column (exclusive, ``line_pair[1]``).
        - ``'linespan'``  : int — inter-line distance in pixels, which equals
          the analysis segment length.
    """
    col1, col2  = line_pair
    linespan    = col2 - col1

    rows1 = np.argwhere(restored_mask[:, col1])[:, 0]
    rows2 = np.argwhere(restored_mask[:, col2])[:, 0]
    if len(rows1) == 0 or len(rows2) == 0:
        return None

    r1_min, r1_max = rows1.min(), rows1.max()
    r2_min, r2_max = rows2.min(), rows2.max()
    row_start = max(r1_min, r2_min)
    row_end   = min(r1_max, r2_max)

    # Need at least linespan rows of shared overlap for one full segment
    if row_end <= row_start or (row_end - row_start) < linespan:
        return None

    # --- Interpolate depth gaps along each line ---
    full_rows1 = np.arange(r1_min, r1_max + 1)
    full_rows2 = np.arange(r2_min, r2_max + 1)
    depth1     = current_depth_raster[rows1, col1]
    depth2     = current_depth_raster[rows2, col2]
    interp1    = np.interp(full_rows1, rows1, depth1, left=np.nan, right=np.nan)
    interp2    = np.interp(full_rows2, rows2, depth2, left=np.nan, right=np.nan)

    # --- Clip to shared row range ---
    shared1 = interp1[row_start - r1_min : row_start - r1_min + (row_end - row_start)]
    shared2 = interp2[row_start - r2_min : row_start - r2_min + (row_end - row_start)]

    # --- DC removal: subtract mean so the spectrum captures roughness only ---
    shared1 = shared1 - np.nanmean(shared1)
    shared2 = shared2 - np.nanmean(shared2)

    # --- Spectral energy via compute_uncertainty ---
    seg_energy1, freqs = compute_uncertainty(shared1, resolution=resolution,
                                             line_spacing=linespan, method=method)
    seg_energy2, _     = compute_uncertainty(shared2, resolution=resolution,
                                             line_spacing=linespan, method=method)

    # --- Average spectra, then collapse to scalar RMS per segment ---
    avg_energy      = (seg_energy1 + seg_energy2) / 2.0  # (n_segments, n_freqs) m²/Hz
    delta_f         = freqs[1] - freqs[0]
    uncertainty_rms = np.sqrt(np.sum(avg_energy * delta_f, axis=1))  # (n_segments,) m

    return {
        'uncertainty_rms': uncertainty_rms,
        'row_start':  row_start,
        'row_end':    row_end,
        'col_start':  col1 + 1,
        'col_end':    col2,
        'linespan':   linespan,
    }


def build_uncertainty_raster(
    peaks: np.ndarray,
    restored_mask: np.ndarray,
    current_depth_raster: np.ndarray,
    resolution: float,
    method: str = 'psd',
) -> np.ndarray:
    """Fill an uncertainty raster for all adjacent trackline pairs.

    Iterates over every consecutive pair of detected tracklines, calls
    :func:`process_line_pair` to compute a per-segment RMS uncertainty, and
    tiles each scalar uncertainty value uniformly across the inter-line gap.

    Parameters
    ----------
    peaks : np.ndarray of int
        Sorted column indices of detected tracklines, as returned by
        :func:`detect_trackline_positions`.
    restored_mask : np.ndarray of bool
        Clean line mask in the rotated frame (same frame as
        ``current_depth_raster``).
    current_depth_raster : np.ndarray of float
        Depth values at trackline pixels; NaN elsewhere, as returned by
        :func:`assign_depths_to_lines`.
    resolution : float
        Spatial sampling interval in metres per pixel, forwarded to
        :func:`process_line_pair`.

    Returns
    -------
    uncertainty_raster : np.ndarray of float
        Array of the same shape as ``current_depth_raster``.  Gap pixels
        between adjacent tracklines carry the RMS interpolation uncertainty
        in metres; all other pixels remain ``NaN``.
    """
    uncertainty_raster = np.full_like(current_depth_raster, np.nan, dtype=float)

    for col1, col2 in zip(peaks[:-1], peaks[1:]):
        result = process_line_pair(
            (col1, col2), restored_mask, current_depth_raster, resolution, method=method
        )
        if result is None:
            continue

        u_rms     = result['uncertainty_rms']
        row_start = result['row_start']
        row_end   = result['row_end']
        col_start = result['col_start']
        col_end   = result['col_end']
        linespan  = result['linespan']
        step      = linespan - 1

        for seg_idx, u_val in enumerate(u_rms):
            seg_row_start = row_start + seg_idx * step
            seg_row_end   = min(seg_row_start + linespan, row_end)
            if seg_row_start >= row_end:
                break
            uncertainty_raster[seg_row_start:seg_row_end, col_start:col_end] = u_val

    return uncertainty_raster
