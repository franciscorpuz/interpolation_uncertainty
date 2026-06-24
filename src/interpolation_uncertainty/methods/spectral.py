import numpy as np


def compute_uncertainty(
    signal: np.ndarray,
    resolution: float,
    line_spacing: int,
    method: str = "psd",
    overlap: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the spectral energy of a 1-D signal over overlapping segments.

    Divides ``signal`` into overlapping segments of length ``line_spacing``
    stepped by ``line_spacing - overlap`` samples, applies a Hann window to
    each segment, computes the real FFT, and returns a one-sided spectral
    energy estimate for every segment.

    Parameters
    ----------
    signal : np.ndarray
        1-D array of evenly-sampled values (e.g. a single depth profile).
    resolution : float
        Spatial sampling interval in metres (metres per sample).  Used to
        convert FFT bin indices to physical frequencies (cycles per metre).
    line_spacing : int
        Length of each analysis segment in samples.  Can be odd or even.
    method : str, optional
        Spectral scaling method.  One of:

        - ``'amplitude'`` — amplitude spectral density (ASD).  Scaled by
          ``sqrt(resolution) / sqrt(sum(window²))``.  Units: signal-unit /
          sqrt(Hz).
        - ``'psd'`` — one-sided power spectral density.  Scaled by
          ``resolution / sum(window²)``.  Units: signal-unit² / Hz.
        - ``'psd_n'`` — normalised PSD.  Reserved for future per-bin
          normalisation; currently identical to ``'psd'``.
        - ``'psd_lf'`` — low-frequency PSD.  Reserved for future band
          filtering around ``line_spacing``; currently identical to ``'psd'``.
        - ``'psd_df'`` — differential PSD.  Reserved for future differential
          scaling; currently identical to ``'psd'``.
        - ``'spectrum'`` — amplitude spectrum.  Reserved for direct spectral
          visualisation; currently identical to ``'psd'``.

        Default is ``'psd'``.
    overlap : int, optional
        Number of samples shared between consecutive segments.  Must be in
        ``[0, line_spacing - 1]``.  ``overlap=0`` gives non-overlapping
        segments (classic chunking); ``overlap=line_spacing-1`` gives a
        fully rolling window (step of 1 sample).  Default is ``1``.

    Returns
    -------
    energy : np.ndarray
        2-D array of shape ``(n_segments, n_freqs)`` where
        ``n_segments = (len(signal) - line_spacing) // step + 1``,
        ``step = line_spacing - overlap``, and
        ``n_freqs = line_spacing // 2`` (Nyquist bin excluded).
        Row *i* is the spectral energy of the *i*-th segment.
    frequencies : np.ndarray
        1-D array of length ``n_freqs`` with the corresponding spatial
        frequencies in cycles per metre.

    Raises
    ------
    ValueError
        If ``signal`` is not 1-D, if ``line_spacing`` exceeds ``len(signal)``,
        if ``overlap`` is outside ``[0, line_spacing - 1]``, or if ``method``
        is not a recognised option.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError(
            f"'signal' must be 1-D, got shape {signal.shape}."
        )

    valid_methods = ("amplitude", "psd", "psd_n", "psd_lf", "psd_df", "spectrum")
    if method not in valid_methods:
        raise ValueError(
            f"Unknown method '{method}'. Valid options: {valid_methods}."
        )

    n = len(signal)
    if line_spacing > n:
        raise ValueError(
            f"'line_spacing' ({line_spacing}) must be <= len(signal) ({n})."
        )
    if not (0 <= overlap < line_spacing):
        raise ValueError(
            f"'overlap' ({overlap}) must be in [0, line_spacing - 1] "
            f"(i.e. [0, {line_spacing - 1}])."
        )

    step       = line_spacing - overlap
    n_segments = (n - line_spacing) // step + 1
    n_freqs    = line_spacing // 2                # bins to keep (Nyquist excluded)

    # --- Build overlapping segments as a 2-D strided view --------------------
    shape   = (n_segments, line_spacing)
    strides = (signal.strides[0] * step, signal.strides[0])
    segments = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

    # Make a contiguous copy before any arithmetic
    segments = np.ascontiguousarray(segments)

    # --- Apply Hann window to every segment (time domain) --------------------
    window   = np.hanning(line_spacing)    # shape: (line_spacing,)
    windowed = segments * window           # broadcasts across all rows

    # --- FFT -----------------------------------------------------------------
    rfft_mag = np.abs(np.fft.rfft(windowed, axis=1))  # (n_segments, line_spacing//2 + 1)

    # --- One-sided scaling: double interior bins to fold in negative mirror --
    energy = rfft_mag.copy()
    if line_spacing % 2 == 0:       # even: DC and Nyquist are unique
        energy[:, 1:-1] *= 2
    else:                           # odd: only DC is unique
        energy[:, 1:]   *= 2

    # Drop the Nyquist bin — output width is always line_spacing // 2
    energy = energy[:, :n_freqs]

    # --- Normalisation denominators ------------------------------------------
    win_norm_pow = np.sum(window ** 2)       # for psd methods
    win_norm_amp = np.sqrt(win_norm_pow)     # for amplitude method

    # --- Method-specific scaling ---------------------------------------------
    if method == "amplitude":
        energy = np.sqrt(resolution) * energy / win_norm_amp

    elif method  == "psd":
        energy = resolution * (energy ** 2) / win_norm_pow

    elif method == "psd_n":
        energy = energy / np.sum(energy, axis=1, keepdims=True)

    elif method == "psd_lf":
        freqs = np.fft.rfftfreq(line_spacing, d=resolution)[:n_freqs]
        energy = energy[:, freqs < (1 / line_spacing)]

    elif method == "psd_df":
        freqs = np.fft.rfftfreq(line_spacing, d=resolution)
        df = 1.0 / (len(signal) * resolution)
        energy = energy * df / win_norm_pow

    elif method == "spectrum":
        energy = energy / len(frequencies) / win_norm_pow

    else:
        raise ValueError(f"Unknown method '{method}'.")

    frequencies = np.fft.rfftfreq(line_spacing, d=resolution)[:n_freqs]

    return energy, frequencies
