


import numpy as np
from skimage.transform import radon, iradon, rotate
from matplotlib import pyplot as plt
from scipy.ndimage import (maximum_filter, zoom, label,
                           binary_dilation, maximum_position,
                           sum as ndi_sum)
from scipy.signal import find_peaks

from .orientation_utils import find_temp_offset
from scipy.ndimage import distance_transform_edt
from skimage.draw import line as draw_line

def orientation_via_Radon(binary_image: np.ndarray,
                        angle_info: dict = None,
                        circle: bool = False,
                        angular_tolerance: int = 5,
                        width_tolerance: int = 5,
                        min_intensity: float = 0.05,
                        top_angles: int = 3,
                        do_offset: bool = False,
                        zoom_factor: float = 2.0,
                        show_plot: bool = True) :
    """Find the dominant orientation of linear features using the Radon transform.

    Computes the Radon sinogram over ``test_angles`` by computing for the
    angle whose projection column has the highest variance.
    High variance indicates that projections along that angle produce
    concentrated bright lines, suggesting presence of  linear features
    aligned with that orientation.

    Parameters
    ----------
    binary_image : np.ndarray
        Binary image containing linear features.
    angle_info : dict, optional
        ``theta_info`` returned by a previous call; reuses its sinogram and
        offset instead of recomputing the Radon transform.
    circle : bool, optional
        If ``True``, only pixels within the circle inscribed in the central
        square are projected — Set to ``False`` as we need to reconstruct the image
        using the original dimensions.
    angular_tolerance : int | None, optional
        Half-width in degrees of the suppression band around each angle in
        ``remove_angles``. Default is 5.
    width_tolerance : int | None, optional
        minimum distance in pixels between two lines of the same family
    min_intensity : float, optional
        Minimum intensity of a pixel to be considered part of a linear feature.
    top_angles : int, optional
        Number of dominant angles to return. If set to 0, all detected angles are returned.
    do_offset : bool, optional
        Set to 'True' if data has near vertical or horizontal features
        to offset for better peak detection. Default is ``False``.
    zoom_factor : float, optional
        Down-scaling factor applied before the Radon transform. Default is 2.0.
    show_plot : bool, optional
        Draw the log profile variance figure. Default is ``True``.

    Returns
    -------
    point_orientations : dict
        ``{angle_deg: float_raster}`` with the pixels of each detected angle
        family, plus ``'combined'`` holding all families. Background is NaN.
    theta_info : dict
        ``{'theta', 'sinogram', 'temp_offset'}``; pass it back as ``angle_info``
        to skip the Radon transform on later calls.
    """
    # Remove excess whitespace around the tracks, then scale down for the Radon transform
    sub_image, bounding_box = _sub_image(binary_image)
    input_image = zoom(sub_image, zoom=1/zoom_factor, order=0)

    sinogram, theta_range, temp_offset = _compute_sinogram(input_image, angle_info,
                                                           circle, do_offset)
    log_profile_variance = _log_profile_variance(sinogram)

    if show_plot:
        _plot_profile_variance(theta_range, log_profile_variance)

    detected_angles_radon = _detect_dominant_angles(log_profile_variance,
                                                    angular_tolerance, top_angles)
    print(f"Detected peaks (xy) adjusted :"
          f"{_radon_angle_to_xy(detected_angles_radon, temp_offset)}")

    line_dict = {}
    for radon_angle in detected_angles_radon:
        xy_angle = _radon_angle_to_xy(radon_angle, temp_offset)
        print(f"Reconstructing image from angle {xy_angle} degrees...", end=" ")
        line_mask = _reconstruct_family_mask(sinogram, theta_range, radon_angle,
                                             input_image.shape, circle,
                                             angular_tolerance, min_intensity,
                                             zoom_factor)
        print("Done.")
        line_mask = _fit_mask_to_shape(line_mask, sub_image.shape)
        line_dict[xy_angle] = _embed_in_full_image(line_mask, binary_image.shape,
                                                   bounding_box)

    print("Extending lines to full image width...", end=" ")
    angle_extended = _extend_line_to_edges2(line_dict)
    print("Done.")

    print("Labeling points...", end=" ")
    combined_angle_raster = _combine_angle_rasters(angle_extended, binary_image.shape)
    point_orientations = {
        'combined': _label_points(combined_angle_raster, binary_image, width_tolerance),
    }
    print("Done.")
    point_orientations.update(_split_by_angle(point_orientations['combined'],
                                              angle_extended.keys()))

    theta_info = {
        "theta": theta_range,
        "sinogram": sinogram,
        "temp_offset": temp_offset,
    }

    return point_orientations, theta_info


def _log_profile_variance(sinogram):
    """Log-variance of each sinogram column (one value per projection angle).

    Variance is the strength of the linear features at each angle.

    Parameters
    ----------
    sinogram : np.ndarray
        Radon sinogram of shape ``(n_projections, n_angles)``.

    Returns
    -------
    np.ndarray
        1-D array of length ``n_angles`` with ``log1p(var)`` of each column.
    """
    return np.log1p(np.var(sinogram, axis=0))


def _compute_sinogram(input_image, angle_info, circle, do_offset):
    """Compute the Radon sinogram of ``input_image``, or reuse a previous one.

    If ``angle_info`` (a previous ``theta_info``) is given its sinogram and
    offset are reused.  Otherwise the Radon transform is computed over
    [0°, 180°); with ``do_offset`` it is recomputed starting at the angle found
    by :func:`find_temp_offset`, so peaks near 0°/180° are not split by the edge.

    Parameters
    ----------
    input_image : np.ndarray
        Binary image passed to ``skimage.transform.radon``.
    angle_info : dict or None
        Previous ``theta_info`` with keys ``'sinogram'`` and ``'temp_offset'``.
        If ``None`` the sinogram is computed.
    circle : bool
        Forwarded to ``skimage.transform.radon``.
    do_offset : bool
        Recompute the sinogram starting at the widest-valley angle. Ignored
        when ``angle_info`` is given.

    Returns
    -------
    sinogram : np.ndarray
        Radon sinogram of shape ``(n_projections, 180)``.
    theta_range : list or np.ndarray
        Projection angles in degrees, one per sinogram column.
    temp_offset : int
        Angle (degrees) at which ``theta_range`` starts; 0 without an offset.
    """
    if angle_info is not None:
        print("Using pre-computed sinogram...")
        temp_offset = angle_info['temp_offset']
        return (angle_info['sinogram'], np.arange(temp_offset, temp_offset + 180),
                temp_offset)

    theta_range = list(np.arange(180))  # 180 angles uniformly spaced over [0°, 180°)
    print("Computing initial Radon transform...", end=" ")
    sinogram = radon(input_image, theta=theta_range, circle=circle)
    print("Done.")

    temp_offset = 0
    if do_offset:
        temp_offset = find_temp_offset(_log_profile_variance(sinogram))
        theta_range = np.arange(temp_offset, temp_offset + 180)
        print("Computing Radon transform with offset...", end=" ")
        sinogram = radon(input_image, theta=theta_range, circle=circle)
        print("Done.")
    return sinogram, theta_range, temp_offset


def _plot_profile_variance(theta_range, log_profile_variance):
    """Plot the log profile variance against the image (x-y) angle.

    Parameters
    ----------
    theta_range : array-like
        Projection angles in degrees, one per profile value; may start at an
        offset and is folded into [0, 180) before plotting.
    log_profile_variance : array-like
        Log profile variance, one value per entry of ``theta_range``.

    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 5))
    # theta_range may start at temp_offset (e.g. 60..239); fold it into [0, 180) and
    # re-sort so the x-axis always spans 0-180 (the profile is 180-periodic)
    theta_plot = _radon_angle_to_xy(np.asarray(theta_range) % 180)
    order = np.argsort(theta_plot)
    plt.plot(theta_plot[order], np.asarray(log_profile_variance)[order], color='blue', linewidth=2)
    plt.xlim(0, 180)
    plt.title("Log Profile Variance vs. Angle")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Log Profile Variance")
    plt.grid(axis='both', alpha=0.75)
    plt.axvline(x=0, color='red', linestyle='--', label='0°')
    plt.axvline(x=90, color='green', linestyle='--', label='90°')
    plt.axvline(x=180, color='orange', linestyle='--', label='180°')
    plt.legend()


def _detect_dominant_angles(log_profile_variance, angular_tolerance, top_angles):
    """Find the sinogram column indices of the profile peaks, most prominent first.

    ``top_angles <= 0`` keeps every detected peak.

    Parameters
    ----------
    log_profile_variance : np.ndarray
        1-D log profile variance, one value per sinogram column.
    angular_tolerance : int
        Minimum distance in columns between two detected peaks.
    top_angles : int
        Number of most prominent peaks to keep; ``<= 0`` keeps all.

    Returns
    -------
    np.ndarray of int
        Column indices of the kept peaks, sorted by descending prominence.
    """
    print("Detecting peaks in the log profile variance...", end=" ")
    peaks, properties = find_peaks(log_profile_variance,
                                   distance=angular_tolerance,
                                   prominence=0.01)
    print("Done.")

    sorted_indices = np.argsort(properties["prominences"])[::-1]
    if top_angles > 0 and len(sorted_indices) > top_angles:
        sorted_indices = sorted_indices[:top_angles]
        print(f"Keeping top {top_angles} detected angles.")
    return peaks[sorted_indices]


def _reconstruct_family_mask(sinogram, theta_range, radon_angle, image_shape, circle,
                             angular_tolerance, min_intensity, zoom_factor):
    """Reconstruct a boolean mask of the lines of one angle family.

    Back-projects only the sinogram columns within ``angular_tolerance`` of
    ``radon_angle``, normalises to [0, 1], thresholds at ``min_intensity`` and
    zooms back up.  ``image_shape`` is the shape that was fed to ``radon``.

    Parameters
    ----------
    sinogram : np.ndarray
        Radon sinogram of shape ``(n_projections, n_angles)``.
    theta_range : list or np.ndarray
        Projection angles in degrees, one per sinogram column.
    radon_angle : int
        Sinogram column index at the centre of the family.
    image_shape : tuple of int
        ``(rows, cols)`` of the image that was fed to ``radon``.
    circle : bool
        Must match the value used for the forward ``radon`` call.
    angular_tolerance : int
        Half-width in columns of the band around ``radon_angle``.
    min_intensity : float
        Threshold as a fraction of the maximum of the normalised reconstruction.
    zoom_factor : float
        Up-scaling factor applied to the mask before it is returned.

    Returns
    -------
    np.ndarray of bool
        Mask of ``image_shape`` scaled by ``zoom_factor``.
    """
    # Angular range for this family (sinogram column indices)
    theta_min = max(0, radon_angle - angular_tolerance)
    theta_max = min(sinogram.shape[1] - 1, radon_angle + angular_tolerance)
    sinogram_band = sinogram[:, theta_min: theta_max + 1].astype(float)

    # circle must match the forward radon call, otherwise iradon assumes a
    # different geometry and returns a min(H, W) square
    iradon_kwargs = {} if circle else {"output_size": max(image_shape)}
    reconstructed = iradon(sinogram_band,
                           theta=theta_range[theta_min: theta_max + 1],
                           filter_name="ramp",
                           circle=circle, **iradon_kwargs)

    isolated_lines = _align_reconstruction(reconstructed, image_shape, circle)
    isolated_lines = np.clip(isolated_lines, 0, None)
    if isolated_lines.max() > 0:
        isolated_lines /= isolated_lines.max()

    line_mask = isolated_lines > min_intensity * isolated_lines.max()
    return zoom(line_mask, zoom=zoom_factor, order=0).astype(bool)


def _fit_mask_to_shape(mask, shape):
    """Crop (bottom/right) or zero-pad (bottom/right) ``mask`` to ``shape``.

    Scaling the image down and up can be off by a pixel or two.

    Parameters
    ----------
    mask : np.ndarray of bool
        2-D mask to fit.
    shape : tuple of int
        Target ``(rows, cols)``.

    Returns
    -------
    np.ndarray of bool
        Mask of exactly ``shape``.
    """
    mask = mask[:shape[0], :shape[1]]
    pad_rows, pad_cols = shape[0] - mask.shape[0], shape[1] - mask.shape[1]
    if pad_rows or pad_cols:
        mask = np.pad(mask, ((0, pad_rows), (0, pad_cols)),
                      mode='constant', constant_values=False)
    return mask


def _embed_in_full_image(mask, full_shape, bounding_box):
    """Place a cropped ``mask`` back at ``bounding_box`` of an empty ``full_shape`` canvas.

    Parameters
    ----------
    mask : np.ndarray of bool
        Cropped mask, matching the bounding box size.
    full_shape : tuple of int
        ``(rows, cols)`` of the canvas.
    bounding_box : tuple of int
        ``(min_row, max_row, min_col, max_col)``, inclusive, as returned by
        :func:`_sub_image`.

    Returns
    -------
    np.ndarray of bool
        Canvas of ``full_shape`` with ``mask`` placed at ``bounding_box``.
    """
    min_row, max_row, min_col, max_col = bounding_box
    full = np.zeros(full_shape, dtype=bool)
    full[min_row:max_row + 1, min_col:max_col + 1] = mask
    return full


def _combine_angle_rasters(angle_masks, shape):
    """Combine per-angle masks into one float raster; the first family claims a pixel.

    Parameters
    ----------
    angle_masks : dict
        ``{angle_deg: bool_mask}``, all masks of ``shape``.
    shape : tuple of int
        ``(rows, cols)`` of the output raster.

    Returns
    -------
    np.ndarray of float
        Raster of ``shape`` holding each pixel's angle; ``NaN`` where no mask is set.
    """
    combined = np.full(shape, np.nan, dtype=float)
    for angle, angle_raster in angle_masks.items():
        mask = angle_raster.astype(bool) & np.isnan(combined)
        combined[mask] = float(angle)
    return combined


def _label_points(combined_angle_raster, binary_image, width_tolerance):
    """Give every (thickened) foreground pixel the angle of its nearest labelled pixel.

    Parameters
    ----------
    combined_angle_raster : np.ndarray of float
        Raster of angles, ``NaN`` where unlabelled.
    binary_image : np.ndarray of bool
        Foreground image to label.
    width_tolerance : int
        Side length in pixels of the square used to thicken ``binary_image``.

    Returns
    -------
    np.ndarray of float
        Raster of the shape of ``binary_image``; the nearest labelled angle on
        thickened foreground pixels, ``NaN`` elsewhere.
    """
    # EDT measures distance to the nearest zero pixel, so pass the "has angle"
    # mask inverted: labelled pixels become 0, and return_indices gives the
    # (row, col) of the nearest one for every pixel.
    has_angle = ~np.isnan(combined_angle_raster)
    _, (ri, ci) = distance_transform_edt(~has_angle, return_indices=True)
    nearest_angle = combined_angle_raster[ri, ci]  # nearest labelled value everywhere
    thickened_binary = binary_dilation(binary_image,
                                       structure=np.ones((width_tolerance, width_tolerance)))
    return np.where(thickened_binary, nearest_angle, np.nan)


def _split_by_angle(combined, angles):
    """Split a combined angle raster into one raster per angle.

    Parameters
    ----------
    combined : np.ndarray of float
        Raster of angles, ``NaN`` where unlabelled.
    angles : iterable
        Angles in degrees to extract.

    Returns
    -------
    dict
        ``{angle: raster}`` keeping only the pixels of ``combined`` labelled
        with that angle; all other pixels are ``NaN``.
    """
    return {angle: np.where(combined == float(angle), float(angle), np.nan)
            for angle in angles}

def _radon_angle_to_xy(radon_angle, temp_offset=0):
    """Convert a Radon angle to the corresponding image (x-y) angle.

    Parameters
    ----------
    radon_angle : float
        Angle in degrees from the Radon transform.
    temp_offset : float, optional
        Offset applied to the Radon angles during peak detection. Default is 0.

    Returns
    -------
    xy_angle : float
        Corresponding angle in degrees in the image coordinate system.
    """
    return (radon_angle - 90 + temp_offset) % 180

def _sub_image(image):
    """Crop the image to the bounding box of its foreground pixels.

    An image with no foreground is returned whole, with the box spanning all of it.

    Parameters
    ----------
    image : np.ndarray
        2-D image whose non-zero pixels are the foreground.

    Returns
    -------
    sub_image : np.ndarray
        ``image`` cropped to the bounding box.
    bounding_box : tuple of int
        ``(min_row, max_row, min_col, max_col)``, inclusive.
    """
    rows, cols = np.where(image)
    if len(rows) == 0 or len(cols) == 0:
        return image, (0, image.shape[0] - 1, 0, image.shape[1] - 1)
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    bounding_box = (min_row, max_row, min_col, max_col)
    sub_image = image[min_row:max_row + 1, min_col:max_col + 1]
    return sub_image, bounding_box

def _radon_center(shape, circle):
    """Pixel ``(row, col)`` of ``shape`` that skimage's radon/iradon treat as the rotation centre.

    With ``circle=False`` the image is padded symmetrically about ``shape // 2``.
    With ``circle=True`` it is first cropped to a centred square whose start
    offset is rounded up, which shifts the centre by one pixel on odd excess.

    Parameters
    ----------
    shape : tuple of int
        ``(rows, cols)`` of the image.
    circle : bool
        The ``circle`` argument passed to ``radon``/``iradon``.

    Returns
    -------
    tuple of int
        ``(row, col)`` of the rotation centre.
    """
    h, w = shape
    if not circle:
        return h // 2, w // 2
    m = min(h, w)
    return int(np.ceil((h - m) / 2)) + m // 2, int(np.ceil((w - m) / 2)) + m // 2


def _align_reconstruction(reconstructed, shape, circle):
    """Place an ``iradon`` output onto a canvas of ``shape`` so pixels overlay the original image.

    The reconstruction can be larger (``circle=False``) or smaller
    (``circle=True``) than the original; pixels with no counterpart are 0.

    Parameters
    ----------
    reconstructed : np.ndarray
        2-D output of ``iradon``.
    shape : tuple of int
        ``(rows, cols)`` of the original image.
    circle : bool
        The ``circle`` argument used for the forward and inverse transforms.

    Returns
    -------
    np.ndarray
        Array of ``shape`` with the reconstruction overlaid on the original pixels.
    """
    h, w = shape
    rh, rw = reconstructed.shape
    cy, cx = _radon_center(shape, circle)
    oy, ox = cy - rh // 2, cx - rw // 2  # reconstructed (r, c) -> image (r + oy, c + ox)
    aligned = np.zeros(shape, dtype=reconstructed.dtype)
    r0, c0 = max(0, oy), max(0, ox)
    r1, c1 = min(h, oy + rh), min(w, ox + rw)
    aligned[r0:r1, c0:c1] = reconstructed[r0 - oy:r1 - oy, c0 - ox:c1 - ox]
    return aligned


def _radon_line_endpoints(s_val, theta_deg, center, shape):
    """Return the two ``(x, y)`` points where the Radon line ``(s, theta)`` crosses the image border.

    skimage's projection offset is ``s = (x - cx)·cos(θ) − (y - cy)·sin(θ)``
    with ``θ`` the Radon angle in degrees and ``(cy, cx)`` from ``_radon_center``.

    Parameters
    ----------
    s_val : float
        Projection offset of the line.
    theta_deg : float
        Radon angle in degrees.
    center : tuple of int
        ``(cy, cx)`` rotation centre, from :func:`_radon_center`.
    shape : tuple of int
        ``(rows, cols)`` of the image.

    Returns
    -------
    list of tuple or None
        The two ``(x, y)`` border points, or ``None`` if the line does not
        cross the image.
    """
    cy, cx = center
    h, w = shape
    t = np.deg2rad(theta_deg)
    cos_t, sin_t = np.cos(t), np.sin(t)

    pts = []
    if abs(sin_t) > 1e-6:
        for x in (0, w - 1):
            y = cy + ((x - cx) * cos_t - s_val) / sin_t
            if 0 <= y <= h - 1:
                pts.append((x, y))
    if abs(cos_t) > 1e-6:
        for y in (0, h - 1):
            x = cx + (s_val + (y - cy) * sin_t) / cos_t
            if 0 <= x <= w - 1:
                pts.append((x, y))

    # Remove duplicate corner matches
    unique_pts = []
    for p in pts:
        if not any(np.allclose(p, u, atol=1.0) for u in unique_pts):
            unique_pts.append(p)
    return unique_pts[:2] if len(unique_pts) >= 2 else None


def _extend_lines_to_edges(angle_dict: dict,
                           min_line_factor: float = 0.1,
                           band_width: int = 40,
                           line_width: int = 3) -> dict:
    """Replace the segmented lines of each angle family with lines spanning the image.

    For every ``(angle, binary_image)`` pair:

    1. rotate the image by ``90 - angle`` so the lines become vertical,
    2. sum each column,
    3. find peaks in that sum (column positions of the lines),
    4. draw a full-height vertical line at each peak column,
    5. rotate back and crop to the original shape.

    Parameters
    ----------
    angle_dict : dict
        ``{angle_deg: binary_image}`` where each image holds lines at ``angle_deg``
        from the x-axis (image convention, as built in ``orientation_via_Radon``).
        All images are assumed to have the same shape.
    min_line_factor : float, optional
        Minimum peak height as a fraction of the tallest column sum.
        Default is ``0.1``.
    band_width : int, optional
        Minimum horizontal separation (pixels) between adjacent peaks; should
        approximate the expected inter-line spacing. Default is ``40``.
    line_width : int, optional
        Width in pixels of the drawn lines. Default is ``3``.

    Returns
    -------
    extended : dict
        ``{angle_deg: bool_image}`` with the same shape as the input images.
        Each image contains the extended lines of that family, running from edge
        to edge of the image.
    """
    extended = {}
    for angle, image in angle_dict.items():
        orig_shape = image.shape
        angle_to_rotate = 90.0 - angle   # project convention, see pipeline/tracklines.py

        # 1. Rotate so the lines are vertical (nearest neighbour keeps it binary)
        rotated = rotate(image.astype(bool), angle=angle_to_rotate, resize=True,
                         order=0, preserve_range=True)

        # 2. Column sum: each line contributes a tall bump at its x-position
        projection = rotated.sum(axis=0)
        if projection.max() == 0:
            extended[angle] = np.zeros(orig_shape, dtype=bool)
            continue

        # 3. Column positions of the lines
        min_line_length = int(projection.max() * min_line_factor)
        peaks, _ = find_peaks(projection,
                              height=min_line_length,
                              distance=band_width,
                              prominence=min_line_length)

        # 4. Full-height vertical line at each peak (top to bottom of the rotated canvas)
        lines = np.zeros_like(rotated, dtype=bool)
        half = line_width // 2
        for peak in peaks:
            lines[:, max(0, peak - half): peak - half + line_width] = True

        # 5. Rotate back and crop the padded canvas to the original shape
        back = rotate(lines.astype(bool), angle=-angle_to_rotate, resize=True,
                      order=0, preserve_range=True)
        row_off = (back.shape[0] - orig_shape[0]) // 2
        col_off = (back.shape[1] - orig_shape[1]) // 2
        extended[angle] = back[row_off: row_off + orig_shape[0],
                               col_off: col_off + orig_shape[1]].copy()
    return extended



def _extend_line_to_edges2(angle_dict: dict,
                           min_line_factor: float = 0.1,
                           band_width: int = 40,
                           line_width: int = 5) -> dict:
    """Rotation-free version of :func:`_extend_lines_to_edges` (same inputs and outputs).

    Rotating an image so the lines become vertical is only a change of
    coordinates: a pixel's column in the rotated image is its distance ``u``
    across the line family, ``u = x*sin(a) + y*cos(a)`` (``a`` = image angle,
    0° = horizontal, 90° = vertical). So instead of resampling the image:

    1. compute ``u`` for every foreground pixel,
    2. histogram ``u`` in 1-pixel bins (== column sums of the rotated image),
    3. find peaks in that histogram (one per line),
    4. draw each line as a chord across the canvas with
       :func:`_radon_line_endpoints` and ``skimage.draw.line``.

    Memory scales with the number of foreground pixels, not with the (up to
    ~4x larger) padded canvas that rotating and rotating back requires.

    Parameters
    ----------
    angle_dict : dict
        ``{angle_deg: binary_image}``, image-convention angles. All images are
        assumed to have the same shape.
    min_line_factor : float, optional
        Minimum peak height as a fraction of the tallest bin. Default ``0.1``.
    band_width : int, optional
        Minimum separation in pixels between adjacent lines. Default ``40``.
    line_width : int, optional
        Width in pixels of the drawn lines. Default ``5``.

    Returns
    -------
    extended : dict
        ``{angle_deg: bool_image}``, lines running from edge to edge.
    """
    extended = {}
    for angle, image in angle_dict.items():
        h, w = image.shape
        out = np.zeros((h, w), dtype=bool)

        rows, cols = np.nonzero(image)
        if rows.size == 0:
            extended[angle] = out
            continue

        a = np.deg2rad(angle)
        # 1-2. distance across the lines for every foreground pixel, binned per pixel
        u = cols * np.sin(a) + rows * np.cos(a)
        u0 = int(np.floor(u.min()))
        # zero margin: find_peaks cannot report a peak at the array boundary, and the
        # rotated canvas used to provide this padding implicitly
        projection = np.pad(np.bincount(np.round(u - u0).astype(int)), 1)
        u0 -= 1                                   # bin i  <->  u = u0 + i

        # 3. line positions
        min_line_length = int(projection.max() * min_line_factor)
        peaks, _ = find_peaks(projection,
                              height=min_line_length,
                              distance=band_width,
                              prominence=min_line_length)

        # 4. one chord per peak; u -> Radon offset s = u_centre - u, Radon angle = a + 90°
        cy, cx = h // 2, w // 2
        u_centre = cx * np.sin(a) + cy * np.cos(a)
        half = (line_width - 1) / 2
        for p in peaks:
            # half-pixel steps across the line: 1-px steps leave gaps at oblique angles
            for k in np.arange(-half, half + 0.01, 0.5):
                endpoints = _radon_line_endpoints(u_centre - (u0 + p + k), angle + 90.0,
                                                  (cy, cx), (h, w))
                if endpoints is None:
                    continue
                (x1, y1), (x2, y2) = endpoints
                rr, cc = draw_line(int(round(y1)), int(round(x1)),
                                   int(round(y2)), int(round(x2)))
                out[rr, cc] = True
        extended[angle] = out
    return extended


def _collapse_candidates(peak_candidates, sinogram, r_gap=2):
    """Reduce candidate cells to one (r_idx, theta_idx) per physical line.

    Parameters
    ----------
    peak_candidates : np.ndarray of bool
        ``(r, theta)`` mask, e.g. local_max & band_mask & above-threshold.
    sinogram : np.ndarray
        The ``(r, theta)`` array the candidates came from.
    r_gap : int, optional
        Candidates whose r differs by up to this many rows (in adjacent
        theta columns) are treated as the same line. Default is 2.

    Returns
    -------
    s_indices : np.ndarray of int
        Row (r) index of the strongest cell of each line.
    a_indices : np.ndarray of int
        Column (theta) index of the strongest cell of each line.
    strength : np.ndarray of float
        Total sinogram energy of each line.
    """
    # Grow candidates along r only, so crests that drift a few rows between
    # neighbouring columns still touch and end up in the same component.
    grown = binary_dilation(peak_candidates,
                            structure=np.ones((2 * r_gap + 1, 1), dtype=bool))

    # 8-connectivity links cells in adjacent theta columns (including diagonals)
    labels, n = label(grown, structure=np.ones((3, 3), dtype=bool))

    # Only the original candidates may win, not the cells added by the dilation
    labels = np.where(peak_candidates, labels, 0)

    ids = np.arange(1, n + 1)
    best = maximum_position(sinogram, labels=labels, index=ids)   # strongest cell per blob
    strength = ndi_sum(sinogram, labels=labels, index=ids)        # optional: total energy

    s_indices = np.array([p[0] for p in best])
    a_indices = np.array([p[1] for p in best])
    return s_indices, a_indices, strength







# Experimental Codes

    # setup 1D local maximum filter for each angle in the tolerance range
    neighborhood_size = (width_tolerance, 1)  # (s-radius, theta-radius)
    local_max = maximum_filter(sinogram, size=neighborhood_size) == sinogram
    min_intensity = min_intensity * sinogram.mean()

    num_projections = sinogram.shape[0]
    r_coords = np.arange(num_projections) - num_projections // 2
    radon_center = _radon_center(input_image.shape, circle)
    line_dict = {}

# # Plot the sinogram with the band mask highlighted
# plt.figure(figsize=(10, 5))
# plt.imshow(1 - band_mask, cmap='gray', aspect='auto', extent=[0, 180, num_projections, 0])
# plt.colorbar(label='Intensity')
# plt.title(f"Sinogram with Band Mask for Angle {radon_angle}° (±{angular_tolerance}°)")
# plt.axvline(x=radon_theta_min, color='red', linestyle='--', label='Band Min')
# plt.axvline(x=radon_theta_max, color='blue', linestyle='--', label='Band Max')
# plt.legend()
# plt.xlabel('Angle (degrees)')
# plt.ylabel('Projection Position (pixels)')
# plt.show()

        # Candidate peaks: must be local 1D maximum, above noise, and in angle band
        # min_intensity is a hyperparameter, needs to be tuned for each dataset
        # peak_candidates = local_max & band_mask & (sinogram >= min_intensity)
        # s_indices, angle_sub_indices = np.where(peak_candidates)
        # peak_candidates = local_max & band_mask & (sinogram >= min_intensity)
        # s_indices, angle_sub_indices, _ = _collapse_candidates(peak_candidates, sinogram,
        #                                                       r_gap=pixel_tolerance // 2)
        # first_line_of_family = True

    #     # Compute the angle in the original image coordinate system (x-y) for this family
    #     xy_angle = (theta[radon_angle] + 90) % 180
    #     peaks_xy.append(xy_angle)
    #     # print(f"Plotting angle family {xy_angle}...")
    #     # fig, ax = plt.subplots(figsize=(9, 9))
    #     # ax.imshow(1 - input_image, cmap="gray")
    #     # ax.set_title(f"Angle Family: {xy_angle}")
    #
    #     print(f"number of lines: {len(s_indices)} for angle family {xy_angle}° (±{angular_tolerance}°)")
    #     s_val = r_coords[s_indices]
    #     # Plot s_indices distribution for this angle family
    #     plt.figure(figsize=(10, 5))
    #     plt.hist(s_val, bins=sinogram.shape[0], color='blue', alpha=0.7)
    #     plt.title(f"Distribution of s_indices for angle family {xy_angle}° (±{angular_tolerance}°)")
    #     plt.xlabel("s_index")
    #     plt.ylabel("Frequency")
    #     plt.grid(axis='y', alpha=0.75)
    #     plt.show()
    #
    # return None, None


        # for s_idx, a_idx in zip(s_indices, angle_sub_indices):
        #     s_val = r_coords[s_idx]
        #     # Use the column where this line actually peaks, not the family centre
        #     # The angle is computed in the Radon space, so we need to convert it to the image space
        #     xy_theta = (radon_angle - 90 - temp_offset) % 180.0
        #     endpoints = _radon_line_endpoints(s_val, xy_theta, radon_center, binary_image.shape)
        #     line_mask = np.zeros_like(binary_image, dtype=bool)
        #
        #     # print(f"Drawing endpoints for line at angle {xy_theta}° with s={s_val}: {endpoints}")
        #     if endpoints is not None:
        #         (x1, y1), (x2, y2) = endpoints
        #         # Set points between the endpoints to True in the line_mask
        #         rr, cc = np.linspace(y1, y2, num=1000), np.linspace(x1, x2, num=1000)
        #         rr = np.clip(np.round(rr).astype(int), 0, binary_image.shape[0] - 1)
        #         cc = np.clip(np.round(cc).astype(int), 0, binary_image.shape[1] - 1)
        #         line_mask[rr, cc] = True
        #     angle_dict[xy_theta] = line_mask

                # label_text = (
                #     f"Angle Family {xy_angle}° (±{angular_tolerance}°)"
                #     if first_line_of_family
                #     else ""
                # )
                # ax.plot(
                #     [x1, x2],
                #     [y1, y2],
                #     color=color,
                #     linewidth=1.75,
                #     alpha=0.9,
                #     label=label_text,
                # )
                # first_line_of_family = False


    # for i, radon_angle in enumerate(detected_angles_radon):
    #     # Define the angular range for this family (sinogram column indices)
    #     radon_theta_min = max(0, radon_angle - angular_tolerance)
    #     radon_theta_max = min(sinogram.shape[1] - 1, radon_angle + angular_tolerance)

        # Create a band mask for this angle
        # band_mask = np.zeros_like(sinogram, dtype=float)
        # band_mask[:, radon_theta_min: radon_theta_max + 1] = sinogram[:, radon_theta_min: radon_theta_max + 1]

        # Candidate peaks: must be local 1D maximum, above noise, and in angle band
        # min_intensity is a hyperparameter, needs to be tuned for each dataset
        # peak_candidates = local_max & band_mask & (sinogram >= min_intensity)
        # s_indices, angle_sub_indices = np.where(peak_candidates)
        # peak_candidates = local_max & band_mask & (sinogram >= min_intensity)
        # s_indices, angle_sub_indices, _ = _collapse_candidates(peak_candidates, sinogram,
        #                                                       r_gap=width_tolerance // 2)

        # xy_angle = _radon_angle_to_xy(radon_angle, temp_offset)
        # print(f"number of lines: {len(s_indices)} for angle family {xy_angle}° (±{angular_tolerance}°)")
        # s_val = r_coords[s_indices]
        # # Plot s_indices distribution for this angle family
        # plt.figure(figsize=(10, 5))
        # plt.hist(s_val, bins=sinogram.shape[0], color='blue', alpha=0.7)
        # plt.title(f"Distribution of s_indices for angle family {xy_angle}° (±{angular_tolerance}°)")
        # plt.xlabel("s_index")
        # plt.ylabel("Frequency")
        # plt.grid(axis='y', alpha=0.75)
        # plt.show()



        # for s_idx, a_idx in zip(s_indices, angle_sub_indices):
        #     s_val = r_coords[s_idx]
        #     # Use the column where this line actually peaks, not the family centre
        #     # The angle is computed in the Radon space, so we need to convert it to the image space
        #     xy_theta = (radon_angle - 90 - temp_offset) % 180.0
        #     endpoints = _radon_line_endpoints(s_val, xy_theta, radon_center, binary_image.shape)
        #     line_mask = np.zeros_like(binary_image, dtype=bool)
        #
        #     # print(f"Drawing endpoints for line at angle {xy_theta}° with s={s_val}: {endpoints}")
        #     if endpoints is not None:
        #         (x1, y1), (x2, y2) = endpoints
        #         # Set points between the endpoints to True in the line_mask
        #         rr, cc = np.linspace(y1, y2, num=1000), np.linspace(x1, x2, num=1000)
        #         rr = np.clip(np.round(rr).astype(int), 0, binary_image.shape[0] - 1)
        #         cc = np.clip(np.round(cc).astype(int), 0, binary_image.shape[1] - 1)
        #         line_mask[rr, cc] = True
        #     line_dict[xy_theta] = line_mask