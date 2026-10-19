


import numpy as np
from skimage.transform import radon, iradon, rotate
import time
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
                        zoom_factor: float = 2.0) :
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
    do_offset: bool, optional
        Set to 'True' if data has near vertical or horizontal features
        to offset for better peak detection. Default is ``False``.

    Returns
    -------
    theta_info : dict
        Dictionary with keys:

        - ``'peaks'`` : np.ndarray of float — dominant angle(s) in degrees,
          sorted by descending prominence.
        - ``'widths_in_degrees'`` : np.ndarray of float — FWHM of each peak
          in degrees.
        - ``'profile_variance'`` : np.ndarray of float — variance profile
          over all tested angles (rolled so index 0 corresponds to 0°).

    angle_profile: dict
        - Python dict containing computed angles for faster debugging

    """


    # Remove excess whitespace around the tracks
    sub_image, bounding_box = _sub_image(binary_image)

    # Scale the sub-image to reduce computation time for the Radon transform
    input_image = zoom(sub_image, zoom=1/zoom_factor, order=0)

    temp_offset = 0
    # If theta has been computed in previous runs, we can skip the
    # Radon transform and use the precomputed values.
    if angle_info is not None:
        print(f"Using pre-computed sinogram...")
        sinogram = angle_info['sinogram']
        temp_offset = angle_info['temp_offset']
        theta_range = np.arange(temp_offset, temp_offset + 180)

    else:
        # Compute the Radon transform
        theta_range = list(np.arange(180)) # 180 angles uniformly spaced over [0°, 180°)
        print("Computing initial Radon transform...", end=" ")
        sinogram = radon(input_image, theta=theta_range, circle=circle)
        print("Done.")

    # Variance corresponds to the strength of the linear features at each angle
    profile_variance = np.var(sinogram, axis=0)
    log_profile_variance = np.log1p(profile_variance)

    # if data has near vertical or horizontal features when visually observed,
    # offset the profile variance to enhance peak detection at edges (0° and 180°)
    if do_offset and not angle_info:
        temp_offset = find_temp_offset(log_profile_variance)
        theta_range = np.arange(temp_offset, temp_offset + 180)
        print("Computing Radon transform with offset...", end=" ")
        sinogram = radon(input_image, theta=theta_range, circle=circle)
        print("Done.")
        profile_variance = np.var(sinogram, axis=0)
        log_profile_variance = np.log1p(profile_variance)


    # Plot the log profile variance for visual inspection
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

    # Apply peak detection to the log profile variance to find dominant angles
    print("Detecting peaks in the log profile variance...", end=" ")
    detected_angles_radon, properties = find_peaks(log_profile_variance,
                                                   distance=angular_tolerance,
                                                   prominence=0.01)
    print("Done.")


    # Sort the detected angles by their peak heights in descending order
    # to give priority to the most prominent angles
    sorted_indices = np.argsort(properties["prominences"])[::-1]

    # Keep only the top N angles based on the specified parameter
    if top_angles > 0 and len(sorted_indices) > top_angles:
        sorted_indices = sorted_indices[:top_angles]
        print(f"Keeping top {top_angles} detected angles.")

    detected_angles_radon = detected_angles_radon[sorted_indices]
    print(f"Detected peaks (xy) adjusted :{_radon_angle_to_xy(detected_angles_radon,
                                                              temp_offset)}")


    line_dict = {}
    for i, radon_angle in enumerate(detected_angles_radon):

        # Define the angular range for this family (sinogram column indices)
        radon_theta_min = max(0, radon_angle - angular_tolerance)
        radon_theta_max = min(sinogram.shape[1] - 1, radon_angle + angular_tolerance)

        # Extract sinogram data inside this angular window
        band_mask = np.zeros_like(sinogram, dtype=float)
        band_mask[:, radon_theta_min: radon_theta_max + 1] = sinogram[:, radon_theta_min: radon_theta_max + 1]

        # circle must match the forward radon call, otherwise iradon assumes a
        # different geometry and returns a min(H, W) square
        x_cols, y_rows = input_image.shape[1], input_image.shape[0]
        iradon_kwargs = {} if circle else {"output_size": max(y_rows, x_cols)}
        xy_angle = _radon_angle_to_xy(radon_angle, temp_offset)
        print(f"Reconstructing image from angle {xy_angle} degrees...", end=" ")
        start_time = time.perf_counter()
        new_radon_theta_range = theta_range[radon_theta_min: radon_theta_max + 1]
        sinogram_band = band_mask[:, radon_theta_min: radon_theta_max + 1]
        reconstructed = iradon(sinogram_band,
                               theta=new_radon_theta_range,
                               filter_name="ramp",
                               circle=circle, **iradon_kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Done.")

        isolated_lines = _align_reconstruction(reconstructed, input_image.shape, circle)

        # Normalize to [0, 1] for visualization
        isolated_lines = np.clip(isolated_lines, 0, None)
        if isolated_lines.max() > 0:
            isolated_lines /= isolated_lines.max()


        # Thresholded overlay onto original
        threshold = min_intensity * isolated_lines.max()
        line_mask = isolated_lines > threshold
        line_mask = zoom(line_mask, zoom=zoom_factor, order=0)
        line_mask = line_mask.astype(bool)

        # Scaling the binary image can introduce dimension misalignment,
        # so we need to crop or pad the line_mask to match the sub_image size
        # Add or remove padding to match the sub_image size
        if line_mask.shape[0] < sub_image.shape[0]:
            pad_height = sub_image.shape[0] - line_mask.shape[0]
            line_mask = np.pad(line_mask, ((0, pad_height), (0, 0)), mode='constant', constant_values=False)
        elif line_mask.shape[0] > sub_image.shape[0]:
            line_mask = line_mask[:sub_image.shape[0], :]
        if line_mask.shape[1] < sub_image.shape[1]:
            pad_width = sub_image.shape[1] - line_mask.shape[1]
            line_mask = np.pad(line_mask, ((0, 0), (0, pad_width)), mode='constant', constant_values=False)
        elif line_mask.shape[1] > sub_image.shape[1]:
            line_mask = line_mask[:, :sub_image.shape[1]]
        # Pad the line_mask to match the original image size
        padded_line_mask = np.zeros_like(binary_image, dtype=bool)
        min_row, max_row, min_col, max_col = bounding_box
        padded_line_mask[min_row:max_row + 1, min_col:max_col + 1] = line_mask

        # Convert the Radon angle to the corresponding image (x-y) angle
        xy_theta = _radon_angle_to_xy(radon_angle, temp_offset)
        line_dict[xy_theta] = padded_line_mask

    print("Extending lines to full image width...", end=" ")
    angle_extended = _extend_line_to_edges2(line_dict)
    print("Done.")
    # angle_extended = line_dict

    point_orientations = {}
    print("Labeling points...", end=" ")
    combined_angle_raster = np.full(binary_image.shape, np.nan, dtype=float)
    for angle in angle_extended.keys():
        angle_raster = angle_extended[angle]
        mask = angle_raster.astype(bool) & np.isnan(combined_angle_raster)
        combined_angle_raster[mask] = float(angle)

    # EDT measures distance to the nearest zero pixel, so pass the "has angle"
    # mask inverted: labelled pixels become 0, and return_indices gives the
    # (row, col) of the nearest one for every pixel.
    has_angle = ~np.isnan(combined_angle_raster)
    _, (ri, ci) = distance_transform_edt(~has_angle, return_indices=True)
    nearest_angle = combined_angle_raster[ri, ci]  # nearest labelled value everywhere
    thickened_binary = binary_dilation(binary_image, structure=np.ones((width_tolerance, width_tolerance)))
    point_orientations['combined'] = np.where(thickened_binary, nearest_angle, np.nan)
    print("Done.")

    for angle in angle_extended.keys():
        point_orientations[angle] = np.where(point_orientations['combined'] == float(angle),
                                             float(angle),
                                             np.nan)

    theta_info = {
        "theta": theta_range,
        "sinogram": sinogram,
        "temp_offset": temp_offset,
    }

    return point_orientations, theta_info


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
    """Crop the image to the bounding box of its foreground pixels."""
    rows, cols = np.where(image)
    if len(rows) == 0 or len(cols) == 0:
        return image  # No foreground pixels
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
    """Return the two ``(x, y)`` points where the Radon line ``(s, theta)``
    crosses the image border.

    skimage's projection offset is ``s = (x - cx)·cos(θ) − (y - cy)·sin(θ)``
    with ``θ`` the Radon angle in degrees and ``(cy, cx)`` from ``_radon_center``.
    Returns ``None`` if the line does not cross the image.
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

    peak_candidates : bool (r, theta) mask, e.g. local_max & band_mask & above-threshold
    sinogram        : the (r, theta) array the candidates came from
    r_gap           : candidates whose r differs by up to this many rows (in adjacent
                      theta columns) are treated as the same line
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