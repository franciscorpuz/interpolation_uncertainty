import numpy as np
from collections import deque
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import maximum_filter, zoom, distance_transform_edt
from skimage.transform import radon, iradon
import time


def compute_local_orientation(binary_image,
                              method:str ='pca',
                              plot_orientation:bool = False,
                              cmap='terrain',
                              figsize=(16, 8),
                              **kwargs):
    """Compute local orientation via PCA for each point in a binary image.

    Uses a chaining method to build a local neighbourhood for each point:
    starting from each point, greedily extends a chain of nearby unvisited
    neighbours and fits PCA to determine the dominant local direction.

    Parameters
    ----------
    binary_image : np.ndarray
        Binary image where ``True`` values represent points of interest.
    method : str, optional
        Method to use for computing local orientation. Can be either 'pca' or 'radon'.
    plot_orientation : bool, optional
        If ``True``, scatter-plots each point coloured by its local
        orientation angle. Default is ``False``.
    cmap : str, optional
        Matplotlib colourmap used when ``plot_orientation=True``.
        Default is ``'terrain'``.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches when plotting.
        Default is ``(16, 8)``.


    Returns
    -------
    point_orientations : np.ndarray
        2-D float array of the same shape as ``binary_image``.  Each
        foreground pixel carries its local orientation in degrees (wrapped
        to ``[0, 180)``); background pixels are ``NaN``.
    """

    if method == 'pca':
        point_orientations, angle_profile = orientation_via_PCA(binary_image,
                                            kwargs.get('line_length', 20),
                                            kwargs.get('num_neighbors', 50))
    elif method == 'radon':
        point_orientations, angle_info = orientation_via_radon(binary_image,
                                                                     kwargs.get('test_angles', None),
                                                                     kwargs.get('circle', False),
                                                                     kwargs.get('angular_tolerance', 5),
                                                                     kwargs.get('pixel_tolerance', 5),
                                                                     kwargs.get('min_intensity', 0.05),
                                                                     kwargs.get('do_offset', False),
                                                                     kwargs.get('precomputed_theta', None),
                                                                     kwargs.get('zoom_factor', 2.0))
    else:
        raise ValueError(f"Invalid method: {method}. Must be one of the following: 'pca', 'radon'")

    if plot_orientation:
        plt.figure(figsize=figsize)
        plt.scatter(point_orientations[:,0], point_orientations[:,1], s=10, cmap='viridis', vmin=0, vmax=180,
                    label='Local Orientation (degrees)')
        plt.colorbar(label='Orientation (degrees)', shrink=0.5)
        plt.title(f"Local Orientations via {method.upper()}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().invert_yaxis()
        plt.show()

    return point_orientations, angle_info


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
    """Return the two ``(x, y)`` points where the Radon line ``(s, theta)`` crosses the image border.

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


def orientation_via_radon(binary_image: np.ndarray,
                                        test_angles: np.ndarray = None,
                                        circle: bool = False,
                                        angular_tolerance: int = 5,
                                        pixel_tolerance: int = 5,
                                        min_intensity: float = 0.05,
                                        do_offset: bool = False,
                                        precomputed_theta: dict = None,
                                        zoom_factor: float = 2.0) -> dict:
    # tuple[float, np.ndarray]
    """Find the dominant orientation of linear features using the Radon transform.

    Computes the Radon sinogram over ``test_angles`` and returns the angle whose
    projection column has the highest variance. High variance indicates that
    projections along that angle produce concentrated bright lines, which is
    characteristic of linear features aligned with that orientation.

    Parameters
    ----------
    image : np.ndarray
        Binary image containing linear features.
    test_angles : np.ndarray, optional
        Angles in degrees at which to compute the Radon transform. If ``None``,
        180 angles uniformly spaced over ``[0°, 180°)`` are used.
    circle : bool, optional
        If ``True``, only pixels within the circle inscribed in the central
        square are projected — on a rectangular raster everything outside that
        square is discarded. Default is ``False`` (whole image is used).
    angular_tolerance : int | None, optional
        Half-width in degrees of the suppression band around each angle in
        ``remove_angles``. Default is 5.
    pixel_tolerance : int | None, optional
        minimum distance in pixels between two lines of the same family
    min_intensity : float, optional
        Minimum intensity of a pixel to be considered part of a linear feature.
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

    angle_profile: np.ndarray
        histogram of the angle scores detected by 'radon'
    """

    min_row = np.argwhere(binary_image)[:, 0].min()
    max_row = np.argwhere(binary_image)[:, 0].max()
    min_col = np.argwhere(binary_image)[:, 1].min()
    max_col = np.argwhere(binary_image)[:, 1].max()
    sub_image = binary_image[min_row:max_row + 1, min_col:max_col + 1]
    # # EDT measures the distance from each nonzero pixel to the nearest zero pixel,
    # # so invert the mask: background becomes nonzero, foreground (lines) becomes 0.
    # # Result: each background pixel -> distance (pixels) to the nearest foreground pixel.
    # sub_image_edt = distance_transform_edt(~sub_image.astype(bool))
    #
    # #Plot the distance transform
    # plt.figure(figsize=(10, 8))
    # plt.imshow(sub_image_edt, cmap='viridis')
    # plt.colorbar(label='Distance to nearest foreground pixel (pixels)')
    # plt.title('Distance Transform')
    # plt.show()
    #
    # # Plot the histogram of the distance transform
    # plt.figure(figsize=(10, 8))
    # log_distance_edt = np.log1p(sub_image_edt.ravel())
    # plt.hist(log_distance_edt, bins=50, color='blue', alpha=0.7)
    # plt.title('Histogram of Distance Transform Values')
    # plt.xlabel('Log Distance to nearest background pixel')
    # plt.ylabel('Frequency')
    # plt.show()



    zoom_factor = zoom_factor
    sub_rows, sub_cols = sub_image.shape
    input_image = zoom(sub_image, zoom=1/zoom_factor, order=0)
    theta = []
    if precomputed_theta is None:
        if test_angles is None:
            theta = np.linspace(0., 180., 180, endpoint=False)
        else:
            print(f"Using pre-computed angles: {precomputed_theta}")
            theta = test_angles

        # Compute the Radon transform (sinogram) and the variance of each projection column
        print("Computing initial Radon transform...", end=" ")
        start_time = time.perf_counter()
        sinogram = radon(input_image, theta=theta, circle=circle)
        print("Done.")
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Radon transform execution time: {execution_time:.4f} seconds")


        profile_variance = np.var(sinogram, axis=0)
        log_profile_variance = np.log1p(profile_variance)

        temp_offset = 0
        if do_offset:
            # data has near vertical or horizontal features,
            # offset the profile variance to avoid peak detection issues

            start_time = time.perf_counter()
            temp_offset = find_temp_offset(log_profile_variance)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"Offset computation execution time: {execution_time:.4f} seconds")

            print(f"Offsetting profile variance by {temp_offset} degrees for better peak detection.")
            optimal_thetas = np.arange(temp_offset, temp_offset + 180)
            print("Computing Radon transform with offset...", end=" ")
            start_time = time.perf_counter()
            sinogram = radon(input_image, theta=optimal_thetas, circle=circle)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"Second Radon computation execution time: {execution_time:.4f} seconds")
            print("Done.")


            centered_profile_variance = np.var(sinogram, axis=0)

            # # Plot before and after offsetting
            # plt.figure(figsize=(10, 8))
            # plt.plot(log_profile_variance, label="Original Log Profile Variance")
            # plt.plot(np.roll(np.log1p(centered_profile_variance), -temp_offset), label="Offset Log Profile Variance")
            # plt.xlabel("Angle (degrees)")
            # plt.ylabel("Log Profile Variance")
            # plt.title("Offsetting Profile Variance")
            # plt.legend()
            # plt.show()
            #
            # plt.figure(figsize=(10, 8))
            # plt.plot(profile_variance, label="Original Profile Variance")
            # plt.plot(np.roll(centered_profile_variance, -temp_offset), label="Offset Profile Variance")
            # plt.xlabel("Angle (degrees)")
            # plt.ylabel("Profile Variance")
            # plt.title("Offsetting Profile Variance")
            # plt.legend()
            # plt.show()


            log_profile_variance = np.log1p(centered_profile_variance)
            theta = optimal_thetas
        else:
            plt.figure(figsize=(10, 8))
            plt.plot(log_profile_variance, label="Original Log Profile Variance")
            plt.xlabel("Angle (degrees)")
            plt.ylabel("Log Profile Variance")
            plt.title("Offsetting Profile Variance")
            plt.legend()
            plt.show()
            # # Plot the offset profile variance
            # plt.figure(figsize=(10, 8))
            # plt.plot(log_profile_variance, label="Offset Log Profile Variance")
            # plt.xlabel("Angle (degrees)")
            # plt.ylabel("Log Profile Variance")
            # plt.title("Offsetting Profile Variance")
            # plt.legend()
            # plt.show()

        print("Detecting peaks in the log profile variance...", end=" ")
        detected_angles_radon, properties = find_peaks(log_profile_variance,
                                                       distance=angular_tolerance*2,
                                                       prominence=None)
        print("Done.")


        # detected_angles_xy = (detected_angles_radon + 90) % 180
        print(f"Detected peaks (radon):{detected_angles_radon}")
        # print(f"Detected peaks (xy):{detected_angles_xy}")


    else:
        detected_angles_radon = precomputed_theta['peaks_radon']
        detected_angles_xy = (detected_angles_radon + 90) % 180
        print(f"Using pre-computed angles: {detected_angles_xy}")
        profile_variance = precomputed_theta['profile_variance']
        log_profile_variance = np.log1p(profile_variance)
        sinogram = precomputed_theta['sinogram']
        # Radon angle (degrees) of each sinogram column
        theta = precomputed_theta.get('theta', np.arange(sinogram.shape[1]))

    theta = np.asarray(theta, dtype=float)

    # # Plot profile variance
    # plt.figure(figsize=(18, 8))
    # plt.imshow(1-sinogram, cmap="gray", aspect="auto")
    # plt.xlabel("Angle (degrees)")
    # plt.ylabel("r-distance (pixels)")
    # plt.title("Sinogram")


    num_projections = sinogram.shape[0]
    r_coords = np.arange(num_projections) - num_projections // 2
    x_cols, y_rows = input_image.shape[1], input_image.shape[0]
    radon_center = _radon_center(input_image.shape, circle)
    peaks_xy = []

    # 2D local maximum mask to prevent double-counting adjacent angles/pixels
    neighborhood_size = (pixel_tolerance, 1)  # (s-radius, theta-radius)
    local_max = maximum_filter(sinogram, size=neighborhood_size) == sinogram
    # local_min = minimum_filter(sinogram, size=neighborhood_size) == sinogram
    min_intensity = min_intensity * sinogram.mean()


    colors = ["cyan", "magenta", "yellow", "lime", "orange", "red", "brown"]

    print("Plotting peaks...")
    angle_dict = {}
    for i, radon_angle in enumerate(detected_angles_radon):
        # angle = (angle - 90) % 180
        color = colors[i % len(colors)]

        # if do_offset:
            # If we offset the profile variance, we need to roll back the detected angles
            # radon_angle = (radon_angle + temp_offset) % 180

        # Define the angular range for this family (sinogram column indices)
        radon_theta_min = max(0, radon_angle - angular_tolerance)
        radon_theta_max = min(sinogram.shape[1] - 1, radon_angle + angular_tolerance)

        # Find 2D peak points inside this angular window
        band_mask = np.zeros_like(sinogram, dtype=bool)
        band_mask[:, radon_theta_min: radon_theta_max + 1] = True

        # visualize lines under this band_mask
        masked_sinogram = np.zeros_like(sinogram)
        masked_sinogram[band_mask] = sinogram[band_mask]

        # circle must match the forward radon call, otherwise iradon assumes a
        # different geometry and returns a min(H, W) square
        iradon_kwargs = {} if circle else {"output_size": max(y_rows, x_cols)}
        print("Reconstructing image from masked sinogram...", end=" ")
        start_time = time.perf_counter()
        reconstructed = iradon(masked_sinogram, theta=theta, filter_name="ramp",
                               circle=circle, **iradon_kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Done. Reconstruction execution time: {execution_time:.4f} seconds")

        isolated_lines = _align_reconstruction(reconstructed, input_image.shape, circle)

        # Normalize to [0, 1] for visualization
        isolated_lines = np.clip(isolated_lines, 0, None)
        if isolated_lines.max() > 0:
            isolated_lines /= isolated_lines.max()

        # fig, axes = plt.subplots(2, 1, figsize=(15, 15))
        # # Original image
        # axes[0].imshow(1 - binary_image, cmap="gray")
        # axes[0].set_title("Original Image")

        # Thresholded overlay onto original
        threshold = 0.3 * isolated_lines.max()
        line_mask = isolated_lines > threshold
        line_mask = zoom(line_mask, zoom=zoom_factor, order=0)
        line_mask = line_mask.astype(bool)
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
        padded_line_mask[min_row:max_row + 1, min_col:max_col + 1] = line_mask

        # Isolated lines from inverse Radon
        plt.figure(figsize=(10, 8))
        plt.imshow(1 - padded_line_mask, cmap="gray")

        # xy_theta_min = (theta[radon_theta_min] + 90) % 180
        # xy_theta_max = (theta[radon_theta_max] + 90) % 180
        xy_theta = (radon_angle - 90 + temp_offset) % 180
        plt.title(f"Isolated Band ({xy_theta})°)")

        angle_dict[xy_theta] = padded_line_mask



        # # Detected lines
        # axes[2].imshow(1 - binary_image, cmap="gray")
        # # Plot detected lines in cyan with transparency
        # axes[2].imshow(
        #     1 - line_mask,
        #     cmap="cool",
        #     alpha=0.3,
        #     interpolation="none",
        # )
        # axes[2].set_title("Detected Lines Highlighted")

        # for ax in axes:
        #     ax.axis("off")


        # # Candidate peaks: must be local 2D maximum, above noise, and in angle band
        # peak_candidates = local_max & band_mask & (sinogram >= min_intensity)
        # s_indices, angle_sub_indices = np.where(peak_candidates)
        # first_line_of_family = True
        #
        # xy_angle = (theta[radon_angle] + 90) % 180
        # peaks_xy.append(xy_angle)
        # print(f"Plotting angle family {xy_angle}...")
        # fig, ax = plt.subplots(figsize=(9, 9))
        # ax.imshow(1 - input_image, cmap="gray")
        # ax.set_title(f"Angle Family: {xy_angle}")

        # print(f"number of lines: {len(s_indices)} ")
        #
        # for s_idx, a_idx in zip(s_indices, angle_sub_indices):
        #     s_val = r_coords[s_idx]
        #     # Use the column where this line actually peaks, not the family centre
        #     xy_theta = (theta[a_idx] - 90 - temp_offset) % 180.0
        #     endpoints = _radon_line_endpoints(s_val, xy_theta, radon_center, binary_image.shape)
        #     # endpoints = _radon_line_endpoints(s_val, xy_angle, radon_center, binary_image.shape)
        #
        #     if endpoints is not None:
        #         (x1, y1), (x2, y2) = endpoints
        #         label_text = (
        #             f"Angle Family {xy_angle}° (±{angular_tolerance}°)"
        #             if first_line_of_family
        #             else ""
        #         )
        #         ax.plot(
        #             [x1, x2],
        #             [y1, y2],
        #             color=color,
        #             linewidth=1.75,
        #             alpha=0.9,
        #             label=label_text,
        #         )
        #         first_line_of_family = False
    # print("Labeling points...")
    # labels, peak_centers_per_angle = label_points_by_probabilistic_line(x_coords,
    #                                                        y_coords,
    #                                                        peaks,
    #                                                        bin_width=5.0,
    #                                                        tol=tol,
    #                                                        chain_length=10)
    # print("Labeling points... Done.")


    # point_orientations = np.full_like(binary_image, fill_value=np.nan, dtype=np.float64)
    # for i in range(len(labels)):
    #     if labels[i] == -1:
    #         continue
    #     point_orientations[X_linear[i, 0], X_linear[i, 1]] = peaks[labels[i]]

    # theta_info = {
    #     "peaks_xy": np.array(peaks_xy),
    #     "peaks_radon": np.array(detected_angles_radon),
    #     "theta": theta,
    #     "sinogram": sinogram,
    #     "profile_variance": np.array(profile_variance),
    # }

    # theta_info = profile_variance
    theta_info = []
    point_orientations = angle_dict

    return point_orientations, theta_info


def find_temp_offset(heights: list[float] | np.ndarray, rel_height: float = 1.0):
    """
    If data has near vertical or horizontal features, this function returns an offset computed from
    the widest valley using inverted peak detection.

    Parameters:
    - heights: 1D array of histogram bin heights.
    - rel_height: Relative height at which to evaluate width.
                  1.0 measures at the full prominence base (the valley's natural rim).
                  0.5 measures full-width at half-maximum (FWHM).

    Returns:
    - best_valley: Dict containing index of the trough, width, and bounding edges.
    """
    from scipy.signal import peak_widths

    y = np.asarray(heights, dtype=float)
    inv_y = -y  # Invert so troughs become peaks

    # 1. Identify inverted peaks and compute their topological prominence
    troughs, properties = find_peaks(inv_y, prominence=0)


    if len(troughs) == 0:
        return None

    # 2. Compute widths at the chosen prominence level
    # rel_height=1.0 measures width at the lowest contour line where the valley merges
    widths, width_heights, left_ips, right_ips = peak_widths(
        inv_y,
        troughs
        # rel_height=0
    )


    # 3. Find the index corresponding to the widest valley
    widest_idx = np.argmax(widths)
    angle_of_valley = int((left_ips[widest_idx] + right_ips[widest_idx]) / 2)

    return angle_of_valley











def orientation_via_probabilistic_line(binary_image: np.ndarray,
                                        test_angles: np.ndarray = None,
                                        circle: bool = True,
                                        tol: int | None = 5,
                                        precomputed_theta: dict = None) -> dict:
    # tuple[float, np.ndarray]
    """Find the dominant orientation of linear features using the Radon transform.

    Computes the Radon sinogram over ``test_angles`` and returns the angle whose
    projection column has the highest variance. High variance indicates that
    projections along that angle produce concentrated bright lines, which is
    characteristic of linear features aligned with that orientation.

    Parameters
    ----------
    image : np.ndarray
        Binary image containing linear features.
    test_angles : np.ndarray, optional
        Angles in degrees at which to compute the Radon transform. If ``None``,
        180 angles uniformly spaced over ``[0°, 180°)`` are used.
    circle : bool, optional
        If ``True``, only pixels within the inscribed circle are projected
        (matches the skimage default). Default is ``True``.
    tol : int | None, optional
        Half-width in degrees of the suppression band around each angle in
        ``remove_angles``. Default is 5.

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

    angle_profile: np.ndarray
        histogram of the angle scores detected by 'radon'
    """

    X_linear = np.argwhere(binary_image)
    x_coords = X_linear[:, 1]  # columns
    y_coords = X_linear[:, 0]  # rows
    if precomputed_theta is None:
        if test_angles is None:
            theta = np.linspace(0., 180., 180, endpoint=False)
        else:
            print(f"Using pre-computed angles: {precomputed_theta}")
            theta = test_angles

        # Compute the Radon transform (sinogram) and the variance of each projection column
        print("Computing Radon transform...")
        sinogram = radon(binary_image, theta=theta, circle=circle)
        profile_variance = np.var(sinogram, axis=0)
        profile_variance_roll = np.roll(profile_variance, shift=90) % 180
        log_profile_variance_roll = np.log1p(profile_variance_roll)
        print("Radon Computation Done.")

        # Plot profile variance
        plt.figure(figsize=(10, 8))
        plt.plot(profile_variance_roll, label="Profile Variance")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Variance")
        plt.title("Variance Profile")
        plt.figure(figsize=(10, 8))
        plt.plot(log_profile_variance_roll, label="Log Profile Variance")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Log Variance")
        plt.title("Variance Profile")

        peaks, properties = find_peaks(log_profile_variance_roll, distance=tol*2, prominence=None)

        print(f"Detected peaks:{peaks}")

    else:
        peaks = precomputed_theta['peaks']
        profile_variance = precomputed_theta['profile_variance']

    print("Labeling points...")
    labels, peak_centers_per_angle = label_points_by_probabilistic_line(x_coords,
                                                           y_coords,
                                                           peaks,
                                                           bin_width=5.0,
                                                           tol=tol,
                                                           chain_length=10)
    print("Labeling points... Done.")


    point_orientations = np.full_like(binary_image, fill_value=np.nan, dtype=np.float64)
    for i in range(len(labels)):
        if labels[i] == -1:
            continue
        point_orientations[X_linear[i, 0], X_linear[i, 1]] = peaks[labels[i]]

    # # Deduct 90 degrees to remove effect of radon transform's 90-degree shift between image and sinogram angles
    # peaks = (peaks - 90) % 180
    #
    # print(f"Detected peaks:{peaks}")
    #
    # # Adjust the profile variance (x-axis) to correspond with the correction above
    # profile_variance = np.roll(profile_variance, len(theta) // 2)

    theta_info = {
        "peaks": np.array(peaks),
        # "widths_in_degrees": np.array(widths_in_degrees),
        "profile_variance": np.array(profile_variance),
    }


    return point_orientations, theta_info



def orientation_via_PCA(binary_image,
                              line_length=20,
                              num_neighbors=50):
    """Compute local orientation via PCA for each point in a binary image.

    Uses a chaining method to build a local neighbourhood for each point:
    starting from each point, greedily extends a chain of nearby unvisited
    neighbours and fits PCA to determine the dominant local direction.

    Parameters
    ----------
    binary_image : np.ndarray
        Binary image where ``True`` values represent points of interest.
    line_length : int, optional
        Number of points to include in each local chain/subgroup. Default is 20.
        More number leads to more accurate results, but also increases the
        processing time.
        Less number decreases processing time but makes the estimation vulnerable
        to errors cause by non-linearly oriented features.
    num_neighbors : int, optional
        Number of nearest neighbours to query when extending/creating the local chain.
        Higher values reduce the chance of getting stuck. Default is 50..

    Returns
    -------
    point_orientations : np.ndarray
        2-D float array of the same shape as ``binary_image``.  Each
        foreground pixel carries its local orientation in degrees (wrapped
        to ``[0, 180)``); background pixels are ``NaN``.

    angle_profile: np.ndarray
        histogram of the angle scores detected by 'pca'
    """

    X_linear = np.argwhere(binary_image)  # Extract coordinates of True values

    # 1. Create a field of points
    tree = KDTree(X_linear)
    angle_scores = []
    pca = PCA(n_components=1)

    for i in tqdm(range(len(X_linear))):
        # 2. Parameters for the chain
        start_idx = i
        chain_length = line_length
        chain = [start_idx]
        visited = {start_idx}

        # 3. The Chaining Logic
        for _ in range(chain_length - 1):
            current_pt = X_linear[chain[-1]].reshape(1, -1)

            # We query more than 1 in case the absolute nearest is already in our chain
            _, indices = tree.query(current_pt, k=num_neighbors)

            found_next = False
            for neighbor_idx in indices[0]:
                if neighbor_idx not in visited:
                    chain.append(neighbor_idx)
                    visited.add(neighbor_idx)
                    found_next = True
                    break

            if not found_next:
                break


        # Compute Local Orientation via PCA
        pca.fit(X_linear[chain])
        direction = pca.components_[0]
        angle_radians = np.arctan2(direction[1], direction[0])  # atan2(y, x)
        angle_scores.append(np.degrees(angle_radians))

    angle_scores = np.array(angle_scores) - 90  # Shift angles by 90 degrees
    angle_scores = angle_scores % 180  # Wrap angles to [0, 180)
    point_orientations = np.full_like(binary_image, fill_value=np.nan, dtype=np.float64)
    point_orientations[X_linear[:, 0].astype(int),
                       X_linear[:, 1].astype(int)] = angle_scores
    angle_profile= np.histogram(point_orientations[~np.isnan(point_orientations)].flatten(),
                                    bins=180)

    return point_orientations, angle_profile


def label_points_by_angle(x,
                          y,
                          theta_degs,
                          bin_width= 1.0,
                          tol=1.5,
                          min_peak_prominence= 5):
    """
    Assign each point to its closest trackline family across all dominant angles.

    For each image-convention angle φ in ``theta_degs`` the function converts
    back to the sinogram projection angle (φ + 90°) so that points on the same
    trackline produce the same ρ value, then finds trackline centers in ρ-space
    via a histogram peak search.  Each point is assigned to the angle family
    whose nearest trackline center is closest; points whose best distance
    exceeds ``tol`` are labelled −1.

    Parameters
    ----------
    x, y : array-like
        Column and row coordinates of foreground pixels.
    theta_degs : array-like
        Image-convention dominant angles in degrees (as returned by
        ``compute_local_orientation_via_radon`` after the −90° correction).
    bin_width : float
        Histogram bin width in pixels for the ρ projection.
    tol : float
        Maximum ρ-distance from a trackline center to count as membership.
    min_peak_prominence : float
        Passed to ``scipy.signal.find_peaks``; tune to point density.

    Returns
    -------
    labels : np.ndarray of int
        Per-point index into ``theta_degs`` (i.e. the assigned angle family),
        or −1 if the point is unassigned.
    all_peak_centers : list of np.ndarray
        Refined ρ trackline centers for each angle in ``theta_degs``.
    """
    theta_degs = np.asarray(theta_degs)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n_points = len(x)
    all_rho = []
    all_peak_centers = []

    for phi in theta_degs:
        # Convert image angle back to sinogram projection angle so that
        # ρ = x·cos(θ_s) + y·sin(θ_s) clusters points on the same trackline.
        # Example: φ=90° (vertical lines) → θ_s=0° → ρ=x (column index) ✓
        # theta_s = np.deg2rad((phi + 90) % 180)


        theta_s = np.deg2rad((phi + 90) % 180)
        rho = x * np.cos(theta_s) + y * np.sin(theta_s)
        print(f"phi: {phi}")
        print(f"theta_s: {theta_s}")
        print(f"rho: {rho}")
        all_rho.append(rho)

        bins = np.arange(rho.min(), rho.max() + bin_width, bin_width)
        counts, edges = np.histogram(rho, bins=bins)
        bin_centers = (edges[:-1] + edges[1:]) / 2

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.plot(edges[:-1], counts)
        # plt.hist(counts, bins=bins)
        plt.ylabel("Count")
        plt.xlabel("Rho (pixels)")
        plt.grid(True)
        plt.title(f"Rho histogram, theta:{phi} ")

        peak_idx, _ = find_peaks(counts, prominence=min_peak_prominence)
        if len(peak_idx) == 0:
            all_peak_centers.append(np.array([]))
            continue

        raw_centers = bin_centers[peak_idx]
        # Refine each center as a weighted mean of nearby ρ values
        peak_centers = np.array([
            rho[np.abs(rho - c) < bin_width * 2].mean() for c in raw_centers
        ])
        all_peak_centers.append(peak_centers)

    # For each point, find the minimum distance to a trackline center per angle
    min_dists = np.full((n_points, len(theta_degs)), np.inf)
    for k, (rho, centers) in enumerate(zip(all_rho, all_peak_centers)):
        if len(centers) == 0:
            continue
        min_dists[:, k] = np.abs(rho[:, None] - centers[None, :]).min(axis=1)

    best_k = np.argmin(min_dists, axis=1)
    best_dist = min_dists[np.arange(n_points), best_k]
    labels = np.where(best_dist <= tol, best_k, -1)

    return labels, all_peak_centers


def label_points_by_probabilistic_line(x, y,
                                      theta_degs,
                                      bin_width=1.0,
                                      tol=5,
                                      num_neighbors=5,
                                      chain_length=100,
                                      n_times=3):
    """
    Assign each point a local angle orientation using probabilistic line fitting.

    Runs a multi-vote ensemble: the core algorithm is repeated until every
    point has been visited at least ``n_times``, and the final label is the
    angle family that received the most votes (mode).  Points that are always
    near a trackline boundary get their correct family reinforced across
    independent random probes, suppressing one-off mislabels.

    Steps (per iteration):
    1. Randomly select an under-labeled point from the foreground.
    2. Greedily chain ``chain_length`` nearest neighbours to build a local
       linear probe.
    3. Compute the mean inter-point distance along the chain.
    4. Fit PCA to the chain; extract the dominant direction as an angle.
    4a. Snap the angle to the closest entry in ``theta_degs`` (within ``tol``);
        discard the probe (no vote) if no entry is close enough.
    5. Find all foreground points within ``bin_width`` pixels perpendicular
       to the snapped line through the starting point.
    6. Flood-fill a connected segment from the starting point through the
       near-line set, using ``2 × mean_dist`` as the adjacency radius.
    7. Cast one vote for the snapped angle family on every under-labeled
       point in the connected segment.
    8. Repeat until every point has been visited ``n_times``.
    9. Resolve each point's label as ``argmax`` of its vote histogram.

    Parameters
    ----------
    x, y : array-like
        Column and row coordinates of foreground pixels.
    theta_degs : array-like
        Image-convention dominant angles in degrees (as returned by
        ``compute_local_orientation_via_radon`` after the −90° correction).
    bin_width : float
        Maximum perpendicular distance (pixels) from the fitted line for a
        point to be considered part of that line segment. Default is 1.0.
    tol : float
        Maximum angular difference (degrees) between the PCA-computed angle
        and the nearest entry in ``theta_degs`` for the snap to be accepted.
        Probes that cannot be snapped cast no vote. Default is 5.
    num_neighbors : int
        Number of nearest neighbours queried at each chain-extension step.
        Higher values reduce the chance of getting stuck. Default is 20.
    chain_length : int
        Maximum number of points in each local chain used for PCA. Default 10.
    n_times : int
        Minimum number of votes each point must accumulate before the loop
        terminates. Higher values reduce mislabels at the cost of runtime.
        Default is 3.

    Returns
    -------
    labels : np.ndarray of int
        Per-point index into ``theta_degs`` (i.e. the winning angle family),
        or −1 if the point received no valid votes.
    all_peak_centers : list
        Empty list; returned for API symmetry with ``label_points_by_angle``.
        This method does not compute ρ-space trackline centers.
    """
    theta_degs = np.asarray(theta_degs, dtype=float)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n_points = len(x)
    n_angles = len(theta_degs)

    # (col, row) ordering gives angle 0° = horizontal, 90° = vertical,
    # matching the convention used by orientation_via_PCA and orientation_via_radon.
    pts = np.column_stack([x, y])
    tree = KDTree(pts)

    # label_votes[i, k] = number of times point i was assigned to angle family k
    label_votes = np.zeros((n_points, n_angles), dtype=int)
    # label_counts[i] = total labeling attempts on point i (votes + no-snap misses)
    label_counts = np.zeros(n_points, dtype=int)

    pca = PCA(n_components=1)
    rng = np.random.default_rng(42)

    with tqdm(total=n_points * n_times, desc="Probabilistic line labeling") as pbar:
        prev_total = 0
        while not np.all(label_counts >= n_times):
            # Step 1: randomly pick a point that still needs more votes
            under_idx = np.where(label_counts < n_times)[0]
            start_idx = int(rng.choice(under_idx))

            # Steps 2-3: bidirectional chain — grow half forward, half backward
            # so start_idx stays near the middle and the PCA line passes through it.
            chain = deque([start_idx])
            visited = {start_idx}
            half = (chain_length - 1) // 2

            # Forward leg: extend from the front end
            front = start_idx
            for _ in range(half):
                _, indices = tree.query(pts[front].reshape(1, -1), k=num_neighbors + 1)
                found = False
                for ni in indices[0]:
                    if ni not in visited:
                        chain.append(ni)
                        visited.add(ni)
                        front = ni
                        found = True
                        break
                if not found:
                    break

            # Backward leg: extend from the back end (opposite side of start_idx)
            # visited already contains the forward points, so the nearest unvisited
            # neighbour of start_idx naturally reaches the other direction.
            back = start_idx
            for _ in range(chain_length - 1 - half):
                _, indices = tree.query(pts[back].reshape(1, -1), k=num_neighbors + 1)
                found = False
                for ni in indices[0]:
                    if ni not in visited:
                        chain.appendleft(ni)
                        visited.add(ni)
                        back = ni
                        found = True
                        break
                if not found:
                    break

            chain_pts = pts[list(chain)]
            if len(chain) < 2:
                label_counts[start_idx] += 1
                pbar.update(label_counts.sum() - prev_total)
                prev_total = label_counts.sum()
                continue

            mean_dist = np.mean(np.linalg.norm(np.diff(chain_pts, axis=0), axis=1))

            # Step 4: PCA dominant direction → angle
            pca.fit(chain_pts)
            direction = pca.components_[0]
            # arctan2(row_component, col_component): 0°=horizontal, 90°=vertical
            angle_deg = np.degrees(np.arctan2(direction[1], direction[0])) % 180

            # Step 4a: snap to the nearest Radon-detected angle within tol
            diffs = np.abs(((theta_degs - angle_deg + 90) % 180) - 90)
            best = int(np.argmin(diffs))
            if diffs[best] > tol:
                # No valid family — count the attempt so this point can still
                # reach n_times and won't block loop termination
                label_counts[start_idx] += 1
                pbar.update(label_counts.sum() - prev_total)
                prev_total = label_counts.sum()
                continue

            snapped_angle = theta_degs[best]

            # Step 5: points within bin_width perpendicular to the snapped line
            angle_rad = np.deg2rad(snapped_angle)
            normal = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
            perp_dist = np.abs((pts - pts[start_idx]) @ normal)
            # near_idx = np.where(perp_dist <= bin_width)[0]  # sorted; start_idx always present
            near_idx = np.where(perp_dist <= mean_dist)[0]

            # Step 6: BFS connected segment — 2×mean_dist adjacency radius
            near_pts = pts[near_idx]
            sub_tree = KDTree(near_pts)
            start_local = int(np.searchsorted(near_idx, start_idx))
            step_radius = mean_dist * 2.0

            subcluster_local = set()
            queue = [start_local]
            sub_seen = {start_local}
            while queue:
                cur = queue.pop(0)
                subcluster_local.add(cur)
                for ni in sub_tree.query_ball_point(near_pts[cur], r=step_radius):
                    if ni not in sub_seen:
                        sub_seen.add(ni)
                        queue.append(ni)

            # Steps 7-8: cast one vote on every under-labeled point in the segment
            subcluster_global = near_idx[list(subcluster_local)]
            for idx in subcluster_global:
                if label_counts[idx] < n_times:
                    label_votes[idx, best] += 1
                    label_counts[idx] += 1

            curr_total = label_counts.sum()
            pbar.update(curr_total - prev_total)
            prev_total = curr_total

    # Step 9: resolve each point's label as the winning angle family (mode)
    total_votes = label_votes.sum(axis=1)
    labels = np.where(total_votes > 0, np.argmax(label_votes, axis=1), -1)

    return labels, []


def label_points_by_probabilistic_line2(x, y,
                                        theta_degs,
                                        bin_width=1.0,
                                        tol=5,
                                        num_neighbors=20,
                                        chain_length=10,
                                        min_tentative=5):
    """
    Two-tier probabilistic line labeling without BFS connectivity.

    Each randomly selected point receives a **definitive** label from its
    local PCA angle (snapped to a Radon-detected family).  Every point that
    lies within ``bin_width`` pixels of that line receives one **tentative**
    vote for the same family.  The loop runs until every point has either
    been selected once or accumulated at least ``min_tentative`` tentative
    votes, whichever comes first.  Final labels are the definitive label for
    selected points and the majority-vote angle for the rest.

    Compared to ``label_points_by_probabilistic_line``, this variant:

    * Removes the BFS connected-component step — near-line membership is a
      pure perpendicular-distance test, vectorised over all points.
    * Separates definitive (strong) from tentative (weak) evidence, so a
      point selected as a start always gets the label its own chain implies,
      while boundary points accumulate evidence from many passing probes.

    Parameters
    ----------
    x, y : array-like
        Column and row coordinates of foreground pixels.
    theta_degs : array-like
        Image-convention dominant angles in degrees (Radon-detected).
    bin_width : float
        Maximum perpendicular distance (pixels) for a point to receive a
        tentative vote from a passing line. Default is 1.0.
    tol : float
        Maximum angular difference (degrees) between the PCA angle and the
        nearest ``theta_degs`` entry for the snap to be accepted. Default 5.
    num_neighbors : int
        Nearest-neighbour count for each chain-extension query. Default 20.
    chain_length : int
        Maximum chain length for PCA; ``start_idx`` sits near the middle
        via bidirectional growth. Default 10.
    min_tentative : int
        Minimum tentative votes a non-selected point must receive before it
        is considered sufficiently labeled. Default 5.

    Returns
    -------
    labels : np.ndarray of int
        Per-point index into ``theta_degs``, or −1 if unresolved.
    all_peak_centers : list
        Empty list; for API symmetry with ``label_points_by_angle``.
    """
    theta_degs = np.asarray(theta_degs, dtype=float)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n_points = len(x)
    n_angles = len(theta_degs)

    pts = np.column_stack([x, y])
    tree = KDTree(pts)

    selected = np.zeros(n_points, dtype=bool)
    definitive_labels = np.full(n_points, -1, dtype=int)
    tentative_votes = np.zeros((n_points, n_angles), dtype=int)
    tentative_counts = np.zeros(n_points, dtype=int)

    pca = PCA(n_components=1)
    rng = np.random.default_rng(42)

    with tqdm(total=n_points, desc="Probabilistic line labeling v2") as pbar:
        prev_satisfied = 0
        while True:
            # Candidates: unselected points that still need more tentative votes.
            # When this pool is empty every point is either selected or has
            # enough votes — the termination condition is met.
            candidates = np.where(~selected & (tentative_counts < min_tentative))[0]
            if len(candidates) == 0:
                break

            start_idx = int(rng.choice(candidates))

            # Bidirectional chain so start_idx sits near the middle
            chain = deque([start_idx])
            visited = {start_idx}
            half = (chain_length - 1) // 2

            front = start_idx
            for _ in range(half):
                _, indices = tree.query(pts[front].reshape(1, -1), k=num_neighbors + 1)
                found = False
                for ni in indices[0]:
                    if ni not in visited:
                        chain.append(ni)
                        visited.add(ni)
                        front = ni
                        found = True
                        break
                if not found:
                    break

            back = start_idx
            for _ in range(chain_length - 1 - half):
                _, indices = tree.query(pts[back].reshape(1, -1), k=num_neighbors + 1)
                found = False
                for ni in indices[0]:
                    if ni not in visited:
                        chain.appendleft(ni)
                        visited.add(ni)
                        back = ni
                        found = True
                        break
                if not found:
                    break

            # Mark as selected regardless of snap outcome
            selected[start_idx] = True
            chain_pts = pts[list(chain)]

            if len(chain) >= 2:
                pca.fit(chain_pts)
                direction = pca.components_[0]
                angle_deg = np.degrees(np.arctan2(direction[1], direction[0])) % 180

                diffs = np.abs(((theta_degs - angle_deg + 90) % 180) - 90)
                best = int(np.argmin(diffs))

                if diffs[best] <= tol:
                    definitive_labels[start_idx] = best
                    snapped_angle = theta_degs[best]

                    # Perpendicular-distance filter — vectorised, no BFS
                    angle_rad = np.deg2rad(snapped_angle)
                    normal = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
                    perp_dist = np.abs((pts - pts[start_idx]) @ normal)
                    near_idx = np.where(perp_dist <= bin_width)[0]

                    # Cast one tentative vote on every near-line point
                    tentative_votes[near_idx, best] += 1
                    tentative_counts[near_idx] += 1

            curr_satisfied = int(np.sum(selected | (tentative_counts >= min_tentative)))
            pbar.update(curr_satisfied - prev_satisfied)
            prev_satisfied = curr_satisfied

    # Resolve final labels
    labels = np.full(n_points, -1, dtype=int)
    labels[selected] = definitive_labels[selected]
    has_votes = (~selected) & (tentative_counts > 0)
    if has_votes.any():
        labels[has_votes] = np.argmax(tentative_votes[has_votes], axis=1)

    return labels, []

