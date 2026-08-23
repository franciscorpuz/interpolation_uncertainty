import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
from scipy.signal import find_peaks, peak_widths
from skimage.transform import radon


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
        point_orientations, angle_profile = compute_local_orientation_via_PCA(binary_image,
                                            kwargs.get('line_length', 20),
                                            kwargs.get('num_neighbors', 50))
    elif method == 'radon':
        point_orientations, angle_info = compute_local_orientation_via_radon(binary_image,
                                                                     kwargs.get('test_angles', None),
                                                                     kwargs.get('circle', True),
                                                                     kwargs.get('tol', 5),
                                                                     kwargs.get('precomputed_theta', None))
    else:
        raise ValueError(f"Invalid method: {method}. Must be either 'pca' or 'radon'.")

    if plot_orientation:
        plt.figure(figsize=figsize)
        plt.imshow(point_orientations, cmap=cmap)
        plt.colorbar(label='Orientation (degrees)', shrink=0.5)
        plt.title(f"Local Orientations via {method.upper()}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().invert_yaxis()
        plt.show()

    return point_orientations, angle_info


def compute_local_orientation_via_PCA(binary_image,
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


def compute_local_orientation_via_radon(binary_image: np.ndarray,
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
            theta = test_angles
        angle_step = theta[1] - theta[0]

        # Compute the Radon transform (sinogram) and the variance of each projection column
        print("Computing Radon transform...")
        sinogram = radon(binary_image, theta=theta, circle=circle)
        profile_variance = np.var(sinogram, axis=0)
        profile_variance_xy = np.roll(profile_variance, shift=90)
        print("Radon Computation Done.")

        # Plot profile variance
        plt.figure(figsize=(10, 6))
        plt.plot(profile_variance_xy, label="Profile Variance")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Variance")
        plt.title("Variance Profile")


        # Detect peaks in the variance profile
        # rough estimate: foreground_points / expected_tracklines_per_angle
        # min_peak_prominence = len(x_coords) // (n_angles * expected_lines_per_angle)
        peaks, properties = find_peaks(profile_variance_xy, prominence=(None, None))

        print(f"Detected peaks:{peaks}")
        print(f"Properties: {properties}")

        widths, _, _, _ = peak_widths(profile_variance, peaks, rel_height=0.5)
        widths_in_degrees = widths * angle_step

        # sort the peaks by variance and return the most dominant angle and its width
        peak_indices = np.argsort(np.array(properties['prominences']))[::-1]
        peaks = peaks[peak_indices]
        widths_in_degrees = widths_in_degrees[peak_indices]
    else:
        peaks = precomputed_theta['peaks']
        widths_in_degrees = precomputed_theta['widths_in_degrees']
        profile_variance = precomputed_theta['profile_variance']

    # tol: use at least half the FWHM of the widest detected peak
    if tol is None:
        tol = int(np.ceil(np.max(widths_in_degrees) / 2))
    else:
        tol = max(tol, int(np.ceil(np.max(widths_in_degrees) / 2)))

    # Convert peaks to rho-space
    # peaks = (peaks + 90 ) % 180
    print("Labeling points...")
    labels, peak_centers_per_angle = label_points_by_angle(x_coords,
                                                           y_coords,
                                                           peaks,
                                                           bin_width=10.0,
                                                           tol=tol,
                                                           min_peak_prominence=None)
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
        "widths_in_degrees": np.array(widths_in_degrees),
        "profile_variance": np.array(profile_variance),
    }


    return point_orientations, theta_info

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




# def compute_dominant_angle_radon_transform(image: np.ndarray,
#                                            test_angles: np.ndarray = None,
#                                            circle: bool = True,
#                                            plot_profile_variance: bool = False,
#                                            tol: int | None = 5) -> tuple[float, np.ndarray]:
#     """Find the dominant orientation of linear features using the Radon transform.
#
#     Computes the Radon sinogram over ``test_angles`` and returns the angle whose
#     projection column has the highest variance. High variance indicates that
#     projections along that angle produce concentrated bright lines, which is
#     characteristic of linear features aligned with that orientation.
#
#     Parameters
#     ----------
#     image : np.ndarray
#         Binary image containing linear features.
#     test_angles : np.ndarray, optional
#         Angles in degrees at which to compute the Radon transform. If ``None``,
#         180 angles uniformly spaced over ``[0°, 180°)`` are used.
#     circle : bool, optional
#         If ``True``, only pixels within the inscribed circle are projected
#         (matches the skimage default). Default is ``True``.
#     plot_profile_variance : bool, optional
#         If ``True``, plots the variance profile across all tested angles.
#         Default is ``False``.
#     tol : int | None, optional
#         Half-width in degrees of the suppression band around each angle in
#         ``remove_angles``. Default is 5.
#
#     Returns
#     -------
#     theta_info : dict
#         Dictionary with keys:
#
#         - ``'peaks'`` : np.ndarray of float — dominant angle(s) in degrees,
#           sorted by descending prominence.
#         - ``'widths_in_degrees'`` : np.ndarray of float — FWHM of each peak
#           in degrees.
#         - ``'profile_variance'`` : np.ndarray of float — variance profile
#           over all tested angles (rolled so index 0 corresponds to 0°).
#     """
#     if test_angles is None:
#         theta = np.linspace(0., 180., 180, endpoint=False)
#     else:
#         theta = test_angles
#     angle_step = theta[1] - theta[0]
#
#     # Compute the Radon transform (sinogram) and the variance of each projection column
#     sinogram = radon(image, theta=theta, circle=circle)
#     profile_variance = np.var(sinogram, axis=0)
#
#     # Detect peaks in the variance profile
#     peaks, properties = find_peaks(profile_variance, prominence=(None, None))
#     widths, _, _, _ = peak_widths(profile_variance, peaks, rel_height=0.5)
#
#     widths_in_degrees = widths * angle_step
#
#     # sort the peaks by variance and return the most dominant angle and its width
#     peak_indices = np.argsort(np.array(properties['prominences']))[::-1]
#     peaks = peaks[peak_indices]
#     widths_in_degrees = widths_in_degrees[peak_indices]
#
#     # Deduct 90 degrees to remove effect of radon transform's 90-degree shift between image and sinogram angles
#     peaks = (peaks - 90) % 180
#
#     # Adjust the profile variance (x-axis) to correspond with the correction above
#     profile_variance = np.roll(profile_variance, len(theta) // 2)
#
#     # If no tolerance is given, set it to half the width of the widest detected peak
#     # to ensure all peaks are included in the output
#     if tol is None:
#         tol = int(np.ceil(np.max(widths_in_degrees) / 2))
#     else:
#         tol = max(tol, int(np.ceil(np.max(widths_in_degrees) / 2)))
#
#     if plot_profile_variance:
#         plt.figure(figsize=(10, 5))
#         plt.plot(theta, profile_variance)
#         plt.title(f'Profile Variance (Dominant Angle: {peaks[0]:.2f}°)')
#         plt.xlabel('Angle (degrees)')
#         plt.ylabel('Variance')
#         plt.show()
#
#     theta_info = {
#         "peaks": np.array(peaks),
#         "widths_in_degrees": np.array(widths_in_degrees),
#         "profile_variance": np.array(profile_variance)
#     }
#
#     return theta_info