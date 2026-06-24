import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
from scipy.signal import find_peaks, peak_widths
from skimage.transform import radon


def compute_local_orientation(binary_image,
                              line_length=20,
                              num_neighbors=50,
                              plot_orientation=False,
                              cmap='terrain',
                              figsize=(16, 8)):
    """ Computes local orientation, via PCA, for each point in a 2D dataset
    using a chaining method to determin the nearest neighbors
    For each point, it builds a chain of nearby points and computes
    the angle of the first principal component of the chain.
    This method captures local linear structures in the data.

    Args:
        binary_image (np.ndarray): A binary image where True values represent points of interest.
        line_length (int): The number of points to include in each chain.
        num_neighbors (int): The number of nearest neighbors to consider when building the chain.
        plot_orientation (bool): Whether to plot the local orientations as arrows.
        cmap (str): The colormap to use for plotting orientations.
        figsize (tuple): The size of the plot if plot_orientation is True.

    Returns:
        point_orientations (np.ndarray): A 2D array of local orientations (in degrees) for each point.
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

    if plot_orientation:
        plt.figure(figsize=figsize)
        plt.scatter(X_linear[:, 1], X_linear[:, 0], c=angle_scores, cmap=cmap, s=5)
        plt.colorbar(label='Orientation (degrees)', shrink=0.5)
        plt.title("Local Orientations via PCA")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        plt.show()

    return point_orientations


def compute_dominant_angle_radon_transform(image: np.ndarray,
                                           test_angles: np.ndarray = None,
                                           circle: bool = True,
                                           plot_profile_variance: bool = False,
                                           tol: int | None = 5) -> tuple[float, np.ndarray]:
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
    plot_profile_variance : bool, optional
        If ``True``, plots the variance profile across all tested angles.
        Default is ``False``.
    tol : int | None, optional
        Half-width in degrees of the suppression band around each angle in
        ``remove_angles``. Default is 5.

    Returns
    -------
    theta_info : dictionary of (peak, widths_in_degrees, profile_variance)
        Profile variance values for all tested angles, where:
        - ``peaks`` is the array of peak angles
        - ``widths_in_degrees`` is the array of corresponding peak widths in degrees
        - ``profile_variance`` is the corresponding array of variance values.
    """
    if test_angles is None:
        theta = np.linspace(0., 180., 180, endpoint=False)
    else:
        theta = test_angles
    angle_step = theta[1] - theta[0]

    # Compute the Radon transform (sinogram) and the variance of each projection column
    sinogram = radon(image, theta=theta, circle=circle)
    profile_variance = np.var(sinogram, axis=0)

    # Detect peaks in the variance profile
    peaks, properties = find_peaks(profile_variance, prominence=(None, None))
    widths, _, _, _ = peak_widths(profile_variance, peaks, rel_height=0.5)

    widths_in_degrees = widths * angle_step

    # sort the peaks by variance and return the most dominant angle and its width
    peak_indices = np.argsort(np.array(properties['prominences']))[::-1]
    peaks = peaks[peak_indices]
    widths_in_degrees = widths_in_degrees[peak_indices]

    # Deduct 90 degrees to remove effect of radon transform's 90-degree shift between image and sinogram angles
    peaks = (peaks - 90) % 180

    # Adjust the profile variance (x-axis) to correspond with the correction above
    profile_variance = np.roll(profile_variance, len(theta) // 2)

    # If no tolerance is given, set it to half the width of the widest detected peak
    # to ensure all peaks are included in the output
    if tol is None:
        tol = int(np.ceil(np.max(widths_in_degrees) / 2))
    else:
        tol = max(tol, int(np.ceil(np.max(widths_in_degrees) / 2)))

    if plot_profile_variance:
        plt.figure(figsize=(10, 5))
        plt.plot(theta, profile_variance)
        plt.title(f'Profile Variance (Dominant Angle: {peaks[0]:.2f}°)')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Variance')
        plt.show()

    theta_info = {
        "peaks": np.array(peaks),
        "widths_in_degrees": np.array(widths_in_degrees),
        "profile_variance": np.array(profile_variance)
    }

    return theta_info
