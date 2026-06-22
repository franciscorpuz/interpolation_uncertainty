
import numpy as np
from osgeo import gdal
from pathlib import Path
from typing import Union
from tqdm import tqdm
from copy import deepcopy

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor

from scipy.spatial import KDTree, cKDTree
from scipy.ndimage import convolve
from scipy.signal import find_peaks, peak_widths, windows, get_window
from scipy.interpolate import griddata

from skimage.morphology import dilation, medial_axis, opening, footprint_rectangle, thin, disk, skeletonize, remove_small_objects, closing, white_tophat, black_tophat
from skimage.draw import line, disk as skimage_disk
from skimage.transform import rotate, radon, hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.feature import canny, blob_doh
from skimage.measure import label, regionprops

gdal.UseExceptions()


def read_file(filename: Union[str, Path], verbose: bool = False) -> dict:
    """Read a raster or BAG file and extract depth data and metadata.

    Parameters
    ----------
    filename : str or Path
        Path to the raster file (.tif, .tiff, or .bag).
    verbose : bool, optional
        If ``True``, prints additional information during file reading. Default is ``False``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'data'`` : np.ndarray — 2D array of raw depth values (NDV not masked).
        - ``'filename'`` : str — base name of the file.
        - ``'filetype'`` : str — always ``'raster'``.
        - ``'metadata'`` : dict containing ``'ndv_value'``, ``'resolution'``,
          ``'full_path'``, and ``'geotransform'``.
    """
    with gdal.Open(str(filename)) as ds:
        if not ds:
            raise RuntimeError(
                f"GDAL failed to open file: '{filename}'")

        depth_band = ds.GetRasterBand(1)
        if not depth_band:
            raise RuntimeError(
                f"Error retrieving depth information from {filename}.")

        ndv_value = depth_band.GetNoDataValue()
        raw_depth_data = depth_band.ReadAsArray()
        depth_gt = ds.GetGeoTransform()
        resolution = depth_gt[1]
        if resolution < 1:
            print(f"WARNING: detected resolution value is <= 1"
                  f"\n Setting resolution value to 1")
            resolution = 1
        
        data_dict = {'data': raw_depth_data,
                'filename': Path(filename).name,
                'filetype': 'raster',
                'metadata': {'ndv_value': ndv_value,
                             'resolution': resolution,
                             'full_path': filename,
                             'geotransform': depth_gt}
                }
        
        if verbose:
            print(f"File '{filename}' read successfully.")
            row_count, col_count = raw_depth_data.shape
            print(f"Data shape: {row_count} rows x {col_count} columns")
            print(f"No-data value: {ndv_value}")
            print(f"Resolution: {resolution} m/pixel")
            print(f"Geotransform values--> ")
            gt_keys = ['Origin x-coordinate:', 
                       'Pixel width:', 
                       'Row rotation:', 
                       'Origin y-coordinate:', 
                       'Column rotation:', 
                       'Pixel height:']
            print("\n".join(f"{k} {v}" for k, v in zip(gt_keys, depth_gt)), end="\n\n")

        return data_dict

def show_depth(dataset_dict: dict,
               title: Union[str, None] = None,
               cmap: str = 'terrain') -> None:
    """Plot bathymetric depth data from either a raster or point-cloud dataset.

    Parameters
    ----------
    dataset_dict : dict
        Dataset dictionary with ``'filetype'`` of either ``'raster'`` or
        ``'points'``, as returned by :func:`read_file` or :func:`raster_to_points`.
    title : str or None, optional
        Custom plot title. If ``None``, a title is generated from the filename
        and resolution. Default is ``None``.
    cmap : str, optional
        Matplotlib colourmap name. Default is ``'terrain'``.

    Returns
    -------
    None
    """
    fig, ax1 = plt.subplots(figsize=(16, 10))
    points_dataset = deepcopy(dataset_dict)
    res = points_dataset["metadata"]['resolution']
    ndv = points_dataset["metadata"]['ndv_value']
    points_dataset["data"] = np.where(points_dataset["data"] == ndv, np.nan, points_dataset["data"])
    if points_dataset["filetype"] == 'raster':
        im = ax1.imshow(points_dataset["data"], cmap=cmap, aspect='equal')
        shape_0 = points_dataset["data"].shape[0]
        shape_1 = points_dataset["data"].shape[1]
        fig.colorbar(im, label='Depth (m)', shrink=0.5)
        locs = ax1.get_xticks()
        ax1.set_xticks(locs)
        ax1.set_xticklabels([str(int(x * res)) for x in locs])
        locs = ax1.get_yticks()
        ax1.set_yticks(locs)
        ax1.set_yticklabels([str(int(y * res)) for y in locs])
        ax1.tick_params(axis='x', labelrotation=90)
        ax1.set_xlim(left=0, right=shape_1)
        ax1.set_ylim(top=0, bottom=shape_0)
    elif points_dataset["filetype"] == 'points':
        x = points_dataset["data"][:, 0]
        y = points_dataset["data"][:, 1]
        depth = points_dataset["data"][:, 2]
        sc = ax1.scatter(x, y, c=depth, cmap=cmap, marker='.', s=1)
        fig.colorbar(sc, label='Depth (m)', shrink=0.5)
        shape_0 = int((np.max(y) - np.min(y)) / res)
        shape_1 = int((np.max(x) - np.min(x)) / res)
    else:
        raise ValueError(f"Unrecognized file type for plotting: {points_dataset['filetype']}")
    ax1.set_xlabel("West-East (m)")
    ax1.set_ylabel("North-South (m)")
    if title is None:
        fn = Path(points_dataset["filename"]).name
        title = f"{fn} at {res}m resolution"
    ax1.set_title(f"""
                Surface:{title} at {res}m resolution
                Dimensions: {shape_0 * res / 1000}km by {shape_1 * res / 1000}km
                    """)

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


def x_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute the absolute x-axis distance between two points.

    Used as a custom DBSCAN metric to cluster points that share a similar
    x-coordinate (i.e., lie on the same across-track survey line after rotation).

    Parameters
    ----------
    p1 : np.ndarray
        First point as ``[x, y, ...]``.
    p2 : np.ndarray
        Second point as ``[x, y, ...]``.

    Returns
    -------
    float
        Absolute difference in the x-coordinate.
    """
    return np.abs(p1[0] - p2[0])


def rotate_points(points: np.ndarray, 
                  angle: float,
                  center: np.ndarray | None = None) -> np.ndarray:
    """Rotate a set of [x, y, depth] points given an angle.

    Parameters
    ----------
    points : np.ndarray
        Array of shape ``(n_points, 3)`` with columns ``[x, y, depth]``.
    angle : float
        Rotation angle in degrees (counter-clockwise positive).
    center : np.ndarray | None, optional
        Center of rotation. If None, center of point cloud is used.

    Returns
    -------
    np.ndarray
        Rotated point array of the same shape as ``points``.
    tuple
        center coordinates
    """
    
    coordinates = points

    # Use centroid if None is given
    if center is None:
        center = coordinates.mean(axis=0)

    # Create rotation matrix
    theta_rad = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    rotation_matrix = np.array([[cos_t, -sin_t],
                                [sin_t,  cos_t]])

    # Apply rotation matrix above to coordinates
    rotated_coords = (coordinates - center) @ rotation_matrix.T + center
    rotated_coords = rotated_coords.astype(int)  # convert back to integer pixel coordinates

    # return rotated coordinates with the center used 
    # for rotation (useful for updating geotransform later)
    return rotated_coords, center

def update_geotransform(points: np.ndarray,
                       pixel_width: float,
                       pixel_height: float) -> tuple:
    """Build a new geotransform that fits a rotated point cloud.

    The origin is placed so that the most extreme points sit at
    pixel centres, consistent with :func:`raster_to_points`.

    Parameters
    ----------
    points : np.ndarray
        Array of shape ``(n, 3)`` with columns ``[x, y, depth]``.
    pixel_width : float
        GT[1] — pixel size in the x direction.
    pixel_height : float
        GT[5] — pixel size in the y direction (negative for north-up).

    Returns
    -------
    tuple
        A 6-element GDAL-style geotransform.
    """
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    x_max, y_max = points[:, 0].max(), points[:, 1].max()

    # Reverse the pixel-centre offset to get the upper-left corner.
    #   x_min = GT[0] + GT[1]/2   →   GT[0] = x_min - GT[1]/2
    #   y_max = GT[3] + GT[5]/2   →   GT[3] = y_max - GT[5]/2
    gt0 = x_min - pixel_width / 2
    gt3 = y_max - pixel_height / 2  # pixel_height is negative, so this adds

    # Compute the new grid dimensions
    n_cols = int(np.round((x_max - x_min) / pixel_width)) + 1
    n_rows = int(np.round((y_max - y_min) / abs(pixel_height))) + 1

    return (gt0, pixel_width, 0.0, gt3, 0.0, pixel_height), (n_rows, n_cols)

def filter_blobs_keep_lines(
    binary_image: np.ndarray,
    line_length: int = 50,
    n_angles: int = 18,
    restore_radius: int = 0,
) -> np.ndarray:
    """Remove blobs from a binary image while preserving lines at their original thickness.

    Uses directional morphological opening across ``n_angles`` orientations.
    Pixels that survive at least one opening (i.e., lie on a sufficiently long
    linear structure) are kept. An optional dilation step recovers the original
    line width after thinning by the opening.

    Parameters
    ----------
    binary_image : np.ndarray
        Input binary image (bool or 0/255 uint8).
    line_length : int, optional
        Length of the line structuring element in pixels. Default is 50.
    n_angles : int, optional
        Number of rotation angles evenly spaced over 180°. Default is 18.
    restore_radius : int, optional
        Dilation radius for recovering original line thickness after opening.
        Set to roughly half the expected line width. Default is 0 (no dilation).

    Returns
    -------
    np.ndarray of bool
        Filtered image with blobs removed, lines at original thickness.
    """
    assert restore_radius >= 0, "restore_radius must be a non-negative integer"
    img = binary_image > 0

    line_cores = np.zeros_like(img)
    half = line_length // 2

    for i in tqdm(range(n_angles)):
        angle = np.pi * i / n_angles
        dx = int(round(half * np.cos(angle)))
        dy = int(round(half * np.sin(angle)))

        size = 2 * half + 1
        center = half
        kernel = np.zeros((size, size), dtype=bool)
        rr, cc = line(
            center - dy, center - dx,
            center + dy, center + dx,
        )
        kernel[rr, cc] = True

        opened = opening(img, footprint=kernel)
        line_cores = line_cores | opened

    # Dilate line cores to recover original line width, then AND with the
    # original image so no new pixels are introduced
    if restore_radius > 0:
        recovery_mask = dilation(line_cores, footprint=disk(restore_radius))
        result = img & recovery_mask
    else:
        result = line_cores

    return result


def filter_squares_keep_lines(
    binary_image: np.ndarray,
    square_size: int = 50,
    angle_rotation: float = 0,
) -> np.ndarray:
    """Remove square-shaped blobs from a binary image while preserving lines.

    Applies a square morphological opening to identify blob-like patches that
    are at least ``square_size`` pixels wide in all directions. Lines are too
    thin to survive the opening and are therefore preserved.

    Parameters
    ----------
    binary_image : np.ndarray
        Input binary image (bool or 0/255 uint8).
    square_size : int, optional
        Side length of the square structuring element in pixels. Default is 50.
    angle_rotation : float, optional
        Angle in degrees to rotate the structuring element. Useful when the
        dominant blob orientation is not axis-aligned. Default is 0.

    Returns
    -------
    np.ndarray of bool
        Filtered image with square blobs removed, lines preserved.
    """
    img = binary_image > 0

    footprint = footprint_rectangle((square_size, square_size), dtype=bool)
    if angle_rotation != 0:
        footprint = rotate(footprint.astype(float), angle_rotation) > 0.5

    # Pixels that survive the square opening are blob-like; subtract them
    squares_filtered = (1 - opening(img, footprint=footprint)) & img

    return squares_filtered



def refine_cluster_rotation(points: np.ndarray,
                             plot_data: bool = False) -> tuple[np.ndarray, float]:
    """Align a set of points with the x-axis using DBSCAN subclustering and PCA.

    Runs DBSCAN on the xy-plane to find subclusters, fits PCA to the largest
    subclusters to estimate the dominant orientation, then rotates all points
    so that orientation aligns with the x-axis.

    Parameters
    ----------
    points : np.ndarray
        Array of shape ``(n_points, 3)`` with columns ``[x, y, depth]``.
    plot_data : bool, optional
        If ``True``, plots each retained subcluster. Default is ``False``.

    Returns
    -------
    refined_cluster : np.ndarray
        Point array of shape ``(n_points, 3)`` rotated so the dominant line
        direction is parallel to the x-axis.
    angle_correction : float
        Rotation angle in degrees applied to produce ``refined_cluster``.
    """
    dbscan = DBSCAN(eps=80, p=2, metric=x_distance, min_samples=10)
    subcluster_labels = dbscan.fit_predict(points[:, :2])

    subcluster_sizes = [np.sum(subcluster_labels == label)
                        for label in np.unique(subcluster_labels) if label != -1]

    # Only use the largest subclusters (above 90th percentile) for angle estimation
    # to avoid small noise clusters skewing the rotation
    min_cluster_size = np.percentile(np.array(subcluster_sizes), 90)

    subcluster_angles = []
    if plot_data:
        plt.figure(figsize=(15, 15))

    for label in np.unique(subcluster_labels):
        if label != -1 and np.sum(subcluster_labels == label) >= min_cluster_size:
            if plot_data:
                plt.scatter(points[subcluster_labels == label, 0],
                            points[subcluster_labels == label, 1],
                            s=10, label=f'Subcluster {label}')
            current_pts = points[subcluster_labels == label]
            pca = PCA(n_components=2)
            pca.fit(current_pts)
            direction = pca.components_[0]
            angle_radians = np.arctan2(direction[1], direction[0])
            angle_degrees = np.degrees(angle_radians) % 180
            subcluster_angles.append(angle_degrees)

    refined_cluster = rotate_points(points, 90 - np.mean(subcluster_angles))
    angle_correction = 90 - np.mean(subcluster_angles)

    return refined_cluster, angle_correction


def filter_morphological_operations(binary_image: np.ndarray,
                                    square_size: int = 50,
                                    line_length: int = 50,
                                    n_angles: int = 180,
                                    restore_radius: int = 5,
                                    blob_max_size: int = 100) -> np.ndarray:
    """Apply a morphological pipeline to suppress blobs while preserving lines.

    The pipeline:

    1. Estimates the dominant orientation of the image via PCA on skeleton pixels.
    2. Rotates the image to align the dominant orientation with the x-axis.
    3. Applies directional opening (line SE) to suppress non-linear blobs.
    4. Applies square opening to remove large square-shaped patches.
    5. Removes small residual blobs below ``blob_max_size`` pixels.
    6. Rotates the result back to the original orientation.

    Parameters
    ----------
    binary_image : np.ndarray
        Input binary image (bool or 0/255 uint8).
    square_size : int, optional
        Side length of the square structuring element. Default is 50.
    line_length : int, optional
        Length of the line structuring element. Default is 50.
    n_angles : int, optional
        Number of orientations tested by the directional opening. Default is 180.
    restore_radius : int, optional
        Dilation radius used in :func:`filter_blobs_keep_lines` to recover
        original line thickness. Default is 5.
    blob_max_size : int, optional
        Maximum size (in pixels) of connected components to discard after
        filtering. Set to 0 to skip this step. Default is 100.

    Returns
    -------
    np.ndarray of bool
        Filtered binary image with blobs suppressed and lines preserved,
        in the original orientation.
    """
    # Use the medial axis to get a thin skeleton for robust PCA orientation estimation
    y_indices, x_indices = np.nonzero(medial_axis(binary_image))
    points = np.column_stack((x_indices, y_indices))
    pca = PCA(n_components=2)
    pca.fit(points)
    direction = pca.components_[0]
    orig_angle_radians = np.arctan2(direction[1], direction[0])
    orig_angle_degrees = np.degrees(orig_angle_radians) % 180

    # Rotate the binary image to align the dominant structure with the x-axis
    rotated_binary = rotate(binary_image, angle=-orig_angle_degrees,
                            resize=False, preserve_range=True).astype(bool)

    # Filter with directional line structuring elements to isolate line-like pixels
    lines_filtered = filter_blobs_keep_lines(
        binary_image=rotated_binary,
        line_length=line_length,
        n_angles=n_angles,
        restore_radius=restore_radius,
    )

    # Compute orientation of the rotated image (used as input to the square filter)
    y_indices, x_indices = np.nonzero(rotated_binary)
    points = np.column_stack((x_indices, y_indices))
    pca2 = PCA(n_components=2)
    pca2.fit(points)
    direction = pca2.components_[0]
    angle_degrees = np.degrees(np.arctan2(direction[1], direction[0])) % 180

    # Remove large square-shaped patches that survived the directional filter
    squares_filtered = filter_squares_keep_lines(
        lines_filtered, square_size=square_size, angle_rotation=angle_degrees
    )

    # Discard small residual blobs
    if blob_max_size > 0:
        output = remove_small_objects(squares_filtered, max_size=blob_max_size)
    else:
        output = squares_filtered

    # Rotate back to original orientation
    output = rotate(output, angle=orig_angle_degrees,
                    resize=False, preserve_range=True).astype(bool)

    return output


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

    Notes
    -----
    **Overlap and step** — consecutive segments are offset by
    ``step = line_spacing - overlap`` samples.  With ``overlap=0`` this
    reduces to non-overlapping chunking; with ``overlap=line_spacing-1``
    every possible window of length ``line_spacing`` is evaluated.

    **Memory efficiency** — segments are built with
    ``np.lib.stride_tricks.as_strided``, which creates a 2-D view into the
    original array without copying data.  The array is made contiguous with
    ``np.ascontiguousarray`` before the FFT so NumPy can operate efficiently.

    **Hann window** — each segment is multiplied by a Hann window before the
    FFT to reduce spectral leakage.  The window energy is normalised out via
    ``sum(window²)`` so amplitudes are comparable across segment lengths.

    **Odd vs even segment length** — for even ``line_spacing`` the Nyquist bin
    is real and unique; for odd ``line_spacing`` there is no exact Nyquist.  In
    both cases interior bins are doubled to account for the negative-frequency
    mirror, and the final bin is dropped, giving a consistent output width of
    ``line_spacing // 2``.
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
    # as_strided creates a (n_segments, line_spacing) view without copying data.
    # The first stride steps 'step' samples along the signal;
    # the second stride steps 1 sample within each segment.
    shape   = (n_segments, line_spacing)
    strides = (signal.strides[0] * step, signal.strides[0])
    segments = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

    # Make a contiguous copy before any arithmetic — as_strided views are not
    # safe to write to and may confuse the FFT backend
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
        freqs = np.fft.rfftfreq(line_spacing, d=resolution)
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




## ── High-level pipeline functions ───────────────────────────────────────────
# These six functions encapsulate the main processing stages of the trackline
# detection and uncertainty estimation pipeline.  Each function does one
# logical thing so the notebook can be written as a short, readable sequence
# of calls rather than a deeply nested loop.
#
# Pipeline order:
#   build_coverage_mask
#   → detect_trackline_positions
#   → assign_depths_to_lines
#   → build_uncertainty_raster  (calls process_line_pair internally)
#   → undo_rotation


def build_coverage_mask(bathy_data: dict) -> np.ndarray:
    """Build a binary coverage mask from a bathymetric raster dataset.

    Marks every pixel as ``True`` where valid depth data exists and ``False``
    where the no-data value (NDV) was recorded.  The mask captures *where*
    the sonar detected the seafloor, independent of the actual depth values.

    Parameters
    ----------
    bathy_data : dict
        Raster dictionary as returned by :func:`read_file`, with keys
        ``'data'`` (2-D depth array) and ``'metadata'`` (containing
        ``'ndv_value'``).

    Returns
    -------
    np.ndarray of bool
        Binary array of the same shape as ``bathy_data['data']``.
        ``True`` where depth values are valid; ``False`` where no-data.

    Notes
    -----
    Both NaN-encoded and sentinel-value NDV formats are supported:

    - If ``ndv_value`` is ``None`` or ``NaN``, the mask is ``~np.isnan(data)``.
    - Otherwise the mask is ``data != ndv_value``.
    """
    data = bathy_data['data']
    ndv  = bathy_data['metadata']['ndv_value']
    if np.isnan(ndv):
        return ~np.isnan(data).astype(bool)
    else:
        return (data != ndv).astype(bool)


def detect_trackline_positions(
    angle_mask: np.ndarray,
    bathy_binary: np.ndarray,
    angle: float,
    min_line_factor: float = 0.1,
    band_width: int = 40,
    min_blob_pct: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Detect survey trackline column positions for a single dominant angle.

    Rotates the angle-specific binary mask so that lines of the target
    orientation become vertical, finds their column positions from the
    column-sum projection, then returns a clean single-pixel-wide line mask
    with noise blobs removed.

    Parameters
    ----------
    angle_mask : np.ndarray of bool
        Binary mask of pixels whose local orientation matches the target
        angle (produced by thresholding :func:`compute_local_orientation`
        output).
    bathy_binary : np.ndarray of bool
        Full binary coverage mask (all valid pixels).  Used to harvest the
        full line row extent rather than just the thinned skeleton pixels.
    angle : float
        Dominant trackline angle in degrees, as returned by
        :func:`compute_dominant_angle_radon_transform`.
    min_line_factor : float, optional
        Minimum peak height as a fraction of the tallest projection peak.
        Peaks shorter than ``min_line_factor * max(projection)`` are ignored.
        Default is ``0.1`` (10 % of the maximum projection height).
    band_width : int, optional
        Minimum horizontal separation between adjacent peaks in pixels.
        Should approximate the expected inter-line spacing so that one wide
        line is not split into two peaks.  Default is ``40``.
    min_blob_pct : float, optional
        After the line mask is built, connected components smaller than
        ``min_blob_pct * largest_component`` are discarded as noise.
        Default is ``0.1`` (10 % of the largest component area).

    Returns
    -------
    peaks : np.ndarray of int
        Column indices of detected tracklines in the rotated frame, sorted
        in ascending order.
    restored_mask : np.ndarray of bool
        Single-pixel-wide vertical line mask in the rotated frame, with
        noise blobs removed.  Same spatial extent as the rotated input images.
    angle_to_rotate : float
        Forward rotation angle applied (``90 - angle``).  Pass this to
        :func:`undo_rotation` to recover the original orientation.

    Raises
    ------
    ValueError
        If no valid bathy pixels exist inside the projection band around a
        detected peak.  Try increasing ``band_width`` or lowering
        ``min_line_factor``.

    Notes
    -----
    **Rotation convention** — the forward rotation aligns tracklines to
    vertical: ``angle_to_rotate = 90 - angle``.  The inverse is
    ``-angle_to_rotate``.

    **Peak detection** — ``scipy.signal.find_peaks`` runs on the column sum
    of the rotated ``angle_mask``.  Both ``height`` and ``prominence`` are
    set to ``min_line_factor * max(projection)`` to reject weak/isolated
    columns while keeping the main tracklines.

    **Line mask construction** — rather than using skeleton pixels directly,
    each line is built from *all valid bathy pixels* inside the peak's
    left/right base window (from ``find_peaks`` ``properties``), then
    stamped at the exact peak column.  This produces a denser, more robust
    line extent that matches the original survey coverage.
    """
    angle_to_rotate = 90.0 - angle

    rotated_mask  = rotate(angle_mask.astype(float),
                           angle=angle_to_rotate, resize=True,
                           preserve_range=True, order=0).astype(bool)
    rotated_bathy = rotate(bathy_binary.astype(float),
                           angle=angle_to_rotate, resize=True,
                           preserve_range=True, order=0).astype(bool)

    # Column projection — each column's sum counts skeleton pixels belonging
    # to a line at that x-position in the rotated frame
    projection      = np.sum(rotated_mask, axis=0)
    min_line_length = int(np.max(projection) * min_line_factor)

    peaks, properties = find_peaks(
        projection,
        height=min_line_length,
        distance=band_width,
        prominence=min_line_length,
    )

    # Stamp each peak column with all valid bathy rows inside its base window
    # so every detected line is exactly 1 pixel wide at the peak position
    line_mask = np.zeros_like(rotated_bathy, dtype=bool)
    for idx, peak in enumerate(peaks):
        min_col     = max(0, properties['left_bases'][idx])
        max_col     = min(rotated_mask.shape[1], properties['right_bases'][idx])
        valid_pts   = np.argwhere(rotated_bathy[:, min_col:max_col])
        if len(valid_pts) == 0:
            raise ValueError(
                f"No valid bathy pixels in the band around peak column {peak}. "
                "Try increasing band_width or lowering min_line_factor."
            )
        valid_rows = np.sort(valid_pts[:, 0])
        line_mask[valid_rows, peak] = True

    # Remove components smaller than min_blob_pct of the largest component
    label_img   = label(line_mask)
    props       = regionprops(label_img)
    blob_sizes  = np.array([r.area for r in props])
    noise_limit = int(blob_sizes.max() * min_blob_pct)
    restored_mask = remove_small_objects(line_mask, max_size=noise_limit)

    return np.sort(peaks), restored_mask, angle_to_rotate


def assign_depths_to_lines(
    restored_mask: np.ndarray,
    bathy_binary: np.ndarray,
    bathy_data: dict,
    angle_to_rotate: float,
) -> np.ndarray:
    """Assign real depth values to detected trackline pixels via nearest-neighbour lookup.

    Each pixel in ``restored_mask`` (clean line mask in the rotated frame) is
    assigned the depth value of its nearest valid source pixel from the rotated
    bathymetric raster, found using a ``cKDTree`` search.

    Parameters
    ----------
    restored_mask : np.ndarray of bool
        Clean single-pixel-wide line mask in the rotated frame, as returned
        by :func:`detect_trackline_positions`.
    bathy_binary : np.ndarray of bool
        Binary coverage mask of valid depth pixels in the *original*
        (unrotated) orientation, e.g. from :func:`build_coverage_mask`.
        Used to identify valid source pixels for the depth lookup.
    bathy_data : dict
        Raster dictionary as returned by :func:`read_file`.  The raw depth
        array is read from ``bathy_data['data']``.
    angle_to_rotate : float
        Forward rotation angle (``90 - dominant_angle``) as returned by
        :func:`detect_trackline_positions`.  Applied to both ``bathy_binary``
        and the depth array to bring them into the same rotated frame as
        ``restored_mask``.

    Returns
    -------
    current_depth_raster : np.ndarray of float
        2-D depth array of the same shape as ``restored_mask``.  Pixels that
        belong to detected tracklines carry their assigned depth value; all
        other pixels are ``NaN``.

    Notes
    -----
    **Why nearest-neighbour?** — After an integer-nearest rotation (``order=0``)
    the line mask pixels may not coincide exactly with the rotated source pixels
    due to rounding.  A ``cKDTree`` query on source coordinates handles any
    sub-pixel misalignment cleanly without introducing interpolation artefacts.

    **Source pixel filtering** — only pixels where the *rotated* ``bathy_binary``
    is ``True`` are used as lookup candidates, so no-data pixels can never be
    assigned to a trackline.
    """
    rotated_bathy = rotate(bathy_binary.astype(float),
                           angle=angle_to_rotate, resize=True,
                           preserve_range=True, order=0).astype(bool)
    rotated_depth = rotate(bathy_data['data'].astype(float),
                           angle=angle_to_rotate, resize=True,
                           preserve_range=True, order=0)

    source_pts    = np.argwhere(rotated_bathy).astype(int)
    source_values = rotated_depth[source_pts[:, 0], source_pts[:, 1]]
    dest_pts      = np.argwhere(restored_mask).astype(int)

    tree = cKDTree(source_pts)
    _, nearest_idx = tree.query(dest_pts)

    current_depth_raster = np.full(restored_mask.shape, np.nan, dtype=float)
    current_depth_raster[dest_pts[:, 0], dest_pts[:, 1]] = source_values[nearest_idx]

    return current_depth_raster


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

    Notes
    -----
    **DC removal** — subtracting the mean before the FFT prevents the DC bin
    from dominating the PSD with energy proportional to ``mean_depth²``
    (order 10³ m² for typical ocean depths).  After removal the spectrum
    reflects only the roughness relevant to interpolation uncertainty.

    **Segment length = linespan** — using the inter-line distance as the
    analysis window ensures each spectral estimate covers exactly the spatial
    scale of the region being interpolated.  Segments overlap by 1 sample
    (``overlap=1`` default of :func:`compute_uncertainty`), so adjacent
    segments share one endpoint and together tile the full shared row range.

    **RMS collapse** — integrating the PSD over all frequency bins:
    ``sigma [m] = sqrt( sum(PSD [m²/Hz] * delta_f [Hz]) )``
    converts from a spectral density to a root-mean-square amplitude in
    metres, making the uncertainty directly comparable to depth values.
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

    Notes
    -----
    **Placement logic** — each segment's uncertainty is a single scalar
    (metres) that fills a rectangular block
    ``[seg_row_start:seg_row_end, col_start:col_end]`` uniformly.  The block
    height equals ``linespan`` and the width spans the full inter-line gap.

    **Overlap-1 stepping** — consecutive segment blocks are offset by
    ``step = linespan - 1`` rows, matching the ``overlap=1`` default used
    inside :func:`compute_uncertainty`.  Adjacent blocks therefore share one
    row, providing a smooth transition between segments.

    **Skipped pairs** — if two adjacent lines share no row overlap, or the
    overlap is too short for a single segment, :func:`process_line_pair`
    returns ``None`` and the gap remains ``NaN``.
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


def undo_rotation(
    raster: np.ndarray,
    angle_to_rotate: float,
    original_shape: tuple[int, int],
) -> np.ndarray:
    """Rotate a raster back to its original orientation and crop to original size.

    Applies the inverse of the forward rotation used in
    :func:`detect_trackline_positions`, then crops the padded output canvas
    back to the original raster dimensions.

    Parameters
    ----------
    raster : np.ndarray
        2-D array in the *rotated* frame (e.g. ``uncertainty_raster`` from
        :func:`build_uncertainty_raster`).
    angle_to_rotate : float
        The *forward* rotation angle that was applied (``90 - dominant_angle``),
        as returned by :func:`detect_trackline_positions`.  This function
        applies ``-angle_to_rotate`` to undo it.
    original_shape : tuple of int
        ``(n_rows, n_cols)`` of the original pre-rotation raster.  Used to
        crop the over-sized canvas back to the correct dimensions.

    Returns
    -------
    np.ndarray of float
        Array cropped to ``original_shape`` in the original orientation.
        Pixels in the corners (outside the original data extent) are ``NaN``.

    Notes
    -----
    **Why resize=True on both rotations?** — ``skimage.transform.rotate`` with
    ``resize=True`` expands the output canvas so no corner pixels are clipped.
    The result is larger than both the input and the target, but the content
    is centred, so the original region is recovered with a symmetric crop:
    ``offset = (large_dim - orig_dim) // 2``.

    **Bilinear vs nearest-neighbour** — the forward rotation uses ``order=0``
    (nearest-neighbour) to keep binary masks crisp.  The inverse rotation here
    uses ``order=1`` (bilinear) because the uncertainty values are continuous
    and benefit from smooth interpolation across the pixel grid.
    """
    rotated_back = rotate(
        raster,
        angle=-angle_to_rotate,
        resize=True,
        preserve_range=True,
        order=1,      # bilinear for continuous uncertainty values
        cval=np.nan,  # fill newly exposed corners with NaN, not 0
    )

    orig_rows, orig_cols   = original_shape
    large_rows, large_cols = rotated_back.shape
    row_offset = (large_rows - orig_rows) // 2
    col_offset = (large_cols - orig_cols) // 2

    return rotated_back[
        row_offset : row_offset + orig_rows,
        col_offset : col_offset + orig_cols,
    ]


## Unused functions from earlier experiments — may be useful for future feature extraction or uncertainty estimation methods.
# def raster_to_scatter(raster_data: dict,
#                      remove_ndv: bool = True,
#                      geotransform: tuple | None = None,
#                      plot_data: bool = False) -> dict:
#     """Convert a 2D raster into an array of (x, y, value) points.

#     Transforms pixel locations to map coordinates using the GDAL
#     geotransform, placing each point at the **centre** of its pixel.

#     Parameters
#     ----------
#     raster_data : dict
#         Raster dictionary as returned by :func:`read_file`, with keys
#         ``'data'``, ``'metadata'``, ``'filename'``, and ``'filetype'``.
#     remove_ndv : bool, optional
#         If ``True``, removes rows where value equals the no-data value.
#         Default is ``True``.
#     geotransform : tuple, optional
#         Optional geotransform to use instead of the one in ``raster_data``.
#         Must be a 6-element GDAL-style geotransform. Default is ``None`` (use
#         geotransform from ``raster_data``).
#     plot_data : bool, optional
#         If ``True``, visualises the resulting point cloud. Default is ``False``.

#     Returns
#     -------
#     dict
#         Copy of ``raster_data`` with ``'filetype'`` set to ``'points'`` and
#         ``'data'`` replaced by an ``(n, 3)`` array of ``[x, y, value]`` values.

#     Raises
#     ------
#     NotImplementedError
#         If the raster contains non-zero rotation terms in the geotransform.

#     Notes
#     -----
#     The GDAL geotransform is a six-element tuple interpreted as follows:

#     - ``GT[0]`` — X coordinate of the upper-left pixel corner (Easting).
#     - ``GT[1]`` — Pixel width in map units (X resolution).
#     - ``GT[2]`` — Row rotation (must be 0 for this function).
#     - ``GT[3]`` — Y coordinate of the upper-left pixel corner (Northing).
#     - ``GT[4]`` — Column rotation (must be 0 for this function).
#     - ``GT[5]`` — Pixel height in map units (Y resolution, negative for
#       north-up images because pixel rows increase downward while map Y
#       increases upward).
#     """
#     if geotransform is None:
#         geotransform = raster_data["metadata"]["geotransform"]

#     # Guard against rotated rasters — the simple affine below assumes none.
#     if geotransform[2] != 0 or geotransform[4] != 0:
#         raise NotImplementedError(
#             "Rotated rasters are not supported. "
#             f"GT[2]={geotransform[2]}, GT[4]={geotransform[4]}"
#         )

#     elev = raster_data["data"].copy()
#     rows, cols = elev.shape

#     # Build 1-D coordinate arrays placed at pixel centres.
#     # Adding half the step size shifts from the upper-left corner to the
#     # centre of each pixel. 
#     x_coords = geotransform[0] + np.arange(cols) * geotransform[1] + geotransform[1] / 2
#     y_coords = geotransform[3] + np.arange(rows) * geotransform[5] + geotransform[5] / 2


#     # Expand to full 2-D grids and flatten into an (n, 3) point array.
#     x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
#     points = np.column_stack((x_mesh.ravel(), y_mesh.ravel(), elev.ravel()))

#     # Remove no-data points.
#     if remove_ndv:
#         ndv = raster_data["metadata"].get("ndv_value")
#         if ndv is None:
#             pass  # No no-data value defined — keep everything.
#         elif np.isnan(ndv):
#             points = points[~np.isnan(points[:, 2])]
#         else:
#             points = points[points[:, 2] != ndv]

#     # Build a lightweight output dict rather than deep-copying the input.
#     points_dataset = {
#         "data": points,
#         "filetype": "points",
#         "metadata": raster_data["metadata"],
#         "filename": raster_data.get("filename", ""),
#     }

#     if plot_data:
#         show_depth(
#             points_dataset,
#             title=f"{raster_data['filename']} as points",
#             cmap="terrain",
#         )

#     return points_dataset

# def scatter_to_raster(raster_data: dict,
#                      plot_data: bool = False) -> dict:
#     """Convert an array of (x, y, depth) points back into a 2D raster.

#     Parameters
#     ----------
#     raster_data : dict
#         Dictionary containing the point data to be rasterised. 
#         Must have keys ``'data'`` (an (n, 3) array of (x, y, value) points)
#     plot_data : bool, optional
#         If ``True``, visualises the resulting raster. Default is ``False``.

#     Returns
#     -------
#     dict
#         Data dict with ``'data'`` replaced by the rasterised values.

#     Raises
#     ------
#     NotImplementedError
#         If the raster contains non-zero rotation terms in the geotransform.
    
#     Notes
#     -----
#     Geotransform details:
#     GT[0]	X-coordinate of the upper-left corner(Easting/Longitude).
#     GT[1]	Pixel width (X-resolution, the size of one pixel in map units).
#     GT[2]	Row rotation 
#     GT[3]	Y-coordinate of the upper-left corner (Northing/Latitude).
#     GT[4]	Column rotation
#     GT[5]	Pixel height (Y-resolution; must be negative for standard "north up" images 
#             because geospatial Y increases going upwards while pixel lines increase going downwards).

#     """
#     raster = np.full(base_raster['data'].shape, fill_value=np.nan)
#     geotransform = base_raster['metadata']['geotransform']
#     resolution = base_raster['metadata']['resolution']
#     rows, cols = base_raster['data'].shape


#     if geotransform[2] != 0 or geotransform[4] != 0:
#         raise NotImplementedError(
#             "Rotated rasters are not supported. "
#             f"GT[2]={geotransform[2]}, GT[4]={geotransform[4]}"
#         )

#     x = points[:, 0].astype(float)
#     y = points[:, 1].astype(float)
#     # depth = points[:, 2].astype(float)

#     # Invert the pixel-centre formula from raster_to_points:
#     #   x = GT[0] + col * GT[1] + GT[1]/2   →   col = (x - GT[0] - GT[1]/2) / GT[1]
#     #   y = GT[3] + row * GT[5] + GT[5]/2   →   row = (y - GT[3] - GT[5]/2) / GT[5]
#     col_indices = np.round((x - geotransform[0] - geotransform[1] / 2) / geotransform[1]).astype(int)
#     row_indices = np.round((y - geotransform[3] - geotransform[5] / 2) / geotransform[5]).astype(int)


#     # Discard points that fall outside the raster bounds.
#     valid = (
#         (row_indices >= 0) & (row_indices < rows) &
#         (col_indices >= 0) & (col_indices < cols)
#     )
#     row_indices = row_indices[valid]
#     col_indices = col_indices[valid]
#     # value = value[valid]

#     # assign value values to the raster grid
#     # raster[row_indices, col_indices] = value

#     plt.figure(figsize=(10, 10))
#     plt.imshow(raster, cmap='terrain', aspect='equal')
#     # plt.imshow(raster, cmap='terrain', aspect='equal')
#     # plt.colorbar(label='Value', shrink=0.5)
#     plt.title('Rasterised Points')
#     plt.xlabel('Column Index')
#     plt.ylabel('Row Index')
#     plt.show()

#     new_data_dict = deepcopy(base_raster)
#     new_data_dict['data'] = raster
#     new_data_dict['metadata']['ndv_value'] = np.nan
#     new_data_dict['filetype'] = 'raster'

#     if plot_data:
#         show_depth(new_data_dict)
    
#     return new_data_dict




# def get_linearity_ratio(data: np.ndarray, k: int = 10) -> np.ndarray:
#     """Compute a PCA-based linearity score for each point from its k-nearest neighbours.

#     A score near 1 indicates a strongly linear local neighbourhood (one dominant
#     direction); a score near 0 indicates an isotropic neighbourhood.

#     Parameters
#     ----------
#     data : np.ndarray
#         Array of shape ``(n_points, 2)`` containing ``(x, y)`` coordinates.
#     k : int, optional
#         Number of nearest neighbours to use per point. Default is 10.

#     Returns
#     -------
#     np.ndarray
#         Array of shape ``(n_points,)`` with linearity scores in ``[0, 1]``.
#     """
#     nbrs = NearestNeighbors(n_neighbors=k).fit(data)
#     _, indices = nbrs.kneighbors(data)

#     linearity_ratios = []
#     for i, idx in enumerate(indices):
#         neighbor_points = data[idx]
#         cov = np.cov(neighbor_points.T)
#         eigenvalues = np.linalg.eigvalsh(cov)  # sorted ascending

#         l2, l1 = eigenvalues  # l1 is the dominant eigenvalue
#         score = (l1 - l2) / l1 if l1 > 1e-8 else 0
#         linearity_ratios.append(score)

#     return np.array(linearity_ratios)


# def compute_local_angles_chaining(X_linear: np.ndarray,
#                                    num_points_per_chain: int = 20,
#                                    k_neighbors: int = 50) -> list[float]:
#     """Compute local dominant angles by chaining nearest neighbours and applying PCA.

#     For each point, a greedy chain of nearby unvisited neighbours is built and
#     PCA is applied to find the principal direction. This captures local linear
#     structure even in curved or multi-directional point sets.

#     Parameters
#     ----------
#     X_linear : np.ndarray
#         Array of shape ``(n_points, 2)`` or ``(n_points, 3)``. Only the first
#         two columns (x, y) are used.
#     num_points_per_chain : int, optional
#         Number of points to include in each local chain. Default is 20.
#     k_neighbors : int, optional
#         Number of nearest neighbours to query when extending the chain.
#         Higher values reduce the chance of getting stuck. Default is 50.

#     Returns
#     -------
#     list of float
#         Angle in radians for the dominant direction of each point's local chain
#         (one value per input point).
#     """
#     X_linear = X_linear[:, :2]

#     tree = KDTree(X_linear)
#     angle_scores = []
#     pca = PCA(n_components=2)

#     for i in tqdm(range(len(X_linear))):
#         chain = [i]
#         visited = {i}

#         # Greedily extend the chain to the nearest unvisited neighbour
#         for _ in range(num_points_per_chain - 1):
#             current_pt = X_linear[chain[-1]].reshape(1, -1)
#             distances, indices = tree.query(current_pt, k=k_neighbors)

#             found_next = False
#             for neighbor_idx in indices[0]:
#                 if neighbor_idx not in visited:
#                     chain.append(neighbor_idx)
#                     visited.add(neighbor_idx)
#                     found_next = True
#                     break

#             if not found_next:
#                 break

#         # Fit PCA to the chain and extract the angle of the principal direction
#         centered = X_linear[chain] - X_linear[chain].mean(axis=0)
#         cov = np.cov(centered.T)
#         eigvals, eigvecs = np.linalg.eigh(cov)
#         direction = eigvecs[:, np.argmax(eigvals)]
#         angle_radians = np.arctan2(direction[1], direction[0])
#         angle_scores.append(angle_radians)

#     return angle_scores


# def detect_intersections(skeleton: np.ndarray,
#                          min_neighbours: int = 3) -> np.ndarray:
#     """Detect intersection points in a skeletonised image.

#     Counts the number of non-zero neighbours for each foreground pixel
#     using a 3x3 connectivity kernel. Pixels with 3+ neighbours are
#     junctions where two or more lines cross.

#     Parameters
#     ----------
#     skeleton : np.ndarray
#         Binary skeletonised image (1-pixel wide lines).
#     min_neighbours : int, optional
#         Minimum neighbour count to classify as intersection.
#         Default is ``3``.  Use ``4`` for stricter detection
#         (e.g. only full crossings, not T-junctions).

#     Returns
#     -------
#     coords : np.ndarray
#         array of ``[row, col]`` positions.


#     Notes
#     -----
#     The 3x3 kernel counts all 8-connected neighbours:
#         1  1  1
#         1  0  1
#         1  1  1

#     Pixels with 3 or more neighbors are returned

#     """
#     binary = (skeleton > 0).astype(np.uint8)

#     # 8-connectivity kernel — centre is 0 so we count neighbours only
#     kernel = np.array([[1, 1, 1],
#                        [1, 0, 1],
#                        [1, 1, 1]], dtype=np.uint8)

#     neighbour_count = convolve(binary, kernel, mode="constant", cval=0)

#     # Only foreground pixels with enough neighbours are intersections
#     intersection_mask = (binary == 1) & (neighbour_count >= min_neighbours)

#     rows, cols = np.where(intersection_mask)
#     coords = np.column_stack((rows, cols))

#     return coords




# def place_kernel(canvas: np.ndarray,
#                  kernel: np.ndarray,
#                  row: int,
#                  col: int,
#                  value: bool = True,
#                  overwrite: bool = False) -> np.ndarray:
#     """Place a rotated kernel onto a canvas at a specified location.

#     The kernel is rotated about its own centre, then stamped onto the
#     canvas so that the kernel centre aligns with ``(row, col)``.

#     Parameters
#     ----------
#     canvas : np.ndarray
#         2D array to draw on. Modified in-place unless you pass a copy.
#     kernel : np.ndarray
#         2D binary array defining the shape to place.
#     row, col : int
#         Canvas coordinates where the kernel centre will land.
#     value : bool, optional
#         Value to write where the kernel is non-zero. Default is ``True``.
#     overwrite : bool, optional
#         If ``True``, replaces existing canvas values. If ``False``
#         (default), uses logical OR so existing features are preserved.

#     Returns
#     -------
#     np.ndarray
#         The canvas with the kernel placed.
#     """

#     kh, kw = kernel.shape

#     # Compute where the kernel sits on the canvas, centred on (row, col)
#     r_start = row - kh // 2
#     c_start = col - kw // 2
#     r_end = r_start + kh
#     c_end = c_start + kw

#     # Clip to canvas bounds — handle kernels that hang off the edge
#     kr_start = max(0, -r_start)
#     kc_start = max(0, -c_start)
#     kr_end = kh - max(0, r_end - canvas.shape[0])
#     kc_end = kw - max(0, c_end - canvas.shape[1])

#     r_start = max(0, r_start)
#     c_start = max(0, c_start)
#     r_end = min(canvas.shape[0], r_end)
#     c_end = min(canvas.shape[1], c_end)

#     # Stamp the kernel onto the canvas
#     kernel_slice = kernel[kr_start:kr_end, kc_start:kc_end]
#     canvas_slice = canvas[r_start:r_end, c_start:c_end]

#     if overwrite:
#         canvas_slice[kernel_slice > 0] = value
#     else:
#         canvas_slice[kernel_slice > 0] |= value

#     return canvas


# def make_line_kernel(length: int, angle_deg: float, width: int = 3) -> np.ndarray:
#     """Create a rotated line structuring element.

#     Builds a horizontal line kernel and rotates it to the desired
#     angle. The result is a binary array suitable for morphological
#     operations.

#     Parameters
#     ----------
#     length : int
#         Length of the line in pixels.
#     angle_deg : float
#         Angle in degrees, measured counterclockwise from horizontal.
#         0° = horizontal, 90° = vertical.
#     width : int, optional
#         Thickness of the line in pixels. Default is ``3``.

#     Returns
#     -------
#     np.ndarray
#         Binary structuring element with the line at the requested angle.
#     """
#     # Start with a horizontal line in an oversized canvas so rotation
#     # doesn't clip the corners.
#     canvas_size = length + width
#     kernel = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

#     # Place a horizontal line through the centre
#     mid = canvas_size // 2
#     half_w = width // 2
#     kernel[mid - half_w : mid + half_w + 1,
#            (canvas_size - length) // 2 : (canvas_size + length) // 2] = 1

#     # Rotate — using order=0 (nearest-neighbour) keeps it binary
#     rotated = rotate(kernel, angle_deg, resize=True, order=0)

#     # Crop to tight bounding box
#     rows, cols = np.where(rotated > 0)
#     cropped = rotated[rows.min():rows.max() + 1,
#                       cols.min():cols.max() + 1]

#     return (cropped > 0).astype(bool)