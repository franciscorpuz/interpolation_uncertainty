import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from skimage.transform import rotate
from matplotlib import pyplot as plt


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
    gt0 = x_min - pixel_width / 2
    gt3 = y_max - pixel_height / 2  # pixel_height is negative, so this adds

    # Compute the new grid dimensions
    n_cols = int(np.round((x_max - x_min) / pixel_width)) + 1
    n_rows = int(np.round((y_max - y_min) / abs(pixel_height))) + 1

    return (gt0, pixel_width, 0.0, gt3, 0.0, pixel_height), (n_rows, n_cols)


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

    subcluster_sizes = [np.sum(subcluster_labels == lbl)
                        for lbl in np.unique(subcluster_labels) if lbl != -1]

    # Only use the largest subclusters (above 90th percentile) for angle estimation
    min_cluster_size = np.percentile(np.array(subcluster_sizes), 90)

    subcluster_angles = []
    if plot_data:
        plt.figure(figsize=(15, 15))

    for lbl in np.unique(subcluster_labels):
        if lbl != -1 and np.sum(subcluster_labels == lbl) >= min_cluster_size:
            if plot_data:
                plt.scatter(points[subcluster_labels == lbl, 0],
                            points[subcluster_labels == lbl, 1],
                            s=10, label=f'Subcluster {lbl}')
            current_pts = points[subcluster_labels == lbl]
            pca = PCA(n_components=2)
            pca.fit(current_pts)
            direction = pca.components_[0]
            angle_radians = np.arctan2(direction[1], direction[0])
            angle_degrees = np.degrees(angle_radians) % 180
            subcluster_angles.append(angle_degrees)

    refined_cluster = rotate_points(points, 90 - np.mean(subcluster_angles))
    angle_correction = 90 - np.mean(subcluster_angles)

    return refined_cluster, angle_correction


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
