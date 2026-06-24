import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from skimage.morphology import (
    dilation, medial_axis, opening, footprint_rectangle, disk,
    remove_small_objects,
)
from skimage.draw import line
from skimage.transform import rotate


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
