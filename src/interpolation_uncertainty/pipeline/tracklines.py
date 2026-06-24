import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from skimage.transform import rotate
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops


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
