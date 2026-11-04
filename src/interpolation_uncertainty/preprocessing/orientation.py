import numpy as np


from interpolation_uncertainty.orientation.orientation_via_PCA import orientation_via_PCA
from interpolation_uncertainty.orientation.orientation_via_Radon import orientation_via_Radon
from interpolation_uncertainty.orientation.orientation_via_Probabilistic import (
    orientation_via_probabilistic_line as orientation_via_Probabilistic,
)



def compute_local_orientation(binary_image,
                              method:str ='pca',
                              **kwargs):
    """Compute local orientation of each point in a binary image.

    Uses various methods (PCA, Radon, Probabilistic) to build a label points
    with their local orientation.  Each foreground pixel carries its local
    orientation in degrees (wrapped to ``[0, 180)``); background pixels are ``NaN``.

    Parameters
    ----------
    binary_image : np.ndarray
        Binary image where ``True`` values represent points of interest.
    method : str, optional
        Method to use for computing local orientation.
        options: {'pca', 'radon', probabilistic'}. Default is ``'radon'``.


    Returns
    -------
    point_orientations : dict[int|str, np.ndarray]
        Dictionary mapping a detected angle to a 2D float array of
        the same shape as ``binary_image``.  Each foreground pixel carries
        its local orientation in degrees (wrapped to ``[0, 180)``);
        background pixels are ``NaN``.
        dict['combined'] contains the combined orientation field from all detected angles.
    angle_info : np.ndarray
        generic variable that may contain additional information about the detected angles,
        depending on the method used.
        can be used to bypass recomputation of the orientation field if the same image is
        processed multiple times with the same method.
    """

    if method == 'pca':
        point_orientations, angle_info = orientation_via_PCA(binary_image,
                                            angle_info=kwargs.get('angle_info', None),
                                            line_length=kwargs.get('line_length', 20),
                                            num_neighbors=kwargs.get('num_neighbors', 50))
    elif method == 'radon':
        point_orientations, angle_info = orientation_via_Radon(binary_image,
                                                             angle_info=kwargs.get('angle_info', None),
                                                             angular_tolerance=kwargs.get('angular_tolerance', 5),
                                                             width_tolerance=kwargs.get('width_tolerance', 5),
                                                             min_intensity=kwargs.get('min_intensity', 0.01),
                                                             do_offset=kwargs.get('do_offset', False),
                                                             top_angles=kwargs.get('top_angles', 0),
                                                             zoom_factor=kwargs.get('zoom_factor', 2.0))
    elif method == 'probabilistic':
        point_orientations, angle_info = orientation_via_Probabilistic(binary_image,
                                                                       angle_info=kwargs.get('angle_info', None),
                                                                       test_angles=kwargs.get('test_angles', None),
                                                                       circle=kwargs.get('circle', True),
                                                                       tol=kwargs.get('tol', 5),
                                                                       precomputed_theta=kwargs.get('precomputed_theta', None))
    else:
        raise ValueError(f"Invalid method: {method}. Must be one of the following: 'pca', 'radon', 'probabilistic'.")

    return point_orientations, angle_info






