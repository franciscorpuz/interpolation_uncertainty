

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.spatial import KDTree

### WARNING: Needs extensive refactoring and testing
### Use orientation_via_Radon for the meantime

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