
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from skimage.transform import radon
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
from collections import deque

### WARNING: Needs extensive refactoring and testing
### Use orientation_via_Radon for the meantime

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
    angle_profile : np.ndarray
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




def label_points_by_angle(x,
                          y,
                          theta_degs,
                          bin_width= 1.0,
                          tol=1.5,
                          min_peak_prominence= 5):
    """Assign each point to its closest trackline family across all dominant angles.

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
    """Assign each point a local angle orientation using probabilistic line fitting.

    Runs a multi-vote ensemble: the core algorithm is repeated until every
    point has been visited at least ``n_times``, and the final label is the
    angle family that received the most votes (mode).  Points that are always
    near a trackline boundary get their correct family reinforced across
    independent random probes, suppressing one-off mislabels.

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

    Notes
    -----
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
    """Two-tier probabilistic line labeling without BFS connectivity.

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
