# NBS interpolation_uncertainty

## What interpolation_uncertainty does

Fully surveying the seafloor at the resolution required for all navigational and scientific needs is often prohibitively costly. In practice, bathymetric surveys collect direct depth measurements along sparse survey lines and use interpolation to represent the seafloor between them. Like all scientific measurements, these depths require an associated uncertainty estimate — but for interpolated regions, that uncertainty must account for how much the true seafloor may depart from the interpolated value.

The core challenge is that we have no direct measurements in the very regions we are trying to characterize. This library addresses that challenge by exploiting the spatial structure of nearby known bathymetry. Under an assumption of spatial isotropy — that the statistical character of the seafloor is consistent regardless of direction — the variance observed in sampled survey lines can be used to estimate the likely variance at different spatial scales across the interpolated gaps.

interpolation_uncertainty implements and compares two families of methods for making this estimate: Spectral Energy and Spatial Distrubution

## Installation

### Quick install (pip)

If you just want to use the library without modifying it:

```bash
pip install git+https://github.com/noaa-ocs-hydrography/interpolation_uncertainty.git
```

> **Note:** This project depends on GDAL, which can be difficult to install via pip alone.
> If you run into issues, use the conda workflow below instead.

### For conda users / collaborators

This is the recommended approach — it handles GDAL and other geospatial C dependencies
reliably via conda-forge.

```bash
git clone https://github.com/noaa-ocs-hydrography/interpolation_uncertainty.git
cd interpolation_uncertainty
conda env create -f environment.yml
conda activate interpolation_uncertainty
```

The environment includes an editable install (`pip install -e .`) of the package itself,
so any changes you make to `src/` are immediately available.

To update your environment after a `git pull`:

```bash
conda env update -f environment.yml --prune
```

## Getting started

See the notebooks in `notebooks/` for usage examples. The primary working notebook is
`uncertainty_sim_nbs_dataset.ipynb`.

```python
from interpolation_uncertainty import (
    read_file, show_depth,
    build_coverage_mask,
    compute_dominant_angle_radon_transform, compute_local_orientation,
    detect_trackline_positions, assign_depths_to_lines,
    build_uncertainty_raster, undo_rotation,
)
from skimage.morphology import opening, thin, disk

# 1. Load bathymetric raster
bathy_data = read_file("data/raster/H13060_MB_4m_MLLW_3of3.bag", verbose=True)
show_depth(bathy_data, cmap='viridis')

# 2. Build binary coverage mask
bathy_binary = build_coverage_mask(bathy_data)

# 3-4. Morphological filtering + skeletonisation to isolate tracklines
bathy_lines = bathy_binary & ~opening(bathy_binary, disk(5))
bathy_thin = thin(bathy_lines) > 0

# 5. Radon transform to find dominant line orientations
dominant = compute_dominant_angle_radon_transform(bathy_thin, circle=False)

# 6. PCA-based local angle estimation
local_orientations = compute_local_orientation(bathy_thin, line_length=50, num_neighbors=20)

# 7-14. For each dominant angle: detect lines, assign depths, compute uncertainty
for angle in dominant['peaks']:
    # Extract pixels near this angle
    low, high = (angle - 5) % 180, (angle + 5) % 180
    if low < high:
        angle_mask = (local_orientations >= low) & (local_orientations < high)
    else:
        angle_mask = (local_orientations >= low) | (local_orientations < high)

    # Detect trackline positions and assign depth values
    peaks, restored_mask, angle_to_rotate = detect_trackline_positions(
        angle_mask, bathy_binary, angle,
    )
    depth_raster = assign_depths_to_lines(
        restored_mask, bathy_binary, bathy_data, angle_to_rotate,
    )

    # Compute spectral uncertainty and rotate back to original orientation
    uncertainty = build_uncertainty_raster(
        peaks, restored_mask, depth_raster, bathy_data['metadata']['resolution'],
    )
    result = undo_rotation(uncertainty, angle_to_rotate, bathy_binary.shape)
```

### Package structure

```
interpolation_uncertainty
├── io              — read_file (GDAL-backed raster/BAG I/O)
├── visualization   — show_depth (plotting)
├── preprocessing
│   ├── morphology  — build_coverage_mask, morphological filters
│   ├── orientation — Radon transform, PCA-based local angles
│   └── geometry    — point rotation, undo_rotation
├── methods         — compute_uncertainty (FFT/PSD spectral energy)
└── pipeline        — detect_trackline_positions, assign_depths_to_lines,
                      process_line_pair, build_uncertainty_raster
```

## How to cite interpolation_uncertainty

NOAA - NOS
National Bathymetric Source Project
Link: https://nauticalcharts.noaa.gov/learn/nbs.html
