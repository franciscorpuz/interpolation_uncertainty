# NBS interpolation_uncertainty

## Motivation

Fully surveying the seafloor at the resolution required for all navigational and scientific needs is often prohibitively costly. In practice, bathymetric surveys collect direct depth measurements along sparse survey lines and use interpolation to represent the seafloor between them. Like all scientific measurements, these depths require an associated uncertainty estimate — but for interpolated regions, that uncertainty must account for how much the true seafloor may depart from the interpolated value.

The core challenge is that we have no direct measurements in the very regions we are trying to characterize. This library addresses that challenge by exploiting the spatial structure of nearby known bathymetry. Under an assumption of spatial isotropy — that the statistical character of the seafloor is consistent regardless of direction — the variance observed in sampled survey lines can be used to estimate the likely variance at different spatial scales across the interpolated gaps.

## What interpolation_uncertainty does

interpolation_uncertainty implements and compares two families of methods for making this estimate:

Spectral methods analyze the power spectral density (PSD) of the seafloor along known survey lines. By cumulatively summing variance across spatial frequency bins — starting from the highest frequency and working toward scales matching the distance from a known measurement — these methods estimate the likely interpolation error at any given location.
Spatial methods use sliding-window statistics (standard deviation, min/max envelopes, Gaussian smoothing) computed directly from the depth field to characterize local variability.
To evaluate and develop these methods, the library simulates sparse surveying on full-coverage raster datasets: it subsamples every N-th column as a synthetic survey line, linearly interpolates across the gaps, and computes the true residual between the interpolated and actual seafloor. This residual serves as ground truth for validating each uncertainty estimate across different seafloor types (flat, rough, sloped).

## Installation

Documentation: https://interpolation_uncertainty.readthedocs.io/

**Requirements:** Python 3.10–3.12 and GDAL ≤ 3.11 (plus numpy, scipy, scikit-image,
scikit-learn, matplotlib, tqdm — installed automatically).

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
git clone https://github.com/noaa-ocs-hydrography/interpolation_uncertainty
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

`environment.yml` is generated from `pyproject.toml`. If you change the dependencies there,
regenerate it with:

```bash
python scripts/sync_env_yml.py
```

### Verify the install and run the tests

```bash
python -c "import interpolation_uncertainty"
pip install pytest
pytest
```

The tests use a small committed fixture (`tests/test_data/sample_tiff.tif`), so no
external data is required.

## Getting started

The notebooks in `notebooks/` walk through the full workflow on real surveys (start with
`sim_H13060.ipynb`). The sketch below shows the current API.

> **Note:** `uncertainty_sim_nbs_dataset.ipynb` and `sim_H13060_xtrack_verfication.ipynb` still use the
> older orientation API (`compute_dominant_angle_radon_transform`); the `sim_*.ipynb` notebooks and
> the example below use the current one.

```python
from skimage.morphology import opening, thin, disk

from interpolation_uncertainty import (
    read_file, show_depth,
    build_coverage_mask,
    compute_local_orientation,
    detect_trackline_positions, assign_depths_to_lines,
    build_uncertainty_raster, undo_rotation,
)

# 1. Load bathymetric raster (GDAL: .tif / .tiff / .bag)
bathy_data = read_file("data/raster/H13060_MB_4m_MLLW_3of3.bag", verbose=True)
show_depth(bathy_data, cmap='viridis')

# 2. Build binary coverage mask
bathy_binary = build_coverage_mask(bathy_data)

# 3-4. Morphological filtering + skeletonisation to isolate tracklines
bathy_lines = bathy_binary & ~opening(bathy_binary, disk(5))
bathy_thin = thin(bathy_lines) > 0

# 5-6. Detect dominant line orientations (Radon transform) and label each
#      skeleton pixel with the angle family it belongs to
orientations, angle_info = compute_local_orientation(bathy_thin, method='radon')
# orientations = {angle: float raster (NaN where not on that family), ..., 'combined': ...}

# 7-14. For each angle family: detect lines, assign depths, compute uncertainty
resolution = bathy_data['metadata']['resolution']
results = {}
for angle, family in orientations.items():
    if angle == 'combined':
        continue
    # `family` is the float raster for this angle (NaN off-family);
    # detect_trackline_positions binarises it internally
    # Detect trackline positions and assign depth values
    peaks, restored_mask, angle_to_rotate = detect_trackline_positions(
        family, bathy_binary, angle,
    )
    depth_raster = assign_depths_to_lines(
        restored_mask, bathy_binary, bathy_data, angle_to_rotate,
    )

    # Spectral uncertainty in the gaps between line pairs, rotated back
    uncertainty = build_uncertainty_raster(
        peaks, restored_mask, depth_raster, resolution, method='psd',
    )
    results[angle] = undo_rotation(uncertainty, angle_to_rotate, bathy_binary.shape)
```

**Choosing methods**

- `compute_local_orientation(..., method=)` accepts `'radon'` (recommended), `'pca'`
  (needs refactoring) and `'probabilistic'`. It returns `(point_orientations, angle_info)`;
  pass `angle_info` back in (Radon) to skip recomputing the sinogram.
- `build_uncertainty_raster(..., method=)` accepts the spectral methods `'psd'` (default),
  `'amplitude'`, `'psd_n'`, `'psd_lf'` and `'psd_df'`.

**Estimator registry.** Estimators live in a registry in `interpolation_uncertainty.methods`:

```python
from interpolation_uncertainty.methods import list_estimators, get_estimator, register

print(list_estimators())   # spectral: 'psd', 'amplitude', ...; statistical: 'SAR', 'SER', 'SMR'

@register('my_method')     # or drop a .py file with this decorator into src/interpolation_uncertainty/methods/
def my_estimator(signal, resolution, line_spacing, **kwargs):
    ...                    # return (energy[n_segments, n_freqs], frequencies[n_freqs])
```

> **Note:** `process_line_pair` / `build_uncertainty_raster` resolve `method=` through this registry
> and forward extra keyword arguments to the estimator (`**estimator_kwargs`). The statistical
> estimators (`SAR`/`SER`/`SMR`) operate on 2-D input and have not been verified with the pipeline yet.

### Package structure

```
interpolation_uncertainty
├── io              — read_file (GDAL-backed raster/BAG I/O)
├── visualization   — show_depth (plotting)
├── preprocessing
│   ├── morphology  — build_coverage_mask, morphological filters
│   ├── orientation — compute_local_orientation (dispatches to orientation/ by method=)
│   └── geometry    — rotate_points, update_geotransform, refine_cluster_rotation,
│                     undo_rotation, x_distance
├── orientation     — orientation_via_Radon (recommended), orientation_via_PCA,
│                     orientation_via_Probabilistic, orientation_utils
├── methods         — estimator registry: register, get_estimator, list_estimators
│   ├── spectral    — spectral_estimator (FFT/PSD: amplitude, psd, psd_n, psd_lf, psd_df)
│   └── statistical — statistical_estimator (sliding-window SAR, SER, SMR)
└── pipeline
    ├── tracklines  — detect_trackline_positions, assign_depths_to_lines
    └── uncertainty — process_line_pair, build_uncertainty_raster
```

Other top-level folders: `notebooks/` (worked examples), `tests/` (pytest suite and
fixture), `docs/` (Sphinx / Read the Docs), `scripts/` (`sync_env_yml.py`),
`config/` (`config.ini`) and `data/raster/` (sample surveys).

## How to cite interpolation_uncertainty

NOAA - NOS
National Bathymetric Source Project
Link: https://nauticalcharts.noaa.gov/learn/nbs.html
