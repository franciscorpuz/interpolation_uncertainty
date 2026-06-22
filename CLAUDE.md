# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a NOAA National Bathymetric Source (NBS) research library for estimating interpolation uncertainty in subsampled bathymetric (ocean depth) data. It reads raster/BAG/CSV files, processes them into strip-based matrices, and computes uncertainty using spectral (FFT-based) or spatial (rolling-window) methods.

The library is being actively updated and several overhauls are expected to improve the structure and organization of the functions and modules.

## Setup

Install via conda (recommended due to GDAL dependency):

```bash
conda create -n interpolation_uncertainty python=3.10
conda activate interpolation_uncertainty
conda install gdal=3.11 libgdal-hdf5 matplotlib scipy scikit-learn scikit-image tqdm jupyterlab
pip install -e .
```

Or install from git:
```bash
pip install git+https://github.com/noaa-ocs-hydrography/interpolation_uncertainty.git
```

The `gdal<3.12` constraint in pyproject.toml is strict — use the conda environment for consistent GDAL bindings.

## Common Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/readers/test_bathymetryFileReaders.py

# Run a single test
pytest tests/readers/test_bathymetryDataset.py::test_resolution

# Install in editable mode (after env setup)
pip install -e .
```

## Architecture

The library currently follows a layered pipeline (some major refactoring might be in order to improve the quality of the library):

```
File I/O (readers/)  →  Data Model (readers/)  →  Preprocessing (processors/)  →  Uncertainty Methods (methods/)
```

**`core/loadfile.py`** — Entry point. `load_file(filename)` dispatches to the correct reader based on file extension (`.tif/.tiff` → `RasterReader`, `.bag` → `RasterReader`, `.csv` → `CSVReader`, `.bps` → `BPSReader`).

**`readers/bathymetryFileReaders.py`** — GDAL-backed I/O. `RasterReader` reads raster/BAG files and returns a `RasterDataset`. `CSVReader` and `BPSReader` are stubs not yet implemented.

**`readers/bathymetryDataset.py`** — Data model. `BathymetryDataset` subclasses `numpy.ndarray` and carries metadata (filename, filetype, metadata dict). `RasterDataset` adds `resolution`, `min_val`, `max_val`, `ndv_value`, and `show_depth()`. Key: the NDV (no-data value) is stored but **not masked** in the array on load — callers must handle it.

**`processors/preProcessors.py`** — Two utilities used upstream of `RasterProcessor`:
- `remove_edge_ndv(raster_data)` — iteratively trims NDV border rows/cols
- `raster_to_points(raster_data)` — converts 2D raster to `(x, y, depth)` point array using the GDAL geotransform

**`processors/rasterProcessor.py`** — Core orchestrator. `RasterProcessor(linespacing_meters, multiple, max_multiple)` slices the raster into across-track strips, runs interpolation, computes residuals, then calls the selected uncertainty method. Key methods: `compute_interpolated_surface()`, `compute_residual_surface()`, `estimate_uncertainty()`.

**`methods/`** — Uncertainty implementations:
- `rasterMethods.py` — Registry dict `raster_methods` mapping string keys to classes. Spectral keys: `'amp_v1'`, `'psd_v1'`, `'amp_v2'`, `'psd_n'`, `'psd_lf'`, `'psd_df'`, `'spectrum'`. Spatial keys: `'spatial_std'`, `'spatial_diff'`, `'spatial_gaussian'`.
- `rasterSpectralMethods.py` — FFT/PSD-based uncertainty (Glen amplitude, Glen PSD, Elias variants). Uses Hann windowing via scipy.
- `rasterSpatialMethods.py` — Rolling-window spatial statistics (`computeSpatialStd`, `computeSpatialDiff`, `computeSpatialGaussian`). Uses `numpy.lib.stride_tricks.sliding_window_view` and `scipy.stats.genextreme`.

**`utils/utils.py`** — Post-processing analysis helpers (`case_1_bp`, `case_2_bp`) for comparing residuals against uncertainty estimates via boxplot statistics.

**`notebooks/helpers.py`** — Actively developed helper module used directly by notebooks via `%autoreload 2`. Contains the full geospatial + image processing pipeline including `read_file`, `raster_to_scatter`, `scatter_to_raster`, `show_depth`, morphological filtering utilities, and `compute_uncertainty` (see below). **This is the current source of truth for pipeline behaviour — `/src/` may lag behind.**

## Key Design Patterns

- `BathymetryDataset` and subclasses are `numpy.ndarray` subclasses — pass them to numpy operations directly; metadata is preserved via `__array_finalize__`.
- Method selection is string-based: pass a key from `raster_methods` to `RasterProcessor` rather than importing method classes directly.
- Test fixtures use a real raster file (`Bluetopo.tiff`) from the `data/` directory — tests require the data file to be present.
- `src/` layout: the package root is `src/interpolation_uncertainty/`. The VSCode config adds `src/` to Python analysis paths; for other editors set `PYTHONPATH=src` or install with `pip install -e .`.

## `compute_uncertainty` — current design (notebooks/helpers.py)

Signature:
```python
compute_uncertainty(signal, resolution, line_spacing, method='psd', overlap=1)
```

- `signal` — 1-D array of depth values along a single trackline.
- `resolution` — spatial sampling interval in metres/sample.
- `line_spacing` — segment length in samples. The signal is divided into
  overlapping segments of this length. Works correctly for both odd and even values.
- `overlap` — samples shared between consecutive segments (default 1).
  `overlap=0` → non-overlapping chunks; `overlap=line_spacing-1` → fully rolling window.
- `method` — one of `'amplitude'`, `'psd'`, `'psd_n'`, `'psd_lf'`, `'psd_df'`, `'spectrum'`.
  All variants except `'amplitude'` currently compute standard one-sided PSD.
  The non-`'amplitude'` variants are reserved for future per-band scaling.

Returns `(energy, frequencies)`:
- `energy` — 2-D array of shape `(n_segments, line_spacing // 2)`. Each row is
  the one-sided spectral energy of one overlapping segment. Nyquist bin excluded.
- `frequencies` — 1-D array of length `line_spacing // 2` in cycles/metre.

Implementation notes:
- Segments are built with `np.lib.stride_tricks.as_strided` (zero-copy).
- Hann window applied per segment in the time domain before FFT.
- One-sided folding: interior bins doubled; handles odd/even `line_spacing` correctly.

## Active Notebook — Trackline Detection & Uncertainty Pipeline

**`notebooks/Untitled-1.ipynb`** — the current working notebook. Pipeline steps:

| Step | What it does |
|------|-------------|
| 1 | Load BAG raster via `read_file` |
| 2 | Build binary coverage mask |
| 3–4 | Morphological filtering + skeletonisation to isolate tracklines |
| 5 | Radon transform to find dominant line orientations |
| 6 | PCA-based local angle estimation |
| 7–8 | Rotate raster to align tracklines vertically; detect line positions via column projection peaks |
| 9 | Interpolate depth gaps within each line (`np.interp`) |
| 10–11 | Call `compute_uncertainty` on each line using `line_spacing = linespan` (pixel distance between adjacent tracklines) |
| 12 | Fill `uncertainty_raster` (separate from `current_depth_raster`) with averaged spectral energy in the inter-line gap |
| 13 | Rotate `uncertainty_raster` back to original orientation (`order=1` bilinear, `cval=np.nan`) |
| 14 | Plot result |

### Critical distinction — two rasters in the loop:

```
current_depth_raster   — holds interpolated depth values [metres]. Read-only after Step 9.
uncertainty_raster     — filled with spectral energy [m²/Hz] in Step 12. Written only here.
```

Never mix these. Writing spectral energy into `current_depth_raster` was a previous
bug that has been resolved.

### Placement logic (Step 12) — key details:

- Both tracklines are clipped to their **shared row range** before calling
  `compute_uncertainty` to ensure both produce the same `n_segments`.
- The one-sided spectrum `(line_spacing // 2,)` is mirrored back to full gap
  width `(linespan - 1,)` before tiling across rows.
- Odd `line_spacing` produces one fewer mirrored element — pad by repeating
  the last bin: `np.append(full_spectrum, full_spectrum[-1])`.
- Final safety clip `[:gap_cols]` guards against any remaining off-by-one.

### Rotation convention:

```python
angle_to_rotate = 90 - TARGET_ANGLES[i]   # forward: aligns tracklines to vertical
# ... process in rotated space ...
rotate(..., angle=-angle_to_rotate, order=1, cval=np.nan)  # inverse: back to original
```

`resize=True` on both rotations — remember to crop back to original dimensions
using `(large - orig) // 2` offsets after the inverse rotation.

## Current Status

Library is mid-overhaul. Notebooks in `/notebooks/` represent the target behaviour
and output. Python files in `/src/` are being refactored to match. Do not assume
`/src/` is stable — always check notebooks for intended behaviour.

The trackline detection notebook (`Untitled-1.ipynb`) is the current focus.
Steps 1–12 are implemented; Steps 13–14 (rotation back + final plot) are the
next milestone.
