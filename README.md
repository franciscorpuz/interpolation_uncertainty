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
from interpolation_uncertainty.core.loadfile import load_file
from interpolation_uncertainty.processors.rasterProcessor import RasterProcessor

# Read bathymetry data from file
bathy_data = load_file("data/raster/BlueTopo.tiff")

# Visualize the dataset
bathy_data.show_depth()

# Create processor with specified subsampling parameters
processor = RasterProcessor(linespacing_meters=256, max_multiple=1, multiple=1)

# Compute the residual surface
residual = processor.compute_residual_surface(bathy_data)

# Compute uncertainty using PSD spectrum
uncertainty = processor.estimate_uncertainty(method='psd_v1', residual_data=residual)
```

## How to cite interpolation_uncertainty

NOAA - NOS
National Bathymetric Source Project
Link: https://nauticalcharts.noaa.gov/learn/nbs.html
