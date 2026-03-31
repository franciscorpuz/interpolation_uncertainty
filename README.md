# NBS interpolation_uncertainty

## What interpolation_uncertainty does

Fully surveying the seafloor at the resolution required for all navigational and scientific needs is often prohibitively costly. In practice, bathymetric surveys collect direct depth measurements along sparse survey lines and use interpolation to represent the seafloor between them. Like all scientific measurements, these depths require an associated uncertainty estimate — but for interpolated regions, that uncertainty must account for how much the true seafloor may depart from the interpolated value.

The core challenge is that we have no direct measurements in the very regions we are trying to characterize. This library addresses that challenge by exploiting the spatial structure of nearby known bathymetry. Under an assumption of spatial isotropy — that the statistical character of the seafloor is consistent regardless of direction — the variance observed in sampled survey lines can be used to estimate the likely variance at different spatial scales across the interpolated gaps.

`interpolation_uncertainty` implements and compares two families of methods for making this estimate:

- **Spectral methods** analyze the power spectral density (PSD) of the seafloor along known survey lines. By cumulatively summing variance across spatial frequency bins — starting from the highest frequency and working toward scales matching the distance from a known measurement — these methods estimate the likely interpolation error at any given location.
- **Spatial methods** use sliding-window statistics (standard deviation, min/max envelopes, Gaussian smoothing) computed directly from the depth field to characterize local variability.

To evaluate and develop these methods, the library simulates sparse surveying on full-coverage raster datasets: it subsamples every N-th column as a synthetic survey line, linearly interpolates across the gaps, and computes the true residual between the interpolated and actual seafloor. This residual serves as ground truth for validating each uncertainty estimate across different seafloor types (flat, rough, sloped).

## How to install interpolation_uncertainty

[//]: # (:::{todo})

[//]: # (- )

[//]: # (:::)

To install this package run:

`python -m pip install git+https://github.com/noaa-ocs-hydrography/interpolation_uncertainty.git`

Create a conda environment using included yml file:

`conda env create -f environment.yml`

## Get started using interpolation_uncertainty

Check 'exploratory_analysis' Jupyter Notebook for basic usage

```python
from interpolation_uncertainty.core.loadfile import load_file
from interpolation_uncertainty.processors.rasterProcessor import RasterProcessor
from interpolation_uncertainty.utils import utils
import numpy as np

# Read bathymetry data from file
filename = "../data/raster/BlueTopo.tiff"
bathy_data = load_file(filename)

# Optional: Visualize the dataset
bathy_data.show_depth()

# Create processor with specified subsampling parameters
linespacing = 256
max_multiple = 1
current_multiple = 1
processor = RasterProcessor(linespacing_meters=linespacing,
                            max_multiple=max_multiple,
                            multiple=current_multiple)

# Use processor object to compute various surface calculations
# E.g., Compute the residual surface
residual = processor.compute_residual_surface(bathy_data)

# Compute Uncertainty using PSD spectrum, etc
uncertainty_psd_v1 = processor.estimate_uncertainty(method='psd_v1', residual_data=residual)




```

## How to cite interpolation_uncertainty

NOAA - NOS
National Bathymetric Source Project
Link: https://nauticalcharts.noaa.gov/learn/nbs.html
