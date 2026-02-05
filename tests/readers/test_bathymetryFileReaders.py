import pytest
from interpolation_uncertainty.readers.bathymetryDataset import BathymetryDataset, RasterDataset
from interpolation_uncertainty.core.loadfile import load_file
from interpolation_uncertainty.readers.bathymetryFileReaders import RasterReader
import numpy as np
from osgeo import gdal
gdal.UseExceptions()

sample_fn = "test_data/sample_tiff.tif"
sample_fn_error = "test_data/nonexistent_tiff.tif"

@pytest.fixture
def raster_bathydata():
    reader = RasterReader()
    return reader.read_file(sample_fn)

def test_read_nonexistent_file():
    with pytest.raises(RuntimeError):
        reader2 = RasterReader()
        reader2.read_file(sample_fn_error)

def test_output_rasterfilereader(raster_bathydata):
    assert isinstance(raster_bathydata, RasterDataset)
    assert isinstance(raster_bathydata, np.ndarray)
    assert raster_bathydata.ndim == 2
    assert raster_bathydata.filetype == 'raster'
    assert (raster_bathydata != raster_bathydata.metadata['ndv_value']).all()

def test_gdal_raster_outputs(raster_bathydata):
    ds = gdal.Open(str(sample_fn))
    depth_band = ds.GetRasterBand(1)
    assert raster_bathydata.metadata['ndv_value'] == depth_band.GetNoDataValue()
    depth_gt = ds.GetGeoTransform()
    resolution = depth_gt[1]
    assert raster_bathydata.metadata['resolution'] == resolution

def test_load_file():
    bathy_dataset = load_file(sample_fn)
    assert isinstance(bathy_dataset, RasterDataset)
    assert isinstance(bathy_dataset, np.ndarray)
    assert bathy_dataset.filetype == 'raster'
