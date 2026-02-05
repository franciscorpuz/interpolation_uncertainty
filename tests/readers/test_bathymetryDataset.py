import pytest
from interpolation_uncertainty.readers.bathymetryFileReaders import RasterReader
from interpolation_uncertainty.readers.bathymetryDataset import RasterDataset
import numpy as np

@pytest.fixture
def sample_rasterbathymetrydataset():
    # Sample bathydataset for testing
    reader = RasterReader()
    print()
    return reader.read_file(filename=r"../data/raster/Bluetopo.tiff")

def test_bathymetrydataset(sample_rasterbathymetrydataset: RasterDataset):
    print(sample_rasterbathymetrydataset)
    assert sample_rasterbathymetrydataset is not None
    assert isinstance(sample_rasterbathymetrydataset, np.ndarray)
    assert sample_rasterbathymetrydataset.metadata["resolution"] == 4
    assert sample_rasterbathymetrydataset.metadata["ndv_value"] == 1_000_000
    assert sample_rasterbathymetrydataset.filename == "Bluetopo.tiff"

def test_metadata(sample_rasterbathymetrydataset: RasterDataset):
    assert isinstance(sample_rasterbathymetrydataset.metadata, dict)
    assert sample_rasterbathymetrydataset.metadata["resolution"] == 4
    assert sample_rasterbathymetrydataset.metadata["ndv_value"] == 1_000_000
    assert "ndv_value" in sample_rasterbathymetrydataset.metadata
