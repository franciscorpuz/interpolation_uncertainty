"""Tests for interpolation_uncertainty.io.readers."""

import pytest
import numpy as np
from pathlib import Path

from interpolation_uncertainty.io import read_file

TEST_TIFF = Path(__file__).parent / "test_data" / "sample_tiff.tif"


@pytest.fixture
def sample_data():
    return read_file(TEST_TIFF)


def test_read_file_returns_dict(sample_data):
    assert isinstance(sample_data, dict)
    assert set(sample_data.keys()) == {"data", "filename", "filetype", "metadata"}


def test_read_file_data_is_2d_array(sample_data):
    assert isinstance(sample_data["data"], np.ndarray)
    assert sample_data["data"].ndim == 2


def test_read_file_metadata_keys(sample_data):
    meta = sample_data["metadata"]
    assert "ndv_value" in meta
    assert "resolution" in meta
    assert "geotransform" in meta


def test_read_file_filetype(sample_data):
    assert sample_data["filetype"] == "raster"


def test_read_nonexistent_file():
    with pytest.raises(RuntimeError):
        read_file("nonexistent_file.tif")
