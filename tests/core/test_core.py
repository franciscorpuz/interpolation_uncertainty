import pytest
from interpolation_uncertainty.readers.bathymetryFileReaders import (RasterReader,
                                                           CSVReader,
                                                           BPSReader)
from interpolation_uncertainty.core.loadfile import return_reader


def test_outputFileReader():
    fn = "sample_fill.tiff"
    fn_ext = fn.split(".")[-1]
    reader = return_reader(fn_ext)
    assert isinstance(reader, RasterReader)

    fn = "sample_fill.tif"
    fn_ext = fn.split(".")[-1]
    reader = return_reader(fn_ext)
    assert isinstance(reader, RasterReader)

    fn = "sample_fill.bps"
    fn_ext = fn.split(".")[-1]
    reader = return_reader(fn_ext)
    assert isinstance(reader, BPSReader)

    fn = "sample_fill.csv"
    fn_ext = fn.split(".")[-1]
    reader = return_reader(fn_ext)
    assert isinstance(reader, CSVReader)



