from interpolation_uncertainty.readers.bathymetryFileReaders import (RasterReader,
                                                           CSVReader,
                                                           BPSReader)
from interpolation_uncertainty.readers.bathymetryDataset import (RasterDataset,
                                                       BPSDataset,
                                                       CSVDataset)

from pathlib import Path
from typing import Union

def load_file(filename: str | Path, **kwargs) -> Union[RasterDataset, BPSDataset, CSVDataset]:
    """
    Selects proper reader based on filetype

    Parameters
    ----------
    filename : str | Path
        Path to the file to be loaded.
    **kwargs
        Additional arguments passed to the reader.

    Returns
    -------
    Union[RasterDataset, BPSDataset, CSVDataset]
        The loaded dataset.
    """
    path = Path(filename)
    # Get extension without the dot and convert to lowercase
    file_ext = path.suffix.lower().lstrip('.')

    reader = return_reader(file_ext)

    return reader.read_file(str(path), **kwargs)


def return_reader(file_ext: str) -> Union[RasterReader, CSVReader, BPSReader]:
    """
    Selects proper reader based on filetype

    Parameters
    ----------
    file_ext: str

    Returns
    -------
    Bathymetry Reader object

    """
    if file_ext == "csv":
        return CSVReader()
    elif file_ext in ["tif", "tiff"]:
        return RasterReader()
    elif file_ext == "bps":
        return BPSReader()
    else:
        raise ValueError(f"Unrecognized file type: {file_ext}")
