"""Tests for interpolation_uncertainty.preprocessing.orientation."""

import matplotlib
matplotlib.use("Agg")  # radon / probabilistic methods create figures

import pytest
import numpy as np

from interpolation_uncertainty.preprocessing.orientation import compute_local_orientation
from interpolation_uncertainty.orientation.orientation_utils import find_temp_offset

PCA_KWARGS = dict(line_length=10, num_neighbors=20)


def circular_diff(a, b, period=180.0):
    d = abs(a - b) % period
    return min(d, period - d)


@pytest.fixture
def vertical_line():
    img = np.zeros((40, 40), dtype=bool)
    img[5:35, 20] = True
    return img


@pytest.fixture
def horizontal_line():
    img = np.zeros((40, 40), dtype=bool)
    img[20, 5:35] = True
    return img


@pytest.fixture
def line_image():
    """Small bool image with a vertical, a horizontal and a diagonal line."""
    img = np.zeros((64, 64), dtype=bool)
    img[5:60, 10] = True
    img[50, 15:60] = True
    idx = np.arange(10, 45)
    img[idx, idx + 15] = True
    return img


def assert_orientation_field(out, image):
    """Float array, same shape as the input, NaN or in [0, 180) per pixel."""
    assert out.shape == image.shape
    assert out.dtype == np.float64
    valid = out[~np.isnan(out)]
    assert np.all((valid >= 0) & (valid < 180))


# --- compute_local_orientation: pca ------------------------------------------

def test_pca_output_structure(line_image):
    out, angle_info = compute_local_orientation(line_image, method="pca", **PCA_KWARGS)
    assert_orientation_field(out, line_image)
    # angle_info is the np.histogram tuple (counts, bin_edges) with 180 bins
    counts, edges = angle_info
    assert len(counts) == 180
    assert len(edges) == 181


def test_pca_nan_only_on_background(line_image):
    out, _ = compute_local_orientation(line_image, method="pca", **PCA_KWARGS)
    assert not np.isnan(out[line_image]).any()
    assert np.isnan(out[~line_image]).all()


def test_pca_vertical_line_angle(vertical_line):
    out, _ = compute_local_orientation(vertical_line, method="pca", **PCA_KWARGS)
    assert circular_diff(np.nanmedian(out), 90) < 5


def test_pca_horizontal_line_angle(horizontal_line):
    out, _ = compute_local_orientation(horizontal_line, method="pca", **PCA_KWARGS)
    assert circular_diff(np.nanmedian(out), 0) < 5


def test_default_method_is_pca(vertical_line):
    default, _ = compute_local_orientation(vertical_line, **PCA_KWARGS)
    explicit, _ = compute_local_orientation(vertical_line, method="pca", **PCA_KWARGS)
    np.testing.assert_array_equal(default, explicit)


# --- compute_local_orientation: probabilistic --------------------------------

def test_probabilistic_output_structure(line_image):
    out, theta_info = compute_local_orientation(line_image, method="probabilistic")
    assert_orientation_field(out, line_image)
    assert {"peaks", "profile_variance"} <= set(theta_info)
    # only foreground pixels may be labelled
    assert np.isnan(out[~line_image]).all()


# --- compute_local_orientation: radon ----------------------------------------

def test_radon_output_structure(line_image):
    masks, _ = compute_local_orientation(line_image, method="radon", zoom_factor=1.0)
    assert isinstance(masks, dict)
    for angle, mask in masks.items():
        assert 0 <= angle < 180
        assert mask.shape == line_image.shape
        assert mask.dtype == bool


# --- compute_local_orientation: dispatcher -----------------------------------

def test_invalid_method_raises(line_image):
    with pytest.raises(ValueError, match="Invalid method"):
        compute_local_orientation(line_image, method="not_a_method")


# --- find_temp_offset --------------------------------------------------------

def test_find_temp_offset_accepts_list_and_array():
    y = np.ones(180)
    y[80:100] = 0
    assert find_temp_offset(y) == find_temp_offset(list(y))


def test_find_temp_offset_returns_int():
    y = np.ones(180)
    y[80:100] = 0
    assert isinstance(find_temp_offset(y), (int, np.integer))


@pytest.mark.parametrize("heights", [np.arange(10.0), np.ones(10)])
def test_find_temp_offset_no_valley_returns_zero(heights):
    assert find_temp_offset(heights) == 0


def test_find_temp_offset_single_valley():
    y = np.ones(180)
    y[80:100] = 0
    assert abs(find_temp_offset(y) - 90) <= 2


def test_find_temp_offset_picks_widest_valley():
    y = np.ones(200)
    y[30:36] = 0       # narrow valley centred ~33
    y[100:130] = 0     # wide valley centred ~115
    assert abs(find_temp_offset(y) - 115) <= 3
