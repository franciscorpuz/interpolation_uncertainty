"""Tests for interpolation_uncertainty.preprocessing.morphology."""

import pytest
import numpy as np
from pathlib import Path

from interpolation_uncertainty.io import read_file
from interpolation_uncertainty.preprocessing.morphology import (
    build_coverage_mask,
    filter_blobs_keep_lines,
    filter_squares_keep_lines,
    filter_morphological_operations,
)

TEST_TIFF = Path(__file__).parent / "test_data" / "sample_tiff.tif"


def assert_binary_same_shape(output, input_array):
    assert output.shape == input_array.shape
    assert output.dtype == bool
    assert set(np.unique(output)) <= {0, 1}


@pytest.fixture(scope="module")
def sample_data():
    return read_file(TEST_TIFF)


@pytest.fixture(scope="module")
def coverage_mask(sample_data):
    return build_coverage_mask(sample_data)


@pytest.fixture
def synthetic_image():
    """Bool image with two thin lines (vertical + diagonal) and one square blob."""
    img = np.zeros((120, 120), dtype=bool)
    img[5:115, 20] = True                  # vertical line
    idx = np.arange(10, 100)
    img[idx, idx + 10] = True              # diagonal line
    img[60:90, 60:90] = True               # 30x30 blob
    return img


@pytest.fixture
def small_mask(coverage_mask, synthetic_image):
    """Cropped tif mask for the slow pipeline; falls back to synthetic data if empty."""
    crop = coverage_mask[:128, :128]
    if crop.sum() < 100:
        return synthetic_image
    return crop


# --- build_coverage_mask -----------------------------------------------------

def test_coverage_mask_binary_same_shape(sample_data, coverage_mask):
    assert_binary_same_shape(coverage_mask, sample_data["data"])


def test_coverage_mask_nan_ndv():
    data = np.array([[1.0, np.nan], [np.nan, 4.0]])
    mask = build_coverage_mask({"data": data, "metadata": {"ndv_value": np.nan}})
    assert_binary_same_shape(mask, data)
    np.testing.assert_array_equal(mask, [[True, False], [False, True]])


def test_coverage_mask_sentinel_ndv():
    data = np.array([[1.0, -9999.0], [-9999.0, 4.0]])
    mask = build_coverage_mask({"data": data, "metadata": {"ndv_value": -9999.0}})
    assert_binary_same_shape(mask, data)
    np.testing.assert_array_equal(mask, [[True, False], [False, True]])


# --- filter_blobs_keep_lines -------------------------------------------------

@pytest.mark.parametrize("restore_radius", [0, 2])
def test_blobs_binary_same_shape(synthetic_image, restore_radius):
    out = filter_blobs_keep_lines(synthetic_image, line_length=15, n_angles=6,
                                  restore_radius=restore_radius)
    assert_binary_same_shape(out, synthetic_image)


def test_blobs_on_sample_tiff(coverage_mask):
    out = filter_blobs_keep_lines(coverage_mask, line_length=15, n_angles=4)
    assert_binary_same_shape(out, coverage_mask)


def test_blobs_accepts_uint8(synthetic_image):
    img = synthetic_image.astype(np.uint8) * 255
    out = filter_blobs_keep_lines(img, line_length=15, n_angles=6)
    assert_binary_same_shape(out, img)


def test_blobs_restore_is_subset_of_input(synthetic_image):
    out = filter_blobs_keep_lines(synthetic_image, line_length=15, n_angles=6,
                                  restore_radius=2)
    assert not (out & ~synthetic_image).any()


def test_blobs_negative_restore_radius_raises(synthetic_image):
    with pytest.raises(AssertionError):
        filter_blobs_keep_lines(synthetic_image, restore_radius=-1)


# --- filter_squares_keep_lines -----------------------------------------------

@pytest.mark.parametrize("angle_rotation", [0, 30])
def test_squares_binary_same_shape(synthetic_image, angle_rotation):
    out = filter_squares_keep_lines(synthetic_image, square_size=15,
                                    angle_rotation=angle_rotation)
    assert_binary_same_shape(out, synthetic_image)


def test_squares_on_sample_tiff(coverage_mask):
    out = filter_squares_keep_lines(coverage_mask, square_size=15)
    assert_binary_same_shape(out, coverage_mask)


def test_squares_accepts_uint8(synthetic_image):
    img = synthetic_image.astype(np.uint8) * 255
    out = filter_squares_keep_lines(img, square_size=15)
    assert_binary_same_shape(out, img)


def test_squares_removes_blob_keeps_line(synthetic_image):
    out = filter_squares_keep_lines(synthetic_image, square_size=15)
    assert not out[60:90, 60:90].any()
    assert out[5:115, 20].all()


# --- filter_morphological_operations -----------------------------------------

@pytest.mark.parametrize("blob_max_size", [0, 20])
def test_morphological_operations_binary_same_shape(small_mask, blob_max_size):
    out = filter_morphological_operations(
        small_mask, square_size=15, line_length=15, n_angles=6,
        restore_radius=2, blob_max_size=blob_max_size,
    )
    assert_binary_same_shape(out, small_mask)
