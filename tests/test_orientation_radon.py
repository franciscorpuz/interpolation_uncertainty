"""Unit tests for the helpers behind orientation_via_Radon."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from interpolation_uncertainty.orientation import orientation_via_Radon as radon_mod
from interpolation_uncertainty.orientation.orientation_via_Radon import (
    _combine_angle_rasters,
    _compute_sinogram,
    _detect_dominant_angles,
    _embed_in_full_image,
    _fit_mask_to_shape,
    _label_points,
    _log_profile_variance,
    _split_by_angle,
    _sub_image,
)


# --- _sub_image --------------------------------------------------------------

def test_sub_image_crops_to_bounding_box():
    img = np.zeros((10, 12), dtype=bool)
    img[2:5, 3:9] = True
    sub, box = _sub_image(img)
    assert box == (2, 4, 3, 8)
    assert sub.shape == (3, 6)
    assert sub.all()


def test_sub_image_empty_returns_whole_image_and_box():
    img = np.zeros((6, 7), dtype=bool)
    sub, box = _sub_image(img)
    assert sub.shape == img.shape
    assert box == (0, 5, 0, 6)


# --- _fit_mask_to_shape ------------------------------------------------------

@pytest.mark.parametrize("src, target", [
    ((5, 5), (5, 5)),    # equal
    ((4, 5), (6, 5)),    # pad rows
    ((5, 4), (5, 7)),    # pad cols
    ((8, 5), (6, 5)),    # crop rows
    ((5, 9), (5, 6)),    # crop cols
    ((4, 9), (6, 6)),    # pad rows, crop cols
])
def test_fit_mask_to_shape(src, target):
    mask = np.ones(src, dtype=bool)
    out = _fit_mask_to_shape(mask, target)
    assert out.shape == target
    assert out.dtype == bool
    # original content is kept where it overlaps, padding is False
    r, c = min(src[0], target[0]), min(src[1], target[1])
    assert out[:r, :c].all()
    assert out.sum() == r * c


# --- _embed_in_full_image ----------------------------------------------------

def test_embed_in_full_image_places_mask_at_bounding_box():
    mask = np.ones((3, 2), dtype=bool)
    full = _embed_in_full_image(mask, (8, 8), (2, 4, 5, 6))
    assert full.shape == (8, 8)
    assert full[2:5, 5:7].all()
    assert full.sum() == 6


# --- _combine_angle_rasters --------------------------------------------------

def test_combine_angle_rasters_first_family_wins():
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[1, :] = True
    b[:, 1] = True           # overlaps a at (1, 1)
    out = _combine_angle_rasters({30: a, 120: b}, (4, 4))
    assert out[1, 1] == 30.0
    assert out[1, 3] == 30.0
    assert out[3, 1] == 120.0
    assert np.isnan(out[3, 3])


# --- _label_points -----------------------------------------------------------

def test_label_points_nearest_angle_and_nan_off_foreground():
    shape = (20, 20)
    combined = np.full(shape, np.nan)
    combined[:, 2] = 90.0     # left line
    combined[:, 17] = 10.0    # right line
    binary = np.zeros(shape, dtype=bool)
    binary[5:15, 4] = True    # nearer the left line
    binary[5:15, 15] = True   # nearer the right line
    out = _label_points(combined, binary, width_tolerance=1)
    assert (out[5:15, 4] == 90.0).all()
    assert (out[5:15, 15] == 10.0).all()
    assert np.isnan(out[~binary]).all()


# --- _split_by_angle ---------------------------------------------------------

def test_split_by_angle_keeps_only_matching_pixels():
    combined = np.array([[10.0, 90.0], [np.nan, 10.0]])
    out = _split_by_angle(combined, [10, 90])
    np.testing.assert_array_equal(out[10], [[10.0, np.nan], [np.nan, 10.0]])
    np.testing.assert_array_equal(out[90], [[np.nan, 90.0], [np.nan, np.nan]])


# --- _detect_dominant_angles -------------------------------------------------

def _two_peak_profile():
    x = np.zeros(180)
    x[40] = 5.0   # most prominent
    x[120] = 2.0
    return x


def test_detect_dominant_angles_sorted_by_prominence():
    out = _detect_dominant_angles(_two_peak_profile(), angular_tolerance=5, top_angles=0)
    assert list(out) == [40, 120]


def test_detect_dominant_angles_top_angles_limits_result():
    out = _detect_dominant_angles(_two_peak_profile(), angular_tolerance=5, top_angles=1)
    assert list(out) == [40]


# --- _log_profile_variance / _compute_sinogram -------------------------------

def test_log_profile_variance_shape_and_value():
    sino = np.array([[1.0, 0.0], [3.0, 0.0]])   # column variances: 1, 0
    out = _log_profile_variance(sino)
    np.testing.assert_allclose(out, np.log1p([1.0, 0.0]))


def test_compute_sinogram_reuses_angle_info_without_radon(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("radon must not be called when angle_info is given")
    monkeypatch.setattr(radon_mod, "radon", boom)

    sino = np.zeros((10, 180))
    info = {"sinogram": sino, "temp_offset": 60}
    out_sino, theta, offset = _compute_sinogram(np.zeros((8, 8)), info,
                                                circle=False, do_offset=True)
    assert out_sino is sino
    assert offset == 60
    np.testing.assert_array_equal(theta, np.arange(60, 240))


def test_compute_sinogram_without_offset_spans_180_degrees():
    img = np.zeros((16, 16), dtype=bool)
    img[2:14, 8] = True
    sino, theta, offset = _compute_sinogram(img, None, circle=False, do_offset=False)
    assert offset == 0
    assert sino.shape[1] == 180
    assert list(theta) == list(range(180))
