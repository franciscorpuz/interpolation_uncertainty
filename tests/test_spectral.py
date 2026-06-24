"""Tests for interpolation_uncertainty.methods.spectral."""

import pytest
import numpy as np

from interpolation_uncertainty.methods.spectral import compute_uncertainty


class TestComputeUncertaintyShapes:
    def test_output_shapes(self):
        signal = np.random.randn(100)
        energy, freqs = compute_uncertainty(signal, resolution=1.0, line_spacing=20)
        assert energy.ndim == 2
        assert energy.shape[1] == 10  # line_spacing // 2
        assert freqs.shape == (10,)

    def test_n_segments_no_overlap(self):
        signal = np.random.randn(100)
        energy, _ = compute_uncertainty(signal, resolution=1.0, line_spacing=20, overlap=0)
        assert energy.shape[0] == 5  # 100 // 20

    def test_n_segments_overlap_1(self):
        signal = np.random.randn(100)
        energy, _ = compute_uncertainty(signal, resolution=1.0, line_spacing=20, overlap=1)
        step = 19  # 20 - 1
        expected = (100 - 20) // step + 1
        assert energy.shape[0] == expected

    def test_odd_line_spacing(self):
        signal = np.random.randn(100)
        energy, freqs = compute_uncertainty(signal, resolution=1.0, line_spacing=21)
        assert energy.shape[1] == 10  # 21 // 2
        assert freqs.shape == (10,)


class TestComputeUncertaintyValidation:
    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            compute_uncertainty(np.ones(50), 1.0, 10, method="invalid")

    def test_2d_signal_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            compute_uncertainty(np.ones((5, 5)), 1.0, 5)

    def test_line_spacing_too_large(self):
        with pytest.raises(ValueError, match="line_spacing"):
            compute_uncertainty(np.ones(10), 1.0, 20)

    def test_overlap_out_of_range(self):
        with pytest.raises(ValueError, match="overlap"):
            compute_uncertainty(np.ones(50), 1.0, 10, overlap=10)


class TestComputeUncertaintyEnergy:
    def test_energy_nonnegative(self):
        signal = np.random.randn(200)
        energy, _ = compute_uncertainty(signal, resolution=1.0, line_spacing=20)
        assert np.all(energy >= 0)

    def test_all_methods_run(self):
        signal = np.random.randn(200)
        for method in ("amplitude", "psd", "psd_n", "psd_lf", "psd_df"):
            energy, freqs = compute_uncertainty(signal, 1.0, 20, method=method)
            assert energy.size > 0
