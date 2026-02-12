"""Integration tests for data pipeline workflows.

Tests multi-function workflows using real function calls (no mocks).
"""

import numpy as np
import pytest

from ntpn.data_processing import (
    pow_transform,
    remove_noise_cat,
    select_samples,
    split_balanced,
    subsample_neurons_3d,
    train_test_gen,
    window_projection,
)


@pytest.fixture
def spike_data():
    """Two sessions of spike data with noise labels."""
    np.random.seed(42)
    stbin_list = [
        np.random.poisson(5, (200, 20)).astype(np.float64) + 1,
        np.random.poisson(5, (150, 20)).astype(np.float64) + 1,
    ]
    labels_list = [
        np.array([0] * 80 + [1] * 80 + [-1] * 40, dtype=np.int32),
        np.array([0] * 60 + [1] * 60 + [-1] * 30, dtype=np.int32),
    ]
    return stbin_list, labels_list


@pytest.mark.integration
class TestEndToEndPreprocessing:
    """remove_noise_cat -> pow_transform -> window_projection"""

    def test_full_pipeline(self, spike_data):
        stbin_list, labels_list = spike_data
        selection = [0, 1]

        # Step 1: Remove noise
        stcut_list, label_cut_list = remove_noise_cat(stbin_list, labels_list, selection)
        for lbl in label_cut_list:
            assert -1 not in lbl

        # Step 2: Power transform
        X_pow_list = pow_transform(stcut_list, selection)
        assert len(X_pow_list) == 2
        for i, x in enumerate(X_pow_list):
            assert x.shape == stcut_list[i].shape

        # Step 3: Window projection
        X_sw_list, Y_sw_list = window_projection(X_pow_list, label_cut_list, selection, window_size=5, stride=1)
        assert len(X_sw_list) == 2
        for X_sw, Y_sw in zip(X_sw_list, Y_sw_list):
            assert X_sw.ndim == 3
            assert X_sw.shape[0] == Y_sw.shape[0]


@pytest.mark.integration
class TestWindowingThroughSplitting:
    """window_projection -> train_test_gen -> split_balanced"""

    def test_windowing_to_split(self, spike_data):
        stbin_list, labels_list = spike_data

        # Use single session for simplicity
        stbin = stbin_list[0][:160]  # Only non-noise
        labels = labels_list[0][:160]

        X_sw, Y_sw = window_projection(stbin, labels, selection=[0], window_size=5, stride=1)

        # Ensure 3D for split_balanced
        if X_sw.ndim == 3:
            X_train, X_val, Y_train, Y_val = train_test_gen(X_sw, Y_sw, test_size=0.2)
            assert X_train.shape[0] + X_val.shape[0] == X_sw.shape[0]

            # Also test split_balanced
            X_train_b, X_test_b, y_train_b, y_test_b = split_balanced(X_sw, Y_sw, test_size=0.2)
            assert len(y_train_b) > 0
            assert len(y_test_b) > 0


@pytest.mark.integration
class TestSampleSelectionThroughSubsampling:
    """select_samples -> subsample_neurons_3d"""

    def test_select_and_subsample(self):
        np.random.seed(42)
        X = np.random.randn(100, 20, 3).astype(np.float32)
        Y = np.array([0] * 50 + [1] * 50, dtype=np.int32)

        # Select class 0 samples
        samples = select_samples(X, Y, num_samples=10, class_label=0)
        assert samples.shape == (10, 20, 3)

        # Subsample neurons
        subsampled = subsample_neurons_3d(samples, sample_size=8)
        assert subsampled.shape == (10, 8, 3)
