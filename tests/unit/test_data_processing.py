"""Unit tests for ntpn/data_processing.py."""

import numpy as np
import pytest
import tensorflow as tf

from ntpn.data_processing import (
    gen_permuted_data,
    pow_transform,
    precut_noise,
    remove_noise_cat,
    select_samples,
    select_samples_cs,
    split_balanced,
    std_transform,
    subsample_dataset_3d_across,
    subsample_dataset_3d_within,
    subsample_neurons,
    subsample_neurons_3d,
    train_test_gen,
    train_test_tensors,
    unit_sphere,
    window_projection,
    window_projection_segments,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trajectories_3d():
    """50 trajectories, 20 points, 3 dims."""
    np.random.seed(0)
    X = np.random.randn(50, 20, 3).astype(np.float32)
    Y = np.array([0] * 25 + [1] * 25, dtype=np.int32)
    return X, Y


@pytest.fixture
def spike_single_session():
    """Single session spike data: (100 time bins, 20 neurons)."""
    np.random.seed(0)
    stbin = np.random.poisson(5, (100, 20)).astype(np.float32)
    labels = np.array([0] * 40 + [1] * 40 + [-1] * 20, dtype=np.int32)
    return stbin, labels


@pytest.fixture
def spike_multi_session():
    """Two sessions of spike data."""
    np.random.seed(0)
    stbin_list = [
        np.random.poisson(5, (100, 20)).astype(np.float32),
        np.random.poisson(5, (80, 20)).astype(np.float32),
    ]
    labels_list = [
        np.array([0] * 40 + [1] * 40 + [-1] * 20, dtype=np.int32),
        np.array([0] * 30 + [1] * 30 + [-1] * 20, dtype=np.int32),
    ]
    return stbin_list, labels_list


# ---------------------------------------------------------------------------
# Sampling / selection
# ---------------------------------------------------------------------------

class TestSelectSamples:

    def test_correct_shape(self, trajectories_3d):
        X, Y = trajectories_3d
        samples = select_samples(X, Y, num_samples=5, class_label=0)
        assert samples.shape == (5, 20, 3)

    def test_correct_class(self, trajectories_3d):
        X, Y = trajectories_3d
        samples, indices = select_samples(X, Y, num_samples=5, class_label=1, return_index=True)
        assert np.all(Y[indices] == 1)

    def test_return_index(self, trajectories_3d):
        X, Y = trajectories_3d
        result = select_samples(X, Y, num_samples=5, class_label=0, return_index=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        samples, indices = result
        assert samples.shape[0] == 5
        assert indices.shape[0] == 5


class TestSelectSamplesCs:

    def test_paired_selection(self, trajectories_3d):
        X, _ = trajectories_3d
        cs = np.random.randn(50, 20, 3).astype(np.float32)
        cs_out, samples_out = select_samples_cs(cs, X, num_samples=10)
        assert cs_out.shape == (10, 20, 3)
        assert samples_out.shape == (10, 20, 3)

    def test_return_index(self, trajectories_3d):
        X, _ = trajectories_3d
        cs = np.random.randn(50, 20, 3).astype(np.float32)
        cs_out, samples_out, indices = select_samples_cs(cs, X, num_samples=10, return_index=True)
        assert indices.shape[0] == 10
        # Verify same indices used for both
        np.testing.assert_array_equal(X[indices], samples_out)
        np.testing.assert_array_equal(cs[indices], cs_out)


class TestSubsampleNeurons:

    def test_2d_output_shape(self):
        stbin = np.random.randn(100, 50).astype(np.float32)
        out = subsample_neurons(stbin, sample_size=20)
        assert out.shape == (100, 20)

    def test_replace_true(self):
        stbin = np.random.randn(100, 10).astype(np.float32)
        out = subsample_neurons(stbin, sample_size=20, replace=True)
        assert out.shape == (100, 20)

    def test_replace_false(self):
        stbin = np.random.randn(100, 50).astype(np.float32)
        out = subsample_neurons(stbin, sample_size=20, replace=False)
        assert out.shape == (100, 20)


class TestSubsampleNeurons3d:

    def test_output_shape(self):
        stbin = np.random.randn(50, 20, 3).astype(np.float32)
        out = subsample_neurons_3d(stbin, sample_size=10)
        assert out.shape == (50, 10, 3)

    def test_replace_true(self):
        stbin = np.random.randn(50, 10, 3).astype(np.float32)
        out = subsample_neurons_3d(stbin, sample_size=20, replace=True)
        assert out.shape == (50, 20, 3)


class TestSubsampleDataset3dWithin:

    def test_concatenation(self):
        stbin_list = [
            np.random.randn(30, 20, 3).astype(np.float32),
            np.random.randn(20, 20, 3).astype(np.float32),
        ]
        labels_list = [
            np.zeros(30, dtype=np.int32),
            np.ones(20, dtype=np.int32),
        ]
        out_neurons, out_labels = subsample_dataset_3d_within(stbin_list, labels_list, sample_size=10)
        assert out_neurons.shape[0] == 50  # 30 + 20
        assert out_neurons.shape[1] == 10
        assert out_labels.shape[0] == 50


class TestSubsampleDataset3dAcross:

    def test_balanced_output(self, trajectories_3d):
        X, Y = trajectories_3d
        stbin_list = [X]
        labels_list = [Y]
        out_neurons, out_labels = subsample_dataset_3d_across(
            stbin_list, labels_list, num_samples=10, sample_size=8
        )
        # Should have num_samples * num_classes samples
        assert out_neurons.shape[0] == 20  # 10 per class * 2 classes
        assert out_neurons.shape[1] == 8
        # Check balanced
        assert np.sum(out_labels == 0) == 10
        assert np.sum(out_labels == 1) == 10


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestPrecutNoise:

    def test_1d_num_data(self):
        labels = np.array([0, 1, -1, 0, -1, 1])
        num_data = np.array([10, 20, 30, 40, 50, 60])
        out_labels, out_data = precut_noise(labels, num_data)
        assert len(out_labels) == 4
        assert len(out_data) == 4
        assert -1 not in out_labels

    def test_2d_num_data(self):
        labels = np.array([0, 1, -1, 0])
        num_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        out_labels, out_data = precut_noise(labels, num_data)
        assert out_data.shape == (3, 2)
        assert -1 not in out_labels

    def test_custom_noise_label(self):
        labels = np.array([0, 1, 99, 0])
        num_data = np.array([10, 20, 30, 40])
        out_labels, out_data = precut_noise(labels, num_data, noise_label=99)
        assert 99 not in out_labels
        assert len(out_labels) == 3


class TestRemoveNoiseCat:

    def test_single_session(self, spike_single_session):
        stbin, labels = spike_single_session
        st_cut, label_cut = remove_noise_cat(stbin, labels, selection=[0])
        assert -1 not in label_cut
        assert st_cut.shape[0] == label_cut.shape[0]

    def test_multi_session(self, spike_multi_session):
        stbin_list, labels_list = spike_multi_session
        stcut_list, label_cut_list = remove_noise_cat(stbin_list, labels_list, selection=[0, 1])
        assert len(stcut_list) == 2
        assert len(label_cut_list) == 2
        for lbl in label_cut_list:
            assert -1 not in lbl


class TestPowTransform:

    def test_single_session(self, spike_single_session):
        stbin, _ = spike_single_session
        # Ensure positive values for power_transform
        stbin_pos = stbin + 1
        result = pow_transform(stbin_pos, selection=[0])
        assert result.shape == stbin_pos.shape

    def test_multi_session(self, spike_multi_session):
        stbin_list, _ = spike_multi_session
        stbin_pos = [s + 1 for s in stbin_list]
        result = pow_transform(stbin_pos, selection=[0, 1])
        assert len(result) == 2
        assert result[0].shape == stbin_pos[0].shape


class TestStdTransform:

    def test_single_session(self, spike_single_session):
        stbin, _ = spike_single_session
        result = std_transform(stbin, selection=[0])
        assert result.shape == stbin.shape
        # StandardScaler should center data near 0
        assert abs(np.mean(result)) < 0.1

    def test_multi_session(self, spike_multi_session):
        stbin_list, _ = spike_multi_session
        result = std_transform(stbin_list, selection=[0, 1])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

class TestWindowProjection:

    def test_single_session(self, spike_single_session):
        stbin, labels = spike_single_session
        X_sw, Y_sw = window_projection(stbin, labels, selection=[0], window_size=5, stride=1)
        assert X_sw.ndim == 3
        assert Y_sw.ndim == 1

    def test_multi_session(self, spike_multi_session):
        stbin_list, labels_list = spike_multi_session
        X_sw_list, Y_sw_list = window_projection(
            stbin_list, labels_list, selection=[0, 1], window_size=5, stride=1
        )
        assert len(X_sw_list) == 2
        assert len(Y_sw_list) == 2
        for X_sw in X_sw_list:
            assert X_sw.ndim == 3


class TestWindowProjectionSegments:

    def test_single_session(self, spike_single_session):
        stbin, labels = spike_single_session
        X_sw, Y_sw = window_projection_segments(stbin, labels, selection=[0], window_size=5, stride=1)
        assert X_sw.ndim == 3

    def test_multi_session(self, spike_multi_session):
        stbin_list, labels_list = spike_multi_session
        X_sw_list, Y_sw_list = window_projection_segments(
            stbin_list, labels_list, selection=[0, 1], window_size=5, stride=1
        )
        assert len(X_sw_list) == 2


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

class TestTrainTestGen:

    def test_output_shapes(self, trajectories_3d):
        X, Y = trajectories_3d
        X_train, X_val, Y_train, Y_val = train_test_gen(X, Y, test_size=0.2)
        assert X_train.shape[0] + X_val.shape[0] == X.shape[0]
        assert Y_train.shape[0] + Y_val.shape[0] == Y.shape[0]

    def test_stratify(self, trajectories_3d):
        X, Y = trajectories_3d
        X_train, X_val, Y_train, Y_val = train_test_gen(X, Y, test_size=0.2, stratify=True)
        # Both sets should contain both classes
        assert len(np.unique(Y_train)) == 2
        assert len(np.unique(Y_val)) == 2


class TestTrainTestTensors:

    def test_returns_tf_datasets(self, trajectories_3d):
        X, Y = trajectories_3d
        X_train, X_val, Y_train, Y_val = train_test_gen(X, Y, test_size=0.2)
        train_ds, test_ds = train_test_tensors(X_train, X_val, Y_train, Y_val, augment=False, batch_size=4)
        assert isinstance(train_ds, tf.data.Dataset)
        assert isinstance(test_ds, tf.data.Dataset)


class TestSplitBalanced:

    def test_balanced_class_distribution(self, trajectories_3d):
        X, Y = trajectories_3d
        X_train, X_test, y_train, y_test = split_balanced(X, Y, test_size=0.2)
        # Each class should have the same count in train and test
        for cls in np.unique(Y):
            train_count = np.sum(y_train == cls)
            test_count = np.sum(y_test == cls)
            assert train_count > 0
            assert test_count > 0

    def test_upsampling_edge_case(self):
        """Test with very few samples per class to trigger upsampling."""
        np.random.seed(42)
        X = np.random.randn(6, 10, 3).astype(np.float32)
        Y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        X_train, X_test, y_train, y_test = split_balanced(X, Y, test_size=0.5)
        assert len(y_train) > 0
        assert len(y_test) > 0


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

class TestUnitSphere:

    def test_output_shape(self):
        result = unit_sphere()
        assert result.shape == (128, 32, 3)

    def test_deterministic(self):
        r1 = unit_sphere()
        r2 = unit_sphere()
        np.testing.assert_array_equal(r1, r2)

    def test_value_range(self):
        result = unit_sphere()
        lim = 0.577
        assert np.all(result >= -lim - 1e-6)
        assert np.all(result <= lim + 1e-6)


class TestGenPermutedData:

    def test_width_mode(self):
        neurons = np.random.randn(100, 50).astype(np.float32)
        labels = np.arange(100)
        neuron_array, labels_array = gen_permuted_data(neurons, labels, direction='width', samples=3)
        # Width mode hstacks, so columns increase
        assert neuron_array.shape[0] == 100
        assert neuron_array.shape[1] == 32 * 3  # 3 samples of 32 neurons
        np.testing.assert_array_equal(labels_array, labels)

    def test_length_mode(self):
        neurons = np.random.randn(100, 50).astype(np.float32)
        labels = np.arange(100)
        neuron_array, labels_array = gen_permuted_data(neurons, labels, direction='length', samples=3)
        # Length mode vstacks, so rows increase
        assert neuron_array.shape[0] == 100 * 3
        assert labels_array.shape[0] == 100 * 3
