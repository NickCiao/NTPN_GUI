"""Unit tests for ntpn/point_net.py.

Tests model construction, building blocks, and critical/upper set extraction.
Uses small models (units=4, num_points=8) for speed.
"""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from ntpn.point_net import (
    OrthogonalRegularizer,
    conv_bn,
    dense_bn,
    generate_critical,
    generate_upper,
    point_net,
    point_net_no_pool,
    point_net_no_pool_no_transform,
    point_net_no_transform,
    point_net_segment,
    tnet,
)


# Common model params for fast tests
NUM_POINTS = 8
NUM_CLASSES = 2
UNITS = 4
DIMS = 3


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

class TestPointNet:

    def test_output_shape(self):
        model = point_net(NUM_POINTS, NUM_CLASSES, units=UNITS, dims=DIMS)
        assert model.output_shape == (None, NUM_CLASSES)

    def test_has_global_max_pooling(self):
        model = point_net(NUM_POINTS, NUM_CLASSES, units=UNITS, dims=DIMS)
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert 'GlobalMaxPooling1D' in layer_types

    def test_has_softmax_output(self):
        model = point_net(NUM_POINTS, NUM_CLASSES, units=UNITS, dims=DIMS)
        last_layer = model.layers[-1]
        assert last_layer.get_config().get('activation') == 'softmax'


class TestPointNetNoTransform:

    def test_output_shape(self):
        model = point_net_no_transform(NUM_POINTS, NUM_CLASSES, units=UNITS, dims=DIMS)
        assert model.output_shape == (None, NUM_CLASSES)

    def test_no_dot_layers(self):
        """No Dot layers means no T-Net transforms."""
        model = point_net_no_transform(NUM_POINTS, NUM_CLASSES, units=UNITS, dims=DIMS)
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert 'Dot' not in layer_types


class TestPointNetNoPool:

    def test_output_shape(self):
        model = point_net_no_pool(NUM_POINTS, NUM_CLASSES, units=UNITS, dims=DIMS)
        assert model.output_shape == (None, NUM_CLASSES)

    def test_uses_flatten(self):
        model = point_net_no_pool(NUM_POINTS, NUM_CLASSES, units=UNITS, dims=DIMS)
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert 'Flatten' in layer_types


class TestPointNetNoPoolNoTransform:

    def test_output_shape(self):
        model = point_net_no_pool_no_transform(NUM_POINTS, NUM_CLASSES, units=UNITS, dims=DIMS)
        assert model.output_shape == (None, NUM_CLASSES)


class TestPointNetSegment:

    def test_output_shape(self):
        model = point_net_segment(NUM_POINTS, NUM_CLASSES, units=UNITS, dims=DIMS)
        assert model.output_shape == (None, NUM_POINTS, NUM_CLASSES)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TestConvBn:

    def test_shape_preserved(self):
        inp = keras.Input(shape=(NUM_POINTS, DIMS))
        out = conv_bn(inp, filters=16)
        model = keras.Model(inputs=inp, outputs=out)
        assert model.output_shape == (None, NUM_POINTS, 16)


class TestDenseBn:

    def test_correct_output_dim(self):
        inp = keras.Input(shape=(32,))
        out = dense_bn(inp, filters=16)
        model = keras.Model(inputs=inp, outputs=out)
        assert model.output_shape == (None, 16)


class TestTnet:

    def test_output_shape_matches_input(self):
        inp = keras.Input(shape=(NUM_POINTS, DIMS))
        out = tnet(inp, num_features=DIMS, units=UNITS)
        model = keras.Model(inputs=inp, outputs=out)
        assert model.output_shape == (None, NUM_POINTS, DIMS)


class TestOrthogonalRegularizer:

    def test_call_returns_scalar(self):
        reg = OrthogonalRegularizer(num_features=3)
        x = tf.random.normal((1, 9))  # 3x3 flattened
        loss = reg(x)
        assert loss.shape == ()
        assert loss.numpy() >= 0

    def test_get_config(self):
        reg = OrthogonalRegularizer(num_features=3, l2reg=0.01)
        config = reg.get_config()
        assert config['num_features'] == 3
        assert config['l2reg'] == 0.01
        assert 'eye' in config


# ---------------------------------------------------------------------------
# Critical / upper set extraction
# ---------------------------------------------------------------------------

class TestGenerateCritical:

    def test_output_shape(self):
        np.random.seed(42)
        num_samples = 5
        num_points = 8
        num_features = 16
        dims = 3
        crit_preds = np.random.randn(num_samples, num_points, num_features).astype(np.float32)
        samples = np.random.randn(num_samples, num_points, dims).astype(np.float32)
        cs, mean_dim = generate_critical(crit_preds, num_samples, samples)
        assert cs.shape == (num_samples, num_features, dims)
        assert isinstance(mean_dim, (float, np.floating))
        assert mean_dim > 0

    def test_mean_dim_reasonable(self):
        np.random.seed(42)
        crit_preds = np.random.randn(3, 8, 4).astype(np.float32)
        samples = np.random.randn(3, 8, 3).astype(np.float32)
        _, mean_dim = generate_critical(crit_preds, 3, samples)
        # mean_dim is mean of unique indices per sample, should be <= num_points
        assert mean_dim <= 8


class TestGenerateUpper:

    def test_output_shape(self):
        np.random.seed(42)
        num_samples = 3
        num_unit = 10
        num_features = 4
        dims = 3
        max_preds = np.random.randn(num_samples, num_features).astype(np.float32)
        max_unit_preds = np.random.randn(num_unit, num_features).astype(np.float32)
        samples = np.random.randn(num_samples, 8, dims).astype(np.float32)
        unit_sphere_data = np.random.randn(num_unit, dims).astype(np.float32)

        # Make some unit preds smaller than max preds to get non-empty results
        max_preds[:] = 10.0
        max_unit_preds[:] = -10.0

        ups = generate_upper(max_preds, max_unit_preds, num_samples, samples, unit_sphere_data)
        assert ups.shape[0] == num_samples
        # All unit sphere points should pass since max_preds >> max_unit_preds
        assert ups.shape[1] == num_unit
