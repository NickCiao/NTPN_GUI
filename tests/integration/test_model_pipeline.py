"""Integration tests for model pipeline workflows.

Tests model construction and forward pass using real TF ops.
Uses direct model calls (not model.fit/predict) to avoid tf.function
graph tracing issues in CI/background environments.
"""

import numpy as np
import pytest

from ntpn.point_net import point_net, point_net_segment

NUM_POINTS = 8
NUM_CLASSES = 2
UNITS = 4
DIMS = 3


@pytest.mark.integration
class TestPointNetPipeline:
    def test_build_compile_and_call(self):
        """Test point_net model builds, compiles, and produces valid output."""
        np.random.seed(42)
        X = np.random.randn(4, NUM_POINTS, DIMS).astype(np.float32)

        model = point_net(NUM_POINTS, NUM_CLASSES, units=UNITS, dims=DIMS)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        assert model.output_shape == (None, NUM_CLASSES)

        preds = model(X, training=False).numpy()
        assert preds.shape == (4, NUM_CLASSES)
        # Softmax outputs should sum to ~1
        np.testing.assert_allclose(preds.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.integration
class TestPointNetSegmentPipeline:
    def test_build_and_call(self):
        """Test segment model builds and produces valid output."""
        np.random.seed(42)
        X = np.random.randn(4, NUM_POINTS, DIMS).astype(np.float32)

        model = point_net_segment(NUM_POINTS, NUM_CLASSES, units=UNITS, dims=DIMS)
        assert model.output_shape == (None, NUM_POINTS, NUM_CLASSES)

        out = model(X, training=False).numpy()
        assert out.shape == (4, NUM_POINTS, NUM_CLASSES)
        # Softmax per-point outputs should sum to ~1
        np.testing.assert_allclose(out.sum(axis=2), 1.0, atol=1e-5)
