"""Smoke tests for ntpn/plotting.py.

Verify that plot functions execute without errors and return expected types.
Uses Agg backend for headless rendering.
"""

import matplotlib

matplotlib.use('Agg')

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ntpn.plotting import (
    cloud_to_lines,
    load_image,
    plot_critical,
    plot_critical_umap2D,
    plot_sample,
    plot_sample_segmented,
    plot_samples,
    plot_target_trajectory,
    plot_target_trajectory_grid,
    plot_upper_bound,
    trajectory_to_lines,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def close_plots():
    """Close all matplotlib figures after each test to prevent memory leaks."""
    yield
    plt.close('all')


@pytest.fixture
def sample_3d():
    """A single 3D trajectory: (20, 3)."""
    np.random.seed(42)
    return np.random.randn(20, 3).astype(np.float32)


@pytest.fixture
def samples_3d():
    """Multiple 3D trajectories: (5, 20, 3)."""
    np.random.seed(42)
    return np.random.randn(5, 20, 3).astype(np.float32)


@pytest.fixture
def cs_3d():
    """Critical sets matching samples_3d shape: (5, 20, 3)."""
    np.random.seed(99)
    return np.random.randn(5, 20, 3).astype(np.float32)


@pytest.fixture
def samples_2d():
    """Multiple 2D trajectories: (3, 20, 2)."""
    np.random.seed(42)
    return np.random.randn(3, 20, 2).astype(np.float32)


@pytest.fixture
def cs_2d():
    """Critical sets for 2D: (3, 20, 2)."""
    np.random.seed(99)
    return np.random.randn(3, 20, 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Line collection helpers
# ---------------------------------------------------------------------------


class TestTrajectoryToLines:
    def test_returns_line3d_collection(self, sample_3d):
        result = trajectory_to_lines(sample_3d)
        assert isinstance(result, Line3DCollection)


class TestCloudToLines:
    def test_returns_line3d_collection(self, sample_3d):
        result = cloud_to_lines(sample_3d)
        assert isinstance(result, Line3DCollection)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


class TestPlotSample:
    def test_scatter_mode(self, sample_3d):
        fig = plot_sample(sample_3d, mode='scatter')
        assert isinstance(fig, Figure)

    def test_line_mode(self, sample_3d):
        fig = plot_sample(sample_3d, mode='line')
        assert isinstance(fig, Figure)

    def test_line_mode_not_trajectory(self, sample_3d):
        fig = plot_sample(sample_3d, mode='line', trajectory=False)
        assert isinstance(fig, Figure)

    def test_scatter_not_trajectory(self, sample_3d):
        fig = plot_sample(sample_3d, mode='scatter', trajectory=False)
        assert isinstance(fig, Figure)

    def test_invalid_mode(self, sample_3d):
        result = plot_sample(sample_3d, mode='invalid')
        assert result == 0


class TestPlotSampleSegmented:
    def test_returns_figure(self, sample_3d):
        labels = np.random.randint(0, 3, size=20)
        fig = plot_sample_segmented(sample_3d, labels)
        assert isinstance(fig, Figure)

    def test_remove_noise(self, sample_3d):
        labels = np.array([0] * 10 + [-1] * 5 + [1] * 5)
        fig = plot_sample_segmented(sample_3d, labels, remove_noise=True)
        assert isinstance(fig, Figure)


class TestPlotSamples:
    def test_scatter_mode(self, samples_3d):
        fig = plot_samples(samples_3d, num_samples=3, mode='scatter')
        assert isinstance(fig, Figure)

    def test_line_mode(self, samples_3d):
        fig = plot_samples(samples_3d, num_samples=3, mode='line')
        assert isinstance(fig, Figure)

    def test_invalid_mode(self, samples_3d):
        result = plot_samples(samples_3d, num_samples=3, mode='invalid')
        assert result == 0


class TestPlotCritical:
    def test_returns_figure(self, cs_3d, samples_3d):
        fig = plot_critical(cs_3d, num_samples=3, samples=samples_3d)
        assert isinstance(fig, Figure)


class TestPlotCriticalUmap2D:
    def test_returns_figure(self, cs_2d, samples_2d):
        fig = plot_critical_umap2D(cs_2d, num_samples=3, samples=samples_2d)
        assert isinstance(fig, Figure)


class TestPlotUpperBound:
    def test_scatter_mode(self, sample_3d):
        fig = plot_upper_bound(sample_3d, mode='scatter')
        assert isinstance(fig, Figure)

    def test_scatter_shell_mode(self, sample_3d):
        fig = plot_upper_bound(sample_3d, mode='scatter shell')
        assert isinstance(fig, Figure)

    def test_distance_colour_mode(self, sample_3d):
        fig = plot_upper_bound(sample_3d, mode='scatter', colour_mode='distance')
        assert isinstance(fig, Figure)

    def test_invalid_mode(self, sample_3d):
        result = plot_upper_bound(sample_3d, mode='invalid')
        assert result == 0


class TestPlotTargetTrajectory:
    def test_with_2d_comps(self, sample_3d):
        comps = np.random.randn(20, 3).astype(np.float32)
        fig = plot_target_trajectory(sample_3d, comps)
        assert isinstance(fig, Figure)

    def test_with_3d_comps(self, sample_3d, samples_3d):
        fig = plot_target_trajectory(sample_3d, samples_3d[:3])
        assert isinstance(fig, Figure)


class TestPlotTargetTrajectoryGrid:
    def test_returns_figure(self, sample_3d, samples_3d):
        fig = plot_target_trajectory_grid(sample_3d, samples_3d[:3])
        assert isinstance(fig, Figure)


class TestLoadImage:
    @patch('ntpn.plotting.skio.imread')
    def test_calls_imread(self, mock_imread):
        mock_imread.return_value = np.zeros((100, 100, 3))
        result = load_image('/fake/path/image.png')
        mock_imread.assert_called_once_with('/fake/path/image.png')
        assert result.shape == (100, 100, 3)
