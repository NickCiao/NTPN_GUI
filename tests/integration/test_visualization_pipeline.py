"""Integration tests for visualization pipeline workflows.

Tests critical set extraction through analysis and plotting.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from ntpn.analysis import (
    calc_cca_trajectories,
    cs_extract_uniques,
    cs_subsample,
    generate_uniques_from_trajectories,
    pca_cs_windowed,
    select_closest_trajectories,
)
from ntpn.plotting import plot_critical
from ntpn.point_net import generate_critical


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close('all')


@pytest.fixture
def synthetic_critical_data():
    """Synthetic critical set prediction data."""
    np.random.seed(42)
    num_samples = 10
    num_points = 16
    num_features = 8
    dims = 3
    crit_preds = np.random.randn(num_samples, num_points, num_features).astype(np.float32)
    samples = np.random.randn(num_samples, num_points, dims).astype(np.float32)
    return crit_preds, samples, num_samples


@pytest.mark.integration
class TestCriticalSetWorkflow:
    """generate_critical -> cs_extract_uniques -> cs_subsample"""

    def test_full_workflow(self, synthetic_critical_data):
        crit_preds, samples, num_samples = synthetic_critical_data

        # Step 1: Generate critical sets
        cs, mean_dim = generate_critical(crit_preds, num_samples, samples)
        assert cs.shape[0] == num_samples
        assert mean_dim > 0

        # Step 2: Extract uniques
        uniques_list, min_size = cs_extract_uniques(cs, mean_dim)
        assert len(uniques_list) == num_samples
        assert min_size > 0

        # Step 3: Subsample
        min_size_int = int(min_size)
        if min_size_int >= 2:
            subs_list = cs_subsample(uniques_list, min_size_int)
            assert len(subs_list) > 0


@pytest.mark.integration
class TestPCAThroughPlotting:
    """pca_cs_windowed -> plot_critical"""

    def test_pca_to_plot(self, synthetic_critical_data):
        crit_preds, samples, num_samples = synthetic_critical_data

        cs, _ = generate_critical(crit_preds, num_samples, samples)

        # PCA to 3 dims
        pca_cs, pca_samples = pca_cs_windowed(cs, samples, dims=3)
        assert pca_cs.shape[2] == 3
        assert pca_samples.shape[2] == 3

        # Plot
        num_plot = 3
        fig = plot_critical(pca_cs[:num_plot], num_plot, pca_samples[:num_plot])
        assert isinstance(fig, Figure)


@pytest.mark.integration
class TestCCAAlignmentWorkflow:
    """calc_cca_trajectories -> select_closest_trajectories -> generate_uniques_from_trajectories"""

    def test_full_cca_workflow(self):
        np.random.seed(42)
        samples = np.random.randn(20, 8, 3).astype(np.float32)
        exemplar = samples[0]

        # Step 1: Calculate CCA aligned trajectories
        example, aligned = calc_cca_trajectories(exemplar, samples, ex_index=0, ndims=3)
        assert aligned.shape == (20, 8, 3)

        # Step 2: Select closest trajectories
        closest = select_closest_trajectories(example, aligned, num_examples=5)
        assert closest.shape[0] == 5

        # Step 3: Generate unique points
        point_set, all_points = generate_uniques_from_trajectories(
            example, closest, mode='fixed', threshold=0.1
        )
        assert len(point_set) > 0
