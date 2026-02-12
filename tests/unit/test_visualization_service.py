"""Unit tests for ntpn.visualization_service module."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import streamlit as st

from ntpn.state_manager import StateManager


@pytest.fixture(autouse=True)
def reset_session_state():
    """Reset Streamlit session state before each test."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    yield


@pytest.fixture
def state():
    """Create a fresh StateManager."""
    return StateManager()


class TestGenerateCriticalSets:
    """Tests for generate_critical_sets."""

    @patch('ntpn.visualization_service.point_net')
    @patch('ntpn.visualization_service.point_net_utils')
    def test_generates_critical_sets(self, mock_utils, mock_pn, state):
        """Critical sets are generated and stored for each class."""
        num_classes = 2
        num_samples = 10

        state.data.sub_samples = np.random.randn(50, 11, 32).astype(np.float32)
        state.data.sub_labels = np.concatenate([np.zeros(25), np.ones(25)]).astype(np.int32)
        state.model.ntpn_model = MagicMock()

        mock_trajectories = np.random.randn(num_samples, 11, 32).astype(np.float32)
        mock_utils.select_samples.return_value = mock_trajectories
        mock_pn.predict_critical.return_value = np.random.randn(num_samples, 11, 32).astype(np.float32)
        mock_pn.generate_critical.return_value = (
            np.random.randn(num_samples, 11, 32).astype(np.float32),
            5.0,
        )

        from ntpn.visualization_service import generate_critical_sets
        generate_critical_sets(num_classes, num_samples, state=state)

        assert state.viz.cs_lists is not None
        assert len(state.viz.cs_lists) == num_classes
        assert state.viz.cs_trajectories is not None
        assert state.viz.cs_predictions is not None
        assert state.viz.cs_means is not None

    @patch('ntpn.visualization_service.point_net')
    @patch('ntpn.visualization_service.point_net_utils')
    def test_uses_activation_14_layer(self, mock_utils, mock_pn, state):
        """predict_critical is called with layer_name='activation_14'."""
        state.data.sub_samples = np.random.randn(50, 11, 32).astype(np.float32)
        state.data.sub_labels = np.zeros(50).astype(np.int32)
        state.model.ntpn_model = MagicMock()

        mock_utils.select_samples.return_value = np.random.randn(10, 11, 32)
        mock_pn.predict_critical.return_value = np.random.randn(10, 11, 32)
        mock_pn.generate_critical.return_value = (np.random.randn(10, 11, 32), 5.0)

        from ntpn.visualization_service import generate_critical_sets
        generate_critical_sets(1, 10, state=state)

        call_args = mock_pn.predict_critical.call_args
        assert call_args[1]['layer_name'] == 'activation_14'


class TestCsDownsamplePCA:
    """Tests for cs_downsample_PCA."""

    @patch('ntpn.visualization_service.point_net_utils')
    def test_returns_downsampled_cs_and_trajectories(self, mock_utils, state):
        """PCA downsampling returns cs and trajectory arrays."""
        n_samples = 10
        state.viz.cs_lists = [np.random.randn(n_samples, 32, 20)]
        state.viz.cs_trajectories = [np.random.randn(n_samples, 32, 20)]

        pca_cs = np.random.randn(n_samples, 32, 3)
        pca_trajs = np.random.randn(n_samples, 32, 3)
        mock_utils.pca_cs_windowed.return_value = (pca_cs, pca_trajs)

        selected_cs = np.random.randn(5, 32, 3)
        selected_trajs = np.random.randn(5, 32, 3)
        mock_utils.select_samples_cs.return_value = (selected_cs, selected_trajs)

        from ntpn.visualization_service import cs_downsample_PCA
        result_cs, result_trajs = cs_downsample_PCA(0, 5, dims=3, state=state)

        assert result_cs is selected_cs
        assert result_trajs is selected_trajs


class TestDrawCsPlots:
    """Tests for draw_cs_plots."""

    @patch('ntpn.visualization_service.plot_critical_sets_PCA')
    def test_pca_plotting(self, mock_plot_pca, state):
        """PCA plotting generates figures for each class."""
        mock_fig = MagicMock()
        mock_plot_pca.return_value = mock_fig

        from ntpn.visualization_service import draw_cs_plots
        draw_cs_plots('PCA', 5, 3, 2, state=state)

        assert mock_plot_pca.call_count == 2
        assert state.viz.cs_ub_plots is not None
        assert len(state.viz.cs_ub_plots) == 2

    @patch('ntpn.visualization_service.plot_critical_sets_UMAP')
    def test_umap_plotting(self, mock_plot_umap, state):
        """UMAP plotting generates figures for each class."""
        mock_fig = MagicMock()
        mock_plot_umap.return_value = mock_fig

        from ntpn.visualization_service import draw_cs_plots
        draw_cs_plots('UMAP', 5, 3, 3, state=state)

        assert mock_plot_umap.call_count == 3
        assert len(state.viz.cs_ub_plots) == 3


class TestStubFunctions:
    """Tests for stub functions."""

    def test_cs_cca_alignment_callable(self, state):
        from ntpn.visualization_service import cs_CCA_alignment
        cs_CCA_alignment(state=state)

    def test_plot_trajectories_umap_callable(self, state):
        from ntpn.visualization_service import plot_trajectories_UMAP
        plot_trajectories_UMAP(state=state)

    def test_plot_critical_sets_grid_callable(self, state):
        from ntpn.visualization_service import plot_critical_sets_grid
        plot_critical_sets_grid(state=state)
