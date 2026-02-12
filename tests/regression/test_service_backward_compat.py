"""Regression tests for service layer backward compatibility.

These tests ensure that all functions previously accessible via ntpn_utils
are still accessible after the service layer extraction.
"""

import pytest


class TestDataServiceReExports:
    """All data service functions remain accessible via ntpn_utils."""

    def test_load_demo_session(self):
        from ntpn.ntpn_utils import load_demo_session
        assert callable(load_demo_session)

    def test_load_2D_data(self):
        from ntpn.ntpn_utils import load_2D_data
        assert callable(load_2D_data)

    def test_load_3D_data(self):
        from ntpn.ntpn_utils import load_3D_data
        assert callable(load_3D_data)

    def test_session_select(self):
        from ntpn.ntpn_utils import session_select
        assert callable(session_select)

    def test_samples_transform(self):
        from ntpn.ntpn_utils import samples_transform
        assert callable(samples_transform)

    def test_create_trajectories(self):
        from ntpn.ntpn_utils import create_trajectories
        assert callable(create_trajectories)

    def test_create_train_test(self):
        from ntpn.ntpn_utils import create_train_test
        assert callable(create_train_test)


class TestModelServiceReExports:
    """All model service functions remain accessible via ntpn_utils."""

    def test_create_model(self):
        from ntpn.ntpn_utils import create_model
        assert callable(create_model)

    def test_compile_model(self):
        from ntpn.ntpn_utils import compile_model
        assert callable(compile_model)

    def test_train_step(self):
        from ntpn.ntpn_utils import train_step
        assert callable(train_step)

    def test_test_step(self):
        from ntpn.ntpn_utils import test_step
        assert callable(test_step)

    def test_save_model(self):
        from ntpn.ntpn_utils import save_model
        assert callable(save_model)


class TestVisualizationServiceReExports:
    """All visualization service functions remain accessible via ntpn_utils."""

    def test_generate_critical_sets(self):
        from ntpn.ntpn_utils import generate_critical_sets
        assert callable(generate_critical_sets)

    def test_cs_downsample_PCA(self):
        from ntpn.ntpn_utils import cs_downsample_PCA
        assert callable(cs_downsample_PCA)

    def test_cs_downsample_UMAP(self):
        from ntpn.ntpn_utils import cs_downsample_UMAP
        assert callable(cs_downsample_UMAP)

    def test_cs_CCA_alignment(self):
        from ntpn.ntpn_utils import cs_CCA_alignment
        assert callable(cs_CCA_alignment)

    def test_plot_trajectories_UMAP(self):
        from ntpn.ntpn_utils import plot_trajectories_UMAP
        assert callable(plot_trajectories_UMAP)

    def test_plot_critical_sets_PCA(self):
        from ntpn.ntpn_utils import plot_critical_sets_PCA
        assert callable(plot_critical_sets_PCA)

    def test_plot_critical_sets_UMAP(self):
        from ntpn.ntpn_utils import plot_critical_sets_UMAP
        assert callable(plot_critical_sets_UMAP)

    def test_plot_critical_sets_grid(self):
        from ntpn.ntpn_utils import plot_critical_sets_grid
        assert callable(plot_critical_sets_grid)

    def test_draw_cs_plots(self):
        from ntpn.ntpn_utils import draw_cs_plots
        assert callable(draw_cs_plots)


class TestStreamlitFunctionsRemain:
    """Streamlit-specific functions still exist in ntpn_utils."""

    def test_initialise_session(self):
        from ntpn.ntpn_utils import initialise_session
        assert callable(initialise_session)

    def test_train_for_streamlit(self):
        from ntpn.ntpn_utils import train_for_streamlit
        assert callable(train_for_streamlit)

    def test_train_model(self):
        from ntpn.ntpn_utils import train_model
        assert callable(train_model)

    def test_draw_image(self):
        from ntpn.ntpn_utils import draw_image
        assert callable(draw_image)


class TestServiceModulesNoStreamlit:
    """Service modules must not import Streamlit directly."""

    def test_data_service_no_streamlit_import(self):
        from pathlib import Path
        source = Path('ntpn/data_service.py').read_text()
        assert 'import streamlit' not in source

    def test_model_service_no_streamlit_import(self):
        from pathlib import Path
        source = Path('ntpn/model_service.py').read_text()
        assert 'import streamlit' not in source

    def test_visualization_service_no_streamlit_import(self):
        from pathlib import Path
        source = Path('ntpn/visualization_service.py').read_text()
        assert 'import streamlit' not in source
