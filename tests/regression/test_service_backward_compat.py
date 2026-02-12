"""Regression tests for service layer after Phase 6 migration.

Re-export tests are removed because pages now import directly from
service modules. These tests verify:
1. Streamlit-specific functions still exist in ntpn_utils
2. Service modules remain Streamlit-free
3. Pages use direct service imports (no ntpn_utils facade for service calls)
"""

from pathlib import Path

import pytest


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
        source = Path('ntpn/data_service.py').read_text()
        assert 'import streamlit' not in source

    def test_model_service_no_streamlit_import(self):
        source = Path('ntpn/model_service.py').read_text()
        assert 'import streamlit' not in source

    def test_visualization_service_no_streamlit_import(self):
        source = Path('ntpn/visualization_service.py').read_text()
        assert 'import streamlit' not in source


PAGE_FILES = [
    'pages/import_and_load_page.py',
    'pages/train_model_page.py',
    'pages/ntpn_visualisations_page.py',
    'pages/ntpn_landing_page.py',
]


class TestPagesUseDirectServiceImports:
    """Pages must not use ntpn_utils as a facade for service functions."""

    @pytest.mark.parametrize('page_path', PAGE_FILES)
    def test_no_ntpn_utils_module_import(self, page_path):
        """No page should use 'from ntpn import ntpn_utils' (module-level facade)."""
        source = Path(page_path).read_text()
        assert 'from ntpn import ntpn_utils' not in source

    @pytest.mark.parametrize('page_path', PAGE_FILES)
    def test_no_ntpn_utils_dot_calls(self, page_path):
        """No page should call ntpn_utils.<service_function>(...)."""
        source = Path(page_path).read_text()
        assert 'ntpn_utils.' not in source


class TestNtpnUtilsReExportsRemoved:
    """ntpn_utils no longer re-exports service functions."""

    SERVICE_FUNCTIONS = [
        'session_select',
        'samples_transform',
        'create_trajectories',
        'create_train_test',
        'load_2D_data',
        'load_3D_data',
        'create_model',
        'compile_model',
        'save_model',
        'generate_critical_sets',
        'draw_cs_plots',
        'cs_downsample_PCA',
        'cs_downsample_UMAP',
        'cs_CCA_alignment',
        'plot_critical_sets_PCA',
        'plot_critical_sets_UMAP',
        'plot_critical_sets_grid',
        'plot_trajectories_UMAP',
    ]

    @pytest.mark.parametrize('func_name', SERVICE_FUNCTIONS)
    def test_not_importable_from_ntpn_utils(self, func_name):
        """Service functions should no longer be importable via ntpn_utils."""
        import ntpn.ntpn_utils as mod

        assert not hasattr(mod, func_name), (
            f'{func_name} is still exported from ntpn_utils — it should be imported directly from its service module'
        )
