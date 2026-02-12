"""
Regression tests for point_net_utils backward compatibility.

Ensures all 46 functions remain accessible via ntpn.point_net_utils.*
after the module was split into data_processing, analysis, and plotting.
"""

import pytest


# Complete list of all functions that must be accessible via point_net_utils
# Note: save/load_pickle are legacy functions preserved for backward compat
ALL_FUNCTIONS = [
    # data_processing (22 functions)
    'save_pickle',
    'load_pickle',
    'load_data_pickle',
    'select_samples',
    'select_samples_cs',
    'subsample_dataset_3d_within',
    'subsample_dataset_3d_across',
    'subsample_neurons',
    'subsample_neurons_3d',
    'gen_permuted_data',
    'augment',
    'unit_sphere',
    'precut_noise',
    'remove_noise_cat',
    'pow_transform',
    'std_transform',
    'window_projection',
    'window_projection_segments',
    'train_test_gen',
    'train_test_tensors',
    'split_balanced',
    # analysis (10 functions)
    'umap_windowed',
    'umap_cs_windowed',
    'pca_cs_windowed',
    'cs_extract_uniques',
    'cs_subsample',
    'generate_cca_trajectories',
    'calc_cca_trajectories',
    'select_closest_trajectories',
    'generate_uniques_from_trajectories',
    'generate_upper_sets',
    # plotting (14 functions)
    'trajectory_to_lines',
    'cloud_to_lines',
    'plot_samples',
    'plot_sample',
    'plot_sample_segmented',
    'plot_target_trajectory',
    'plot_target_trajectory_grid',
    'plot_critical',
    'plot_upper',
    'plot_upper_bound',
    'plot_mean_critical',
    'plot_mean_upper',
    'plot_critical_umap2D',
    'load_image',
]


class TestBackwardCompatibility:
    """All functions remain accessible via point_net_utils facade."""

    @pytest.mark.parametrize('func_name', ALL_FUNCTIONS)
    def test_function_accessible(self, func_name):
        """Each function is accessible via point_net_utils."""
        from ntpn import point_net_utils
        assert hasattr(point_net_utils, func_name), \
            f"point_net_utils.{func_name} not found — backward compatibility broken"
        assert callable(getattr(point_net_utils, func_name)), \
            f"point_net_utils.{func_name} is not callable"

    def test_total_function_count(self):
        """Verify the expected number of public functions are re-exported."""
        from ntpn import point_net_utils
        public_attrs = [a for a in dir(point_net_utils) if not a.startswith('_')]
        # Should have at least 46 public names (functions + any re-exported modules)
        assert len(public_attrs) >= len(ALL_FUNCTIONS), \
            f"Expected >= {len(ALL_FUNCTIONS)} public attrs, got {len(public_attrs)}"


class TestNewModulesNoStreamlit:
    """New modules must not depend on Streamlit."""

    def test_data_processing_no_streamlit(self):
        """data_processing has no streamlit import."""
        import ntpn.data_processing as dp
        source = open(dp.__file__).read()
        assert 'import streamlit' not in source
        assert 'from streamlit' not in source

    def test_analysis_no_streamlit(self):
        """analysis has no streamlit import."""
        import ntpn.analysis as an
        source = open(an.__file__).read()
        assert 'import streamlit' not in source
        assert 'from streamlit' not in source

    def test_plotting_no_streamlit(self):
        """plotting has no streamlit import."""
        import ntpn.plotting as pl
        source = open(pl.__file__).read()
        assert 'import streamlit' not in source
        assert 'from streamlit' not in source


class TestDirectImports:
    """Functions can be imported directly from the new modules."""

    def test_data_processing_imports(self):
        """Key data_processing functions are importable."""
        from ntpn.data_processing import (
            load_data_pickle,
            select_samples,
            remove_noise_cat,
            pow_transform,
            std_transform,
            window_projection,
            subsample_dataset_3d_within,
            train_test_gen,
            train_test_tensors,
        )

    def test_analysis_imports(self):
        """Key analysis functions are importable."""
        from ntpn.analysis import (
            pca_cs_windowed,
            umap_cs_windowed,
            cs_extract_uniques,
            generate_cca_trajectories,
            generate_upper_sets,
        )

    def test_plotting_imports(self):
        """Key plotting functions are importable."""
        from ntpn.plotting import (
            plot_critical,
            plot_sample,
            trajectory_to_lines,
            load_image,
        )
