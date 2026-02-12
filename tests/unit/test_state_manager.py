"""Unit tests for StateManager."""

from unittest.mock import MagicMock

import numpy as np
import pytest

# Import streamlit and set up test mode
import streamlit as st

# Import after streamlit is ready
from ntpn.state_manager import StateManager


@pytest.fixture(autouse=True)
def reset_session_state():
    """Reset session state before each test."""
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    yield
    # Clean up after test
    for key in list(st.session_state.keys()):
        del st.session_state[key]


@pytest.fixture
def state_manager():
    """Create StateManager instance."""
    # Reset singleton
    import ntpn.state_manager

    ntpn.state_manager._state_manager_instance = None
    return StateManager()


class TestDataState:
    """Tests for DataState."""

    def test_data_state_initial(self, state_manager):
        """Test initial DataState."""
        assert state_manager.data.dataset_name == 'demo_data'
        assert state_manager.data.dataset is None
        assert state_manager.data.labels is None

    def test_data_state_is_loaded(self, state_manager):
        """Test is_loaded() method."""
        assert not state_manager.data.is_loaded()

        # Set data
        state_manager.data.dataset = [np.array([[1, 2, 3]])]
        state_manager.data.labels = [np.array([0, 1])]

        assert state_manager.data.is_loaded()

    def test_data_state_is_selected(self, state_manager):
        """Test is_selected() method."""
        assert not state_manager.data.is_selected()

        # Set selected data
        state_manager.data.select_samples = [np.array([[1, 2]])]
        state_manager.data.select_labels = [np.array([0])]

        assert state_manager.data.is_selected()

    def test_data_state_is_subsampled(self, state_manager):
        """Test is_subsampled() method."""
        assert not state_manager.data.is_subsampled()

        # Set subsampled data
        state_manager.data.sub_samples = [np.array([[1]])]
        state_manager.data.sub_labels = [np.array([0])]

        assert state_manager.data.is_subsampled()


class TestModelState:
    """Tests for ModelState."""

    def test_model_state_initial(self, state_manager):
        """Test initial ModelState."""
        assert state_manager.model.ntpn_model is None
        assert state_manager.model.batch_size == 8
        assert state_manager.model.learning_rate == 0.02

    def test_model_state_has_model(self, state_manager):
        """Test has_model() method."""
        assert not state_manager.model.has_model()

        state_manager.model.ntpn_model = MagicMock()

        assert state_manager.model.has_model()

    def test_model_state_has_training_data(self, state_manager):
        """Test has_training_data() method."""
        assert not state_manager.model.has_training_data()

        state_manager.model.train_tensors = MagicMock()
        assert not state_manager.model.has_training_data()  # Need both

        state_manager.model.test_tensors = MagicMock()
        assert state_manager.model.has_training_data()

    def test_model_state_can_train(self, state_manager):
        """Test can_train() method."""
        assert not state_manager.model.can_train()

        # Need both model and data
        state_manager.model.ntpn_model = MagicMock()
        assert not state_manager.model.can_train()

        state_manager.model.train_tensors = MagicMock()
        state_manager.model.test_tensors = MagicMock()
        assert state_manager.model.can_train()


class TestVisualizationState:
    """Tests for VisualizationState."""

    def test_viz_state_initial(self, state_manager):
        """Test initial VisualizationState."""
        assert state_manager.viz.cs_lists is None
        assert state_manager.viz.num_critical_samples == 50
        assert state_manager.viz.pca_dimensions == 3

    def test_viz_state_has_critical_sets(self, state_manager):
        """Test has_critical_sets() method."""
        assert not state_manager.viz.has_critical_sets()

        state_manager.viz.cs_lists = [[np.array([1, 2, 3])]]

        assert state_manager.viz.has_critical_sets()


class TestUIState:
    """Tests for UIState."""

    def test_ui_state_initial(self, state_manager):
        """Test initial UIState."""
        assert state_manager.ui.current_page == 'import_and_load'
        assert not state_manager.ui.show_advanced_options
        assert not state_manager.ui.debug_mode


class TestStateManager:
    """Tests for StateManager class."""

    def test_state_manager_initialization(self):
        """Test StateManager initializes correctly."""
        sm = StateManager()

        # Check initialization flag
        assert st.session_state['_state_initialized'] is True

        # Check all state objects created
        assert '_data_state' in st.session_state
        assert '_model_state' in st.session_state
        assert '_viz_state' in st.session_state
        assert '_ui_state' in st.session_state

    def test_validate_data_loaded(self, state_manager):
        """Test validate_data_loaded() method."""
        assert not state_manager.validate_data_loaded()

        state_manager.data.dataset = [np.array([[1]])]
        state_manager.data.labels = [np.array([0])]

        assert state_manager.validate_data_loaded()

    def test_validate_model_ready(self, state_manager):
        """Test validate_model_ready() method."""
        assert not state_manager.validate_model_ready()

        state_manager.model.ntpn_model = MagicMock()
        state_manager.model.train_tensors = MagicMock()
        state_manager.model.test_tensors = MagicMock()

        assert state_manager.validate_model_ready()

    def test_validate_can_generate_critical_sets(self, state_manager):
        """Test validate_can_generate_critical_sets() method."""
        assert not state_manager.validate_can_generate_critical_sets()

        # Need model, subsampled data, and training data
        state_manager.model.ntpn_model = MagicMock()
        state_manager.data.sub_samples = [np.array([[1]])]
        state_manager.data.sub_labels = [np.array([0])]
        state_manager.model.train_tensors = MagicMock()
        state_manager.model.test_tensors = MagicMock()

        assert state_manager.validate_can_generate_critical_sets()

    def test_get_state_summary(self, state_manager):
        """Test get_state_summary() method."""
        summary = state_manager.get_state_summary()

        assert 'data' in summary
        assert 'model' in summary
        assert 'viz' in summary
        assert 'ui' in summary

        assert summary['data']['loaded'] is False
        assert summary['data']['dataset_name'] == 'demo_data'
        assert summary['model']['batch_size'] == 8

    def test_reset_state(self, state_manager):
        """Test reset_state() method."""
        # Set up some state
        state_manager.data.dataset = [np.array([[1]])]
        state_manager.data.labels = [np.array([0])]
        state_manager.model.ntpn_model = MagicMock()
        state_manager.viz.cs_lists = [[np.array([1])]]

        # Reset without keeping data
        state_manager.reset_state(keep_data=False)

        assert state_manager.data.dataset is None
        assert state_manager.data.labels is None
        assert state_manager.model.ntpn_model is None
        assert state_manager.viz.cs_lists is None

    def test_reset_state_keep_data(self, state_manager):
        """Test reset_state() with keep_data=True."""
        # Set up some state
        test_data = [np.array([[1]])]
        test_labels = [np.array([0])]
        state_manager.data.dataset = test_data
        state_manager.data.labels = test_labels
        state_manager.data.dataset_name = 'test_data'
        state_manager.model.ntpn_model = MagicMock()
        state_manager.viz.cs_lists = [[np.array([1])]]

        # Reset keeping data
        state_manager.reset_state(keep_data=True)

        # Data should be preserved
        assert state_manager.data.dataset == test_data
        assert state_manager.data.labels == test_labels
        assert state_manager.data.dataset_name == 'test_data'

        # Other state should be reset
        assert state_manager.model.ntpn_model is None
        assert state_manager.viz.cs_lists is None

    def test_repr(self, state_manager):
        """Test __repr__() method."""
        repr_str = repr(state_manager)

        assert 'StateManager' in repr_str
        assert 'data_loaded' in repr_str
        assert 'model_ready' in repr_str


class TestGetStateManager:
    """Tests for get_state_manager() singleton function."""

    def test_get_state_manager_singleton(self):
        """Test that get_state_manager() returns singleton."""
        # Reset singleton first
        import ntpn.state_manager
        from ntpn.state_manager import get_state_manager

        ntpn.state_manager._state_manager_instance = None

        # Get instance twice
        sm1 = get_state_manager()
        sm2 = get_state_manager()

        # Should be same instance
        assert sm1 is sm2

    def test_get_state_manager_creates_instance(self):
        """Test that get_state_manager() creates instance."""
        # Reset singleton
        import ntpn.state_manager

        ntpn.state_manager._state_manager_instance = None

        from ntpn.state_manager import get_state_manager

        sm = get_state_manager()
        assert sm is not None
        assert hasattr(sm, 'data')
        assert hasattr(sm, 'model')


@pytest.mark.integration
def test_state_persistence():
    """Test that state persists across StateManager instances."""
    # Create first instance and set data
    sm1 = StateManager()
    sm1.data.dataset_name = 'test_dataset'
    sm1.model.batch_size = 16

    # Create second instance (simulating new execution)
    # Reset singleton but keep session_state
    import ntpn.state_manager

    ntpn.state_manager._state_manager_instance = None
    sm2 = StateManager()

    # State should persist through session_state
    assert sm2.data.dataset_name == 'test_dataset'
    assert sm2.model.batch_size == 16


class TestLegacySync:
    """Tests for legacy session_state synchronization."""

    def test_sync_to_legacy_data_keys(self, state_manager):
        """Test syncing data state to legacy session_state keys."""
        # Set data via StateManager
        test_data = [np.array([[1, 2, 3]])]
        test_labels = [np.array([0, 1])]
        state_manager.data.dataset = test_data
        state_manager.data.labels = test_labels
        state_manager.data.dataset_name = 'test_data'

        # Sync to legacy
        state_manager.sync_to_legacy()

        # Check legacy keys exist
        assert st.session_state['dataset'] == test_data
        assert st.session_state['labels'] == test_labels
        assert st.session_state['dataset_name'] == 'test_data'

    def test_sync_to_legacy_model_keys(self, state_manager):
        """Test syncing model state to legacy session_state keys."""
        # Set model data
        state_manager.model.batch_size = 16
        state_manager.model.learning_rate = 0.01

        # Sync to legacy
        state_manager.sync_to_legacy()

        # Check legacy keys
        assert st.session_state['batch_size'] == 16
        assert st.session_state['learning_rate'] == 0.01

    def test_sync_to_legacy_viz_keys(self, state_manager):
        """Test syncing visualization state to legacy session_state keys."""
        # Set viz data
        test_cs = [[np.array([1, 2, 3])]]
        state_manager.viz.cs_lists = test_cs
        state_manager.viz.num_critical_samples = 100

        # Sync to legacy
        state_manager.sync_to_legacy()

        # Check legacy keys
        assert st.session_state['cs_lists'] == test_cs
        assert st.session_state['num_critical_samples'] == 100

    def test_sync_from_legacy_data_keys(self, state_manager):
        """Test syncing from legacy session_state to StateManager."""
        # Set legacy keys
        test_data = [np.array([[1, 2, 3]])]
        test_labels = [np.array([0, 1])]
        st.session_state['dataset'] = test_data
        st.session_state['labels'] = test_labels
        st.session_state['dataset_name'] = 'legacy_data'

        # Sync from legacy
        state_manager.sync_from_legacy()

        # Check StateManager has the data
        assert state_manager.data.dataset == test_data
        assert state_manager.data.labels == test_labels
        assert state_manager.data.dataset_name == 'legacy_data'

    def test_sync_from_legacy_model_keys(self, state_manager):
        """Test syncing model data from legacy session_state."""
        # Set legacy keys
        st.session_state['batch_size'] = 32
        st.session_state['learning_rate'] = 0.001
        st.session_state['ntpn_model'] = MagicMock()

        # Sync from legacy
        state_manager.sync_from_legacy()

        # Check StateManager
        assert state_manager.model.batch_size == 32
        assert state_manager.model.learning_rate == 0.001
        assert state_manager.model.ntpn_model is not None

    def test_sync_from_legacy_partial_keys(self, state_manager):
        """Test syncing when only some legacy keys exist."""
        # Set only some legacy keys
        st.session_state['dataset_name'] = 'partial_data'
        st.session_state['batch_size'] = 64

        # Sync from legacy
        state_manager.sync_from_legacy()

        # Check only those keys were synced
        assert state_manager.data.dataset_name == 'partial_data'
        assert state_manager.model.batch_size == 64

        # Others should have defaults
        assert state_manager.data.dataset is None
        assert state_manager.data.labels is None

    def test_bidirectional_sync(self, state_manager):
        """Test bidirectional synchronization."""
        # Set legacy keys
        st.session_state['dataset_name'] = 'original'

        # Sync from legacy
        state_manager.sync_from_legacy()
        assert state_manager.data.dataset_name == 'original'

        # Modify via StateManager
        state_manager.data.dataset_name = 'modified'

        # Sync to legacy
        state_manager.sync_to_legacy()
        assert st.session_state['dataset_name'] == 'modified'

    def test_sync_to_legacy_new_data_fields(self, state_manager):
        """Test syncing new tsf_samples field to legacy."""
        test_tsf = [np.array([[1, 2]])]
        state_manager.data.tsf_samples = test_tsf

        state_manager.sync_to_legacy()

        assert st.session_state['tsf_samples'] == test_tsf

    def test_sync_from_legacy_new_data_fields(self, state_manager):
        """Test syncing tsf_samples from legacy."""
        test_tsf = [np.array([[3, 4]])]
        st.session_state['tsf_samples'] = test_tsf

        state_manager.sync_from_legacy()

        assert state_manager.data.tsf_samples == test_tsf

    def test_sync_to_legacy_new_model_fields(self, state_manager):
        """Test syncing new model fields to legacy."""
        state_manager.model.model_name = 'test_model'
        state_manager.model.loss_fn = MagicMock()
        state_manager.model.optimizer = MagicMock()
        state_manager.model.train_metric = MagicMock()
        state_manager.model.test_metric = MagicMock()

        state_manager.sync_to_legacy()

        assert st.session_state['model_name'] == 'test_model'
        assert st.session_state['loss_fn'] is state_manager.model.loss_fn
        assert st.session_state['optimizer'] is state_manager.model.optimizer
        assert st.session_state['train_metric'] is state_manager.model.train_metric
        assert st.session_state['test_metric'] is state_manager.model.test_metric

    def test_sync_from_legacy_new_model_fields(self, state_manager):
        """Test syncing new model fields from legacy."""
        mock_loss = MagicMock()
        mock_opt = MagicMock()
        st.session_state['model_name'] = 'legacy_model'
        st.session_state['loss_fn'] = mock_loss
        st.session_state['optimizer'] = mock_opt
        st.session_state['train_metric'] = MagicMock()
        st.session_state['test_metric'] = MagicMock()

        state_manager.sync_from_legacy()

        assert state_manager.model.model_name == 'legacy_model'
        assert state_manager.model.loss_fn is mock_loss
        assert state_manager.model.optimizer is mock_opt

    def test_sync_to_legacy_new_viz_fields(self, state_manager):
        """Test syncing new viz fields to legacy."""
        state_manager.viz.cs_trajectories = [MagicMock()]
        state_manager.viz.cs_predictions = [MagicMock()]
        state_manager.viz.cs_means = [MagicMock()]
        state_manager.viz.cs_ub_plots = [MagicMock()]

        state_manager.sync_to_legacy()

        assert st.session_state['cs_trajectories'] is state_manager.viz.cs_trajectories
        assert st.session_state['cs_predictions'] is state_manager.viz.cs_predictions
        assert st.session_state['cs_means'] is state_manager.viz.cs_means
        assert st.session_state['cs_ub_plots'] is state_manager.viz.cs_ub_plots

    def test_sync_from_legacy_new_viz_fields(self, state_manager):
        """Test syncing new viz fields from legacy."""
        mock_trajs = [MagicMock()]
        mock_plots = [MagicMock()]
        st.session_state['cs_trajectories'] = mock_trajs
        st.session_state['cs_predictions'] = [MagicMock()]
        st.session_state['cs_means'] = [MagicMock()]
        st.session_state['cs_ub_plots'] = mock_plots

        state_manager.sync_from_legacy()

        assert state_manager.viz.cs_trajectories is mock_trajs
        assert state_manager.viz.cs_ub_plots is mock_plots


class TestNewDataStateFields:
    """Tests for new DataState fields and methods."""

    def test_tsf_samples_default(self, state_manager):
        """Test tsf_samples defaults to None."""
        assert state_manager.data.tsf_samples is None

    def test_is_transformed_false(self, state_manager):
        """Test is_transformed() returns False initially."""
        assert not state_manager.data.is_transformed()

    def test_is_transformed_true(self, state_manager):
        """Test is_transformed() returns True when set."""
        state_manager.data.tsf_samples = [np.array([[1, 2]])]
        assert state_manager.data.is_transformed()


class TestNewModelStateFields:
    """Tests for new ModelState fields and methods."""

    def test_model_name_default(self, state_manager):
        """Test model_name defaults to expected value."""
        assert state_manager.model.model_name == 'No model loaded'

    def test_training_infrastructure_defaults(self, state_manager):
        """Test training infrastructure fields default to None."""
        assert state_manager.model.loss_fn is None
        assert state_manager.model.optimizer is None
        assert state_manager.model.train_metric is None
        assert state_manager.model.test_metric is None

    def test_has_training_infrastructure_false(self, state_manager):
        """Test has_training_infrastructure() returns False initially."""
        assert not state_manager.model.has_training_infrastructure()

    def test_has_training_infrastructure_partial(self, state_manager):
        """Test has_training_infrastructure() with partial setup."""
        state_manager.model.loss_fn = MagicMock()
        state_manager.model.optimizer = MagicMock()
        assert not state_manager.model.has_training_infrastructure()

    def test_has_training_infrastructure_true(self, state_manager):
        """Test has_training_infrastructure() returns True when fully set."""
        state_manager.model.loss_fn = MagicMock()
        state_manager.model.optimizer = MagicMock()
        state_manager.model.train_metric = MagicMock()
        state_manager.model.test_metric = MagicMock()
        assert state_manager.model.has_training_infrastructure()


class TestNewVisualizationStateFields:
    """Tests for new VisualizationState fields and methods."""

    def test_cs_intermediates_defaults(self, state_manager):
        """Test cs intermediate fields default to None."""
        assert state_manager.viz.cs_trajectories is None
        assert state_manager.viz.cs_predictions is None
        assert state_manager.viz.cs_means is None

    def test_cs_ub_plots_default(self, state_manager):
        """Test cs_ub_plots defaults to None."""
        assert state_manager.viz.cs_ub_plots is None

    def test_has_cs_intermediates_false(self, state_manager):
        """Test has_cs_intermediates() returns False initially."""
        assert not state_manager.viz.has_cs_intermediates()

    def test_has_cs_intermediates_partial(self, state_manager):
        """Test has_cs_intermediates() with partial data."""
        state_manager.viz.cs_trajectories = [MagicMock()]
        assert not state_manager.viz.has_cs_intermediates()

    def test_has_cs_intermediates_true(self, state_manager):
        """Test has_cs_intermediates() returns True when fully set."""
        state_manager.viz.cs_trajectories = [MagicMock()]
        state_manager.viz.cs_predictions = [MagicMock()]
        state_manager.viz.cs_means = [MagicMock()]
        assert state_manager.viz.has_cs_intermediates()

    def test_has_plots_false(self, state_manager):
        """Test has_plots() returns False initially."""
        assert not state_manager.viz.has_plots()

    def test_has_plots_true(self, state_manager):
        """Test has_plots() returns True when set."""
        state_manager.viz.cs_ub_plots = [MagicMock()]
        assert state_manager.viz.has_plots()


class TestUpdatedStateSummary:
    """Tests for updated get_state_summary()."""

    def test_summary_includes_transformed(self, state_manager):
        """Test summary includes transformed status."""
        summary = state_manager.get_state_summary()
        assert 'transformed' in summary['data']
        assert summary['data']['transformed'] is False

    def test_summary_includes_model_name(self, state_manager):
        """Test summary includes model_name."""
        summary = state_manager.get_state_summary()
        assert 'model_name' in summary['model']
        assert summary['model']['model_name'] == 'No model loaded'

    def test_summary_includes_training_infrastructure(self, state_manager):
        """Test summary includes has_training_infrastructure."""
        summary = state_manager.get_state_summary()
        assert 'has_training_infrastructure' in summary['model']
        assert summary['model']['has_training_infrastructure'] is False

    def test_summary_includes_viz_intermediates(self, state_manager):
        """Test summary includes viz intermediate statuses."""
        summary = state_manager.get_state_summary()
        assert 'has_cs_intermediates' in summary['viz']
        assert 'has_plots' in summary['viz']
        assert summary['viz']['has_cs_intermediates'] is False
        assert summary['viz']['has_plots'] is False
