"""Unit tests for ntpn.data_service module."""

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


class TestLoadDemoSession:
    """Tests for load_demo_session."""

    @patch('ntpn.data_service.data_processing')
    def test_loads_demo_data(self, mock_utils, state):
        """Demo data is loaded and stored in state."""
        mock_data = [np.random.randn(100, 20).astype(np.float32)]
        mock_labels = [np.random.randint(0, 2, 100).astype(np.int32)]
        mock_utils.load_data_pickle.return_value = (mock_data, mock_labels)

        from ntpn.data_service import load_demo_session
        load_demo_session(state=state)

        assert state.data.dataset is mock_data
        assert state.data.labels is mock_labels
        assert state.data.dataset_name == "demo_data"

    @patch('ntpn.data_service.data_processing')
    def test_calls_load_data_with_constants(self, mock_utils, state):
        """load_demo_session uses file paths from ntpn_constants."""
        mock_utils.load_data_pickle.return_value = ([], [])

        from ntpn.data_service import load_demo_session
        from ntpn import ntpn_constants

        load_demo_session(state=state)

        mock_utils.load_data_pickle.assert_called_once_with(
            ntpn_constants.demo_st_file,
            ntpn_constants.demo_context_file,
            'context_labels',
        )


class TestLoad2DData:
    """Tests for load_2D_data stub."""

    def test_is_callable(self, state):
        from ntpn.data_service import load_2D_data
        load_2D_data(state=state)


class TestLoad3DData:
    """Tests for load_3D_data stub."""

    def test_is_callable(self, state):
        from ntpn.data_service import load_3D_data
        load_3D_data(state=state)


class TestSessionSelect:
    """Tests for session_select."""

    def test_selects_sessions_from_dataset(self, state):
        """Selected sessions are stored in state."""
        state.data.dataset = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            np.array([[9, 10], [11, 12]]),
        ]
        state.data.labels = [
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([0, 0]),
        ]

        from ntpn.data_service import session_select
        session_select([0, 2], trim_noise=False, state=state)

        assert len(state.data.select_samples) == 2
        np.testing.assert_array_equal(state.data.select_samples[0], np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(state.data.select_samples[1], np.array([[9, 10], [11, 12]]))

    @patch('ntpn.data_service.data_processing')
    def test_trim_noise_calls_remove_noise_cat(self, mock_utils, state):
        """When trim_noise=True, calls remove_noise_cat."""
        state.data.dataset = [np.array([[1, 2]])]
        state.data.labels = [np.array([0])]
        mock_utils.remove_noise_cat.return_value = ([np.array([[1, 2]])], [np.array([0])])

        from ntpn.data_service import session_select
        session_select([0], trim_noise=True, state=state)

        mock_utils.remove_noise_cat.assert_called_once()


class TestSamplesTransform:
    """Tests for samples_transform."""

    @patch('ntpn.data_service.data_processing')
    def test_power_transform(self, mock_utils, state):
        """Power transform is applied when selected."""
        test_samples = [np.array([[1, 2, 3]])]
        test_indices = [0]
        state.data.select_samples = test_samples
        state.data.select_indices = test_indices
        mock_utils.pow_transform.return_value = [np.array([[2, 3, 4]])]

        from ntpn.data_service import samples_transform
        samples_transform('Power', state=state)

        mock_utils.pow_transform.assert_called_once_with(test_samples, test_indices)
        assert state.data.tsf_samples is not None

    def test_raw_passthrough(self, state):
        """Raw/None transform passes samples through unchanged."""
        test_samples = [np.array([[1, 2, 3]])]
        state.data.select_samples = test_samples
        state.data.select_indices = [0]

        from ntpn.data_service import samples_transform
        samples_transform('Raw', state=state)

        assert state.data.tsf_samples is test_samples


class TestCreateTrajectories:
    """Tests for create_trajectories."""

    @patch('ntpn.data_service.data_processing')
    def test_creates_trajectories(self, mock_utils, state):
        """Trajectories are created and stored."""
        state.data.tsf_samples = [np.random.randn(100, 20)]
        state.data.select_labels = [np.random.randint(0, 2, 100)]
        state.data.select_indices = [0]

        mock_sw = [np.random.randn(10, 32, 20)]
        mock_ysw = [np.random.randint(0, 2, 10)]
        mock_utils.window_projection.return_value = (mock_sw, mock_ysw)
        mock_utils.subsample_dataset_3d_within.return_value = (
            np.random.randn(10, 11, 32),
            np.random.randint(0, 2, 10),
        )

        from ntpn.data_service import create_trajectories
        create_trajectories(32, 8, 11, state=state)

        assert state.data.sub_samples is not None
        assert state.data.sub_labels is not None


class TestCreateTrainTest:
    """Tests for create_train_test."""

    @patch('ntpn.data_service.data_processing')
    def test_creates_train_test_split(self, mock_utils, state):
        """Train/test tensors are created and stored."""
        state.data.sub_samples = np.random.randn(50, 11, 32).astype(np.float32)
        state.data.sub_labels = np.random.randint(0, 2, 50).astype(np.int32)

        mock_utils.train_test_gen.return_value = (
            np.random.randn(40, 11, 32),
            np.random.randn(10, 11, 32),
            np.random.randint(0, 2, 40),
            np.random.randint(0, 2, 10),
        )
        mock_train_ds = MagicMock()
        mock_test_ds = MagicMock()
        mock_utils.train_test_tensors.return_value = (mock_train_ds, mock_test_ds)

        from ntpn.data_service import create_train_test
        create_train_test(0.2, state=state)

        assert state.model.train_tensors is mock_train_ds
        assert state.model.test_tensors is mock_test_ds
