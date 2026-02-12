"""Pytest configuration and shared fixtures for NTPN GUI tests."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.fixtures.sample_data import generate_model_output, generate_spike_data, generate_trajectories


@pytest.fixture
def sample_spike_data() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Generate synthetic spike count data for testing.

    Returns:
        Tuple of (spike_data_list, labels_list)
        - spike_data_list: List of 3 sessions with (20 neurons, 100 time bins)
        - labels_list: List of 3 label arrays with 100 time bins each
    """
    return generate_spike_data(n_sessions=3, n_neurons=20, n_time_bins=100, n_classes=2, seed=42)


@pytest.fixture
def sample_labels() -> list[np.ndarray]:
    """
    Generate sample class labels matching time bins.

    Returns:
        List of label arrays for 3 sessions
    """
    _, labels = generate_spike_data(n_sessions=3, n_time_bins=100, seed=42)
    return labels


@pytest.fixture
def sample_trajectories() -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic trajectory data for testing.

    Returns:
        Tuple of (trajectories, labels)
        - trajectories: (50, 20, 32) array
        - labels: (50,) array
    """
    return generate_trajectories(n_trajectories=50, n_neurons=20, trajectory_length=32, n_classes=2, seed=42)


@pytest.fixture
def small_spike_data() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Generate small synthetic spike data for fast tests.

    Returns:
        Tuple of (spike_data_list, labels_list) with 2 sessions, 10 neurons, 50 bins
    """
    return generate_spike_data(n_sessions=2, n_neurons=10, n_time_bins=50, n_classes=2, seed=42)


@pytest.fixture
def mock_session_state() -> MagicMock:
    """
    Mock Streamlit session state for testing.

    Returns:
        MagicMock object that behaves like st.session_state
    """
    state = MagicMock()

    # Default state values
    state.dataset_name = 'demo_data'
    state.dataset = None
    state.labels = None
    state.ntpn_model = None
    state.select_samples = None
    state.select_indices = None
    state.select_labels = None
    state.sub_samples = None
    state.sub_indices = None
    state.sub_labels = None
    state.train_tensors = None
    state.test_tensors = None
    state.cs_lists = None

    return state


@pytest.fixture
def sample_model_output() -> np.ndarray:
    """
    Generate synthetic model output for testing.

    Returns:
        Model features array of shape (50, 32)
    """
    return generate_model_output(n_samples=50, n_features=32, n_classes=2, seed=42)


@pytest.fixture
def temp_data_file(tmp_path):
    """
    Create a temporary data file for testing file I/O.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to temporary .npz file with sample data
    """
    spike_data, labels = generate_spike_data(n_sessions=2, seed=42)

    file_path = tmp_path / 'test_data.npz'
    np.savez(file_path, data=spike_data, labels=labels)

    return file_path


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    yield
    # Cleanup after test if needed
