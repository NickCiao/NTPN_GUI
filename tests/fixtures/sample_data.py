"""Synthetic test data generators for NTPN GUI tests."""

import numpy as np
from typing import List, Tuple


def generate_spike_data(
    n_sessions: int = 3,
    n_neurons: int = 20,
    n_time_bins: int = 100,
    n_classes: int = 2,
    seed: int = 42
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate synthetic spike count data for testing.

    Args:
        n_sessions: Number of recording sessions
        n_neurons: Number of neurons per session
        n_time_bins: Number of time bins per session
        n_classes: Number of class labels
        seed: Random seed for reproducibility

    Returns:
        Tuple of (spike_data_list, labels_list)
        - spike_data_list: List of (n_neurons, n_time_bins) arrays
        - labels_list: List of (n_time_bins,) arrays with class labels
    """
    np.random.seed(seed)

    spike_data_list = []
    labels_list = []

    for _ in range(n_sessions):
        # Generate spike counts (Poisson-like distribution)
        # Shape: (time_bins, neurons) - matches giotto-tda SlidingWindow format
        spike_data = np.random.poisson(lam=5.0, size=(n_time_bins, n_neurons)).astype(np.float32)

        # Generate class labels
        labels = np.random.randint(0, n_classes, size=n_time_bins, dtype=np.int32)

        spike_data_list.append(spike_data)
        labels_list.append(labels)

    return spike_data_list, labels_list


def generate_trajectories(
    n_trajectories: int = 50,
    n_neurons: int = 20,
    trajectory_length: int = 32,
    n_classes: int = 2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic trajectory data for testing.

    Args:
        n_trajectories: Number of trajectories
        n_neurons: Number of neurons (points per trajectory)
        trajectory_length: Length of each trajectory (time dimension)
        n_classes: Number of class labels
        seed: Random seed for reproducibility

    Returns:
        Tuple of (trajectories, labels)
        - trajectories: (n_trajectories, n_neurons, trajectory_length) array
        - labels: (n_trajectories,) array with class labels
    """
    np.random.seed(seed)

    # Generate trajectories
    trajectories = np.random.randn(n_trajectories, n_neurons, trajectory_length).astype(np.float32)

    # Generate labels
    labels = np.random.randint(0, n_classes, size=n_trajectories, dtype=np.int32)

    return trajectories, labels


def generate_model_output(
    n_samples: int = 50,
    n_features: int = 32,
    n_classes: int = 2,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic model output (e.g., features before max pooling).

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes (affects output distribution)
        seed: Random seed for reproducibility

    Returns:
        Model output array of shape (n_samples, n_features)
    """
    np.random.seed(seed)
    return np.random.randn(n_samples, n_features).astype(np.float32)
