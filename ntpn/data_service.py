"""
Data pipeline service for NTPN.

This module provides pure data processing functions with no Streamlit dependency.
All functions accept and return data directly, using StateManager for persistence
only when a state parameter is provided.

@author: proxy_loken
"""

from typing import List, Tuple, Optional, Any
import numpy as np
import numpy.typing as npt

from ntpn import point_net_utils
from ntpn import ntpn_constants
from ntpn.logging_config import get_logger
from ntpn.state_manager import StateManager, get_state_manager

logger = get_logger(__name__)


def load_demo_session(state: Optional[StateManager] = None) -> None:
    """Load demo dataset and labels.

    Args:
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    st_file = ntpn_constants.demo_st_file
    context_file = ntpn_constants.demo_context_file

    logger.info("Loading demo session from %s and %s", st_file, context_file)
    stbin_list, context_list = point_net_utils.load_data_pickle(
        st_file, context_file, 'context_labels'
    )

    state.data.dataset = stbin_list
    state.data.labels = context_list
    state.data.dataset_name = "demo_data"

    state.sync_to_legacy()


def load_2D_data(state: Optional[StateManager] = None) -> None:
    """Load 2D data (stub function - not implemented).

    Args:
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()


def load_3D_data(state: Optional[StateManager] = None) -> None:
    """Load 3D trajectory data (stub function - not implemented).

    Args:
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()


def session_select(
    sessions: List[int],
    trim_noise: bool,
    state: Optional[StateManager] = None,
) -> None:
    """Select sessions from dataset and optionally trim noise.

    Args:
        sessions: List of session indices to select
        trim_noise: Whether to remove noise categories
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    logger.info("Selecting sessions %s, trim_noise=%s", sessions, trim_noise)
    select_samples = [state.data.dataset[i] for i in sessions]
    select_labels = [state.data.labels[i] for i in sessions]
    select_indices = [(idx) for idx, item in enumerate(select_samples)]

    if trim_noise:
        select_samples, select_labels = point_net_utils.remove_noise_cat(
            select_samples, select_labels, select_indices
        )

    state.data.select_samples = select_samples
    state.data.select_labels = select_labels
    state.data.select_indices = select_indices

    state.sync_to_legacy()


def samples_transform(
    transform_radio: str,
    state: Optional[StateManager] = None,
) -> None:
    """Apply transformation to selected samples.

    Args:
        transform_radio: Type of transform ('Power', 'Standard', or 'None')
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    logger.info("Applying transform: %s", transform_radio)
    if transform_radio == 'Power':
        X_tsf = point_net_utils.pow_transform(
            state.data.select_samples, state.data.select_indices
        )
    elif transform_radio == 'Standard':
        X_tsf = point_net_utils.std_transform(
            state.data.select_samples, state.data.select_indices
        )
    else:
        X_tsf = state.data.select_samples

    state.data.tsf_samples = X_tsf

    state.sync_to_legacy()


def create_trajectories(
    trajectories_window_size: int,
    trajectories_window_stride: int,
    trajectories_num_neurons: int,
    state: Optional[StateManager] = None,
) -> None:
    """Create neural trajectories using sliding windows and neuron subsampling.

    Args:
        trajectories_window_size: Window size for sliding window
        trajectories_window_stride: Stride for sliding window
        trajectories_num_neurons: Number of neurons to subsample
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    logger.info("Creating trajectories: window_size=%d, stride=%d, neurons=%d",
                trajectories_window_size, trajectories_window_stride, trajectories_num_neurons)
    tsf_samples = state.data.tsf_samples

    X_sw_list, Y_sw_list = point_net_utils.window_projection(
        tsf_samples,
        state.data.select_labels,
        state.data.select_indices,
        window_size=trajectories_window_size,
        stride=trajectories_window_stride,
    )
    X_subs, Ys = point_net_utils.subsample_dataset_3d_within(
        X_sw_list, Y_sw_list, trajectories_num_neurons, replace=False
    )
    X_subs = np.swapaxes(X_subs, 1, 2)

    state.data.sub_samples = X_subs
    state.data.sub_labels = Ys

    state.sync_to_legacy()


def create_train_test(
    test_size: float,
    state: Optional[StateManager] = None,
) -> None:
    """Create train/test split from subsampled data.

    Args:
        test_size: Fraction of data to use for testing
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    logger.info("Creating train/test split with test_size=%.2f", test_size)
    X_train, X_val, Y_train, Y_val = point_net_utils.train_test_gen(
        state.data.sub_samples, state.data.sub_labels, test_size=test_size
    )
    train_dataset, test_dataset = point_net_utils.train_test_tensors(
        X_train, X_val, Y_train, Y_val, augment=False
    )

    state.model.train_tensors = train_dataset
    state.model.test_tensors = test_dataset

    state.sync_to_legacy()
