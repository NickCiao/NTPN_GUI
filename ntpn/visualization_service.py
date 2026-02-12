"""
Visualization service for NTPN.

This module provides critical set generation, dimensionality reduction,
and plotting functions with no Streamlit dependency.

@author: proxy_loken
"""

from typing import List, Tuple, Optional, Any
import numpy as np
import numpy.typing as npt

from ntpn import point_net
from ntpn import point_net_utils
from ntpn import ntpn_constants
from ntpn.logging_config import get_logger
from ntpn.state_manager import StateManager, get_state_manager

logger = get_logger(__name__)


def generate_critical_sets(
    num_classes: int,
    num_samples: int,
    state: Optional[StateManager] = None,
) -> None:
    """Generate critical sets for each class.

    Args:
        num_classes: Number of classes
        num_samples: Number of samples per class
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    logger.info("Generating critical sets: %d classes, %d samples each", num_classes, num_samples)
    cs_trajectories = []
    cs_predictions = []
    cs_lists = []
    cs_means = []
    for i in range(num_classes):
        class_trajectories = point_net_utils.select_samples(
            state.data.sub_samples, state.data.sub_labels, num_samples, i
        )
        cs_trajectories.append(class_trajectories)
        class_predictions = point_net.predict_critical(
            state.model.ntpn_model,
            class_trajectories,
            layer_name=ntpn_constants.CRITICAL_SET_LAYER_NAME,
        )
        cs_predictions.append(class_predictions)
        class_cs, class_cs_mean = point_net.generate_critical(
            class_predictions, num_samples, class_trajectories
        )
        cs_lists.append(class_cs)
        cs_means.append(class_cs_mean)

    state.viz.cs_lists = cs_lists
    state.viz.cs_trajectories = cs_trajectories
    state.viz.cs_predictions = cs_predictions
    state.viz.cs_means = cs_means

    state.sync_to_legacy()


def cs_downsample_PCA(
    label: int,
    num_examples: int,
    dims: int = 3,
    state: Optional[StateManager] = None,
) -> Tuple[Any, Any]:
    """Downsample critical sets using PCA.

    Args:
        label: Class label
        num_examples: Number of examples to select
        dims: Number of PCA dimensions
        state: StateManager instance (uses singleton if not provided)

    Returns:
        Tuple of (downsampled_cs, downsampled_trajectories)
    """
    if state is None:
        state = get_state_manager()

    pca_cs, pca_trajs = point_net_utils.pca_cs_windowed(
        state.viz.cs_lists[label],
        state.viz.cs_trajectories[label],
        dims=dims,
    )

    pca_css, pca_trajss = point_net_utils.select_samples_cs(
        pca_cs, pca_trajs, num_examples
    )

    return pca_css, pca_trajss


def cs_downsample_UMAP(
    label: int,
    num_examples: int,
    dims: int = 3,
    state: Optional[StateManager] = None,
) -> Tuple[Any, Any]:
    """Downsample critical sets using UMAP.

    Args:
        label: Class label
        num_examples: Number of examples to select
        dims: Number of UMAP dimensions
        state: StateManager instance (uses singleton if not provided)

    Returns:
        Tuple of (downsampled_cs, downsampled_trajectories)
    """
    if state is None:
        state = get_state_manager()

    umap_cs, umap_trajs = point_net_utils.umap_cs_windowed(
        state.viz.cs_lists[label],
        state.viz.cs_trajectories[label],
        dims=dims,
    )

    umap_css, umap_trajss = point_net_utils.select_samples_cs(
        umap_cs, umap_trajs, num_examples
    )

    return umap_css, umap_trajss


def cs_CCA_alignment(state: Optional[StateManager] = None) -> None:
    """Align critical sets using CCA (stub function).

    Args:
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()


def plot_trajectories_UMAP(state: Optional[StateManager] = None) -> None:
    """Plot trajectories using UMAP (stub function).

    Args:
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()


def plot_critical_sets_PCA(
    label: int,
    num_examples: int,
    dims: int = 3,
    state: Optional[StateManager] = None,
) -> Any:
    """Plot critical sets using PCA.

    Args:
        label: Class label
        num_examples: Number of examples
        dims: Number of dimensions
        state: StateManager instance (uses singleton if not provided)

    Returns:
        Matplotlib figure
    """
    if state is None:
        state = get_state_manager()

    pca_css, pca_trajss = cs_downsample_PCA(
        label, num_examples, dims=dims, state=state
    )

    fig = point_net_utils.plot_critical(pca_css, num_examples, pca_trajss)

    return fig


def plot_critical_sets_UMAP(
    label: int,
    num_examples: int,
    dims: int = 3,
    state: Optional[StateManager] = None,
) -> Any:
    """Plot critical sets using UMAP.

    Args:
        label: Class label
        num_examples: Number of examples
        dims: Number of dimensions
        state: StateManager instance (uses singleton if not provided)

    Returns:
        Matplotlib figure
    """
    if state is None:
        state = get_state_manager()

    umap_css, umap_trajss = cs_downsample_UMAP(
        label, num_examples, dims=dims, state=state
    )

    fig = point_net_utils.plot_critical(umap_css, num_examples, umap_trajss)

    return fig


def plot_critical_sets_grid(state: Optional[StateManager] = None) -> None:
    """Plot critical sets in a grid (stub function).

    Args:
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()


def draw_cs_plots(
    plotting_algo: str,
    num_examples: int,
    dims: int,
    num_classes: int,
    state: Optional[StateManager] = None,
) -> None:
    """Generate and draw critical set plots.

    Args:
        plotting_algo: Algorithm to use ('PCA' or 'UMAP')
        num_examples: Number of examples per class
        dims: Number of dimensions
        num_classes: Number of classes
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    figs = []
    if plotting_algo == 'PCA':
        for i in range(num_classes):
            fig = plot_critical_sets_PCA(i, num_examples, dims, state=state)
            figs.append(fig)
    elif plotting_algo == 'UMAP':
        for i in range(num_classes):
            fig = plot_critical_sets_UMAP(i, num_examples, dims, state=state)
            figs.append(fig)

    state.viz.cs_ub_plots = figs

    state.sync_to_legacy()
