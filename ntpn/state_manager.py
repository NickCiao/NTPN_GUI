"""
Centralized state management for NTPN GUI.

This module provides a type-safe, centralized way to manage Streamlit session state,
replacing scattered st.session_state references with a structured StateManager.

Usage:
    from ntpn.state_manager import StateManager

    # Initialize once at app startup
    state = StateManager()

    # Access state
    state.data.dataset = loaded_data
    state.model.ntpn_model = trained_model

    # Check state validity
    if state.has_valid_data():
        process_data()
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
import streamlit as st
import numpy as np
import numpy.typing as npt
from tensorflow import keras


@dataclass
class DataState:
    """State for data loading and preprocessing."""

    # Dataset identification
    dataset_name: str = "demo_data"

    # Raw data
    dataset: Optional[List[npt.NDArray]] = None
    labels: Optional[List[npt.NDArray]] = None

    # Session selection
    select_samples: Optional[List[npt.NDArray]] = None
    select_indices: Optional[List[int]] = None
    select_labels: Optional[List[npt.NDArray]] = None

    # Subsampled data
    sub_samples: Optional[List[npt.NDArray]] = None
    sub_indices: Optional[List[int]] = None
    sub_labels: Optional[List[npt.NDArray]] = None

    def is_loaded(self) -> bool:
        """Check if raw data is loaded."""
        return self.dataset is not None and self.labels is not None

    def is_selected(self) -> bool:
        """Check if session selection is complete."""
        return self.select_samples is not None and self.select_labels is not None

    def is_subsampled(self) -> bool:
        """Check if data is subsampled."""
        return self.sub_samples is not None and self.sub_labels is not None


@dataclass
class ModelState:
    """State for model configuration and training."""

    # Model
    ntpn_model: Optional[keras.Model] = None

    # Training data
    train_tensors: Optional[Any] = None
    test_tensors: Optional[Any] = None

    # Training configuration
    batch_size: int = 8
    learning_rate: float = 0.02

    # Training metrics
    train_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    test_accuracy: List[float] = field(default_factory=list)

    def has_model(self) -> bool:
        """Check if model is created."""
        return self.ntpn_model is not None

    def has_training_data(self) -> bool:
        """Check if training data is ready."""
        return self.train_tensors is not None and self.test_tensors is not None

    def can_train(self) -> bool:
        """Check if model can be trained."""
        return self.has_model() and self.has_training_data()


@dataclass
class VisualizationState:
    """State for critical sets and visualizations."""

    # Critical sets
    cs_lists: Optional[List] = None
    cs_pca: Optional[List] = None
    cs_umap: Optional[List] = None
    cs_cca_aligned: Optional[List] = None

    # Upper bounds
    upper_lists: Optional[List] = None

    # Visualization parameters
    num_critical_samples: int = 50
    pca_dimensions: int = 3
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1

    def has_critical_sets(self) -> bool:
        """Check if critical sets are generated."""
        return self.cs_lists is not None


@dataclass
class UIState:
    """State for UI configuration and user interactions."""

    # Current page/view
    current_page: str = "import_and_load"

    # UI flags
    show_advanced_options: bool = False
    debug_mode: bool = False

    # Progress tracking
    operation_in_progress: bool = False
    progress_message: str = ""


class StateManager:
    """
    Centralized state manager for NTPN GUI.

    Provides type-safe access to all application state through dataclass properties.
    Automatically syncs with Streamlit session_state for persistence.

    Example:
        state = StateManager()
        state.data.dataset = loaded_data
        if state.model.can_train():
            train_model()
    """

    def __init__(self):
        """Initialize state manager and sync with session_state."""
        self._init_session_state()

    def _init_session_state(self) -> None:
        """Initialize all state categories in session_state if not present."""
        if '_state_initialized' not in st.session_state:
            # Initialize state objects
            st.session_state['_data_state'] = DataState()
            st.session_state['_model_state'] = ModelState()
            st.session_state['_viz_state'] = VisualizationState()
            st.session_state['_ui_state'] = UIState()
            st.session_state['_state_initialized'] = True

    @property
    def data(self) -> DataState:
        """Access data state."""
        return st.session_state['_data_state']

    @property
    def model(self) -> ModelState:
        """Access model state."""
        return st.session_state['_model_state']

    @property
    def viz(self) -> VisualizationState:
        """Access visualization state."""
        return st.session_state['_viz_state']

    @property
    def ui(self) -> UIState:
        """Access UI state."""
        return st.session_state['_ui_state']

    # Convenience validation methods

    def validate_data_loaded(self) -> bool:
        """Validate that data is loaded."""
        return self.data.is_loaded()

    def validate_model_ready(self) -> bool:
        """Validate that model is ready for training."""
        return self.model.can_train()

    def validate_can_generate_critical_sets(self) -> bool:
        """Validate that critical sets can be generated."""
        return (self.model.has_model() and
                self.data.is_subsampled() and
                self.model.has_training_data())

    # Migration helpers (for backward compatibility)

    def get_legacy_attr(self, attr_name: str) -> Any:
        """
        Get attribute from legacy session_state format.

        This helps during migration from old session_state usage to StateManager.

        Args:
            attr_name: Name of the attribute in old session_state format

        Returns:
            Value from session_state, or None if not found
        """
        return st.session_state.get(attr_name, None)

    def set_legacy_attr(self, attr_name: str, value: Any) -> None:
        """
        Set attribute in legacy session_state format.

        This helps during migration from old session_state usage to StateManager.

        Args:
            attr_name: Name of the attribute in old session_state format
            value: Value to set
        """
        st.session_state[attr_name] = value

    def sync_from_legacy(self) -> None:
        """
        Sync data from legacy session_state keys to StateManager.

        This allows gradual migration by importing existing session_state
        values into the new StateManager structure.
        """
        # Data state keys
        legacy_data_keys = {
            'dataset': 'dataset',
            'labels': 'labels',
            'dataset_name': 'dataset_name',
            'select_samples': 'select_samples',
            'select_indices': 'select_indices',
            'select_labels': 'select_labels',
            'sub_samples': 'sub_samples',
            'sub_indices': 'sub_indices',
            'sub_labels': 'sub_labels',
        }

        for legacy_key, attr_name in legacy_data_keys.items():
            if legacy_key in st.session_state:
                setattr(self.data, attr_name, st.session_state[legacy_key])

        # Model state keys
        legacy_model_keys = {
            'ntpn_model': 'ntpn_model',
            'train_tensors': 'train_tensors',
            'test_tensors': 'test_tensors',
            'batch_size': 'batch_size',
            'learning_rate': 'learning_rate',
            'train_loss': 'train_loss',
            'train_accuracy': 'train_accuracy',
            'test_loss': 'test_loss',
            'test_accuracy': 'test_accuracy',
        }

        for legacy_key, attr_name in legacy_model_keys.items():
            if legacy_key in st.session_state:
                setattr(self.model, attr_name, st.session_state[legacy_key])

        # Visualization state keys
        legacy_viz_keys = {
            'cs_lists': 'cs_lists',
            'cs_pca': 'cs_pca',
            'cs_umap': 'cs_umap',
            'cs_cca_aligned': 'cs_cca_aligned',
            'upper_lists': 'upper_lists',
            'num_critical_samples': 'num_critical_samples',
            'pca_dimensions': 'pca_dimensions',
            'umap_n_neighbors': 'umap_n_neighbors',
            'umap_min_dist': 'umap_min_dist',
        }

        for legacy_key, attr_name in legacy_viz_keys.items():
            if legacy_key in st.session_state:
                setattr(self.viz, attr_name, st.session_state[legacy_key])

        # UI state keys
        legacy_ui_keys = {
            'current_page': 'current_page',
            'show_advanced_options': 'show_advanced_options',
            'debug_mode': 'debug_mode',
            'operation_in_progress': 'operation_in_progress',
            'progress_message': 'progress_message',
        }

        for legacy_key, attr_name in legacy_ui_keys.items():
            if legacy_key in st.session_state:
                setattr(self.ui, attr_name, st.session_state[legacy_key])

    def sync_to_legacy(self) -> None:
        """
        Sync data from StateManager to legacy session_state keys.

        This maintains backward compatibility by updating old session_state
        keys when StateManager values change.
        """
        # Data state
        st.session_state['dataset'] = self.data.dataset
        st.session_state['labels'] = self.data.labels
        st.session_state['dataset_name'] = self.data.dataset_name
        st.session_state['select_samples'] = self.data.select_samples
        st.session_state['select_indices'] = self.data.select_indices
        st.session_state['select_labels'] = self.data.select_labels
        st.session_state['sub_samples'] = self.data.sub_samples
        st.session_state['sub_indices'] = self.data.sub_indices
        st.session_state['sub_labels'] = self.data.sub_labels

        # Model state
        st.session_state['ntpn_model'] = self.model.ntpn_model
        st.session_state['train_tensors'] = self.model.train_tensors
        st.session_state['test_tensors'] = self.model.test_tensors
        st.session_state['batch_size'] = self.model.batch_size
        st.session_state['learning_rate'] = self.model.learning_rate
        st.session_state['train_loss'] = self.model.train_loss
        st.session_state['train_accuracy'] = self.model.train_accuracy
        st.session_state['test_loss'] = self.model.test_loss
        st.session_state['test_accuracy'] = self.model.test_accuracy

        # Visualization state
        st.session_state['cs_lists'] = self.viz.cs_lists
        st.session_state['cs_pca'] = self.viz.cs_pca
        st.session_state['cs_umap'] = self.viz.cs_umap
        st.session_state['cs_cca_aligned'] = self.viz.cs_cca_aligned
        st.session_state['upper_lists'] = self.viz.upper_lists
        st.session_state['num_critical_samples'] = self.viz.num_critical_samples
        st.session_state['pca_dimensions'] = self.viz.pca_dimensions
        st.session_state['umap_n_neighbors'] = self.viz.umap_n_neighbors
        st.session_state['umap_min_dist'] = self.viz.umap_min_dist

        # UI state
        st.session_state['current_page'] = self.ui.current_page
        st.session_state['show_advanced_options'] = self.ui.show_advanced_options
        st.session_state['debug_mode'] = self.ui.debug_mode
        st.session_state['operation_in_progress'] = self.ui.operation_in_progress
        st.session_state['progress_message'] = self.ui.progress_message

    # State inspection

    def get_state_summary(self) -> dict:
        """
        Get a summary of current state for debugging.

        Returns:
            Dictionary with state summary
        """
        return {
            'data': {
                'loaded': self.data.is_loaded(),
                'selected': self.data.is_selected(),
                'subsampled': self.data.is_subsampled(),
                'dataset_name': self.data.dataset_name,
            },
            'model': {
                'has_model': self.model.has_model(),
                'has_training_data': self.model.has_training_data(),
                'can_train': self.model.can_train(),
                'batch_size': self.model.batch_size,
            },
            'viz': {
                'has_critical_sets': self.viz.has_critical_sets(),
                'num_samples': self.viz.num_critical_samples,
            },
            'ui': {
                'current_page': self.ui.current_page,
                'debug_mode': self.ui.debug_mode,
            }
        }

    def reset_state(self, keep_data: bool = False) -> None:
        """
        Reset application state.

        Args:
            keep_data: If True, keep loaded data but reset everything else
        """
        if keep_data:
            # Save data
            saved_dataset = self.data.dataset
            saved_labels = self.data.labels
            saved_name = self.data.dataset_name

            # Reset states
            st.session_state['_data_state'] = DataState(
                dataset=saved_dataset,
                labels=saved_labels,
                dataset_name=saved_name
            )
        else:
            st.session_state['_data_state'] = DataState()

        st.session_state['_model_state'] = ModelState()
        st.session_state['_viz_state'] = VisualizationState()
        # Keep UI state

    def __repr__(self) -> str:
        """String representation for debugging."""
        summary = self.get_state_summary()
        return f"StateManager(data_loaded={summary['data']['loaded']}, model_ready={summary['model']['can_train']})"


# Singleton instance for easy import
_state_manager_instance: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """
    Get the singleton StateManager instance.

    Returns:
        StateManager instance
    """
    global _state_manager_instance
    if _state_manager_instance is None:
        _state_manager_instance = StateManager()
    return _state_manager_instance
