#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for the NTPN Streamlit Application.

This module provides Streamlit-specific utilities for session state management,
data pipeline orchestration, model training, and visualization workflows.

@author: proxy_loken
"""

from typing import List, Tuple, Optional, Any
import streamlit as st
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import skimage.io as skio
from PIL import Image
import time

from ntpn import point_net_utils
from ntpn import ntpn_constants
from ntpn import point_net
from ntpn.state_manager import StateManager, get_state_manager

import tensorflow as tf
from tensorflow import keras

def initialise_session(state: Optional[StateManager] = None) -> None:
    """Initialize session with default data.

    Args:
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    # Load demo data if not already loaded
    if state.data.dataset_name == "demo_data" and not state.data.is_loaded():
        load_demo_session(state)

    # Initialize model if not set
    if state.model.ntpn_model is None:
        state.model.ntpn_model = None  # Will be created later

    # Sync to legacy for backward compatibility
    state.sync_to_legacy()

    return


# LOADING/SAVING UTILITIES

def load_demo_session(state: Optional[StateManager] = None) -> None:
    """Load demo dataset and labels.

    Args:
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    # files for dataset and labels
    st_file = ntpn_constants.demo_st_file
    context_file = ntpn_constants.demo_context_file

    # load dataset and labels (uses safe NPZ format from Phase 1)
    stbin_list, context_list = point_net_utils.load_data_pickle(st_file, context_file, 'context_labels')

    # Store in StateManager
    state.data.dataset = stbin_list
    state.data.labels = context_list
    state.data.dataset_name = "demo_data"

    # Sync to legacy for backward compatibility
    state.sync_to_legacy()

    return


# loading for features by samples matrices as input data
def load_2D_data(state: Optional[StateManager] = None) -> None:
    """Load 2D data (stub function - not implemented).

    Args:
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    return


# loading for input data that is pre-sliced into trajectories
def load_3D_data(state: Optional[StateManager] = None) -> None:
    """Load 3D trajectory data (stub function - not implemented).

    Args:
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    return


# DATASET PROCESSING

def session_select(sessions: List[int], trim_noise: bool, state: Optional[StateManager] = None) -> None:
    """Select sessions from dataset and optionally trim noise.

    Args:
        sessions: List of session indices to select
        trim_noise: Whether to remove noise categories
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    select_samples = [state.data.dataset[i] for i in sessions]
    select_labels = [state.data.labels[i] for i in sessions]
    select_indices = [(idx) for idx, item in enumerate(select_samples)]

    if trim_noise:
        select_samples, select_labels = point_net_utils.remove_noise_cat(select_samples, select_labels, select_indices)

    state.data.select_samples = select_samples
    state.data.select_labels = select_labels
    state.data.select_indices = select_indices

    # Sync to legacy
    state.sync_to_legacy()

    return


def samples_transform(transform_radio: str, state: Optional[StateManager] = None) -> None:
    """Apply transformation to selected samples.

    Args:
        transform_radio: Type of transform ('Power', 'Standard', or 'None')
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    if transform_radio == 'Power':
        X_tsf = point_net_utils.pow_transform(state.data.select_samples, state.data.select_indices)
    elif transform_radio == 'Standard':
        X_tsf = point_net_utils.std_transform(state.data.select_samples, state.data.select_indices)
    else:
        X_tsf = state.data.select_samples

    # Store transformed samples (using legacy key for now)
    st.session_state.tsf_samples = X_tsf

    return


def create_trajectories(trajectories_window_size: int, trajectories_window_stride: int, trajectories_num_neurons: int, state: Optional[StateManager] = None) -> None:
    """Create neural trajectories using sliding windows and neuron subsampling.

    Args:
        trajectories_window_size: Window size for sliding window
        trajectories_window_stride: Stride for sliding window
        trajectories_num_neurons: Number of neurons to subsample
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    # Get transformed samples from legacy key (tsf_samples not yet in StateManager)
    tsf_samples = st.session_state.tsf_samples

    # Project into 3D via sliding windows
    X_sw_list, Y_sw_list = point_net_utils.window_projection(tsf_samples, state.data.select_labels, state.data.select_indices, window_size=trajectories_window_size, stride=trajectories_window_stride)
    # Within Session Dataset Gen
    X_subs, Ys = point_net_utils.subsample_dataset_3d_within(X_sw_list, Y_sw_list, trajectories_num_neurons, replace=False)
    X_subs = np.swapaxes(X_subs, 1, 2)

    state.data.sub_samples = X_subs
    state.data.sub_labels = Ys

    # Sync to legacy
    state.sync_to_legacy()

    return



def create_train_test(test_size: float, state: Optional[StateManager] = None) -> None:
    """Create train/test split from subsampled data.

    Args:
        test_size: Fraction of data to use for testing
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    # Make training and Test sets
    X_train, X_val, Y_train, Y_val = point_net_utils.train_test_gen(state.data.sub_samples, state.data.sub_labels, test_size=test_size)
    # Make tensors for the point net
    train_dataset, test_dataset = point_net_utils.train_test_tensors(X_train, X_val, Y_train, Y_val, augment=False)

    state.model.train_tensors = train_dataset
    state.model.test_tensors = test_dataset

    # Sync to legacy
    state.sync_to_legacy()

    return

# NTPN MODEL UTILITIES

# create a ntpn model by calling on the point net class
def create_model(trajectory_length: int, num_classes: int, layer_width: int, trajectory_dim: int, state: Optional[StateManager] = None) -> None:
    """Create a PointNet model.

    Args:
        trajectory_length: Length of trajectory (number of time points)
        num_classes: Number of output classes
        layer_width: Width of hidden layers
        trajectory_dim: Dimensionality of trajectory
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    state.model.ntpn_model = point_net.point_net(trajectory_length, num_classes, units=layer_width, dims=trajectory_dim)

    # Sync to legacy
    state.sync_to_legacy()

    return

# compile the model; single line if not viewing training (built-in keras fit), manually if usin the streamlit version
def compile_model(loss: str = 'sparse_categorical_crossentropy', learning_rate: float = 0.02, metric: str = 'sparse_categorical_accuracy', view: bool = True, state: Optional[StateManager] = None) -> None:
    """Compile the PointNet model.

    Args:
        loss: Loss function name
        learning_rate: Learning rate for optimizer
        metric: Metric to track
        view: Whether to set up for Streamlit training view
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    # TODO: refactor to account for possible different loss functions, metrics, etc.

    # Store learning rate in state
    state.model.learning_rate = learning_rate

    if view:
        # Store training objects in legacy session_state (not yet in StateManager)
        st.session_state.loss_fn = keras.losses.SparseCategoricalCrossentropy()
        st.session_state.train_metric = keras.metrics.SparseCategoricalAccuracy()
        st.session_state.test_metric = keras.metrics.SparseCategoricalAccuracy()
        st.session_state.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    state.model.ntpn_model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=[metric])

    # Sync to legacy
    state.sync_to_legacy()

    return

# replaces keras fit method to enable progress to be displayed inside streamlit
# adapted from shubhadtiya goswami
def train_for_streamlit(epochs: int, state: Optional[StateManager] = None) -> None:
    """Train model with Streamlit progress display.

    Args:
        epochs: Number of training epochs
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    # Store batch size in state
    state.model.batch_size = 8

    st.write('Starting Training with {} epochs...'.format(epochs))

    for epoch in range(epochs):
        st.write("Epoch {}".format(epoch+1))
        start_time = time.time()
        # initialise display params
        progress_bar = st.progress(value=0.0)
        percent_complete = 0
        epoch_time = 0
        # placeholder for update step
        st_t = st.empty()

        train_loss_list = []
        # Iterate over batches
        for step, (x_batch_train, y_batch_train) in enumerate(state.model.train_tensors):
            start_step = time.time()
            loss_value = train_step(x_batch_train, y_batch_train, state=state)
            end_step = time.time()
            epoch_time += (end_step - start_step)
            train_loss_list.append(float(loss_value))

            # number of steps to log
            if step % 1 == 0:
                step_acc = float(st.session_state.train_metric.result())
                percent_complete = ((step/(len(state.data.sub_samples)//state.model.batch_size)))
                progress_bar.progress(percent_complete)
                st_t.write('Duration : {0:.2f}s, Training Acc : {1:.4f}'.format((epoch_time), float(step_acc)))

        progress_bar.progress(1.0)

        # Metrics for the end of each epoch
        train_acc = st.session_state.train_metric.result()
        # reset training metric at the end of each epoch
        st.session_state.train_metric.reset_state()

        train_loss = round((sum(train_loss_list)/len(train_loss_list)), 5)

        val_loss_list = []
        # run the validation loop
        for x_batch_val, y_batch_val in state.model.test_tensors:
            val_loss_list.append(float(test_step(x_batch_val, y_batch_val, state=state)))

        val_loss = round((sum(val_loss_list)/len(val_loss_list)), 5)

        val_acc = st.session_state.test_metric.result()
        st.session_state.test_metric.reset_state()

        st_t.write('Duration : {0:.2f}s, Training Acc : {1:.4f}, Validation Acc : {2:.4f}'.format((time.time() - start_time), float(train_acc), float(val_acc)))

    return


def train_model(epochs: int, view: bool = True, state: Optional[StateManager] = None) -> None:
    """Train the model.

    Args:
        epochs: Number of training epochs
        view: Whether to use Streamlit progress display
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    if view:
        train_for_streamlit(epochs, state=state)
    else:
        state.model.ntpn_model.fit(state.model.train_tensors, epochs=epochs, validation_data=state.model.test_tensors)

    return


# NTPN MODEL TRAINING AND TESTING FUNCTIONS (necessary for display in streamlit)

# gradient, loss, and metric calculation for the model and data
# adapted from shubhadtiya goswami
def train_step(x: Any, y: Any, state: Optional[StateManager] = None) -> tf.Tensor:
    """Execute one training step.

    Args:
        x: Input batch
        y: Target batch
        state: StateManager instance (uses singleton if not provided)

    Returns:
        Loss value for this step
    """
    if state is None:
        state = get_state_manager()

    model = state.model.ntpn_model
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss_value = st.session_state.loss_fn(y, predictions)

    # calculate gradients
    grads = tape.gradient(loss_value, model.trainable_weights)
    # apply gradients via optimizer
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # update loss metric
    for metric in model.metrics:
        if metric.name == 'loss':
            metric.update_state(loss_value)
        else:
            metric.update_state(y, predictions)
    st.session_state.train_metric.update_state(y, predictions)
    return loss_value


def test_step(x: Any, y: Any, state: Optional[StateManager] = None) -> tf.Tensor:
    """Execute one test/validation step.

    Args:
        x: Input batch
        y: Target batch
        state: StateManager instance (uses singleton if not provided)

    Returns:
        Loss value for this step
    """
    if state is None:
        state = get_state_manager()

    val_predictions = state.model.ntpn_model(x, training=False)
    st.session_state.test_metric.update_state(y, val_predictions)
    return st.session_state.loss_fn(y, val_predictions)




# helper to export model as a .h5(keras) file for re-use
def save_model(model_name: str, state: Optional[StateManager] = None) -> None:
    """Save the trained model to disk.

    Args:
        model_name: Name for the saved model file
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    state.model.ntpn_model.save('models/'+model_name+'.keras', overwrite=True, include_optimizer=True)

    return


# CRITICAL SETS AND UPPER-BOUND Functions

# generate critical sets for a number of classes and samples
# NOTE: may need to explicitly name the correct activation layer(before the pooling operation) during model creation if multiple models breaks this function
def generate_critical_sets(num_classes: int, num_samples: int, state: Optional[StateManager] = None) -> None:
    """Generate critical sets for each class.

    Args:
        num_classes: Number of classes
        num_samples: Number of samples per class
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    cs_trajectories = []
    cs_predictions = []
    cs_lists = []
    cs_means = []
    for i in range(num_classes):
        class_trajectories = point_net_utils.select_samples(state.data.sub_samples, state.data.sub_labels, num_samples, i)
        cs_trajectories.append(class_trajectories)
        class_predictions = point_net.predict_critical(state.model.ntpn_model, class_trajectories, layer_name='activation_14')
        cs_predictions.append(class_predictions)
        class_cs, class_cs_mean = point_net.generate_critical(class_predictions, num_samples, class_trajectories)
        cs_lists.append(class_cs)
        cs_means.append(class_cs_mean)

    # Store in StateManager
    state.viz.cs_lists = cs_lists

    # Store additional data in legacy session_state (not yet in StateManager)
    st.session_state.cs_trajectories = cs_trajectories
    st.session_state.cs_predictions = cs_predictions
    st.session_state.cs_means = cs_means

    # Sync to legacy
    state.sync_to_legacy()

    return



def cs_downsample_PCA(label: int, num_examples: int, dims: int = 3, state: Optional[StateManager] = None) -> Tuple[Any, Any]:
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

    cs_trajectories = st.session_state.cs_trajectories  # Not yet in StateManager

    pca_cs, pca_trajs = point_net_utils.pca_cs_windowed(state.viz.cs_lists[label], cs_trajectories[label], dims=dims)

    pca_css, pca_trajss = point_net_utils.select_samples_cs(pca_cs, pca_trajs, num_examples)

    return pca_css, pca_trajss


def cs_downsample_UMAP(label: int, num_examples: int, dims: int = 3, state: Optional[StateManager] = None) -> Tuple[Any, Any]:
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

    cs_trajectories = st.session_state.cs_trajectories  # Not yet in StateManager

    umap_cs, umap_trajs = point_net_utils.umap_cs_windowed(state.viz.cs_lists[label], cs_trajectories[label], dims=dims)

    umap_css, umap_trajss = point_net_utils.select_samples_cs(umap_cs, umap_trajs, num_examples)

    return umap_css, umap_trajss


def cs_CCA_alignment(state: Optional[StateManager] = None) -> None:
    """Align critical sets using CCA (stub function).

    Args:
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    return



def plot_trajectories_UMAP(state: Optional[StateManager] = None) -> None:
    """Plot trajectories using UMAP (stub function).

    Args:
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    return


def plot_critical_sets_PCA(label: int, num_examples: int, dims: int = 3, state: Optional[StateManager] = None) -> Any:
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

    pca_css, pca_trajss = cs_downsample_PCA(label, num_examples, dims=dims, state=state)

    fig = point_net_utils.plot_critical(pca_css, num_examples, pca_trajss)

    return fig


def plot_critical_sets_UMAP(label: int, num_examples: int, dims: int = 3, state: Optional[StateManager] = None) -> Any:
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

    umap_css, umap_trajss = cs_downsample_UMAP(label, num_examples, dims=dims, state=state)

    fig = point_net_utils.plot_critical(umap_css, num_examples, umap_trajss)

    return fig


def plot_critical_sets_grid(state: Optional[StateManager] = None) -> None:
    """Plot critical sets in a grid (stub function).

    Args:
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    return


# DRAWING UTILITIES

def draw_image(image: npt.NDArray, header: str, description: str) -> None:
    """Draw an image with header and description in Streamlit.

    Args:
        image: Image array
        header: Header text
        description: Description text
    """
    # Draw header and image
    st.subheader(header)
    st.markdown(description)
    st.image(image.astype(np.uint8), use_container_width=True)
    return


def draw_cs_plots(plotting_algo: str, num_examples: int, dims: int, num_classes: int, state: Optional[StateManager] = None) -> None:
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

    st.session_state.cs_ub_plots = figs

    return




