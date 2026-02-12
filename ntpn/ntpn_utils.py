#!/usr/bin/env python3
"""
Streamlit-specific utilities for the NTPN Application.

This module contains only Streamlit-specific functions that cannot live
in the service layer (which must remain Streamlit-free).

@author: proxy_loken
"""

import time

import numpy as np
import numpy.typing as npt
import streamlit as st

from ntpn import ntpn_constants
from ntpn.data_service import load_demo_session
from ntpn.logging_config import get_logger
from ntpn.model_service import test_step, train_step
from ntpn.state_manager import StateManager, get_state_manager

logger = get_logger(__name__)

# STREAMLIT-SPECIFIC FUNCTIONS (remain in this module)


def initialise_session(state: StateManager | None = None) -> None:
    """Initialize session with default data.

    Args:
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    # Load demo data if not already loaded
    if state.data.dataset_name == 'demo_data' and not state.data.is_loaded():
        load_demo_session(state)

    # Initialize model if not set
    if state.model.ntpn_model is None:
        state.model.ntpn_model = None  # Will be created later

    # Sync to legacy for backward compatibility
    state.sync_to_legacy()

    return


def train_for_streamlit(epochs: int, state: StateManager | None = None) -> None:
    """Train model with Streamlit progress display.

    Replaces keras fit method to enable progress to be displayed inside streamlit.
    Adapted from shubhadtiya goswami.

    Args:
        epochs: Number of training epochs
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    # Store batch size in state
    state.model.batch_size = ntpn_constants.DEFAULT_BATCH_SIZE

    logger.info('Starting Streamlit training with %d epochs', epochs)
    st.write(f'Starting Training with {epochs} epochs...')

    for epoch in range(epochs):
        st.write(f'Epoch {epoch + 1}')
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
            epoch_time += end_step - start_step
            train_loss_list.append(float(loss_value))

            # number of steps to log
            if step % 1 == 0:
                step_acc = float(state.model.train_metric.result())
                percent_complete = step / max(1, len(state.data.sub_samples) // state.model.batch_size)
                progress_bar.progress(percent_complete)
                st_t.write(f'Duration : {epoch_time:.2f}s, Training Acc : {float(step_acc):.4f}')

        progress_bar.progress(1.0)

        # Metrics for the end of each epoch
        train_acc = state.model.train_metric.result()
        # reset training metric at the end of each epoch
        state.model.train_metric.reset_state()

        train_loss = round((sum(train_loss_list) / len(train_loss_list)), 5) if train_loss_list else 0.0

        val_loss_list = []
        # run the validation loop
        for x_batch_val, y_batch_val in state.model.test_tensors:
            val_loss_list.append(float(test_step(x_batch_val, y_batch_val, state=state)))

        val_loss = round((sum(val_loss_list) / len(val_loss_list)), 5) if val_loss_list else 0.0

        val_acc = state.model.test_metric.result()
        state.model.test_metric.reset_state()

        st_t.write(
            f'Duration : {time.time() - start_time:.2f}s, Training Acc : {float(train_acc):.4f}, Validation Acc : {float(val_acc):.4f}'
        )

    return


def train_model(epochs: int, view: bool = True, state: StateManager | None = None) -> None:
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
        from ntpn.model_service import train_model_headless

        train_model_headless(epochs, state=state)

    return


def draw_image(image: npt.NDArray, header: str, description: str) -> None:
    """Draw an image with header and description in Streamlit.

    Args:
        image: Image array
        header: Header text
        description: Description text
    """
    st.subheader(header)
    st.markdown(description)
    st.image(image.astype(np.uint8), use_container_width=True)
    return
