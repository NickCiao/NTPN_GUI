"""
Model lifecycle service for NTPN.

This module provides model creation, compilation, training steps, and saving
with no Streamlit dependency.

@author: proxy_loken
"""

from typing import Any

import tensorflow as tf
from tensorflow import keras

from ntpn import ntpn_constants, point_net
from ntpn.logging_config import get_logger
from ntpn.state_manager import StateManager, get_state_manager

logger = get_logger(__name__)


def create_model(
    trajectory_length: int,
    num_classes: int,
    layer_width: int,
    trajectory_dim: int,
    state: StateManager | None = None,
) -> None:
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

    logger.info(
        'Creating PointNet model: trajectory_length=%d, num_classes=%d, layer_width=%d, dims=%d',
        trajectory_length,
        num_classes,
        layer_width,
        trajectory_dim,
    )
    state.model.ntpn_model = point_net.point_net(trajectory_length, num_classes, units=layer_width, dims=trajectory_dim)

    state.sync_to_legacy()


def compile_model(
    loss: str = 'sparse_categorical_crossentropy',
    learning_rate: float = ntpn_constants.DEFAULT_LEARNING_RATE,
    metric: str = 'sparse_categorical_accuracy',
    view: bool = True,
    state: StateManager | None = None,
) -> None:
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

    logger.info('Compiling model: loss=%s, lr=%f, metric=%s, view=%s', loss, learning_rate, metric, view)
    state.model.learning_rate = learning_rate

    if view:
        state.model.loss_fn = keras.losses.SparseCategoricalCrossentropy()
        state.model.train_metric = keras.metrics.SparseCategoricalAccuracy()
        state.model.test_metric = keras.metrics.SparseCategoricalAccuracy()
        state.model.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    state.model.ntpn_model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[metric],
    )

    state.sync_to_legacy()


def train_step(x: Any, y: Any, state: StateManager | None = None) -> tf.Tensor:
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
        loss_value = state.model.loss_fn(y, predictions)

    grads = tape.gradient(loss_value, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    for metric in model.metrics:
        if metric.name == 'loss':
            metric.update_state(loss_value)
        else:
            metric.update_state(y, predictions)
    state.model.train_metric.update_state(y, predictions)
    return loss_value


def test_step(x: Any, y: Any, state: StateManager | None = None) -> tf.Tensor:
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
    state.model.test_metric.update_state(y, val_predictions)
    return state.model.loss_fn(y, val_predictions)


def train_model_headless(
    epochs: int,
    state: StateManager | None = None,
) -> None:
    """Train the model without UI (standard Keras fit).

    Args:
        epochs: Number of training epochs
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    state.model.ntpn_model.fit(
        state.model.train_tensors,
        epochs=epochs,
        validation_data=state.model.test_tensors,
    )


def save_model(
    model_name: str,
    state: StateManager | None = None,
) -> None:
    """Save the trained model to disk.

    Args:
        model_name: Name for the saved model file
        state: StateManager instance (uses singleton if not provided)
    """
    if state is None:
        state = get_state_manager()

    logger.info('Saving model as %s%s.keras', ntpn_constants.MODEL_SAVE_DIR, model_name)
    state.model.ntpn_model.save(
        ntpn_constants.MODEL_SAVE_DIR + model_name + '.keras',
        overwrite=True,
        include_optimizer=True,
    )
