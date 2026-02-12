"""Unit tests for ntpn.model_service module."""

from unittest.mock import MagicMock, patch

import pytest
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


class TestCreateModel:
    """Tests for create_model."""

    @patch('ntpn.model_service.point_net')
    def test_creates_model_and_stores_in_state(self, mock_pn, state):
        """Model is created and stored in state."""
        mock_model = MagicMock()
        mock_pn.point_net.return_value = mock_model

        from ntpn.model_service import create_model

        create_model(32, 2, 32, 11, state=state)

        mock_pn.point_net.assert_called_once_with(32, 2, units=32, dims=11)
        assert state.model.ntpn_model is mock_model

    @patch('ntpn.model_service.point_net')
    def test_syncs_to_legacy(self, mock_pn, state):
        """State is synced to legacy after model creation."""
        mock_pn.point_net.return_value = MagicMock()

        from ntpn.model_service import create_model

        create_model(32, 2, 32, 11, state=state)

        assert 'ntpn_model' in st.session_state


class TestCompileModel:
    """Tests for compile_model."""

    @patch('ntpn.model_service.point_net')
    def test_compile_with_view_sets_training_infra(self, mock_pn, state):
        """When view=True, training infrastructure (loss, metrics, optimizer) is set."""
        mock_model = MagicMock()
        mock_pn.point_net.return_value = mock_model
        state.model.ntpn_model = mock_model

        from ntpn.model_service import compile_model

        compile_model(learning_rate=0.01, view=True, state=state)

        assert state.model.loss_fn is not None
        assert state.model.train_metric is not None
        assert state.model.test_metric is not None
        assert state.model.optimizer is not None
        assert state.model.learning_rate == 0.01

    @patch('ntpn.model_service.point_net')
    def test_compile_calls_model_compile(self, mock_pn, state):
        """Model.compile is called with correct args."""
        mock_model = MagicMock()
        state.model.ntpn_model = mock_model

        from ntpn.model_service import compile_model

        compile_model(loss='sparse_categorical_crossentropy', learning_rate=0.02, state=state)

        mock_model.compile.assert_called_once()
        call_kwargs = mock_model.compile.call_args
        assert call_kwargs[1]['loss'] == 'sparse_categorical_crossentropy'


class TestTrainStep:
    """Tests for train_step."""

    def test_train_step_returns_loss(self, state):
        """train_step executes one training step and returns loss."""
        import tensorflow as tf
        from tensorflow import keras

        # Create a minimal model
        model = keras.Sequential([keras.layers.Dense(2, activation='softmax', input_shape=(3,))])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])

        state.model.ntpn_model = model
        state.model.loss_fn = keras.losses.SparseCategoricalCrossentropy()
        state.model.train_metric = keras.metrics.SparseCategoricalAccuracy()
        state.model.optimizer = keras.optimizers.Adam()

        x = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
        y = tf.constant([0], dtype=tf.int32)

        from ntpn.model_service import train_step

        loss = train_step(x, y, state=state)

        assert loss is not None
        assert float(loss) >= 0


class TestTestStep:
    """Tests for test_step."""

    def test_test_step_returns_loss(self, state):
        """test_step executes one validation step and returns loss."""
        import tensorflow as tf
        from tensorflow import keras

        model = keras.Sequential([keras.layers.Dense(2, activation='softmax', input_shape=(3,))])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

        state.model.ntpn_model = model
        state.model.loss_fn = keras.losses.SparseCategoricalCrossentropy()
        state.model.test_metric = keras.metrics.SparseCategoricalAccuracy()

        x = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
        y = tf.constant([0], dtype=tf.int32)

        from ntpn.model_service import test_step

        loss = test_step(x, y, state=state)

        assert loss is not None
        assert float(loss) >= 0


class TestSaveModel:
    """Tests for save_model."""

    def test_save_model_calls_model_save(self, state, tmp_path):
        """save_model calls model.save with correct path."""
        mock_model = MagicMock()
        state.model.ntpn_model = mock_model

        from ntpn.model_service import save_model

        save_model('test_model', state=state)

        mock_model.save.assert_called_once_with(
            'models/test_model.keras',
            overwrite=True,
            include_optimizer=True,
        )
