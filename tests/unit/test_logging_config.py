"""Unit tests for ntpn.logging_config module."""

import logging

import pytest

# Reset the module state before import
import ntpn.logging_config as logging_config


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging state before each test."""
    # Clear all handlers from the ntpn logger
    ntpn_logger = logging.getLogger('ntpn')
    ntpn_logger.handlers.clear()
    ntpn_logger.setLevel(logging.WARNING)  # Reset level

    # Reset module flag
    logging_config._initialized = False
    yield
    # Cleanup after
    ntpn_logger.handlers.clear()
    logging_config._initialized = False


class TestSetupLogging:
    """Tests for setup_logging."""

    def test_creates_console_handler(self):
        """setup_logging creates a console handler."""
        logging_config.setup_logging()
        ntpn_logger = logging.getLogger('ntpn')
        assert len(ntpn_logger.handlers) >= 1
        assert any(isinstance(h, logging.StreamHandler) for h in ntpn_logger.handlers)

    def test_sets_level(self):
        """setup_logging sets the specified level."""
        logging_config.setup_logging(level=logging.DEBUG)
        ntpn_logger = logging.getLogger('ntpn')
        assert ntpn_logger.level == logging.DEBUG

    def test_default_level_is_info(self):
        """Default level is INFO."""
        logging_config.setup_logging()
        ntpn_logger = logging.getLogger('ntpn')
        assert ntpn_logger.level == logging.INFO

    def test_file_handler(self, tmp_path):
        """setup_logging creates a file handler when log_file is specified."""
        log_file = str(tmp_path / 'test.log')
        logging_config.setup_logging(log_file=log_file)
        ntpn_logger = logging.getLogger('ntpn')
        file_handlers = [h for h in ntpn_logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

    def test_idempotent(self):
        """Calling setup_logging twice does not duplicate handlers."""
        logging_config.setup_logging()
        handler_count_1 = len(logging.getLogger('ntpn').handlers)

        logging_config.setup_logging()
        handler_count_2 = len(logging.getLogger('ntpn').handlers)

        assert handler_count_1 == handler_count_2

    def test_custom_format(self):
        """Custom format string is applied."""
        custom_format = '%(levelname)s - %(message)s'
        logging_config.setup_logging(format_string=custom_format)
        ntpn_logger = logging.getLogger('ntpn')
        handler = ntpn_logger.handlers[0]
        assert handler.formatter._fmt == custom_format


class TestGetLogger:
    """Tests for get_logger."""

    def test_returns_logger(self):
        """get_logger returns a Logger instance."""
        result = logging_config.get_logger('test')
        assert isinstance(result, logging.Logger)

    def test_namespaced_logger(self):
        """get_logger prefixes name with 'ntpn.'."""
        result = logging_config.get_logger('mymodule')
        assert result.name == 'ntpn.mymodule'

    def test_already_namespaced(self):
        """Names already under ntpn namespace are not double-prefixed."""
        result = logging_config.get_logger('ntpn.mymodule')
        assert result.name == 'ntpn.mymodule'

    def test_child_loggers_inherit_level(self):
        """Child loggers inherit the root ntpn logger's level."""
        logging_config.setup_logging(level=logging.DEBUG)
        child_logger = logging_config.get_logger('child')
        assert child_logger.getEffectiveLevel() == logging.DEBUG

    def test_different_names_different_loggers(self):
        """Different names return different logger instances."""
        logger_a = logging_config.get_logger('module_a')
        logger_b = logging_config.get_logger('module_b')
        assert logger_a is not logger_b
        assert logger_a.name != logger_b.name
