"""
Logging configuration for NTPN application.

Provides centralized logging setup using Python's stdlib logging module.
Call setup_logging() once at application startup.

Usage:
    from ntpn.logging_config import setup_logging, get_logger

    setup_logging()  # Call once at startup
    logger = get_logger(__name__)
    logger.info("Operation completed")
"""

import logging

# Root namespace for all NTPN loggers
_NAMESPACE = 'ntpn'

# Track whether logging has been set up
_initialized = False


def setup_logging(
    level: int = logging.INFO,
    format_string: str | None = None,
    log_file: str | None = None,
) -> None:
    """Configure logging for the NTPN application.

    Safe to call multiple times; subsequent calls are no-ops unless
    the root ntpn logger has no handlers.

    Args:
        level: Logging level (default: logging.INFO)
        format_string: Custom format string (default: timestamp + name + level + message)
        log_file: Optional file path to write logs to
    """
    global _initialized

    root_logger = logging.getLogger(_NAMESPACE)

    # Skip if already configured (idempotent)
    if _initialized and root_logger.handlers:
        return

    root_logger.setLevel(level)

    if format_string is None:
        format_string = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the ntpn namespace.

    Args:
        name: Logger name (typically __name__). Will be prefixed with 'ntpn.'
              if not already.

    Returns:
        Logger instance
    """
    if name.startswith(_NAMESPACE):
        return logging.getLogger(name)
    return logging.getLogger(f'{_NAMESPACE}.{name}')
