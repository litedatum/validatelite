"""
Logging setup utility

Provides functions for setting up logging in the application.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from shared.config.logging_config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """
    Configures the global logging settings for the application.

    Args:
        config: LoggingConfig object containing logging configuration
    """
    # Convert string level to logging level constant
    log_level = getattr(logging, config.level.upper(), logging.INFO)

    # Clear existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(config.format)

    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)  # Set console handler level from config
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if enabled
    if config.to_file:
        # Ensure the log directory exists
        log_dir = os.path.dirname(config.file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Use rotating file handler with size limits
        file_handler = RotatingFileHandler(
            filename=config.file_path,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
        )
        file_handler.setLevel(log_level)  # Set file handler level from config
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set root logger level
    root_logger.setLevel(log_level)

    # Set specific levels for noisy third-party loggers
    logging.getLogger("pydantic").setLevel(logging.WARNING)
    logging.getLogger("toml").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

    # Apply module-specific log levels from config
    for module_name, module_level in config.module_levels.items():
        module_log_level = getattr(logging, module_level.upper(), logging.INFO)
        logging.getLogger(module_name).setLevel(module_log_level)

    # Log configuration summary
    root_logger.debug(
        f"Logging initialized: level={config.level}, to_file={config.to_file}"
    )
    if config.to_file:
        root_logger.debug(
            f"Log file: {config.file_path} (max: {config.max_bytes/1024/1024:.1f}MB, "
            f"backups: {config.backup_count})"
        )
