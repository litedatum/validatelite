"""
Logging configuration settings.

Controls the behavior of application logging, including log levels,
formatting, and output destinations.
"""

import os
from typing import Dict, cast

from pydantic import BaseModel, Field

from shared.config.loader import load_config


class LoggingConfig(BaseModel):
    """
    Logging configuration settings.

    Controls the behavior of application logging, including log levels,
    formatting, and output destinations.
    """

    level: str = Field(
        "INFO", description="Logging level (e.g., DEBUG, INFO, WARNING)."
    )
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format string.",
    )
    to_file: bool = Field(False, description="Enable logging to a file.")
    file_path: str = Field("logs/app.log", description="Path to the log file.")
    max_bytes: int = Field(
        10485760, description="Maximum log file size in bytes (10MB default)."
    )
    backup_count: int = Field(5, description="Number of backup log files to keep.")
    module_levels: Dict[str, str] = Field(
        default_factory=dict,
        description="Module-specific log levels to override global level.",
    )


def get_logging_config() -> LoggingConfig:
    """Return the logging configuration (delegation-aware).

    Ensures that any monkey-patched version of ``get_logging_config`` is
    honoured even by aliases imported *before* the patch (mirroring the
    strategy used for core and CLI configuration helpers).
    """

    current_callable = globals().get("get_logging_config")
    if (
        current_callable is not _ORIGINAL_GET_LOGGING_CONFIG
        and current_callable is not None
    ):
        return cast(LoggingConfig, current_callable())

    config_path_from_env = os.getenv("LOGGING_CONFIG_PATH")

    if config_path_from_env:
        config_path = config_path_from_env
    else:
        config_path = "config/logging.toml"

    try:
        return load_config(config_path, LoggingConfig)
    except FileNotFoundError:
        print(
            f"Warning: Logging configuration not found at {config_path}. "
            "Using default values."
        )
        return LoggingConfig()


# Sentinel for patch detection
_ORIGINAL_GET_LOGGING_CONFIG = get_logging_config
