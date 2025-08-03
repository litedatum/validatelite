"""
CLI configuration module

Provides configuration for the CLI application, including database connection
settings, query timeout, and sample size.
"""

import os
from typing import Optional, cast

from pydantic import BaseModel, Field

from shared.config.loader import load_config


class DatabaseConfig(BaseModel):
    """Database configuration"""

    url: Optional[str] = Field(
        None, description="Database connection URL (e.g., 'sqlite:///temp.db')."
    )
    connect_timeout: int = Field(
        30, description="Database connection timeout in seconds."
    )
    echo_queries: bool = Field(
        False, description="Log all SQL queries issued to the database."
    )


class CliConfig(BaseModel):
    """CLI configuration"""

    # General
    debug_mode: bool = Field(
        False, description="Enable debug mode for verbose error output."
    )

    # Data Source
    default_sample_size: int = Field(
        10000, description="Number of records to sample for analysis."
    )
    max_file_size_mb: int = Field(
        100, description="Maximum file size in MB to load into memory."
    )

    # Database Connection
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    query_timeout: int = Field(
        300, description="Timeout for database queries initiated by the CLI."
    )


def get_cli_config() -> CliConfig:
    """Return the CLI configuration instance (delegation-aware).

    Mirrors the behaviour implemented in `core.config.get_core_config` so that
    previously imported aliases of this function still delegate to any
    monkey-patched replacement created by the test-suite.
    """

    current_callable = globals().get("get_cli_config")
    if (
        current_callable is not _ORIGINAL_GET_CLI_CONFIG
        and current_callable is not None
    ):
        return cast(CliConfig, current_callable())

    config_path_from_env = os.getenv("CLI_CONFIG_PATH")

    if config_path_from_env:
        config_path = config_path_from_env
    else:
        config_path = "config/cli.toml"

    try:
        return load_config(config_path, CliConfig)
    except FileNotFoundError:
        print(
            f"Warning: Configuration file not found at {config_path}. "
            "Using default values."
        )
        return CliConfig()


# Sentinel for original function (defined after function definition)

# NOTE: It is important this assignment happens *after* the function definition
# so that it captures the original function object before any external patching.

_ORIGINAL_GET_CLI_CONFIG = get_cli_config
