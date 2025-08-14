"""
ValidateLite CLI Package

Command-line interface for the data quality validation tool.
Provides a unified `vlite-cli check` command for data quality checking.
"""

__version__ = "0.4.0"

from .app import cli_app

__all__ = ["cli_app"]
