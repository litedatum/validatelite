#!/usr/bin/env python3
"""
ValidateLite CLI Main Entry Point

Main entry point for the vlite command-line tool.
"""

import os
import sys

from shared.config.logging_config import get_logging_config  # noqa: E402
from shared.utils.logger import setup_logging as setup_logger_manager  # noqa: E402

# ---------------------------------------------------------------------------
# Configure logging *before* importing any module that may create loggers.
# This guarantees that all subsequent loggers pick up the desired settings
# and avoids the INFO-level leakage observed previously.
# ---------------------------------------------------------------------------


setup_logger_manager(get_logging_config().model_dump())

# ---------------------------------------------------------------------------
# Now that logging is configured we can safely import the rest of the stack.
# ---------------------------------------------------------------------------

from cli.app import main  # noqa: E402
from cli.core.config import get_cli_config  # noqa: E402
from core.config import get_core_config  # noqa: E402
from shared.config import register_config  # noqa: E402

# Ensure project root is on PYTHONPATH for direct execution
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def bootstrap() -> None:
    """Initialise configuration registry (logging is already configured)."""

    # Load configurations
    logging_config = get_logging_config()
    core_config = get_core_config()
    cli_config = get_cli_config()

    # Register configurations for global access
    register_config("logging", logging_config)
    register_config("core", core_config)
    register_config("cli", cli_config)


if __name__ == "__main__":
    bootstrap()
    main()
