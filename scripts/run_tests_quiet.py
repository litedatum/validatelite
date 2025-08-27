#!/usr/bin/env python3
"""
Quiet test runner script that suppresses debug and info messages.

Usage:
    python scripts/run_tests_quiet.py [pytest_options...]

Examples:
    python scripts/run_tests_quiet.py
    python scripts/run_tests_quiet.py -k "test_data_validator"
    python scripts/run_tests_quiet.py --cov=core --cov-report=html
"""

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run tests with quiet logging configuration."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Set environment variables for quiet logging
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    # Build pytest command with quiet options
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--log-cli-level=WARNING",
        "--tb=short",
        "-v",
    ]

    # Add any additional arguments passed to the script
    cmd.extend(sys.argv[1:])

    # Run pytest
    try:
        result = subprocess.run(cmd, env=env, cwd=project_root)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
