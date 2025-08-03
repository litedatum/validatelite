#!/usr/bin/env python3
"""
Script to update requirements files using pip-tools.

This script helps maintain pinned requirements files from the .in files.
It requires pip-tools to be installed: pip install pip-tools
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> str:
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✓ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        sys.exit(1)


def main() -> None:
    """Main function to update requirements files."""
    project_root = Path(__file__).parent.parent

    print("Updating requirements files using pip-tools...")
    print("=" * 50)

    # Check if pip-tools is installed
    try:
        subprocess.run(["pip-compile", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("pip-tools not found. Installing...")
        run_command("pip install pip-tools", "Installing pip-tools")

    # Update production requirements
    requirements_in = project_root / "requirements.in"
    requirements_txt = project_root / "requirements.txt"

    if requirements_in.exists():
        cmd = (
            f"pip-compile {requirements_in} --output-file {requirements_txt} --upgrade"
        )
        run_command(cmd, "Updating production requirements")
    else:
        print("Warning: requirements.in not found")

    # Update development requirements
    requirements_dev_in = project_root / "requirements-dev.in"
    requirements_dev_txt = project_root / "requirements-dev.txt"

    if requirements_dev_in.exists():
        cmd = f"pip-compile {requirements_dev_in} --output-file"
        f" {requirements_dev_txt} --upgrade"
        run_command(cmd, "Updating development requirements")
    else:
        print("Warning: requirements-dev.in not found")

    print("\n" + "=" * 50)
    print("Requirements update completed!")
    print("\nTo install dependencies:")
    print("  Production: pip install -r requirements.txt")
    print("  Development: pip install -r requirements-dev.txt")
    print("\nTo add new dependencies:")
    print("  1. Add to requirements.in or requirements-dev.in")
    print("  2. Run this script again")


if __name__ == "__main__":
    main()
