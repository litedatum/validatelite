#!/usr/bin/env python3
"""
E2E Test Runner

Script to run E2E tests with different configurations and options.
"""

import argparse
import subprocess
import sys


def run_pytest_command(args: list[str]) -> int:
    """Run pytest with the given arguments."""
    cmd = ["python", "-m", "pytest"] + args
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E2E tests for data quality CLI")
    parser.add_argument(
        "--test-type",
        choices=["all", "basic", "database", "sqlite", "mysql", "postgres"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--data-source",
        choices=["all", "sqlite", "mysql", "postgres"],
        default="all",
        help="Data source to test against",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--coverage", action="store_true", help="Run with coverage report"
    )
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument(
        "--markers", nargs="+", help="Run only tests with specific markers"
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Test timeout in seconds"
    )

    args = parser.parse_args()

    # Base pytest arguments
    pytest_args = [
        "tests/e2e/",
        "--tb=short",
        f"--timeout={args.timeout}",
    ]

    # Add verbose flag
    if args.verbose:
        pytest_args.append("-v")

    # Add coverage if requested
    if args.coverage:
        pytest_args.extend(
            ["--cov=cli", "--cov=core", "--cov=shared", "--cov-report=html"]
        )

    # Add parallel execution if requested
    if args.parallel:
        pytest_args.extend(["-n", "auto"])

    # Filter by test type
    if args.test_type == "basic":
        pytest_args.extend(["-k", "not database"])
    elif args.test_type == "database":
        pytest_args.extend(["-k", "database"])
    elif args.test_type == "sqlite":
        pytest_args.extend(["-k", "sqlite"])
    elif args.test_type == "mysql":
        pytest_args.extend(["-k", "mysql"])
    elif args.test_type == "postgres":
        pytest_args.extend(["-k", "postgres"])

    # Filter by data source
    if args.data_source != "all":
        pytest_args.extend(
            [
                "-k",
                (
                    f"test_data/customers.xlsx"
                    if args.data_source == "sqlite"
                    else args.data_source
                ),
            ]
        )

    # Add markers if specified
    if args.markers:
        for marker in args.markers:
            pytest_args.extend(["-m", marker])

    # Run the tests
    print("Starting E2E tests...")
    print(f"Test type: {args.test_type}")
    print(f"Data source: {args.data_source}")
    print(f"Verbose: {args.verbose}")
    print(f"Coverage: {args.coverage}")
    print(f"Parallel: {args.parallel}")

    exit_code = run_pytest_command(pytest_args)

    if exit_code == 0:
        print("\n✅ All E2E tests passed!")
    else:
        print(f"\n❌ E2E tests failed with exit code {exit_code}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
