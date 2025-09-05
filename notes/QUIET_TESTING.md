# Quiet Testing Guide

This guide explains how to run tests with minimal logging output to keep test results clean and readable.

## Overview

By default, pytest and our test environment can produce verbose logging output from:
- aiosqlite (SQL execution debug messages)
- SQLAlchemy (database connection and pool messages)
- Shared modules (INFO level messages)
- Third-party libraries

We've configured multiple ways to suppress these messages for a cleaner testing experience.

## Configuration Options

### 1. Global pytest.ini Configuration

The main `pytest.ini` file is configured to suppress most logging by default:

```ini
[pytest]
addopts =
    --log-cli-level=ERROR
    # ... other options
```

This means:
- ✅ DEBUG messages are suppressed
- ✅ INFO messages are suppressed  
- ✅ WARNING messages are suppressed
- ❌ Only ERROR and CRITICAL messages are shown

### 2. Test-Specific Logging Configuration

A dedicated test logging configuration file `config/logging.test.toml` provides granular control:

```toml
# Test Environment Logging Configuration
level = "WARNING"

[module_levels]
# Core modules
"shared.database.connection" = "WARNING"
"shared.database.query_executor" = "WARNING"

# Third-party modules
"aiosqlite" = "ERROR"
"sqlalchemy" = "ERROR"
"sqlalchemy.engine" = "ERROR"
```

### 3. Conftest.py Configuration

The `tests/conftest.py` file automatically loads and applies test logging configuration:

```python
# Load test-specific logging configuration
try:
    test_logging_config = load_config("logging.test.toml")
    if test_logging_config:
        # Apply test logging configuration
        for module, level in test_logging_config.get("module_levels", {}).items():
            _logging.getLogger(module).setLevel(getattr(_logging, level.upper()))
except Exception:
    # Fallback to default configuration if test config not found
    pass
```

## Usage Methods

### Method 1: Use pytest directly (Recommended)

```bash
# Run all tests with quiet logging (default behavior)
pytest

# Run specific tests
pytest tests/unit/ -v

# Override logging level if needed
pytest --log-cli-level=INFO tests/unit/ -v
```

### Method 2: Use the quiet test runner script

```bash
# Run all tests quietly
python scripts/run_tests_quiet.py

# Run specific tests
python scripts/run_tests_quiet.py tests/unit/ -v

# With coverage
python scripts/run_tests_quiet.py --cov
```

### Method 3: Set environment variables

```bash
# Set global pytest options
export PYTEST_ADDOPTS="--log-cli-level=ERROR"

# Run tests
pytest tests/unit/ -v
```

## Logging Levels Explained

| Level | Description | What You'll See |
|-------|-------------|-----------------|
| `DEBUG` | Detailed debug information | ❌ Suppressed |
| `INFO` | General information messages | ❌ Suppressed |
| `WARNING` | Warning messages | ❌ Suppressed |
| `ERROR` | Error messages | ✅ Visible |
| `CRITICAL` | Critical errors | ✅ Visible |

## What Gets Suppressed

### ✅ Successfully Suppressed
- aiosqlite SQL execution debug messages
- SQLAlchemy database connection debug messages
- Database connection pool debug messages
- Shared module INFO level messages
- Third-party library verbose output

### ⚠️ Still Visible (if needed)
- Test failures and errors
- Coverage reports
- Critical error messages
- Test collection information

## Customizing for Debugging

When you need to debug tests, you can temporarily increase logging verbosity:

```bash
# Show INFO messages
pytest --log-cli-level=INFO tests/unit/ -v

# Show WARNING messages  
pytest --log-cli-level=WARNING tests/unit/ -v

# Show all messages (including DEBUG)
pytest --log-cli-level=DEBUG tests/unit/ -v
```

## Troubleshooting

### Issue: Still seeing debug messages
**Solution**: Check if the message is coming from stderr capture rather than pytest logging:
```bash
# Run with -s to disable stderr capture
pytest -s tests/unit/ -v
```

### Issue: Need to see specific module logs
**Solution**: Override specific module logging in conftest.py:
```python
# Temporarily enable INFO for specific module
_logging.getLogger("shared.database.connection").setLevel(_logging.INFO)
```

### Issue: Logging configuration not working
**Solution**: Verify the configuration files are being loaded:
```bash
# Check if test logging config is loaded
python -c "from shared.config.loader import load_config; print(load_config('logging.test.toml'))"
```

## Best Practices

1. **Use the default quiet configuration** for regular testing
2. **Use the quiet test runner script** for CI/CD pipelines
3. **Temporarily increase verbosity** only when debugging
4. **Keep test output clean** by maintaining the ERROR level default
5. **Use module-specific overrides** when you need detailed logging for specific components

## Configuration Files

- `pytest.ini` - Main pytest configuration with quiet logging
- `config/logging.test.toml` - Test-specific logging configuration
- `tests/conftest.py` - Test environment setup and logging configuration
- `scripts/run_tests_quiet.py` - Convenient script for quiet test execution

## Examples

### Clean test run (default)
```bash
pytest tests/unit/ -v
# Output: Clean test results, no debug noise
```

### Debug specific test
```bash
pytest --log-cli-level=INFO tests/unit/test_specific.py -v
# Output: Test results + INFO level messages for debugging
```

### Run integration tests quietly
```bash
python scripts/run_tests_quiet.py tests/integration/ -v
# Output: Clean integration test results
```

This configuration ensures that your test output is clean and focused on test results rather than logging noise, while still providing the ability to enable detailed logging when needed for debugging.
