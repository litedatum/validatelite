# Development Setup Guide

## Overview

This guide helps you set up the ValidateLite development environment for contributing to the project.

## Prerequisites

- Python 3.11+
- Git
- Docker (optional, for database testing)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/validatelite.git
cd validatelite
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
# Option 1: Install from pinned requirements (recommended for production)
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Option 2: Install using pip-tools (recommended for development)
# First install pip-tools
pip install pip-tools
# Then install from .in files (this will generate pinned requirements)
pip-compile requirements.in
pip-compile requirements-dev.in
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install
```

## Development Workflow

### Running Tests

#### Unit Tests (No Database Required)
```bash
pytest tests/unit/ -v
```

#### Integration Tests (Requires Database)
```bash
# Option 1: Use Docker Compose for databases
docker-compose -f docker-compose.test.yml up -d
pytest tests/integration/ -v

# Option 2: Use local databases
# Set up MySQL and PostgreSQL locally, then:
export MYSQL_DB_URL="mysql://user:pass@localhost:3306/test_db"
export POSTGRESQL_DB_URL="postgresql://user:pass@localhost:5432/test_db"
pytest tests/integration/ -v
```

#### All Tests
```bash
pytest tests/ -v --cov=core --cov=shared --cov-report=html
```

### Code Quality Checks

```bash
# Run all pre-commit checks
pre-commit run --all-files

# Or run individual checks
black .
isort .
mypy .
flake8 .
```

### Building and Testing Package

```bash
# Build package
python -m build

# Test installation
pip install dist/*.whl

# Test CLI
validatelite --help
```

## Dependency Management

### Overview

This project uses a two-tier dependency management system:

1. **requirements.in** and **requirements-dev.in**: High-level dependency specifications
2. **requirements.txt** and **requirements-dev.txt**: Pinned versions for reproducible builds

### Adding New Dependencies

#### Production Dependencies
1. Add the dependency to `requirements.in`:
   ```bash
   echo "new-package>=1.0.0" >> requirements.in
   ```
2. Regenerate pinned requirements:
   ```bash
   python scripts/update_requirements.py
   ```

#### Development Dependencies
1. Add the dependency to `requirements-dev.in`:
   ```bash
   echo "new-dev-package>=1.0.0" >> requirements-dev.in
   ```
2. Regenerate pinned requirements:
   ```bash
   python scripts/update_requirements.py
   ```

### Updating Dependencies

To update all dependencies to their latest compatible versions:

```bash
# Update production dependencies
pip-compile requirements.in --upgrade

# Update development dependencies
pip-compile requirements-dev.in --upgrade

# Or use the convenience script
python scripts/update_requirements.py
```

### Using pip-tools

The project includes a convenience script for managing requirements:

```bash
# Install pip-tools if not already installed
pip install pip-tools

# Update all requirements files
python scripts/update_requirements.py
```

This script will:
- Check if pip-tools is installed
- Generate pinned requirements from .in files
- Provide helpful output and instructions

## Database Setup for Testing

### Using Docker (Recommended)

```bash
# Start test databases
docker-compose -f docker-compose.test.yml up -d

# Run tests
pytest tests/integration/ -v

# Stop databases
docker-compose -f docker-compose.test.yml down
```

### Using Local Databases

#### MySQL Setup
```bash
# Install MySQL
# On Ubuntu/Debian:
sudo apt-get install mysql-server

# On macOS:
brew install mysql

# Create test database
mysql -u root -p
CREATE DATABASE test_db;
CREATE USER 'test_user'@'localhost' IDENTIFIED BY 'test_password';
GRANT ALL PRIVILEGES ON test_db.* TO 'test_user'@'localhost';
FLUSH PRIVILEGES;
```

#### PostgreSQL Setup
```bash
# Install PostgreSQL
# On Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib

# On macOS:
brew install postgresql

# Create test database
sudo -u postgres psql
CREATE DATABASE test_db;
CREATE USER test_user WITH PASSWORD 'test_password';
GRANT ALL PRIVILEGES ON DATABASE test_db TO test_user;
```

## Configuration

### Environment Variables

Create a `.env` file for local development:

```bash
# Database URLs for testing
MYSQL_DB_URL=mysql://test_user:test_password@localhost:3306/test_db
POSTGRESQL_DB_URL=postgresql://test_user:test_password@localhost:5432/test_db

# Development settings
ENVIRONMENT=development
LOG_LEVEL=DEBUG
```

### Local Configuration

Copy example configuration files:

```bash
cp config/cli.toml.example config/cli.toml
# Edit config/cli.toml as needed
```

## Contributing

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow the coding standards (PEP 8, Black formatting)
- Add tests for new functionality
- Update documentation as needed

### 3. Run Quality Checks

```bash
# Run all checks
pre-commit run --all-files

# Run tests
pytest tests/ -v --cov=core --cov=shared
```

### 4. Commit and Push

```bash
git add .
git commit -m "feat: add new feature"
git push origin feature/your-feature-name
```

### 5. Create Pull Request

- Go to GitHub and create a pull request
- Ensure all CI checks pass
- Request review from maintainers

## Troubleshooting

### Common Issues

#### 1. Database Connection Issues
```bash
# Check if databases are running
docker-compose -f docker-compose.test.yml ps

# Check database logs
docker-compose -f docker-compose.test.yml logs mysql
docker-compose -f docker-compose.test.yml logs postgres
```

#### 2. Python Path Issues
```bash
# Ensure you're in the project root
pwd  # Should show /path/to/validatelite

# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 3. Pre-commit Hook Failures
```bash
# Update pre-commit hooks
pre-commit autoupdate

# Run specific hook
pre-commit run black --all-files
```

#### 4. Test Failures
```bash
# Clear test cache
pytest --cache-clear

# Run with verbose output
pytest tests/ -v -s

# Run specific test
pytest tests/unit/test_specific.py::test_function -v
```

### Getting Help

- Check the [README.md](README.md) for general information
- Review [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
- Open an issue on GitHub for bugs or feature requests
- Join discussions in GitHub Discussions

## IDE Setup

### VS Code

Install recommended extensions:
- Python
- Pylance
- Black Formatter
- isort

### PyCharm

Configure:
- Set project interpreter to virtual environment
- Enable Black formatter
- Configure test runner to use pytest

## Performance Tips

### Faster Test Execution

```bash
# Run tests in parallel
pytest tests/ -n auto

# Run only changed tests
pytest tests/ --lf

# Run tests without coverage for speed
pytest tests/ --no-cov
```

### Faster Development Cycle

```bash
# Use watch mode for tests (requires pytest-watch)
pip install pytest-watch
ptw tests/ -- -v

# Use auto-reload for development
pip install watchdog
python -m watchdog.watchmedo auto-restart --pattern="*.py" --recursive -- python cli_main.py
```
