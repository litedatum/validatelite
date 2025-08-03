# ValidateLite

A flexible, extensible command-line tool for automated data quality validation, profiling, and rule-based checks across diverse data sources. Designed for data engineers, analysts, and developers to ensure data reliability and compliance in modern data pipelines.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://img.shields.io/badge/coverage-80%25-green.svg)](https://github.com/litedatum/validatelite)

---

## 📝 Development Blog

Follow the journey of building ValidateLite through our development blog posts:

- **[DevLog #1: Building a Zero-Config Data Validation Tool](https://blog.litedatum.com/posts/Devlog01-data-validation-tool/)** - The initial vision and architecture of ValidateLite
- **[DevLog #2: Why I Scrapped My Half-Built Data Validation Platform](https://blog.litedatum.com/posts/Devlog02-Rethinking-My-Data-Validation-Tool/)** - Lessons learned from scope creep and the pivot to a focused CLI tool

---

## 🚀 Quick Start

### For Regular Users

**Option 1: Install from PyPI (Recommended)**
```bash
pip install validatelite
vlite --help
```

**Option 2: Install from pre-built package**
```bash
# Download the latest release from GitHub
pip install validatelite-0.1.0-py3-none-any.whl
vlite --help
```

**Option 3: Run from source**
```bash
git clone https://github.com/litedatum/validatelite.git
cd validatelite
pip install -r requirements.txt
python cli_main.py --help
```

**Option 4: Install with pip-tools (for development)**
```bash
git clone https://github.com/litedatum/validatelite.git
cd validatelite
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
python cli_main.py --help
```

### For Developers & Contributors

If you want to contribute to the project or need the latest development version:

```bash
git clone https://github.com/litedatum/validatelite.git
cd validatelite

# Install dependencies (choose one approach)
# Option 1: Install from pinned requirements
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Option 2: Use pip-tools for development
pip install pip-tools
python scripts/update_requirements.py
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

See [DEVELOPMENT_SETUP.md](docs/DEVELOPMENT_SETUP.md) for detailed development setup instructions.

---

## ✨ Features

- **🔧 Rule-based Data Quality Engine**: Supports completeness, uniqueness, validity, and custom rules
- **🖥️ Extensible CLI**: Easily integrate with CI/CD and automation workflows
- **🗄️ Multi-Source Support**: Validate data from files (CSV, Excel) and databases (MySQL, PostgreSQL, SQLite)
- **⚙️ Configurable & Modular**: Flexible configuration via TOML and environment variables
- **🛡️ Comprehensive Error Handling**: Robust exception and error classification system
- **🧪 Tested & Reliable**: High code coverage, modular tests, and pre-commit hooks

---

## 📖 Documentation

- **[USAGE.md](docs/USAGE.md)** - Complete user guide with examples and best practices
- **[DEVELOPMENT_SETUP.md](docs/DEVELOPMENT_SETUP.md)** - Development environment setup and contribution guidelines
- **[CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md)** - Configuration file reference
- **[ROADMAP.md](docs/ROADMAP.md)** - Development roadmap and future plans
- **[CHANGELOG.md](CHANGELOG.md)** - Release history and changes

---

## 🎯 Basic Usage

### Validate a CSV file
```bash
vlite check data.csv --rule "not_null(id)" --rule "unique(email)"
```

### Validate a database table
```bash
vlite check "mysql://user:pass@host:3306/db.table" --rules validation_rules.json
```

### Check with verbose output
```bash
vlite check data.csv --rules rules.json --verbose
```

For detailed usage examples and advanced features, see [USAGE.md](docs/USAGE.md).

---

## 🏗️ Project Structure

```
validatelite/
├── cli/           # CLI logic and commands
├── core/          # Rule engine and core validation logic
├── shared/        # Common utilities, enums, exceptions, and schemas
├── config/        # Example and template configuration files
├── tests/         # Unit, integration, and E2E tests
├── scripts/       # Utility scripts
├── docs/          # Documentation
└── examples/      # Usage examples and sample data
```

---

## 🧪 Testing

### For Regular Users
The project includes comprehensive tests to ensure reliability. If you encounter issues, please check the [troubleshooting section](docs/USAGE.md#error-handling) in the usage guide.

### For Developers
```bash
# Set up test databases (requires Docker)
./scripts/setup_test_databases.sh start

# Run all tests with coverage
pytest -vv --cov

# Run specific test categories
pytest tests/unit/ -v          # Unit tests only
pytest tests/integration/ -v   # Integration tests
pytest tests/e2e/ -v           # End-to-end tests

# Code quality checks
pre-commit run --all-files

# Stop test databases when done
./scripts/setup_test_databases.sh stop
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

### Development Setup
For detailed development setup instructions, see [DEVELOPMENT_SETUP.md](docs/DEVELOPMENT_SETUP.md).

---

## 🔒 Security

For security issues, please review [SECURITY.md](SECURITY.md) and follow the recommended process.

---

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Inspired by best practices in data engineering and open-source data quality tools
- Thanks to all contributors and users for their feedback and support
