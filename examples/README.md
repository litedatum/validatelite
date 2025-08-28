# ValidateLite Examples

This directory contains examples and sample files to help you get started with ValidateLite.

## Files

- `basic_usage.py` - Python script demonstrating basic usage patterns
- `sample_rules.json` - Example validation rules file
- `sample_data.csv` - Sample data file for testing
- `README.md` - This file

## Quick Start

1. **Run the basic usage example:**
   ```bash
   python examples/basic_usage.py
   ```

2. **Validate the sample data:**
   ```bash
   python cli_main.py check --conn examples/sample_data.csv --table data --rules examples/sample_rules.json
   ```

3. **Test with your own data:**
   ```bash
   # Create your own rules file based on sample_rules.json
   # Then run validation
   python cli_main.py check --conn your_data.csv --table data --rules your_rules.json
   ```

## Example Rules

The `sample_rules.json` file demonstrates various rule types:

- **NOT_NULL** - Ensures required fields are not empty
- **UNIQUE** - Ensures values are unique across the dataset
- **REGEX** - Validates text patterns (email, phone numbers)
- **RANGE** - Validates numeric ranges (age, revenue)
- **LENGTH** - Validates string lengths
- **ENUM** - Validates against allowed values
- **DATE_FORMAT** - Validates date/time formats

## Customization

Modify the example files to match your data structure and validation requirements:

1. Update column names in rules to match your data
2. Adjust validation parameters (patterns, ranges, etc.)
3. Add or remove rules based on your needs
4. Change severity levels (ERROR, WARNING, INFO)

## Next Steps

- Read the [main documentation](../README.md)
- Check the [usage guide](../docs/USAGE.md)
- Explore the [configuration options](../config/)
- Review the [test examples](../tests/) for more advanced usage patterns
