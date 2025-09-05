# Command Format Update Summary

## Overview

This document summarizes the command format updates made to align documentation with the latest CLI implementation.

## Changes Made

### 1. Command Name Update
- **Old**: `vlite` 
- **New**: `vlite`

### 2. Command Parameter Structure Update

#### Check Command
**Old Format:**
```bash
vlite check <data_source> [options]
```

**New Format:**
```bash
vlite check --conn <data_source> --table <table_name> [options]
```

**Key Changes:**
- Added required `--conn` parameter for data source
- Added required `--table` parameter for table/identifier
- Data source no longer includes table name in connection string

#### Schema Command
**Old Format:**
```bash
vlite schema <data_source> --rules <schema_file.json> [options]
```

**New Format:**
```bash
vlite schema --conn <data_source> --rules <schema_file.json> [options]
```

**Key Changes:**
- Added required `--conn` parameter for database connection
- Database connection string no longer includes table name

### 3. Connection String Format Updates

#### Database Connections
**Old Format:**
```
mysql://user:pass@host:3306/db.table
postgresql://user:pass@host:5432/db.table
sqlite:///path/to/db.sqlite.table
```

**New Format:**
```
mysql://user:pass@host:3306/db
postgresql://user:pass@host:5432/db
sqlite:///path/to/db.sqlite
```

**Note**: Table name is now specified separately using the `--table` parameter.

### 4. Examples Updated

#### File Validation
**Old:**
```bash
vlite check data.csv --rule "not_null(id)"
```

**New:**
```bash
vlite check --conn data.csv --table data --rule "not_null(id)"
```

#### Database Validation
**Old:**
```bash
vlite check "mysql://user:pass@host:3306/db.customers" --rule "unique(email)"
```

**New:**
```bash
vlite check --conn "mysql://user:pass@host:3306/db" --table customers --rule "unique(email)"
```

## Files Updated

### Documentation Files
1. **docs/USAGE.md** - Complete command reference and examples
2. **docs/CONFIG_REFERENCE.md** - Configuration examples
3. **README.md** - Quick start and main examples
4. **examples/README.md** - Example usage instructions
5. **examples/basic_usage.py** - Python example commands

### Notes and Other Files
1. **notes/issue_list_08_23_25.md** - Issue tracking updates

## Benefits of New Format

1. **Clearer Separation**: Connection and table are now separate parameters
2. **Better Consistency**: Both commands use similar parameter structure
3. **Improved Readability**: Commands are more self-documenting
4. **Easier Parsing**: Clear parameter boundaries for automation

## Migration Guide

### For Users
1. Update command from `vlite` to `vlite`
2. Add `--conn` parameter for data source
3. Add `--table` parameter for table name
4. Remove table name from database connection strings

### For Scripts and CI/CD
1. Update all command invocations
2. Separate connection strings and table names
3. Test with new parameter structure

## Verification

To verify the new format works correctly:

```bash
# Test help output
vlite --help
vlite check --help
vlite schema --help

# Test basic validation
vlite check --conn examples/sample_data.csv --table data --rule "not_null(customer_id)"

# Test schema validation
vlite schema --conn "sqlite:///test.db" --rules test_data/schema.json
```

## Backward Compatibility

**Note**: This is a breaking change. The old command format is no longer supported. Users must update their commands to use the new format.

---

*This document was created to track the command format updates made during the feature improvement phase.*
