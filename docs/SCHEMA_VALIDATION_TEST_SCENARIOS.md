# Schema Validation Test Scenarios

This document defines comprehensive test scenarios for the Schema Validation feature in ValidateLite. The scenarios cover unit tests, integration tests, and end-to-end tests.

## Table of Contents

1. [Unit Tests](#unit-tests)
2. [Integration Tests](#integration-tests)
3. [End-to-End Tests](#end-to-end-tests)
4. [Test Data Requirements](#test-data-requirements)
5. [Performance Tests](#performance-tests)
6. [Error Handling Tests](#error-handling-tests)

## Unit Tests

### SchemaExecutor Tests (`tests/core/executors/test_schema_executor.py`)

#### Test Class: `TestSchemaExecutor`

**Basic Functionality**

1. **test_supports_schema_rule_type**
   - Verify that SchemaExecutor supports RuleType.SCHEMA
   - Verify that it doesn't support other rule types (NOT_NULL, RANGE, etc.)

2. **test_execute_schema_rule_all_pass**
   - Test scenario: All declared columns exist with correct types
   - Expected: status=PASSED, failed_records=0
   - Mock database returns: id (INTEGER), name (VARCHAR), email (VARCHAR)
   - Schema rule expects: id (INTEGER), name (STRING), email (STRING)

3. **test_execute_schema_rule_field_missing**
   - Test scenario: Some declared columns are missing from actual table
   - Expected: status=FAILED, field marked as FIELD_MISSING
   - Mock database returns: id (INTEGER), name (VARCHAR)
   - Schema rule expects: id (INTEGER), name (STRING), email (STRING)

4. **test_execute_schema_rule_type_mismatch**
   - Test scenario: Column exists but has wrong type
   - Expected: status=FAILED, field marked as TYPE_MISMATCH
   - Mock database returns: id (VARCHAR), name (VARCHAR)
   - Schema rule expects: id (INTEGER), name (STRING)

5. **test_execute_schema_rule_strict_mode_extra_columns**
   - Test scenario: Extra columns exist with strict_mode=true
   - Expected: status=FAILED, extras in execution_plan
   - Mock database returns: id (INTEGER), name (VARCHAR), extra_col (TEXT)
   - Schema rule expects: id (INTEGER), name (STRING) with strict_mode=true

6. **test_execute_schema_rule_case_insensitive**
   - Test scenario: Column names with different casing
   - Expected: status=PASSED when case_insensitive=true
   - Mock database returns: ID (INTEGER), Name (VARCHAR)
   - Schema rule expects: id (integer), name (string) with case_insensitive=true

**Type Mapping Tests**

7. **test_vendor_type_mapping_mysql**
   - Verify mapping of MySQL types: INT→INTEGER, VARCHAR→STRING, DATETIME→DATETIME

8. **test_vendor_type_mapping_postgresql**
   - Verify mapping of PostgreSQL types: INTEGER→INTEGER, TEXT→STRING, TIMESTAMP→DATETIME

9. **test_vendor_type_mapping_sqlite**
   - Verify mapping of SQLite types: INTEGER→INTEGER, TEXT→STRING, REAL→FLOAT

10. **test_unsupported_vendor_type**
    - Test scenario: Database returns unsupported type
    - Expected: Use raw type for comparison

**Parameter Validation Tests**

11. **test_missing_columns_parameter**
    - Test scenario: SCHEMA rule without columns parameter
    - Expected: RuleExecutionError

12. **test_empty_columns_parameter**
    - Test scenario: SCHEMA rule with empty columns dict
    - Expected: RuleExecutionError

13. **test_missing_expected_type**
    - Test scenario: Column definition without expected_type
    - Expected: RuleExecutionError

14. **test_invalid_expected_type**
    - Test scenario: Column with unsupported expected_type
    - Expected: RuleExecutionError

**Metadata Validation Tests**

15. **test_string_max_length_validation_success**
    - Test scenario: String column with matching max_length
    - Mock database returns: name (VARCHAR(100))
    - Schema rule expects: name (STRING, max_length: 100)
    - Expected: status=PASSED

16. **test_string_max_length_validation_failure**
    - Test scenario: String column with max_length mismatch
    - Mock database returns: name (VARCHAR(50))
    - Schema rule expects: name (STRING, max_length: 100)
    - Expected: status=FAILED, METADATA_MISMATCH

17. **test_float_precision_scale_validation_success**
    - Test scenario: Float column with matching precision/scale
    - Mock database returns: price (DECIMAL(10,2))
    - Schema rule expects: price (FLOAT, precision: 10, scale: 2)
    - Expected: status=PASSED

18. **test_float_precision_validation_failure**
    - Test scenario: Float column with precision mismatch
    - Mock database returns: price (DECIMAL(8,2))
    - Schema rule expects: price (FLOAT, precision: 10, scale: 2)
    - Expected: status=FAILED, METADATA_MISMATCH

19. **test_float_scale_validation_failure**
    - Test scenario: Float column with scale mismatch
    - Mock database returns: price (DECIMAL(10,4))
    - Schema rule expects: price (FLOAT, precision: 10, scale: 2)
    - Expected: status=FAILED, METADATA_MISMATCH

20. **test_mixed_metadata_validation**
    - Test scenario: Mix of columns with and without metadata
    - Mock database returns: id (INTEGER), name (VARCHAR(100)), price (DECIMAL(10,2))
    - Schema rule expects: id (INTEGER), name (STRING, max_length: 100), price (FLOAT)
    - Expected: status=PASSED for all columns

21. **test_unlimited_length_string_validation**
    - Test scenario: TEXT/BLOB columns (unlimited length)
    - Mock database returns: description (TEXT)
    - Schema rule expects: description (STRING, max_length: 1000)
    - Expected: status=PASSED (unlimited >= specified limit)

22. **test_missing_metadata_in_database**
    - Test scenario: Database metadata unavailable
    - Mock database returns: name (VARCHAR) [no length info]
    - Schema rule expects: name (STRING, max_length: 100)
    - Expected: status=FAILED, clear error message about missing metadata

23. **test_metadata_type_parsing**
    - Test scenario: Various vendor-specific type formats
    - Test parsing: VARCHAR(255), DECIMAL(10,2), FLOAT(8,4), TEXT, etc.
    - Expected: Correct extraction of metadata from type strings

24. **test_performance_large_schema_with_metadata**
    - Test scenario: 100+ columns with metadata validation
    - Expected: Validation completes within 5 seconds
    - No memory leaks or performance degradation

### CLI Schema Command Tests (`tests/cli/commands/test_schema_command.py`)

#### Test Class: `TestSchemaCommand`

**File Format Tests**

25. **test_single_table_format_valid**
    - Test valid single-table JSON format
    - Expected: Proper decomposition into atomic rules

26. **test_multi_table_format_valid**
    - Test valid multi-table JSON format
    - Expected: Rules grouped by table correctly

27. **test_invalid_json_format**
    - Test malformed JSON file
    - Expected: click.UsageError with clear message

28. **test_missing_rules_array**
    - Test JSON without required 'rules' array
    - Expected: click.UsageError

29. **test_empty_rules_file**
    - Test empty JSON file
    - Expected: Early exit with appropriate message

**Metadata Parsing Tests**

30. **test_extended_json_format_with_metadata**
    - Input: `{"field": "name", "type": "string", "max_length": 100, "required": true}`
    - Expected: SCHEMA rule with metadata + NOT_NULL rule

31. **test_float_metadata_parsing**
    - Input: `{"field": "price", "type": "float", "precision": 10, "scale": 2}`
    - Expected: SCHEMA rule with precision and scale metadata

32. **test_invalid_metadata_combinations**
    - Input: `{"field": "id", "type": "integer", "max_length": 100}`
    - Expected: click.UsageError (max_length invalid for integer type)

33. **test_invalid_precision_scale_combination**
    - Input: `{"field": "price", "type": "float", "precision": 5, "scale": 10}`
    - Expected: click.UsageError (scale cannot exceed precision)

34. **test_negative_metadata_values**
    - Input: `{"field": "name", "type": "string", "max_length": -100}`
    - Expected: click.UsageError (metadata must be non-negative)

35. **test_backwards_compatibility_without_metadata**
    - Input: Legacy JSON format without metadata fields
    - Expected: Proper parsing, metadata validation skipped

36. **test_mixed_metadata_fields**
    - Input: Schema with some fields having metadata, others not
    - Expected: Correct rule decomposition for all field types

**Rule Decomposition Tests**

37. **test_decompose_type_only**
    - Input: `{"field": "id", "type": "integer"}`
    - Expected: One SCHEMA rule with id→INTEGER mapping

38. **test_decompose_required_true**
    - Input: `{"field": "name", "type": "string", "required": true}`
    - Expected: SCHEMA rule + NOT_NULL rule

39. **test_decompose_range_constraints**
    - Input: `{"field": "age", "type": "integer", "min": 0, "max": 120}`
    - Expected: SCHEMA rule + RANGE rule with min_value/max_value

40. **test_decompose_enum_values**
    - Input: `{"field": "status", "type": "string", "enum": ["active", "inactive"]}`
    - Expected: SCHEMA rule + ENUM rule with allowed_values

41. **test_decompose_combined_constraints**
    - Input: Multiple constraints on single field
    - Expected: All corresponding atomic rules generated

**Data Type Mapping Tests**

25. **test_type_mapping_all_supported**
    - Verify mapping: string→STRING, integer→INTEGER, float→FLOAT, etc.

26. **test_type_mapping_case_insensitive**
    - Input: "STRING", "Integer", "FLOAT"
    - Expected: Proper DataType enum values

27. **test_unsupported_type_name**
    - Input: `{"field": "id", "type": "uuid"}`
    - Expected: click.UsageError with allowed types list

**Output Format Tests**

28. **test_table_output_format**
    - Execute schema command with --output=table
    - Expected: Human-readable table output

29. **test_json_output_format**
    - Execute schema command with --output=json
    - Expected: Valid JSON with all required fields

30. **test_prioritization_in_output**
    - Test field with FIELD_MISSING → dependent rules skipped
    - Expected: Proper skip_reason in JSON output

## Integration Tests

### Database Integration Tests (`tests/integration/test_schema_validation.py`)

#### Test Class: `TestSchemaValidationIntegration`

**Real Database Tests**

48. **test_mysql_schema_validation**
    - Setup: Real MySQL table with known schema
    - Test: Run schema validation against actual table
    - Cleanup: Drop test table

49. **test_postgresql_schema_validation**
    - Setup: Real PostgreSQL table
    - Test: Validate complex types (TIMESTAMP, TEXT, etc.)
    - Cleanup: Drop test table

50. **test_sqlite_schema_validation**
    - Setup: In-memory SQLite database
    - Test: Full schema validation workflow
    - No cleanup needed (in-memory)

**Metadata Integration Tests**

51. **test_mysql_metadata_validation**
    - Setup: MySQL table with VARCHAR(100), DECIMAL(10,2) columns
    - Test: Schema rules with corresponding metadata
    - Expected: Metadata extracted and validated correctly

52. **test_postgresql_metadata_validation**
    - Setup: PostgreSQL table with TEXT, NUMERIC(12,3) columns
    - Test: Metadata validation across different PostgreSQL types
    - Expected: Proper type mapping and metadata validation

53. **test_sqlite_metadata_validation**
    - Setup: SQLite table with limited type system
    - Test: Metadata validation with SQLite type affinity
    - Expected: Graceful handling of SQLite's dynamic typing

54. **test_mixed_metadata_integration**
    - Setup: Table with mixed columns (some with metadata, some without)
    - Test: End-to-end validation with selective metadata checking
    - Expected: Only columns with expected metadata are validated

55. **test_metadata_extraction_performance**
    - Setup: Large table with 50+ columns, various types with metadata
    - Test: Full metadata extraction and validation
    - Expected: Completes within 10 seconds, single database query

**Multi-Table Validation**

34. **test_multi_table_validation**
    - Setup: Multiple tables with different schemas
    - Test: Multi-table rules file validation
    - Expected: Per-table results aggregation

35. **test_table_not_found**
    - Test: Schema rules for non-existent table
    - Expected: Proper error handling and reporting

**Connection String Tests**

36. **test_file_based_source**
    - Test: CSV file as data source
    - Schema: Inferred from CSV headers
    - Expected: Proper type detection

37. **test_database_connection_string**
    - Test: Various database connection formats
    - Expected: Proper source parsing and validation

## End-to-End Tests

### CLI End-to-End Tests (`tests/e2e/test_schema_cli.py`)

#### Test Class: `TestSchemaCliE2E`

**Complete Workflow Tests**

38. **test_full_schema_validation_success**
    - Setup: Complete test database + rules file
    - Command: `vlite schema --conn <db> --rules <file>`
    - Expected: Exit code 0, success output

39. **test_full_schema_validation_failure**
    - Setup: Database with schema mismatches
    - Command: Schema validation with failing rules
    - Expected: Exit code 1, clear failure reporting

40. **test_verbose_output**
    - Command: Schema validation with --verbose flag
    - Expected: Detailed logging output

41. **test_fail_on_error_flag**
    - Command: Schema validation with --fail-on-error
    - Expected: Exit code 1 on any execution errors

**File Handling Tests**

42. **test_rules_file_not_found**
    - Command: Reference non-existent rules file
    - Expected: Exit code 2, clear error message

43. **test_rules_file_permission_denied**
    - Setup: Rules file with no read permissions
    - Expected: Exit code 2, permission error message

44. **test_large_rules_file**
    - Setup: Rules file with 100+ field definitions
    - Expected: Successful processing, performance within limits

## Test Data Requirements

### Sample Database Schemas

**MySQL Test Table:**
```sql
CREATE TABLE test_users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255),
    age SMALLINT,
    created_at DATETIME,
    is_active BOOLEAN DEFAULT TRUE
);
```

**PostgreSQL Test Table:**
```sql
CREATE TABLE test_products (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    price DECIMAL(10,2),
    created_date DATE,
    updated_timestamp TIMESTAMP,
    metadata JSONB
);
```

**SQLite Test Table:**
```sql
CREATE TABLE test_orders (
    id INTEGER PRIMARY KEY,
    customer_name TEXT,
    total_amount REAL,
    order_date TEXT,
    status TEXT CHECK(status IN ('pending', 'completed', 'cancelled'))
);
```

### Sample Rules Files

**Single-Table Format (Legacy):**
```json
{
  "rules": [
    {"field": "id", "type": "integer", "required": true},
    {"field": "name", "type": "string", "required": true},
    {"field": "email", "type": "string"},
    {"field": "age", "type": "integer", "min": 0, "max": 150},
    {"field": "status", "type": "string", "enum": ["active", "inactive"]}
  ]
}
```

**Single-Table Format with Metadata:**
```json
{
  "rules": [
    {"field": "id", "type": "integer", "required": true},
    {"field": "name", "type": "string", "max_length": 100, "required": true},
    {"field": "email", "type": "string", "max_length": 255},
    {"field": "price", "type": "float", "precision": 10, "scale": 2, "min": 0},
    {"field": "description", "type": "string", "max_length": 1000},
    {"field": "status", "type": "string", "enum": ["active", "inactive"]}
  ]
}
```

**Multi-Table Format with Mixed Metadata:**
```json
{
  "users": {
    "rules": [
      {"field": "id", "type": "integer"},
      {"field": "username", "type": "string", "max_length": 50, "required": true},
      {"field": "email", "type": "string", "max_length": 255, "required": true},
      {"field": "bio", "type": "string", "max_length": 500}
    ],
    "strict_mode": true
  },
  "products": {
    "rules": [
      {"field": "id", "type": "integer"},
      {"field": "name", "type": "string", "max_length": 200, "required": true},
      {"field": "price", "type": "float", "precision": 12, "scale": 2, "min": 0},
      {"field": "weight", "type": "float", "precision": 8, "scale": 3}
    ],
    "case_insensitive": true
  }
}
```

## Performance Tests

### Performance Test Scenarios (`tests/performance/test_schema_performance.py`)

45. **test_large_table_schema_validation**
    - Setup: Table with 1M+ rows, 50+ columns
    - Expected: Validation completes within 30 seconds

46. **test_many_columns_validation**
    - Setup: Table with 200+ columns
    - Expected: Memory usage remains reasonable

47. **test_concurrent_schema_validations**
    - Setup: Multiple schema validations in parallel
    - Expected: No resource conflicts, proper isolation

## Error Handling Tests

### Error Scenario Tests (`tests/error_handling/test_schema_errors.py`)

48. **test_database_connection_failure**
    - Scenario: Invalid database credentials
    - Expected: Clear error message, proper exit code

49. **test_network_timeout**
    - Scenario: Database connection timeout
    - Expected: Timeout handling, retry logic if applicable

50. **test_insufficient_permissions**
    - Scenario: Database user without table access
    - Expected: Permission error with helpful message

51. **test_malformed_column_metadata**
    - Scenario: Database returns unexpected metadata format
    - Expected: Graceful handling, fallback behavior

## Test Execution Guidelines

### Running Tests

```bash
# Run all schema validation tests
pytest tests/ -k "schema" -v

# Run only unit tests
pytest tests/core/executors/test_schema_executor.py -v
pytest tests/cli/commands/test_schema_command.py -v

# Run integration tests (requires test databases)
pytest tests/integration/test_schema_validation.py -v

# Run performance tests
pytest tests/performance/test_schema_performance.py -v

# Run with coverage
pytest tests/ -k "schema" --cov=core --cov=cli --cov-report=html
```

### Test Environment Setup

1. **Database Setup:**
   - MySQL test instance
   - PostgreSQL test instance
   - SQLite (no setup required)

2. **Test Data:**
   - Sample CSV files
   - Test database schemas
   - Various rules files (valid/invalid)

3. **Mock Objects:**
   - Database connection mocks
   - Query result mocks
   - File system mocks

### Coverage Requirements

- **Unit Tests:** 90%+ coverage for new code
- **Integration Tests:** Cover all database dialects
- **E2E Tests:** Cover all CLI options and error paths
- **Performance Tests:** Establish baseline metrics

### Continuous Integration

- All tests must pass before merge
- Performance regression detection
- Database compatibility matrix testing
- Documentation updates required for new test scenarios

## Metadata Validation Troubleshooting Guide

### Common Issues and Solutions

**Issue 1: Metadata Mismatch Errors**
- **Symptom**: METADATA_MISMATCH failures for correct-looking schemas
- **Cause**: Database metadata extraction returning unexpected formats
- **Solution**: Check actual database column definitions using database-specific tools
- **Debug**: Enable verbose logging to see extracted metadata vs expected

**Issue 2: Missing Metadata in Database Response**
- **Symptom**: Validation failures with "metadata unavailable" messages
- **Cause**: Database system not providing length/precision in metadata queries
- **Solution**: Verify database permissions and version compatibility
- **Workaround**: Use schema validation without metadata (legacy format)

**Issue 3: Unlimited Length Field Validation**
- **Symptom**: TEXT/BLOB fields failing length validation unexpectedly
- **Cause**: Database returns -1 or NULL for unlimited length fields
- **Expected Behavior**: Unlimited length should pass all max_length checks
- **Solution**: This is handled automatically - no action needed

**Issue 4: Vendor-Specific Type Parsing**
- **Symptom**: Type parsing errors for complex database types
- **Cause**: Unsupported vendor-specific type format
- **Solution**: Review type mapping in SchemaExecutor._extract_type_metadata()
- **Add Support**: Extend regex patterns for new type formats

**Issue 5: Performance Issues with Large Schemas**
- **Symptom**: Metadata validation takes longer than expected
- **Cause**: Multiple database queries or inefficient metadata extraction
- **Expected**: Single query per table, completes within 10 seconds for 100+ columns
- **Debug**: Check database query logs for multiple metadata requests

**Issue 6: Scale/Precision Validation Failures**
- **Symptom**: FLOAT columns failing precision/scale validation
- **Cause**: Database storing different precision than schema definition
- **Solution**: Verify actual database column definitions match expected
- **Note**: Some databases automatically adjust precision/scale during table creation

### Performance Expectations

**Metadata Validation Performance Targets:**
- **Small schemas (1-10 columns)**: < 1 second
- **Medium schemas (10-50 columns)**: < 3 seconds
- **Large schemas (50-100 columns)**: < 5 seconds
- **Very large schemas (100+ columns)**: < 10 seconds

**Memory Usage:**
- Metadata validation should not significantly increase memory usage
- Expected: < 10MB additional memory for 100+ column schemas

**Database Queries:**
- **Expected**: 1 metadata query per table (using get_column_list())
- **Not Expected**: Per-column queries or data scanning queries

### Debugging Commands

**Enable Verbose Logging:**
```bash
vlite schema --conn <connection> --rules <file> --verbose
```

**Test Metadata Extraction:**
```python
# Debug database metadata extraction
from shared.database.query_executor import QueryExecutor
from shared.schema.connection_schema import ConnectionSchema

conn = ConnectionSchema(...)
executor = QueryExecutor(conn)
columns = executor.get_column_list("table_name")
print("Extracted metadata:", columns)
```

**Validate Rule Parameters:**
```python
# Test rule parameter validation
from shared.schema.rule_schema import RuleSchema
from shared.enums.rule_types import RuleType

rule = RuleSchema(
    type=RuleType.SCHEMA,
    parameters={
        "columns": {
            "name": {"expected_type": "STRING", "max_length": 100}
        }
    }
)
```
