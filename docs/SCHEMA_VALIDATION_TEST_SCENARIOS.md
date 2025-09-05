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

### CLI Schema Command Tests (`tests/cli/commands/test_schema_command.py`)

#### Test Class: `TestSchemaCommand`

**File Format Tests**

15. **test_single_table_format_valid**
    - Test valid single-table JSON format
    - Expected: Proper decomposition into atomic rules

16. **test_multi_table_format_valid**
    - Test valid multi-table JSON format
    - Expected: Rules grouped by table correctly

17. **test_invalid_json_format**
    - Test malformed JSON file
    - Expected: click.UsageError with clear message

18. **test_missing_rules_array**
    - Test JSON without required 'rules' array
    - Expected: click.UsageError

19. **test_empty_rules_file**
    - Test empty JSON file
    - Expected: Early exit with appropriate message

**Rule Decomposition Tests**

20. **test_decompose_type_only**
    - Input: `{"field": "id", "type": "integer"}`
    - Expected: One SCHEMA rule with id→INTEGER mapping

21. **test_decompose_required_true**
    - Input: `{"field": "name", "type": "string", "required": true}`
    - Expected: SCHEMA rule + NOT_NULL rule

22. **test_decompose_range_constraints**
    - Input: `{"field": "age", "type": "integer", "min": 0, "max": 120}`
    - Expected: SCHEMA rule + RANGE rule with min_value/max_value

23. **test_decompose_enum_values**
    - Input: `{"field": "status", "type": "string", "enum": ["active", "inactive"]}`
    - Expected: SCHEMA rule + ENUM rule with allowed_values

24. **test_decompose_combined_constraints**
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

31. **test_mysql_schema_validation**
    - Setup: Real MySQL table with known schema
    - Test: Run schema validation against actual table
    - Cleanup: Drop test table

32. **test_postgresql_schema_validation**
    - Setup: Real PostgreSQL table
    - Test: Validate complex types (TIMESTAMP, TEXT, etc.)
    - Cleanup: Drop test table

33. **test_sqlite_schema_validation**
    - Setup: In-memory SQLite database
    - Test: Full schema validation workflow
    - No cleanup needed (in-memory)

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

**Single-Table Format:**
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

**Multi-Table Format:**
```json
{
  "users": {
    "rules": [
      {"field": "id", "type": "integer"},
      {"field": "name", "type": "string", "required": true}
    ],
    "strict_mode": true
  },
  "orders": {
    "rules": [
      {"field": "id", "type": "integer"},
      {"field": "user_id", "type": "integer", "required": true},
      {"field": "total", "type": "float", "min": 0}
    ]
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