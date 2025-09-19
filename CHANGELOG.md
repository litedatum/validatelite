# Changelog

All notable changes to ValidateLite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- None

### Changed
- None

### Fixed
- None

### Removed
- None

## [0.5.0] 2025-9-18

### Added
- feat(schema): Implement syntactic sugar for type definitions in schema rules
- feat(core): Add TypeParser utility for parsing compact type definitions (e.g., `string(50)`, `float(12,2)`)
- feat(schema): Support shorthand type syntax: `string(50)` → `{"type": "string", "max_length": 50}`
- feat(schema): Support float precision/scale syntax: `float(12,2)` → `{"type": "float", "precision": 12, "scale": 2}`
- feat(schema): Support datetime format syntax: `datetime('yyyymmdd')` → `{"type": "datetime", "format": "yyyymmdd"}`
- feat(core): Enhanced schema executor with native database type reporting capabilities
- feat(core): Add comprehensive type aliases support (str→string, int→integer, bool→boolean)
- feat(tests): Comprehensive test coverage for type parser with unit and integration tests
- feat(tests): Native type integration testing for enhanced schema validation
- **feat(architecture): Implement two-phase execution framework in CLI with skip semantics**
- feat(schema): Add SchemaPhaseExecutor class for coordinated Phase 1 execution (schema rules only)
- feat(schema): Add DesiredTypePhaseExecutor class for coordinated Phase 2 execution (additional rules with filtering)
- feat(schema): Add ResultMerger class for combining phase results while maintaining output format consistency
- feat(schema): Comprehensive logging system for debugging two-phase execution with timing and rule counts
- feat(schema): Intelligent rule separation - automatically separate SCHEMA rules from other rule types for phased execution
- **feat(schema): Implement desired_type soft validation with compatibility analysis and rule generation**
- feat(schema): Add desired_type parsing support with extended TypeParser for complex type definitions
- feat(schema): Implement CompatibilityAnalyzer for intelligent type conversion analysis (COMPATIBLE/INCOMPATIBLE/CONFLICTING)
- feat(schema): Add DesiredTypeRuleGenerator for automatic validation rule creation based on compatibility analysis
- feat(schema): Generate LENGTH rules for precision/length reduction scenarios in type conversions
- feat(schema): Generate REGEX rules for string-to-numeric type conversion validation
- feat(schema): Generate DATE_FORMAT rules for date validation (MySQL support)
- feat(schema): Enhanced result merging with desired_type validation results integration
- feat(schema): Updated JSON and table output formats to display desired_type validation status
- feat(schema): Comprehensive error handling with clear distinction between schema vs desired_type failures
- feat(tests): Complete test coverage for desired_type validation including compatibility analysis and rule generation

### Changed
- enhance(cli): Updated schema command to support both syntactic sugar and detailed JSON type definitions
- enhance(core): Improved schema executor to handle parsed type definitions with metadata
- enhance(validation): Maintain backward compatibility with existing detailed JSON schema format
- **refactor(schema): Enhanced `_decompose_schema_payload()` to return tuple of (schema_rules, other_rules) for two-phase execution**
- refactor(schema): Added `_decompose_schema_payload_atomic()` for backward compatibility with single-list return format
- refactor(tests): Updated all schema-related test mocks to handle new tuple return format from rule decomposition
- improve(architecture): All validation maintains identical output format and behavior - no user-visible changes
- **enhance(schema): Extended two-phase execution framework with actual desired_type validation implementation**
- enhance(schema): DesiredTypePhaseExecutor now performs actual compatibility analysis and rule generation (no longer skip-only)
- enhance(schema): Enhanced type parser with full desired_type syntax support including complex type definitions
- enhance(validation): Intelligent compatibility matrix ensures optimal validation performance by skipping unnecessary checks
- enhance(output): Merged validation results clearly distinguish between schema structure validation and desired_type compatibility validation

### Fixed
- **fix(async): Resolved RuntimeError event loop management issue in two-phase execution**
- fix(async): Consolidated both validation phases into single event loop to prevent database connection pool conflicts
- fix(async): Eliminated multiple `asyncio.run()` calls that caused "Event loop is closed" errors in production
- fix(tests): Updated test contracts and mocks to work with new two-phase execution architecture
- **fix(sqlite): Implemented custom functions to solve SQLite regex compatibility limitations**
- fix(sqlite): Created comprehensive SQLite custom validation functions for precision and length validation
- fix(sqlite): Added `DETECT_INVALID_INTEGER_DIGITS`, `DETECT_INVALID_STRING_LENGTH`, `DETECT_INVALID_FLOAT_PRECISION` functions
- fix(sqlite): Automatic registration of custom functions via SQLAlchemy event listeners on connection establishment
- fix(database): Enhanced database dialect to intelligently use custom functions for SQLite regex replacement
- fix(validation): Seamless fallback from regex patterns to custom function calls for incompatible databases

### Removed
- None

### Architecture Notes
- **Two-Phase Execution Framework**: Complete implementation with desired_type soft validation capabilities
- **Phase 1**: Schema rules execute first to collect native type information and validate table/column existence
- **Phase 2**: Desired_type compatibility analysis with automatic rule generation for incompatible type conversions
- **Compatibility Analysis**: Intelligent type conversion analysis (COMPATIBLE/INCOMPATIBLE/CONFLICTING) optimizes validation performance
- **Rule Generation**: Automatic LENGTH, REGEX, and DATE_FORMAT rule creation based on compatibility analysis results
- **Skip Logic**: Rules targeting missing tables/columns are automatically skipped to prevent cascading failures
- **Result Merging**: Unified results combining schema validation and desired_type validation with clear error distinction
- **Performance**: Current implementation optimizes for stability over concurrency - both phases execute serially within single event loop
- **Database Support**: DATE_FORMAT validation currently supports MySQL with planned SQLite/PostgreSQL support in Phase 4
- **SQLite Regex Compatibility**: Custom function implementation (`shared/database/sqlite_functions.py`) provides seamless regex replacement for SQLite databases that lack native regex support
- **Custom Function Architecture**: Automatic registration of `DETECT_INVALID_INTEGER_DIGITS`, `DETECT_INVALID_STRING_LENGTH`, and `DETECT_INVALID_FLOAT_PRECISION` functions via SQLAlchemy event listeners
- **Intelligent Fallback**: Database dialect automatically detects SQLite and converts regex patterns to equivalent custom function calls for precision/length validation

## [0.4.3] - 2025-09-06

### Added
- feat(schema): Enhanced SCHEMA rule with metadata validation capabilities
- feat(schema): String length validation via `max_length` parameter for precise VARCHAR constraints
- feat(schema): Float precision and scale validation via `precision`/`scale` parameters for DECIMAL constraints
- feat(cli): Extended JSON schema format support with metadata fields (max_length, precision, scale)
- feat(core): Database-agnostic metadata extraction across MySQL, PostgreSQL, and SQLite
- feat(core): Vendor-specific type parsing with regex-based metadata extraction
- feat(core): Performance-optimized validation using database catalog queries (no data scanning)
- feat(validation): Comprehensive metadata comparison logic with detailed failure reporting
- feat(cli): Enhanced rule parameter validation for metadata fields with logical constraints
- feat(tests): Comprehensive metadata validation test suite (87% coverage on SchemaExecutor)
- feat(tests): Unit, integration, and CLI tests for metadata validation scenarios
- feat(docs): Enhanced documentation with metadata validation examples and troubleshooting guide
- feat(docs): Migration guide for legacy schema formats and performance characteristics

### Changed
- refactor(schema): Enhanced SchemaExecutor with metadata validation capabilities
- refactor(cli): Extended CLI schema parsing to support metadata fields with validation
- refactor(core): Improved database metadata extraction and type mapping
- improve(performance): Metadata validation uses single database query per table (no data scans)
- improve(validation): Enhanced error messages with specific metadata mismatch descriptions
- improve(architecture): Clear separation between structure validation (SCHEMA) and content validation (RANGE/ENUM)

### Fixed
- None

### Removed
- None

### Migration Guide
- **Backward Compatibility**: Existing schema files without metadata continue to work unchanged
- **Enhanced Validation**: Add `max_length`, `precision`, and `scale` fields incrementally to existing schemas
- **Performance**: Metadata validation provides superior performance vs scanning-based approaches
- **Architecture**: Enhanced SCHEMA rule eliminates need for separate LENGTH rule type

## [0.4.2] - 2025-08-27

### Added
- feat(cli): refactor check command interface from positional arguments to `--conn` and `--table` options
- feat(cli): add comprehensive test coverage for new CLI interface functionality
- feat(cli): support explicit table name specification independent of database URL
- feat(schema): add comprehensive multi-table support for schema validation
- feat(schema): support multi-table rules format with table-level configuration options
- feat(schema): add Excel multi-sheet file support as data source
- feat(schema): implement table-grouped output display for multi-table validation results
- feat(schema): add table-level options support (strict_mode, case_insensitive)
- feat(tests): add comprehensive multi-table functionality test coverage
- feat(tests): add multi-table Excel file validation test scenarios

### Changed
- **BREAKING CHANGE**: CLI interface changed from `vlite check <source>` to `vlite check --conn <connection> --table <table_name>`
- refactor(cli): update SourceParser to accept optional table_name parameter
- refactor(cli): modify check command to pass table_name to SourceParser.parse_source()
- refactor(tests): update all existing CLI tests to use new interface format
- refactor(tests): add new test cases specifically for table name parameter validation
- refactor(schema): enhance schema command to support both single-table and multi-table formats
- refactor(schema): improve output formatting with table-grouped results display
- refactor(schema): enhance rule decomposition logic for multi-table support
- refactor(data-validator): improve multi-table detection and processing capabilities
- refactor(schema): preserve field order from initial JSON definition instead of alphabetical sorting
- refactor(schema): consolidate field validation information display to single line per field

### Fixed
- fix(cli): resolve issue where `--table` parameter was not correctly passed to backend
- fix(cli): ensure table name from `--table` option takes precedence over table name in database URL
- fix(tests): update regression tests to use new CLI interface format
- fix(tests): resolve test failures caused by interface changes
- fix(schema): resolve multi-table rules validation and type checking issues
- fix(schema): improve table name detection and validation in multi-table scenarios
- fix(schema): enhance error handling for multi-table validation workflows
- fix(schema): ensure schema-only rule fields are not omitted from validation results
- fix(schema): properly display skip conventions for non-existent columns (FIELD_MISSING/TYPE_MISMATCH)

### Removed
- **BREAKING CHANGE**: remove backward compatibility for old positional argument interface
- remove(cli): eliminate support for `<source>` positional argument in check command

## [0.4.0] - 2025-08-14

### Added
- feat(cli): add `schema` command skeleton
- feat(cli): add minimal rules file validation for schema command (no jsonschema in v1)
- feat(core): introduce `SCHEMA` rule type with table-level existence and type checks
- feat(cli): decompose schema rules into atomic rules (SCHEMA, NOT_NULL, RANGE, ENUM)
- feat(cli): aggregation and prioritization in CLI with column-guard skip semantics (FIELD_MISSING/TYPE_MISMATCH)
- feat(cli): output formatting improvements for table mode (column-grouped view, readable descriptors)
- feat(cli): aggregated JSON output for schema command with summary/results/fields/schema_extras
- docs: add JSON Schema for results at `docs/schemas/schema_results.schema.json`
- tests(cli): comprehensive unit tests for `schema` command covering argument parsing, rules file validation, decomposition/mapping, aggregation priority, output formats (table/json), and exit codes (AC satisfied)
 - tests(core): unit tests for `SCHEMA` rule covering normal/edge/error cases, strict type checks, and mypy compliance
- tests(integration): database schema drift tests for MySQL and PostgreSQL (existence, type consistency, strict mode extras, case-insensitive)
- tests(e2e): end-to-end `vlite schema` scenarios on database URLs covering happy path, drift (FIELD_MISSING/TYPE_MISMATCH), strict extras, empty rules minimal payload; JSON and table outputs

### Changed
- docs: update README and USAGE with schema command overview and detailed usage
- cli(schema): align table header record count with execution metrics to avoid misleading warnings
- cli(schema): data-source resolution parity with `check` (analyzing echo and empty file guard)
- tests(e2e): JSON parse failures now assert with detailed stdout/stderr instead of being skipped, to surface real errors in CI

### Fixed
- cli(schema): correct failed records accounting in table output
- cli(schema): ensure dependent rules display as SKIPPED where applicable in both JSON and table modes
- cli(schema): handle empty source file with clear error, mirroring `check`
- cli(schema): JSON output now serializes datetime fields via `default=str` to avoid non-serializable payloads
- core(validity): schema type mapping recognizes PostgreSQL `CHARACTER` as STRING to prevent false type mismatches

### Removed
- None

## [0.3.0] - 2025-08-05

### Added
- Enhanced project maturity with comprehensive test coverage
- Robust CI/CD pipeline with automated testing and security scanning
- Advanced rule engine with support for complex validation scenarios
- Improved error handling and classification system
- Comprehensive documentation and development guides
- Pre-commit hooks and code quality enforcement
- Support for multiple database dialects and connection types
- Performance optimizations and monitoring capabilities

### Changed
- Upgraded to version 0.3.0 to reflect project maturity
- Enhanced error reporting and user experience
- Improved configuration management and validation

### Fixed
- Various bug fixes and stability improvements
- Enhanced test coverage and reliability

### Removed
- None

## [0.1.0] - 2025-07-22

### Added
- Initial release of ValidateLite
- Rule-based data quality validation engine
- Command-line interface for data validation
- Support for file-based data sources (CSV, Excel)
- Support for database connections (MySQL, PostgreSQL, SQLite)
- Core validation rules: not_null, unique, range, enum, regex
- Comprehensive error handling and classification
- Configuration management with TOML support
- Extensive test coverage (>80%)
- Development documentation and setup guides

### Changed
- None

### Fixed
- None

### Removed
- None
