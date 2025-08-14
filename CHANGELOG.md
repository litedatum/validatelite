# Changelog

All notable changes to ValidateLite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- tests(e2e): end-to-end `vlite-cli schema` scenarios on database URLs covering happy path, drift (FIELD_MISSING/TYPE_MISMATCH), strict extras, empty rules minimal payload; JSON and table outputs

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
