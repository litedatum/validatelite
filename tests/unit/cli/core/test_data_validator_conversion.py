"""
🧙‍♂️ Test Data Validator Conversion Logic - The Testing Ghost's Comprehensive Suite

This module tests the critical conversion and cleaning logic in DataValidator:
- Source dict to ConnectionSchema conversion
- Rule dict to RuleSchema conversion
- Column & table name sanitization for SQLite compatibility

Modern Testing Strategies Applied:
✅ Schema Builder Pattern - Eliminating fixture duplication
✅ Property-based Testing - Parametric testing for edge cases
✅ Contract Testing - Ensuring mock/real implementation consistency
✅ Comprehensive Boundary Testing - All edge cases covered
"""

import re
from typing import Any, Dict, List, Union
from uuid import uuid4

import pytest

from cli.core.config import CliConfig
from cli.core.data_validator import DataValidator
from cli.exceptions import ValidationError
from core.config import CoreConfig
from shared.enums import ConnectionType, RuleCategory, RuleType
from shared.schema import ConnectionSchema, RuleSchema

# Import our modern testing infrastructure
from tests.shared.builders.test_builders import TestDataBuilder
from tests.shared.contracts.test_contracts import MockContract


class TestDataValidatorConversion:
    """
    🎯 Comprehensive test suite for DataValidator conversion logic

    Focus Areas:
    1. Source dict conversion with all connection types
    2. Rule dict conversion with parameter validation
    3. Name cleaning with SQLite compatibility edge cases
    4. Error handling for malformed inputs
    """

    @pytest.fixture
    def mock_configs(self) -> Dict[str, Any]:
        """Provide mock configurations using Contract Testing"""
        return {
            "core_config": MockContract.create_core_config_mock(),
            "cli_config": MockContract.create_cli_config_mock(),
        }

    # ============================================================================
    # Source Dict Conversion Tests
    # ============================================================================

    @pytest.mark.parametrize(
        "source_data,expected_type,expected_attributes",
        [
            # CSV file sources
            (
                {"type": "csv", "path": "/data/users.csv"},
                ConnectionType.CSV,
                {"file_path": "/data/users.csv", "host": None},
            ),
            (
                {
                    "connection_type": "csv",
                    "file_path": "/data/test.csv",
                    "parameters": {"encoding": "utf-8"},
                },
                ConnectionType.CSV,
                {"file_path": "/data/test.csv", "parameters": {"encoding": "utf-8"}},
            ),
            # Excel file sources
            (
                {"type": "excel", "path": "/data/report.xlsx"},
                ConnectionType.EXCEL,
                {"file_path": "/data/report.xlsx"},
            ),
            # JSON file sources
            (
                {"type": "json", "path": "/data/records.json"},
                ConnectionType.JSON,
                {"file_path": "/data/records.json"},
            ),
            # Database sources
            (
                {
                    "type": "mysql",
                    "host": "localhost",
                    "database": "testdb",
                    "username": "user",
                    "password": "pass",
                    "port": 3306,
                },
                ConnectionType.MYSQL,
                {
                    "host": "localhost",
                    "db_name": "testdb",
                    "username": "user",
                    "port": 3306,
                },
            ),
            (
                {
                    "connection_type": "postgresql",
                    "host": "db.example.com",
                    "port": 5432,
                    "db_name": "analytics",
                },
                ConnectionType.POSTGRESQL,
                {"host": "db.example.com", "port": 5432, "db_name": "analytics"},
            ),
        ],
    )
    def test_convert_source_dict_valid_cases(
        self,
        mock_configs: Dict[str, Any],
        source_data: Dict[str, Any],
        expected_type: ConnectionType,
        expected_attributes: Dict[str, Any],
    ) -> None:
        """Test successful source dict conversion across all supported types"""
        # Arrange
        validator = DataValidator(
            source_config=TestDataBuilder.connection().build(),  # Placeholder
            rules=[],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Act
        result = validator._convert_source_dict(source_data)

        # Assert
        assert isinstance(result, ConnectionSchema)
        assert result.connection_type == expected_type

        for attr, expected_value in expected_attributes.items():
            actual_value = getattr(result, attr)
            assert (
                actual_value == expected_value
            ), f"Attribute {attr} mismatch: expected {expected_value}, got {actual_value}"

        # Verify the connection has required fields
        assert result.name is not None
        assert result.description is not None

    @pytest.mark.parametrize(
        "invalid_source,expected_error_message",
        [
            # Missing type field
            (
                {"path": "/data/test.csv"},
                "source_config dict must contain a 'type' key",
            ),
            ({"host": "localhost"}, "source_config dict must contain a 'type' key"),
            # Empty dict
            ({}, "source_config dict must contain a 'type' key"),
            # Invalid type
            (
                {"type": "unsupported_db"},
                None,
            ),  # Should raise but with different message
        ],
    )
    def test_convert_source_dict_error_cases(
        self,
        mock_configs: Dict[str, Any],
        invalid_source: Dict[str, Any],
        expected_error_message: str,
    ) -> None:
        """Test error handling for malformed source dicts"""
        # Arrange
        validator = DataValidator(
            source_config=TestDataBuilder.connection().build(),
            rules=[],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            validator._convert_source_dict(invalid_source)

        if expected_error_message:
            assert expected_error_message in str(exc_info.value)

    # ============================================================================
    # Rule Dict Conversion Tests
    # ============================================================================

    @pytest.mark.parametrize(
        "rule_data,expected_type,expected_params",
        [
            # NOT_NULL rule
            ({"type": "not_null", "target": "user_id"}, RuleType.NOT_NULL, {}),
            # UNIQUE rule
            ({"rule_type": "unique", "column": "email"}, RuleType.UNIQUE, {}),
            # LENGTH rule with parameters
            (
                {
                    "type": "length",
                    "target": "username",
                    "parameters": {"min_length": 3, "max_length": 50},
                },
                RuleType.LENGTH,
                {"min_length": 3, "max_length": 50},
            ),
            # RANGE rule
            (
                {
                    "type": "range",
                    "column": "age",
                    "parameters": {"min": 18, "max": 120},
                },
                RuleType.RANGE,
                {"min": 18, "max": 120},
            ),
            # ENUM rule
            (
                {
                    "type": "enum",
                    "target": "status",
                    "parameters": {"allowed_values": ["active", "inactive"]},
                },
                RuleType.ENUM,
                {"allowed_values": ["active", "inactive"]},
            ),
            # REGEX rule
            (
                {
                    "type": "regex",
                    "column": "phone",
                    "parameters": {"pattern": r"^\+?[\d\s\-\(\)]+$"},
                },
                RuleType.REGEX,
                {"pattern": r"^\+?[\d\s\-\(\)]+$"},
            ),
        ],
    )
    def test_convert_rule_dict_valid_cases(
        self,
        mock_configs: Dict[str, Any],
        rule_data: Dict[str, Any],
        expected_type: RuleType,
        expected_params: Dict[str, Any],
    ) -> None:
        """Test successful rule dict conversion with parameter validation"""
        # Arrange
        validator = DataValidator(
            source_config=TestDataBuilder.connection().build(),
            rules=[],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Act
        result = validator._convert_rule_dict(rule_data)

        # Assert
        assert isinstance(result, RuleSchema)
        assert result.type == expected_type

        # Verify target configuration
        assert len(result.target.entities) == 1
        entity = result.target.entities[0]
        expected_column = rule_data.get("target") or rule_data.get("column")
        assert entity.column == expected_column

        # Verify parameters
        for param_key, expected_value in expected_params.items():
            assert param_key in result.parameters
            assert result.parameters[param_key] == expected_value

    @pytest.mark.parametrize(
        "invalid_rule,expected_error_type",
        [
            # Missing type
            ({"target": "user_id"}, ValueError),
            ({"column": "email"}, ValueError),
            # Missing target/column
            ({"type": "not_null"}, ValueError),
            ({"rule_type": "unique"}, ValueError),
            # Invalid type
            ({"type": "invalid_rule", "target": "col"}, ValueError),
            # Wrong parameter type
            ({"type": "not_null", "target": "col", "parameters": "invalid"}, TypeError),
        ],
    )
    def test_convert_rule_dict_error_cases(
        self,
        mock_configs: Dict[str, Any],
        invalid_rule: Dict[str, Any],
        expected_error_type: type,
    ) -> None:
        """Test error handling for malformed rule dicts"""
        # Arrange
        validator = DataValidator(
            source_config=TestDataBuilder.connection().build(),
            rules=[],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Act & Assert
        with pytest.raises(expected_error_type):
            validator._convert_rule_dict(invalid_rule)

    # ============================================================================
    # Column Name Cleaning Tests (SQLite Compatibility)
    # ============================================================================

    @pytest.mark.parametrize(
        "input_column,expected_output,test_description",
        [
            # Normal cases
            ("user_id", "user_id", "normal_underscore"),
            ("firstName", "firstName", "camelCase"),
            ("Name", "Name", "simple_word"),
            # Special character replacement
            ("user-name", "user_name", "dash_replacement"),
            ("user.email", "user_email", "dot_replacement"),
            ("user@domain", "user_domain", "at_symbol_replacement"),
            ("user name", "user_name", "space_replacement"),
            ("user/path", "user_path", "slash_replacement"),
            ("user#id", "user_id", "hash_replacement"),
            ("user%percent", "user_percent", "percent_replacement"),
            ("user(1)", "user_1_", "parentheses_replacement"),
            ("user[0]", "user_0_", "brackets_replacement"),
            ("user{id}", "user_id_", "braces_replacement"),
            # Number prefix handling
            ("1user", "col_1user", "number_prefix"),
            ("123column", "col_123column", "multiple_number_prefix"),
            ("0index", "col_0index", "zero_prefix"),
            # Empty/None handling
            ("", "unnamed_column", "empty_string"),
            ("   ", "unnamed_column", "whitespace_only"),
            # Unicode and special cases (non-ASCII replaced with underscores)
            ("用户名", "___", "unicode_characters"),
            ("user🎯name", "user_name", "emoji_replacement"),
            ("αβγ", "___", "greek_letters"),
            # Mixed complexity
            ("2user-name.email@domain", "col_2user_name_email_domain", "complex_mixed"),
            (
                "  3_test-column.name  ",
                "col_3_test_column_name",
                "complex_with_whitespace",
            ),
            # Edge cases from real data
            ("User ID", "User_ID", "space_in_title_case"),
            ("customer_1st_name", "customer_1st_name", "number_in_middle"),
            ("order#2023-01", "order_2023_01", "complex_business_case"),
        ],
    )
    def test_clean_column_name_comprehensive(
        self,
        mock_configs: Dict[str, Any],
        input_column: str,
        expected_output: str,
        test_description: str,
    ) -> None:
        """Comprehensive test for column name cleaning with SQLite compatibility"""
        # Arrange
        validator = DataValidator(
            source_config=TestDataBuilder.connection().build(),
            rules=[],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Act
        result = validator._clean_column_name(input_column)

        # Assert
        assert result == expected_output, f"Test case '{test_description}' failed"

        # Verify SQLite compatibility
        self._assert_sqlite_compatible_name(result)

    @pytest.mark.parametrize(
        "input_table,expected_output,test_description",
        [
            # Normal cases
            ("users", "users", "simple_table"),
            ("user_profiles", "user_profiles", "underscore_table"),
            ("UserData", "UserData", "camelCase_table"),
            # Special character handling
            ("user-data", "user_data", "dash_in_table"),
            ("user.table", "user_table", "dot_in_table"),
            ("user@domain", "user_domain", "at_in_table"),
            ("user table", "user_table", "space_in_table"),
            # Number prefix handling
            ("1users", "table_1users", "number_prefix_table"),
            ("2023_data", "table_2023_data", "year_prefix"),
            # Empty handling
            ("", "data", "empty_table_name"),
            ("   ", "data", "whitespace_table_name"),
            # Complex cases
            ("2user-data.backup", "table_2user_data_backup", "complex_table_name"),
            ("temp@2023-01-01", "temp_2023_01_01", "timestamped_table"),
        ],
    )
    def test_clean_table_name_comprehensive(
        self,
        mock_configs: Dict[str, Any],
        input_table: str,
        expected_output: str,
        test_description: str,
    ) -> None:
        """Comprehensive test for table name cleaning with SQLite compatibility"""
        # Arrange
        validator = DataValidator(
            source_config=TestDataBuilder.connection().build(),
            rules=[],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Act
        result = validator._clean_table_name(input_table)

        # Assert
        assert result == expected_output, f"Test case '{test_description}' failed"

        # Verify SQLite compatibility
        self._assert_sqlite_compatible_name(result)

    # ============================================================================
    # Property-based Testing for Name Cleaning
    # ============================================================================

    @pytest.mark.parametrize(
        "test_input",
        [
            "a" * 100,  # Very long name
            "🎯" * 50,  # Many emojis
            "αβγδε" * 20,  # Unicode characters
            "123" + "x" * 97,  # Number prefix with long name
            "",  # Empty
            "  ",  # Whitespace only
            "user_name_with_very_long_description_that_exceeds_normal_limits",  # Business case
        ],
    )
    def test_name_cleaning_property_based(
        self, mock_configs: Dict[str, Any], test_input: str
    ) -> None:
        """Property-based testing for name cleaning invariants"""
        # Arrange
        validator = DataValidator(
            source_config=TestDataBuilder.connection().build(),
            rules=[],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Act
        column_result = validator._clean_column_name(test_input)
        table_result = validator._clean_table_name(test_input)

        # Assert invariants
        # 1. Results are never empty
        assert column_result != ""
        assert table_result != ""

        # 2. Results are SQLite compatible
        self._assert_sqlite_compatible_name(column_result)
        self._assert_sqlite_compatible_name(table_result)

        # 3. Results don't start with numbers (unless prefixed)
        assert not column_result[0].isdigit() or column_result.startswith(
            ("col_", "table_")
        )
        assert not table_result[0].isdigit() or table_result.startswith(
            ("col_", "table_")
        )

    # ============================================================================
    # Integration Tests - Full Conversion Flow
    # ============================================================================

    def test_full_conversion_integration_csv_source(
        self, mock_configs: Dict[str, Any]
    ) -> None:
        """Test complete conversion flow from dict inputs to schema objects"""
        # Arrange
        source_dict = {"type": "csv", "path": "/data/test-file.csv"}
        rule_dicts: List[Dict[str, Any]] = [
            {"type": "not_null", "target": "user-id"},
            {
                "type": "length",
                "column": "user name",
                "parameters": {"min_length": 2, "max_length": 50},
            },
            {"type": "unique", "target": "email@domain"},
        ]

        # Act
        validator = DataValidator(
            source_config=source_dict,
            rules=rule_dicts,  # type: ignore[arg-type]
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Assert source conversion
        assert isinstance(validator.source_config, ConnectionSchema)
        assert validator.source_config.connection_type == ConnectionType.CSV
        assert validator.source_config.file_path == "/data/test-file.csv"

        # Assert rule conversions
        assert len(validator.rules) == 3
        assert all(isinstance(rule, RuleSchema) for rule in validator.rules)

        # Check specific rule conversions
        not_null_rule = next(
            rule for rule in validator.rules if rule.type == RuleType.NOT_NULL
        )
        assert not_null_rule.target.entities[0].column == "user-id"

        length_rule = next(
            rule for rule in validator.rules if rule.type == RuleType.LENGTH
        )
        assert length_rule.target.entities[0].column == "user name"
        assert length_rule.parameters["min_length"] == 2
        assert length_rule.parameters["max_length"] == 50

    def test_mixed_schema_and_dict_inputs(self, mock_configs: Dict[str, Any]) -> None:
        """Test handling mixed RuleSchema objects and dict inputs"""
        # Arrange - Mix pre-built schema and dict
        source_dict = {
            "type": "mysql",
            "host": "localhost",
            "database": "testdb",
            "port": 3306,
        }
        rule_schema = (
            TestDataBuilder.rule()
            .as_not_null_rule()
            .with_target("test_db", "users", "id")
            .build()
        )
        rule_dict = {"type": "unique", "target": "email"}

        # Act
        validator = DataValidator(
            source_config=source_dict,
            rules=[rule_schema, rule_dict],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Assert
        assert len(validator.rules) == 2
        assert all(isinstance(rule, RuleSchema) for rule in validator.rules)

        # First rule should be unchanged (was already RuleSchema)
        assert validator.rules[0] == rule_schema

        # Second rule should be converted from dict
        unique_rule = validator.rules[1]
        assert unique_rule.type == RuleType.UNIQUE
        assert unique_rule.target.entities[0].column == "email"

    # ============================================================================
    # Error Handling and Edge Cases
    # ============================================================================

    def test_invalid_rule_type_in_constructor(
        self, mock_configs: Dict[str, Any]
    ) -> None:
        """Test error handling for invalid rule types in constructor"""
        # Arrange
        source_dict = {"type": "csv", "path": "/data/test.csv"}
        invalid_rule = "not_a_dict_or_schema"  # Wrong type

        # Act & Assert
        with pytest.raises(TypeError) as exc_info:
            DataValidator(
                source_config=source_dict,
                rules=[invalid_rule],  # type: ignore[list-item]
                core_config=mock_configs["core_config"],
                cli_config=mock_configs["cli_config"],
            )

        assert "must be a RuleSchema or a dict" in str(exc_info.value)

    def test_none_inputs_handling(self, mock_configs: Dict[str, Any]) -> None:
        """Test handling of None values in conversion methods"""
        # Arrange
        validator = DataValidator(
            source_config=TestDataBuilder.connection().build(),
            rules=[],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Test None column name
        result = validator._clean_column_name(None)  # type: ignore[arg-type]
        assert result == "unnamed_column"

        # Test None table name
        result = validator._clean_table_name(None)  # type: ignore[arg-type]
        assert result == "data"

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _assert_sqlite_compatible_name(self, name: str) -> None:
        """Assert that a name is SQLite compatible"""
        # Must not be empty
        assert name != ""

        # Must not start with a number (unless properly prefixed)
        if name[0].isdigit():
            assert name.startswith(
                ("col_", "table_")
            ), f"Name '{name}' starts with digit but not properly prefixed"

        # Must contain only word characters and underscores (after cleaning)
        assert re.match(
            r"^[a-zA-Z_][a-zA-Z0-9_]*$", name
        ), f"Name '{name}' contains invalid characters for SQLite"

        # Should not be empty after stripping
        assert name.strip() == name, f"Name '{name}' has leading/trailing whitespace"
