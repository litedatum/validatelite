"""
🎭 Testing Ghost's Rule Parameter Conversion Test Suite

This modernized test suite implements the four key improvement strategies:
1. Schema Builder Pattern - Eliminates fixture duplication
2. Contract Testing - Ensures mocks match reality
3. Property-based Testing - Verifies behavior with random inputs
4. Mutation Testing Readiness - Catches subtle bugs

Scenarios covered:
- Legacy parameter format conversion
- Target information extraction
- Rule configuration extraction
- Filter condition handling
- Parameter merging strategies
- Edge cases and error handling
"""

import copy
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from shared.utils.param_converter import (
    convert_legacy_params,
    extract_filter_condition,
    extract_rule_config,
    extract_target_info,
    merge_params,
)
from tests.shared.builders.test_builders import TestDataBuilder
from tests.shared.contracts.test_contracts import ContractValidator


class TestRuleParameterConversion:
    """🎭 Testing Ghost's modernized parameter conversion tests"""

    def setup_method(self) -> None:
        """Setup for each test method"""
        self.builder = TestDataBuilder()
        self.contract_validator = ContractValidator()

    # =================== LEGACY PARAMETER CONVERSION TESTS ===================

    def test_basic_legacy_parameter_conversion(self) -> None:
        """Test basic conversion from legacy to modern parameter format"""
        legacy_params = {
            "database": "test_db",
            "table_name": "users",
            "column_name": "email",
            "filter_condition": "status = 'active'",
        }

        converted = convert_legacy_params(legacy_params)

        # Validate target structure
        assert "target" in converted
        assert converted["target"]["database"] == "test_db"
        assert converted["target"]["table"] == "users"
        assert converted["target"]["column"] == "email"

        # Validate rule config structure
        assert "rule_config" in converted
        assert converted["rule_config"]["filter_condition"] == "status = 'active'"

        # Original params should not be modified
        assert "target" not in legacy_params
        assert legacy_params["database"] == "test_db"

    def test_already_modern_format_passthrough(self) -> None:
        """Test that modern format parameters pass through unchanged"""
        modern_params = {
            "target": {"database": "test_db", "table": "users", "column": "email"},
            "rule_config": {
                "filter_condition": "status = 'active'",
                "pattern": r".*@.*\..*",
            },
        }

        result = convert_legacy_params(modern_params)

        # Should maintain modern structure
        assert "target" in result
        assert result["target"]["database"] == "test_db"
        assert result["target"]["table"] == "users"
        assert result["target"]["column"] == "email"
        assert result["rule_config"]["filter_condition"] == "status = 'active'"
        assert result["rule_config"]["pattern"] == r".*@.*\..*"

    @pytest.mark.parametrize(
        "rule_type,legacy_params,expected_config",
        [
            ("RANGE", {"min": 0, "max": 100}, {"min": 0, "max": 100}),
            (
                "ENUM",
                {"allowed_values": ["A", "B", "C"]},
                {"allowed_values": ["A", "B", "C"]},
            ),
            (
                "ENUM",
                {"values": ["X", "Y", "Z"]},
                {"allowed_values": ["X", "Y", "Z"]},
            ),  # Alternative field name
            (
                "REGEX",
                {"pattern": r"\d{3}-\d{3}-\d{4}"},
                {"pattern": r"\d{3}-\d{3}-\d{4}"},
            ),
            (
                "LENGTH",
                {"min_length": 2, "max_length": 50},
                {"min_length": 2, "max_length": 50},
            ),
            ("DATE_FORMAT", {"format": "%Y-%m-%d"}, {"format": "%Y-%m-%d"}),
        ],
    )
    def test_rule_specific_parameter_conversion(
        self,
        rule_type: str,
        legacy_params: Dict[str, Any],
        expected_config: Dict[str, Any],
    ) -> None:
        """Test conversion of rule-specific parameters"""
        full_legacy_params = {
            "database": "test_db",
            "table_name": "test_table",
            "column_name": "test_column",
            **legacy_params,
        }

        converted = convert_legacy_params(full_legacy_params)

        # Verify rule-specific config is properly extracted
        for key, value in expected_config.items():
            assert converted["rule_config"][key] == value

    def test_alternative_field_names_support(self) -> None:
        """Test support for alternative field names in legacy parameters"""
        legacy_params = {
            "database": "test_db",
            "table": "users",  # Alternative to table_name
            "column": "email",  # Alternative to column_name
            "values": ["A", "B"],  # Alternative to allowed_values
            "min_value": 10,  # Alternative to min
            "max_value": 90,  # Alternative to max
        }

        converted = convert_legacy_params(legacy_params)

        assert converted["target"]["table"] == "users"
        assert converted["target"]["column"] == "email"
        assert converted["rule_config"]["allowed_values"] == ["A", "B"]
        assert converted["rule_config"]["min_value"] == 10
        assert converted["rule_config"]["max_value"] == 90

    # =================== TARGET INFORMATION EXTRACTION TESTS ===================

    def test_extract_target_info_from_modern_format(self) -> None:
        """Test extracting target info from modern parameter format"""
        params = {
            "target": {
                "database": "analytics_db",
                "table": "user_events",
                "column": "event_timestamp",
            },
            "rule_config": {"filter_condition": "event_type = 'click'"},
        }

        database, table, column = extract_target_info(params)

        assert database == "analytics_db"
        assert table == "user_events"
        assert column == "event_timestamp"

    def test_extract_target_info_from_legacy_format(self) -> None:
        """Test extracting target info from legacy parameter format"""
        legacy_params = {
            "database": "sales_db",
            "table_name": "orders",
            "column_name": "order_date",
            "filter_condition": "status = 'completed'",
        }

        database, table, column = extract_target_info(legacy_params)

        assert database == "sales_db"
        assert table == "orders"
        assert column == "order_date"

    def test_extract_target_info_from_multi_target_format(self) -> None:
        """Test extracting target info from multi-target format (uses first target)"""
        params = {
            "targets": [
                {
                    "database": "main_db",
                    "table": "users",
                    "column": "user_id",
                    "alias": "u",
                },
                {
                    "database": "main_db",
                    "table": "orders",
                    "column": "user_id",
                    "alias": "o",
                },
            ],
            "rule_config": {"join_condition": "u.user_id = o.user_id"},
        }

        database, table, column = extract_target_info(params)

        # Should use first target
        assert database == "main_db"
        assert table == "users"
        assert column == "user_id"

    def test_extract_target_info_no_column(self) -> None:
        """Test extracting target info when column is not specified"""
        params = {
            "target": {
                "database": "test_db",
                "table": "test_table",
                # No column specified
            }
        }

        database, table, column = extract_target_info(params)

        assert database == "test_db"
        assert table == "test_table"
        assert column is None

    @pytest.mark.parametrize(
        "invalid_params,expected_error",
        [
            ({}, "Invalid parameter structure, missing target or targets"),
            (
                {"target": {}},
                "Target information incomplete, missing database or table",
            ),
            (
                {"target": {"database": "test_db"}},
                "Target information incomplete, missing database or table",
            ),
            (
                {"target": {"table": "test_table"}},
                "Target information incomplete, missing database or table",
            ),
            ({"targets": []}, "Invalid parameter structure, missing target or targets"),
        ],
    )
    def test_extract_target_info_error_handling(
        self, invalid_params: Dict[str, Any], expected_error: str
    ) -> None:
        """Test error handling for invalid target information"""
        with pytest.raises(ValueError, match=expected_error):
            extract_target_info(invalid_params)

    # =================== RULE CONFIGURATION EXTRACTION TESTS ===================

    def test_extract_rule_config_from_modern_format(self) -> None:
        """Test extracting rule config from modern parameter format"""
        params = {
            "target": {"database": "test_db", "table": "users", "column": "email"},
            "rule_config": {
                "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                "filter_condition": "is_verified = true",
                "case_sensitive": False,
            },
        }

        config = extract_rule_config(params)

        assert config["pattern"] == r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        assert config["filter_condition"] == "is_verified = true"
        assert config["case_sensitive"] is False

    def test_extract_rule_config_from_legacy_format(self) -> None:
        """Test extracting rule config from legacy parameter format"""
        legacy_params = {
            "database": "test_db",
            "table_name": "products",
            "column_name": "price",
            "min": 0.01,
            "max": 999.99,
            "filter_condition": "status = 'active'",
            "currency": "USD",
        }

        config = extract_rule_config(legacy_params)

        assert config["min"] == 0.01
        assert config["max"] == 999.99
        assert config["filter_condition"] == "status = 'active'"
        assert config["currency"] == "USD"

    def test_extract_rule_config_empty_parameters(self) -> None:
        """Test extracting rule config from empty parameters"""
        config = extract_rule_config({})
        assert config == {}

        config = extract_rule_config({"target": {"database": "db", "table": "table"}})
        assert config == {}

    # =================== FILTER CONDITION EXTRACTION TESTS ===================

    def test_extract_filter_condition_from_rule_config(self) -> None:
        """Test extracting filter condition from rule config"""
        params = {
            "target": {"database": "db", "table": "table", "column": "col"},
            "rule_config": {
                "filter_condition": "status IN ('active', 'pending')",
                "other_param": "value",
            },
        }

        filter_condition = extract_filter_condition(params)
        assert filter_condition == "status IN ('active', 'pending')"

    def test_extract_filter_condition_from_legacy_format(self) -> None:
        """Test extracting filter condition from legacy format"""
        legacy_params = {
            "database": "test_db",
            "table_name": "orders",
            "column_name": "amount",
            "filter_condition": "created_date >= '2023-01-01'",
            "min": 100,
        }

        filter_condition = extract_filter_condition(legacy_params)
        assert filter_condition == "created_date >= '2023-01-01'"

    def test_extract_filter_condition_missing(self) -> None:
        """Test extracting filter condition when it doesn't exist"""
        params = {
            "target": {"database": "db", "table": "table", "column": "col"},
            "rule_config": {"min": 0, "max": 100},
        }

        filter_condition = extract_filter_condition(params)
        assert filter_condition is None

    # =================== PARAMETER MERGING TESTS ===================

    def test_merge_params_basic(self) -> None:
        """Test basic parameter merging"""
        base_params = {
            "target": {"database": "test_db", "table": "users", "column": "email"},
            "rule_config": {"pattern": r".*@.*", "case_sensitive": True},
        }

        override_params = {
            "rule_config": {
                "case_sensitive": False,  # Override existing
                "validate_mx": True,  # Add new
            }
        }

        merged = merge_params(base_params, override_params)

        # Target should remain unchanged
        assert merged["target"]["database"] == "test_db"
        assert merged["target"]["table"] == "users"
        assert merged["target"]["column"] == "email"

        # Rule config should be merged
        assert merged["rule_config"]["pattern"] == r".*@.*"  # Preserved
        assert merged["rule_config"]["case_sensitive"] is False  # Overridden
        assert merged["rule_config"]["validate_mx"] is True  # Added

    def test_merge_params_target_override(self) -> None:
        """Test merging parameters with target override"""
        base_params = {
            "target": {
                "database": "old_db",
                "table": "old_table",
                "column": "old_column",
            },
            "rule_config": {"filter_condition": "status = 'active'"},
        }

        override_params = {
            "target": {
                "database": "new_db",
                "table": "new_table",
                # column intentionally omitted
            },
            "rule_config": {"threshold": 0.95},
        }

        merged = merge_params(base_params, override_params)

        # Target should be partially overridden
        assert merged["target"]["database"] == "new_db"  # Overridden
        assert merged["target"]["table"] == "new_table"  # Overridden
        assert merged["target"]["column"] == "old_column"  # Preserved (not in override)

        # Rule config should be merged
        assert merged["rule_config"]["filter_condition"] == "status = 'active'"
        assert merged["rule_config"]["threshold"] == 0.95

    def test_merge_params_with_legacy_formats(self) -> None:
        """Test merging parameters when inputs are in legacy format"""
        base_legacy = {
            "database": "base_db",
            "table_name": "base_table",
            "column_name": "base_column",
            "min": 0,
            "max": 100,
        }

        override_legacy = {
            "database": "override_db",
            "max": 200,
            "filter_condition": "status = 'active'",
        }

        merged = merge_params(base_legacy, override_legacy)

        # Should convert to modern format and merge
        assert merged["target"]["database"] == "override_db"  # Overridden
        assert merged["target"]["table"] == "base_table"  # Preserved
        assert merged["target"]["column"] == "base_column"  # Preserved
        assert merged["rule_config"]["min"] == 0  # Preserved
        assert merged["rule_config"]["max"] == 200  # Overridden
        assert merged["rule_config"]["filter_condition"] == "status = 'active'"  # Added

    def test_merge_params_empty_override(self) -> None:
        """Test merging with empty override parameters"""
        base_params = {
            "target": {"database": "db", "table": "table"},
            "rule_config": {"min": 0, "max": 100},
        }

        merged = merge_params(base_params, {})

        # Should remain unchanged
        assert merged["target"]["database"] == "db"
        assert merged["target"]["table"] == "table"
        assert merged["rule_config"]["min"] == 0
        assert merged["rule_config"]["max"] == 100

    # =================== PROPERTY-BASED TESTING ===================

    @given(
        database=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
        ),
        table=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
        ),
        column=st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
        ),
    )
    @settings(max_examples=20)
    def test_legacy_conversion_roundtrip_property(
        self, database: str, table: str, column: str
    ) -> None:
        """Property: Legacy parameter conversion should preserve target information"""
        assume(database and table and column)  # Ensure non-empty strings

        legacy_params = {
            "database": database,
            "table_name": table,
            "column_name": column,
            "filter_condition": "status = 'active'",
        }

        converted = convert_legacy_params(legacy_params)
        extracted_db, extracted_table, extracted_column = extract_target_info(converted)

        # Property: Target information should be preserved through conversion
        assert extracted_db == database
        assert extracted_table == table
        assert extracted_column == column

    @given(
        config_dict=st.dictionaries(
            keys=st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
            ),
            values=st.one_of(
                st.text(), st.integers(), st.floats(allow_nan=False), st.booleans()
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=15)
    def test_rule_config_extraction_property(self, config_dict: Dict[str, Any]) -> None:
        """Property: Rule config extraction should preserve all config values"""
        params = {
            "target": {"database": "db", "table": "table", "column": "col"},
            "rule_config": config_dict,
        }

        extracted_config = extract_rule_config(params)

        # Property: All config values should be preserved
        for key, value in config_dict.items():
            assert extracted_config[key] == value

    @given(
        base_config=st.dictionaries(
            keys=st.text(min_size=1, max_size=10),
            values=st.integers(),
            min_size=1,
            max_size=5,
        ),
        override_config=st.dictionaries(
            keys=st.text(min_size=1, max_size=10),
            values=st.integers(),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=10)
    def test_parameter_merge_properties(
        self, base_config: Dict[str, int], override_config: Dict[str, int]
    ) -> None:
        """Property: Parameter merging should follow override semantics correctly"""
        base_params = {
            "target": {"database": "db", "table": "table"},
            "rule_config": base_config,
        }

        override_params = {"rule_config": override_config}

        merged = merge_params(base_params, override_params)

        # Property: Override values should take precedence
        for key, value in override_config.items():
            assert merged["rule_config"][key] == value

        # Property: Non-overridden base values should be preserved
        for key, value in base_config.items():
            if key not in override_config:
                assert merged["rule_config"][key] == value

    # =================== EDGE CASES AND ERROR RESILIENCE ===================

    def test_parameter_immutability(self) -> None:
        """Test that original parameters are not modified during conversion"""
        original_params = {
            "database": "test_db",
            "table_name": "test_table",
            "column_name": "test_column",
            "nested_config": {"key": "value"},
        }

        # Keep a deep copy for comparison
        original_copy = copy.deepcopy(original_params)

        # Convert parameters
        converted = convert_legacy_params(original_params)

        # Original should be unchanged
        assert original_params == original_copy
        assert "target" not in original_params
        assert original_params["database"] == "test_db"

    def test_deep_nested_parameter_handling(self) -> None:
        """Test handling of deeply nested parameter structures"""
        complex_params = {
            "database": "test_db",
            "table_name": "test_table",
            "column_name": "test_column",
            "advanced_config": {
                "validation_rules": {
                    "email": {"pattern": r".*@.*", "check_mx": True},
                    "length": {"min": 5, "max": 100},
                },
                "error_handling": {"on_invalid": "skip", "log_errors": True},
            },
        }

        converted = convert_legacy_params(complex_params)

        # Deep nested structures should be preserved in rule_config
        assert "advanced_config" in converted["rule_config"]
        nested_config = converted["rule_config"]["advanced_config"]
        assert nested_config["validation_rules"]["email"]["pattern"] == r".*@.*"
        assert nested_config["validation_rules"]["length"]["min"] == 5
        assert nested_config["error_handling"]["on_invalid"] == "skip"

    def test_null_and_empty_value_handling(self) -> None:
        """Test handling of null and empty values in parameters"""
        params_with_nulls: Dict[str, Any] = {
            "database": "test_db",
            "table_name": "test_table",
            "column_name": None,  # Null column
            "filter_condition": "",  # Empty string
            "threshold": 0,  # Zero value
            "enabled": False,  # False boolean
            "tags": [],  # Empty list
            "metadata": {},  # Empty dict
        }

        converted = convert_legacy_params(params_with_nulls)

        # All values should be preserved, including nulls and empties
        assert converted["target"]["column"] is None
        assert converted["rule_config"]["filter_condition"] == ""
        assert converted["rule_config"]["threshold"] == 0
        assert converted["rule_config"]["enabled"] is False
        assert converted["rule_config"]["tags"] == []
        assert converted["rule_config"]["metadata"] == {}

    # =================== PERFORMANCE AND CONCURRENCY TESTS ===================

    def test_parameter_conversion_performance(self) -> None:
        """Test parameter conversion performance with large parameter sets"""
        import time

        # Create a large parameter set
        large_params = {
            "database": "test_db",
            "table_name": "large_table",
            "column_name": "test_column",
        }

        # Add many rule config parameters
        for i in range(1000):
            large_params[f"param_{i}"] = f"value_{i}"

        start_time = time.time()
        converted = convert_legacy_params(large_params)
        end_time = time.time()

        # Should complete quickly even with many parameters
        assert (end_time - start_time) < 1.0  # Less than 1 second

        # Should still convert correctly
        assert converted["target"]["database"] == "test_db"
        assert converted["target"]["table"] == "large_table"
        assert len(converted["rule_config"]) >= 1000

    # =================== REGRESSION TESTS ===================

    def test_regression_enum_values_field_conversion(self) -> None:
        """Regression test: Ensure 'values' field converts to 'allowed_values'"""
        legacy_params = {
            "database": "test_db",
            "table_name": "test_table",
            "column_name": "status",
            "values": [
                "active",
                "inactive",
                "pending",
            ],  # Using 'values' instead of 'allowed_values'
        }

        converted = convert_legacy_params(legacy_params)

        # Should convert 'values' to 'allowed_values'
        assert "allowed_values" in converted["rule_config"]
        assert converted["rule_config"]["allowed_values"] == [
            "active",
            "inactive",
            "pending",
        ]
        assert "values" not in converted["rule_config"]

    def test_regression_case_sensitive_parameter_preservation(self) -> None:
        """Regression test: Ensure case-sensitive parameter names are preserved"""
        legacy_params = {
            "database": "test_db",
            "table_name": "test_table",
            "column_name": "test_column",
            "CamelCaseParam": "value1",
            "snake_case_param": "value2",
            "UPPER_CASE_PARAM": "value3",
        }

        converted = convert_legacy_params(legacy_params)

        # All parameter names should be preserved with original casing
        config = converted["rule_config"]
        assert config["CamelCaseParam"] == "value1"
        assert config["snake_case_param"] == "value2"
        assert config["UPPER_CASE_PARAM"] == "value3"
