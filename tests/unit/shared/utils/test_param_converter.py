"""
Test the parameter conversion tool.

Test the functionality of the parameter conversion tool.
Conversion of legacy parameter format.
2. Target Information Extraction
3. Rule Configuration Extraction
Apply filtering criteria.
Parameter merging.
"""

from typing import Any, Dict, List, Optional

import pytest

from shared.utils.param_converter import (
    convert_legacy_params,
    extract_filter_condition,
    extract_rule_config,
    extract_target_info,
    merge_params,
)


class TestParamConverter:
    """Test the parameter conversion tool."""

    def test_convert_legacy_params(self) -> None:
        """Test the conversion of legacy parameter formats."""
        # Test legacy parameter format.
        legacy_params = {
            "database": "test_db",
            "table_name": "test_table",
            "column_name": "test_column",
            "filter_condition": "status = 'active'",
            "min_value": 0,
            "max_value": 100,
        }

        new_params = convert_legacy_params(legacy_params)

        # Validate the new format structure.
        assert "target" in new_params
        assert new_params["target"]["database"] == "test_db"
        assert new_params["target"]["table"] == "test_table"
        assert new_params["target"]["column"] == "test_column"

        assert "rule_config" in new_params
        assert new_params["rule_config"]["filter_condition"] == "status = 'active'"
        assert new_params["rule_config"]["min_value"] == 0
        assert new_params["rule_config"]["max_value"] == 100

        # Tests the case where the argument is already in the new format (a shallow copy should be returned).
        new_format_params = {
            "target": {
                "database": "test_db",
                "table": "test_table",
                "column": "test_column",
            },
            "rule_config": {"filter_condition": "status = 'active'"},
        }

        result = convert_legacy_params(new_format_params)
        # The `target` field is required as this function preserves any existing formatting within it.
        assert "target" in result
        assert result["target"]["database"] == "test_db"

    def test_extract_target_info(self) -> None:
        """Target information extraction testing."""
        # Test extraction from the new format.
        params = {
            "target": {
                "database": "test_db",
                "table": "test_table",
                "column": "test_column",
            },
            "rule_config": {"filter_condition": "status = 'active'"},
        }
        database, table, column = extract_target_info(params)
        assert database == "test_db"
        assert table == "test_table"
        assert column == "test_column"

        # Test extraction from the legacy format.
        l_params: Dict[str, str] = {
            "database": "test_db",
            "table_name": "test_table",
            "column_name": "test_column",
            "filter_condition": "status = 'active'",
        }
        database, table, column = extract_target_info(l_params)
        assert database == "test_db"
        assert table == "test_table"
        assert column == "test_column"

        # Test extraction from multi-table formats.
        m_params = {
            "targets": [
                {
                    "database": "test_db",
                    "table": "table1",
                    "column": "id",
                    "alias": "t1",
                },
                {
                    "database": "test_db",
                    "table": "table2",
                    "column": "id",
                    "alias": "t2",
                },
            ],
            "rule_config": {"join_condition": "t1.id = t2.parent_id"},
        }
        database, table, column = extract_target_info(m_params)
        assert database == "test_db"
        assert table == "table1"
        assert column == "id"

        # Test for missing required information.
        with pytest.raises(ValueError):
            extract_target_info(
                {}
            )  # Empty or null parameter.  (Or, depending on context, "No parameter" or "Parameter not provided.")

        with pytest.raises(ValueError):
            extract_target_info(
                {"target": {"table": "test_table"}}  # The database is missing.
            )

    def test_extract_rule_config(self) -> None:
        """Testing the extraction of rule configurations."""
        # Test extraction from the new format.
        params = {
            "target": {
                "database": "test_db",
                "table": "test_table",
                "column": "test_column",
            },
            "rule_config": {
                "filter_condition": "status = 'active'",
                "min_value": 0,
                "max_value": 100,
            },
        }
        rule_config = extract_rule_config(params)
        assert rule_config["filter_condition"] == "status = 'active'"
        assert rule_config["min_value"] == 0
        assert rule_config["max_value"] == 100

        # Test extraction from the legacy/old format.
        params = {
            "database": "test_db",
            "table_name": "test_table",
            "column_name": "test_column",
            "filter_condition": "status = 'active'",
            "min_value": 0,
            "max_value": 100,
        }
        rule_config = extract_rule_config(params)
        assert rule_config["filter_condition"] == "status = 'active'"
        assert rule_config["min_value"] == 0
        assert rule_config["max_value"] == 100

        # Test with empty arguments/parameters.
        rule_config = extract_rule_config({})
        assert rule_config == {}

    def test_extract_filter_condition(self) -> None:
        """Testing filter criteria extraction."""
        # Test extraction from the new format.
        params = {
            "target": {
                "database": "test_db",
                "table": "test_table",
                "column": "test_column",
            },
            "rule_config": {
                "filter_condition": "status = 'active'",
                "min_value": 0,
                "max_value": 100,
            },
        }
        filter_condition = extract_filter_condition(params)
        assert filter_condition == "status = 'active'"

        # Test extraction from the legacy format.
        params = {
            "database": "test_db",
            "table_name": "test_table",
            "column_name": "test_column",
            "filter_condition": "status = 'active'",
            "min_value": 0,
            "max_value": 100,
        }
        filter_condition = extract_filter_condition(params)
        assert filter_condition == "status = 'active'"

        # Test with no filter criteria applied.
        params = {
            "target": {
                "database": "test_db",
                "table": "test_table",
                "column": "test_column",
            },
            "rule_config": {"min_value": 0, "max_value": 100},
        }
        filter_condition = extract_filter_condition(params)
        assert filter_condition is None

        # Test with empty arguments/parameters.
        filter_condition = extract_filter_condition({})
        assert filter_condition is None

    def test_merge_params(self) -> None:
        """Test merging parameters."""
        # Testing basic merge functionality.
        base_params = {
            "target": {
                "database": "test_db",
                "table": "test_table",
                "column": "test_column",
            },
            "rule_config": {
                "filter_condition": "status = 'active'",
                "min": 0,
                "max": 100,
            },
        }

        override_params: Dict[str, Dict[str, Any]] = {"rule_config": {"max": 200}}

        merged = merge_params(base_params, override_params)
        # The `target` might be empty because `convert_legacy_params` handles
        # pre-existing data already in the new format.
        assert "target" in merged
        assert "rule_config" in merged
        assert merged["rule_config"]["filter_condition"] == "status = 'active'"
        assert merged["rule_config"]["min"] == 0
        assert merged["rule_config"]["max"] == 200
        # The expected value is 200 because the `merge_params` function prioritizes
        # values from the overriding parameter set, effectively replacing any
        # corresponding values in the base parameter set.

        # Test the addition of new key-value pairs.
        override_params = {"rule_config": {"new_key": "value"}}

        merged = merge_params(base_params, override_params)
        assert merged["rule_config"]["min"] == 0
        assert merged["rule_config"]["max"] == 100
        assert merged["rule_config"]["new_key"] == "value"

        # Test coverage target information.
        override_params = {
            "target": {
                "database": "new_db",
                "table": "new_table",
                "column": "new_column",
            }
        }

        merged = merge_params(base_params, override_params)
        assert merged["target"]["database"] == "new_db"
        assert merged["target"]["table"] == "new_table"
        assert merged["target"]["column"] == "new_column"
