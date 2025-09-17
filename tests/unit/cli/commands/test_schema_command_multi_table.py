"""Unit tests for schema command multi-table functionality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from cli.app import cli_app


def _write_tmp_file(tmp_path: Path, name: str, content: str) -> str:
    file_path = tmp_path / name
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


class TestSchemaCommandMultiTable:
    def test_multi_table_rules_format_parsing(self, tmp_path: Path) -> None:
        """Test that multi-table rules format is correctly parsed."""
        runner = CliRunner()

        # Create multi-table rules file
        # Use the existing multi-table schema file
        rules_path = "test_data/multi_table_schema.json"
        # Use the new multi-table Excel file instead of CSV
        data_path = "test_data/multi_table_data.xlsx"

        result = runner.invoke(
            cli_app,
            ["schema", "--conn", data_path, "--rules", rules_path, "--output", "json"],
        )

        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["status"] == "ok"
        assert payload["rules_count"] == 21

        # Check that fields have table information
        fields = payload["fields"]
        assert len(fields) > 0
        for field in fields:
            assert "table" in field
            assert field["table"] in ["users", "products", "orders"]

    def test_multi_table_excel_sheets_detection(self, tmp_path: Path) -> None:
        """Test that Excel file sheets are correctly detected and used as tables."""
        runner = CliRunner()

        # Create a simple multi-table rules file
        multi_table_rules = {
            "users": {
                "rules": [
                    {"field": "id", "type": "integer", "required": True},
                    {"field": "name", "type": "string", "required": True},
                ]
            },
            "products": {
                "rules": [
                    {"field": "product_id", "type": "integer", "required": True},
                    {"field": "product_name", "type": "string", "required": True},
                ]
            },
        }

        rules_path = _write_tmp_file(
            tmp_path, "multi_table_rules.json", json.dumps(multi_table_rules)
        )
        data_path = "test_data/multi_table_data.xlsx"

        result = runner.invoke(
            cli_app,
            ["schema", "--conn", data_path, "--rules", rules_path, "--output", "json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "ok"

        # Check that both tables are processed
        fields = payload["fields"]
        user_fields = [f for f in fields if f.get("table") == "users"]
        product_fields = [f for f in fields if f.get("table") == "products"]

        assert len(user_fields) > 0
        assert len(product_fields) > 0

    def test_multi_table_with_table_level_options(self, tmp_path: Path) -> None:
        """Test multi-table format with table-level options like strict_mode."""
        runner = CliRunner()

        multi_table_rules = {
            "users": {
                "rules": [{"field": "id", "type": "integer", "required": True}],
                "strict_mode": True,
            },
            "products": {
                "rules": [
                    {"field": "product_name", "type": "string", "required": True}
                ],
                "case_insensitive": True,
            },
        }

        rules_path = _write_tmp_file(
            tmp_path, "multi_table_options.json", json.dumps(multi_table_rules)
        )
        data_path = "test_data/multi_table_data.xlsx"

        result = runner.invoke(
            cli_app,
            ["schema", "--conn", data_path, "--rules", rules_path, "--output", "json"],
        )

        # With strict_mode=True, extra columns will cause SCHEMA validation to fail
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["status"] == "ok"  # Overall status is ok
        assert (
            payload["summary"]["failed_rules"] == 1
        )  # One rule failed due to strict mode
        assert payload["summary"]["passed_rules"] == 3  # Three rules passed

    def test_multi_table_backward_compatibility(self, tmp_path: Path) -> None:
        """Test that single-table format still works for backward compatibility."""
        runner = CliRunner()

        # Single-table format (legacy)
        single_table_rules = {
            "rules": [
                {"field": "id", "type": "integer", "required": True},
                {"field": "name", "type": "string", "required": True},
            ]
        }

        rules_path = _write_tmp_file(
            tmp_path, "single_table.json", json.dumps(single_table_rules)
        )
        # Use only the users sheet for single table test
        data_path = "test_data/multi_table_data.xlsx"

        result = runner.invoke(
            cli_app,
            ["schema", "--conn", data_path, "--rules", rules_path, "--output", "json"],
        )

        assert result.exit_code == 0

        # Handle mixed output (warning + JSON)
        output_lines = result.output.strip().split("\n")
        json_line = None
        for line in output_lines:
            if line.strip().startswith("{"):
                json_line = line.strip()
                break

        assert json_line is not None, f"No JSON found in output: {result.output}"

        payload = json.loads(json_line)
        assert payload["status"] == "ok"
        assert payload["rules_count"] == 3

    def test_multi_table_validation_errors(self, tmp_path: Path) -> None:
        """Test validation errors for invalid multi-table format."""
        runner = CliRunner()

        # Invalid: table schema is not an object
        invalid_rules = {"users": "not_an_object"}

        rules_path = _write_tmp_file(
            tmp_path, "invalid.json", json.dumps(invalid_rules)
        )
        data_path = "test_data/multi_table_data.xlsx"

        result = runner.invoke(
            cli_app, ["schema", "--conn", data_path, "--rules", rules_path]
        )

        assert result.exit_code >= 2  # Usage error
        assert "must be an object" in result.output

    def test_multi_table_missing_rules_array(self, tmp_path: Path) -> None:
        """Test validation error when table is missing rules array."""
        runner = CliRunner()

        invalid_rules = {
            "users": {
                "strict_mode": True
                # Missing rules array
            }
        }

        rules_path = _write_tmp_file(
            tmp_path, "missing_rules.json", json.dumps(invalid_rules)
        )
        data_path = "test_data/multi_table_data.xlsx"

        result = runner.invoke(
            cli_app, ["schema", "--conn", data_path, "--rules", rules_path]
        )

        assert result.exit_code >= 2  # Usage error
        assert "must have a 'rules' array" in result.output

    def test_multi_table_invalid_table_level_options(self, tmp_path: Path) -> None:
        """Test validation error for invalid table-level options."""
        runner = CliRunner()

        invalid_rules = {
            "users": {
                "rules": [{"field": "id", "type": "integer", "required": True}],
                "strict_mode": "not_a_boolean",  # Should be boolean
            }
        }

        rules_path = _write_tmp_file(
            tmp_path, "invalid_options.json", json.dumps(invalid_rules)
        )
        data_path = "test_data/multi_table_data.xlsx"

        result = runner.invoke(
            cli_app, ["schema", "--conn", data_path, "--rules", rules_path]
        )

        assert result.exit_code >= 2  # Usage error
        assert "must be a boolean" in result.output

    def test_multi_table_output_formatting(self, tmp_path: Path) -> None:
        """Test that multi-table output is properly formatted and grouped."""
        runner = CliRunner()

        multi_table_rules = {
            "users": {
                "rules": [
                    {"field": "id", "type": "integer", "required": True},
                    {"field": "name", "type": "string", "required": True},
                ]
            },
            "products": {
                "rules": [{"field": "product_id", "type": "integer", "required": True}]
            },
        }

        rules_path = _write_tmp_file(
            tmp_path, "multi_table.json", json.dumps(multi_table_rules)
        )
        data_path = "test_data/multi_table_data.xlsx"

        # Test table output format
        result = runner.invoke(
            cli_app,
            ["schema", "--conn", data_path, "--rules", rules_path, "--output", "table"],
        )

        assert result.exit_code == 0
        output = result.output

        # Should show table headers for multi-table
        assert "📋 Table: users" in output
        assert "📋 Table: products" in output
        assert "📊 Multi-table Summary:" in output

    def test_multi_table_json_output_structure(self, tmp_path: Path) -> None:
        """Test that JSON output includes table information for multi-table."""
        runner = CliRunner()

        multi_table_rules = {
            "users": {"rules": [{"field": "id", "type": "integer", "required": True}]},
            "products": {
                "rules": [{"field": "product_name", "type": "string", "required": True}]
            },
        }

        rules_path = _write_tmp_file(
            tmp_path, "multi_table.json", json.dumps(multi_table_rules)
        )
        data_path = "test_data/multi_table_data.xlsx"

        result = runner.invoke(
            cli_app,
            ["schema", "--conn", data_path, "--rules", rules_path, "--output", "json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)

        # Check that fields have table information
        fields = payload["fields"]
        assert len(fields) >= 2

        # Find fields for each table
        user_fields = [f for f in fields if f.get("table") == "users"]
        product_fields = [f for f in fields if f.get("table") == "products"]

        assert len(user_fields) > 0
        assert len(product_fields) > 0

        # Check that each field has table info
        for field in fields:
            assert "table" in field
            assert field["table"] in ["users", "products"]

    def test_multi_table_no_table_option_required(self, tmp_path: Path) -> None:
        """Test that --table option is no longer required."""
        runner = CliRunner()

        multi_table_rules = {
            "users": {"rules": [{"field": "id", "type": "integer", "required": True}]}
        }

        rules_path = _write_tmp_file(
            tmp_path, "multi_table.json", json.dumps(multi_table_rules)
        )
        data_path = "test_data/multi_table_data.xlsx"

        # Should work without --table option
        result = runner.invoke(
            cli_app, ["schema", "--conn", data_path, "--rules", rules_path]
        )

        assert result.exit_code == 0
        # Command should execute successfully without --table option

    def test_multi_table_excel_specific_functionality(self, tmp_path: Path) -> None:
        """Test specific Excel multi-table functionality."""
        runner = CliRunner()

        # Test with all three tables from the Excel file
        multi_table_rules = {
            "users": {
                "rules": [
                    {"field": "id", "type": "integer", "required": True},
                    {"field": "name", "type": "string", "required": True},
                    {"field": "email", "type": "string", "required": True},
                ]
            },
            "products": {
                "rules": [
                    {"field": "product_id", "type": "integer", "required": True},
                    {"field": "product_name", "type": "string", "required": True},
                    {"field": "price", "type": "float", "min": 0.0},
                ]
            },
            "orders": {
                "rules": [
                    {"field": "order_id", "type": "integer", "required": True},
                    {"field": "user_id", "type": "integer", "required": True},
                    {"field": "total_amount", "type": "float", "min": 0.0},
                ]
            },
        }

        rules_path = _write_tmp_file(
            tmp_path, "excel_multi_table.json", json.dumps(multi_table_rules)
        )
        data_path = "test_data/multi_table_data.xlsx"

        result = runner.invoke(
            cli_app,
            ["schema", "--conn", data_path, "--rules", rules_path, "--output", "json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "ok"

        # Check that all three tables are processed
        fields = payload["fields"]
        table_names = set(field.get("table") for field in fields)
        assert "users" in table_names
        assert "products" in table_names
        assert "orders" in table_names

    def test_multi_table_help_text_updated(self, tmp_path: Path) -> None:
        """Test that help text reflects multi-table support."""
        runner = CliRunner()

        result = runner.invoke(cli_app, ["schema", "--help"])
        assert result.exit_code == 0

        # Should mention multi-table support
        assert "multi-table" in result.output.lower()
        # Should not mention --table option
        assert "--table" not in result.output
