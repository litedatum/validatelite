"""Unit tests for schema command skeleton."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from cli.app import cli_app
from cli.core.data_validator import ExecutionResultSchema
from shared.enums.connection_types import ConnectionType


def _write_tmp_file(tmp_path: Path, name: str, content: str) -> str:
    file_path = tmp_path / name
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


class TestSchemaCommandSkeleton:
    def test_schema_command_help_registered(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_app, ["--help"])
        assert result.exit_code == 0
        assert "schema" in result.output

    def test_schema_requires_source_and_rules(self, tmp_path: Path) -> None:
        runner = CliRunner()

        # Missing args -> Click usage error (exit code >= 2)
        result = runner.invoke(cli_app, ["schema"])
        assert result.exit_code >= 2

        # Provide a minimal CSV and rules file
        data_path = _write_tmp_file(tmp_path, "sample.csv", "id\n1\n")
        rules_obj: dict[str, list[dict[str, Any]]] = {"rules": []}
        rules_path = _write_tmp_file(tmp_path, "schema.json", json.dumps(rules_obj))

        result = runner.invoke(
            cli_app, ["schema", "--conn", data_path, "--rules", rules_path]
        )
        assert result.exit_code == 0
        assert "Checking" in result.output

    def test_output_json_mode(self, tmp_path: Path) -> None:
        runner = CliRunner()
        data_path = _write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        rules_path = _write_tmp_file(
            tmp_path, "schema.json", json.dumps({"user": {"rules": []}})
        )

        result = runner.invoke(
            cli_app,
            ["schema", "--conn", data_path, "--rules", rules_path, "--output", "json"],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "ok"
        assert payload["rules_count"] == 0

    def test_output_json_declared_columns_always_listed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Patch decomposition to include a SCHEMA rule that declares a column not in results
        from shared.enums import RuleType
        from shared.schema.rule_schema import RuleSchema
        from tests.shared.builders import test_builders

        schema_rule: RuleSchema = (
            test_builders.TestDataBuilder.rule()
            .with_name("schema")
            .with_type(RuleType.SCHEMA)
            .with_target("", "", "id")
            .with_parameter("columns", {"id": {"expected_type": "INTEGER"}})
            .build()
        )

        # Create a mock ConnectionSchema for testing
        mock_source_config = (
            test_builders.TestDataBuilder.connection()
            .with_type(ConnectionType.CSV)
            .with_database("test_db")
            .with_available_tables("test_table")
            .with_parameters({})
            .build()
        )

        monkeypatch.setattr(
            "cli.commands.schema._decompose_schema_payload",
            lambda payload, source_config: ([schema_rule], []),
        )

        class DummyValidator:
            def __init__(
                self, source_config: Any, rules: Any, core_config: Any, cli_config: Any
            ) -> None:
                # Accept constructor arguments but ignore them
                pass

            async def validate(self) -> list[ExecutionResultSchema]:
                # Return no results to simulate missing schema details
                return []

        monkeypatch.setattr("cli.commands.schema.DataValidator", DummyValidator)

        runner = CliRunner()
        data_path = _write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        rules_path = _write_tmp_file(
            tmp_path,
            "schema.json",
            json.dumps({"data": {"rules": [{"field": "id", "type": "integer"}]}}),
        )

        result = runner.invoke(
            cli_app,
            ["schema", "--conn", data_path, "--rules", rules_path, "--output", "json"],
        )
        # No failures but explicit -- in this setup lack of results implies exit 0
        assert result.exit_code == 0
        payload = json.loads(result.output)
        # Declared column should still appear with UNKNOWN statuses
        fields = {f["column"]: f for f in payload["fields"]}
        assert "id" in fields
        assert fields["id"]["checks"]["existence"]["status"] in {
            "UNKNOWN",
            "PASSED",
            "FAILED",
        }

    def test_fail_on_error_sets_exit_code_1(self, tmp_path: Path) -> None:
        runner = CliRunner()
        data_path = _write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        rules_path = _write_tmp_file(tmp_path, "schema.json", json.dumps({"rules": []}))

        result = runner.invoke(
            cli_app,
            [
                "schema",
                "--conn",
                data_path,
                "--rules",
                rules_path,
                "--fail-on-error",
            ],
        )
        assert result.exit_code == 1

    def test_invalid_rules_json_yields_usage_error(self, tmp_path: Path) -> None:
        runner = CliRunner()
        data_path = _write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        # invalid content
        bad_rules_path = _write_tmp_file(tmp_path, "bad.json", "{invalid json}")

        result = runner.invoke(
            cli_app, ["schema", "--conn", data_path, "--rules", bad_rules_path]
        )

        # Click usage error exit code is >= 2
        assert result.exit_code >= 2
        assert "Invalid JSON" in result.output


class TestSchemaCommandValidation:
    def _write_tmp_file(self, tmp_path: Path, name: str, content: str) -> str:
        file_path = tmp_path / name
        file_path.write_text(content, encoding="utf-8")
        return str(file_path)

    def test_warn_on_top_level_table_ignored(self, tmp_path: Path) -> None:
        runner = CliRunner()
        data_path = self._write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        rules = {
            "users": {
                "rules": [
                    {"field": "id", "type": "integer", "required": True},
                ]
            }
        }
        rules_path = self._write_tmp_file(tmp_path, "schema.json", json.dumps(rules))

        result = runner.invoke(
            cli_app,
            ["schema", "--conn", data_path, "--rules", rules_path, "--output", "json"],
        )
        # exit code from skeleton remains success
        assert result.exit_code == 0
        # Since multi-table has been supported,so no warning emitted to stderr
        # assert "table' is ignored" in (result.stderr or "")

    def test_rules_must_be_array(self, tmp_path: Path) -> None:
        runner = CliRunner()
        data_path = self._write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        rules_path = self._write_tmp_file(tmp_path, "schema.json", json.dumps({}))

        result = runner.invoke(
            cli_app, ["schema", "--conn", data_path, "--rules", rules_path]
        )
        assert result.exit_code >= 2
        assert "must have a 'rules' array" in result.output

    def test_rules_item_requires_field(self, tmp_path: Path) -> None:
        runner = CliRunner()
        data_path = self._write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        bad = {"rules": [{"type": "integer"}]}
        rules_path = self._write_tmp_file(tmp_path, "schema.json", json.dumps(bad))

        result = runner.invoke(
            cli_app, ["schema", "--conn", data_path, "--rules", rules_path]
        )
        assert result.exit_code >= 2
        assert "field must be a non-empty string" in result.output

    def test_type_must_be_supported_string(self, tmp_path: Path) -> None:
        runner = CliRunner()
        data_path = self._write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        bad = {"rules": [{"field": "id", "type": "number"}]}
        rules_path = self._write_tmp_file(tmp_path, "schema.json", json.dumps(bad))

        result = runner.invoke(
            cli_app, ["schema", "--conn", data_path, "--rules", rules_path]
        )
        assert result.exit_code >= 2
        assert "type 'number' is not supported" in result.output

    def test_required_must_be_boolean(self, tmp_path: Path) -> None:
        runner = CliRunner()
        data_path = self._write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        bad = {"rules": [{"field": "id", "required": "yes"}]}
        rules_path = self._write_tmp_file(tmp_path, "schema.json", json.dumps(bad))

        result = runner.invoke(
            cli_app, ["schema", "--conn", data_path, "--rules", rules_path]
        )
        assert result.exit_code >= 2
        assert "required must be a boolean" in result.output

    def test_enum_must_be_array(self, tmp_path: Path) -> None:
        runner = CliRunner()
        data_path = self._write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        bad = {"rules": [{"field": "flag", "enum": "01"}]}
        rules_path = self._write_tmp_file(tmp_path, "schema.json", json.dumps(bad))

        result = runner.invoke(
            cli_app, ["schema", "--conn", data_path, "--rules", rules_path]
        )
        assert result.exit_code >= 2
        assert "enum must be an array" in result.output

    def test_min_max_must_be_numeric(self, tmp_path: Path) -> None:
        runner = CliRunner()
        data_path = self._write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        bad = {"rules": [{"field": "age", "type": "integer", "min": "0"}]}
        rules_path = self._write_tmp_file(tmp_path, "schema.json", json.dumps(bad))

        result = runner.invoke(
            cli_app, ["schema", "--conn", data_path, "--rules", rules_path]
        )
        assert result.exit_code >= 2
        assert "min must be numeric" in result.output

    def test_desired_type_validation_accepts_valid_format(self, tmp_path: Path) -> None:
        """Test that desired_type field accepts valid type definitions."""
        runner = CliRunner()
        data_path = self._write_tmp_file(tmp_path, "data.csv", "id,name,amount\n1,test,12.34\n")
        
        # Test valid desired_type formats
        valid_rules = {
            "rules": [
                {"field": "id",  "desired_type": "integer"},
                {"field": "name", "desired_type": "string(50)"},
                {"field": "amount",  "desired_type": "float(10,2)"},
            ]
        }
        rules_path = self._write_tmp_file(tmp_path, "schema.json", json.dumps(valid_rules))

        result = runner.invoke(
            cli_app, ["schema", "--conn", data_path, "--rules", rules_path]
        )
        # Debug: print the result if it failed
        if result.exit_code != 0:
            print(f"Exit code: {result.exit_code}")
            print(f"Output: {result.output}")
            print(f"Exception: {result.exception}")
        # Should not have validation errors from desired_type parsing
        assert result.exit_code == 0

    def test_desired_type_validation_rejects_invalid_format(self, tmp_path: Path) -> None:
        """Test that desired_type field rejects invalid type definitions."""
        runner = CliRunner()
        data_path = self._write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        
        # Test invalid desired_type format
        invalid_rules = {
            "rules": [
                {"field": "id", "type": "string", "desired_type": "invalid_type"},
            ]
        }
        rules_path = self._write_tmp_file(tmp_path, "schema.json", json.dumps(invalid_rules))

        result = runner.invoke(
            cli_app, ["schema", "--conn", data_path, "--rules", rules_path]
        )
        assert result.exit_code >= 2
        assert "desired_type 'invalid_type' is not supported" in result.output
