"""Unit tests for schema command skeleton."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from cli.app import cli_app


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

        result = runner.invoke(cli_app, ["schema", data_path, "--rules", rules_path])
        assert result.exit_code == 0
        assert "Checking" in result.output

    def test_output_json_mode(self, tmp_path: Path) -> None:
        runner = CliRunner()
        data_path = _write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        rules_path = _write_tmp_file(tmp_path, "schema.json", json.dumps({"rules": []}))

        result = runner.invoke(
            cli_app, ["schema", data_path, "--rules", rules_path, "--output", "json"]
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "ok"
        assert payload["rules_count"] == 0

    def test_fail_on_error_sets_exit_code_1(self, tmp_path: Path) -> None:
        runner = CliRunner()
        data_path = _write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        rules_path = _write_tmp_file(tmp_path, "schema.json", json.dumps({"rules": []}))

        result = runner.invoke(
            cli_app,
            [
                "schema",
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
            cli_app, ["schema", data_path, "--rules", bad_rules_path]
        )

        # Click usage error exit code is >= 2
        assert result.exit_code >= 2
        assert "Invalid JSON" in result.output
