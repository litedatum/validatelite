from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from click.testing import CliRunner

from cli.app import cli_app
from shared.enums import RuleType
from shared.schema.rule_schema import RuleSchema
from tests.shared.builders import test_builders


def _write_tmp_file(tmp_path: Path, name: str, content: str) -> str:
    file_path = tmp_path / name
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


def _schema_rule_with(columns: Dict[str, Dict[str, str]]) -> RuleSchema:
    return (
        test_builders.TestDataBuilder.rule()
        .with_name("schema")
        .with_type(RuleType.SCHEMA)
        .with_target("", "", "id")
        .with_parameter("columns", columns)
        .with_parameter("strict_mode", True)
        .build()
    )


class TestSchemaJsonExtrasAndSummary:
    def test_json_includes_schema_extras_and_summary_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Decomposition yields one SCHEMA rule for columns id/email
        schema_rule = _schema_rule_with(
            {
                "id": {"expected_type": "INTEGER"},
                "email": {"expected_type": "STRING"},
            }
        )
        monkeypatch.setattr(
            "cli.commands.schema._decompose_schema_payload",
            lambda payload, source_config: ([schema_rule], []),
        )

        # Results: SCHEMA failed with 1 type mismatch, 0 existence failures, extras present
        schema_result = {
            "rule_id": str(schema_rule.id),
            "status": "FAILED",
            "dataset_metrics": [
                {"entity_name": "t", "total_records": 2, "failed_records": 1}
            ],
            "execution_plan": {
                "schema_details": {
                    "field_results": [
                        {
                            "column": "id",
                            "existence": "PASSED",
                            "type": "PASSED",
                            "failure_code": "NONE",
                        },
                        {
                            "column": "email",
                            "existence": "PASSED",
                            "type": "FAILED",
                            "failure_code": "TYPE_MISMATCH",
                        },
                    ],
                    "extras": ["zzz_extra", "aaa_extra"],
                }
            },
        }

        class DummyValidator:
            def __init__(
                self, source_config: Any, rules: Any, core_config: Any, cli_config: Any
            ) -> None:
                # Accept all required parameters but don't use them
                pass

            async def validate(self) -> List[Dict[str, Any]]:  # type: ignore[override]
                return [schema_result]

        monkeypatch.setattr("cli.commands.schema.DataValidator", DummyValidator)

        runner = CliRunner()
        data_path = _write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        rules_path = _write_tmp_file(
            tmp_path,
            "schema.json",
            json.dumps(
                {
                    "rules": [
                        {"field": "id", "type": "integer"},
                        {"field": "email", "type": "string"},
                    ]
                }
            ),
        )

        result = runner.invoke(
            cli_app,
            ["schema", "--conn", data_path, "--rules", rules_path, "--output", "json"],
        )
        assert result.exit_code == 1

        # Extract JSON part from output (skip warning messages)
        output_lines = result.output.strip().split("\n")
        json_line = None
        for line in output_lines:
            if line.strip().startswith("{"):
                json_line = line.strip()
                break

        if not json_line:
            raise ValueError("No JSON output found in result")

        payload = json.loads(json_line)

        # schema_extras must present, sorted by CLI before emission
        assert payload.get("schema_extras") == ["aaa_extra", "zzz_extra"]
        # summary counts
        assert payload["summary"]["total_rules"] == 1
        assert payload["summary"]["failed_rules"] == 1
        assert payload["summary"]["skipped_rules"] >= 0
        assert payload["summary"]["total_failed_records"] >= 1

    def test_table_output_does_not_emit_schema_extras_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        schema_rule = _schema_rule_with({"id": {"expected_type": "INTEGER"}})
        monkeypatch.setattr(
            "cli.commands.schema._decompose_schema_payload",
            lambda payload, source_config: ([schema_rule], []),
        )

        schema_result = {
            "rule_id": str(schema_rule.id),
            "status": "PASSED",
            "dataset_metrics": [
                {"entity_name": "t", "total_records": 1, "failed_records": 0}
            ],
            "execution_plan": {
                "schema_details": {"field_results": [], "extras": ["x"]}
            },
        }

        class DummyValidator:
            def __init__(
                self, source_config: Any, rules: Any, core_config: Any, cli_config: Any
            ) -> None:
                # Accept all required parameters but don't use them
                pass

            async def validate(self) -> List[Dict[str, Any]]:  # type: ignore[override]
                return [schema_result]

        monkeypatch.setattr("cli.commands.schema.DataValidator", DummyValidator)

        runner = CliRunner()
        data_path = _write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        rules_path = _write_tmp_file(
            tmp_path,
            "schema.json",
            json.dumps({"rules": [{"field": "id", "type": "integer"}]}),
        )
        result = runner.invoke(
            cli_app, ["schema", "--conn", data_path, "--rules", rules_path]
        )
        assert result.exit_code == 0
        # Plain text output should not dump JSON key name
        assert "schema_extras" not in result.output
