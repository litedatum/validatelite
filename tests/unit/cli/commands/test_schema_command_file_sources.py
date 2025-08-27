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
        .with_target("main", "data", "id")
        .with_parameter("columns", columns)
        .build()
    )


class TestSchemaCommandForFileSources:
    def test_csv_excel_to_sqlite_type_implications(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Declare DATE/DATETIME expectations; SQLite columns will be TEXT post-conversion
        schema_rule = _schema_rule_with(
            {"reg_date": {"expected_type": "DATE"}, "ts": {"expected_type": "DATETIME"}}
        )
        monkeypatch.setattr(
            "cli.commands.schema._decompose_schema_payload",
            lambda payload, source_config: [schema_rule],
        )

        # Build SCHEMA result indicating SQLite TEXT types cause TYPE_MISMATCH
        schema_result = {
            "rule_id": str(schema_rule.id),
            "status": "FAILED",
            "dataset_metrics": [
                {"entity_name": "main.data", "total_records": 2, "failed_records": 2}
            ],
            "execution_plan": {
                "schema_details": {
                    "field_results": [
                        {
                            "column": "reg_date",
                            "existence": "PASSED",
                            "type": "FAILED",
                            "failure_code": "TYPE_MISMATCH",
                        },
                        {
                            "column": "ts",
                            "existence": "PASSED",
                            "type": "FAILED",
                            "failure_code": "TYPE_MISMATCH",
                        },
                    ],
                    "extras": [],
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

        # Prepare CSV file path as source (will be converted to SQLite inside command)
        data_path = _write_tmp_file(
            tmp_path,
            "data.csv",
            "reg_date,ts\n2023-01-01,2023-01-01T10:00:00Z\n2023-01-02,2023-01-02T11:00:00Z\n",
        )
        rules_path = _write_tmp_file(
            tmp_path,
            "schema.json",
            json.dumps(
                {
                    "rules": [
                        {"field": "reg_date", "type": "date"},
                        {"field": "ts", "type": "datetime"},
                    ]
                }
            ),
        )

        runner = CliRunner()
        result = runner.invoke(
            cli_app,
            ["schema", "--conn", data_path, "--rules", rules_path, "--output", "json"],
        )

        assert result.exit_code == 1
        payload = json.loads(result.output)

        # The JSON `fields` section should reflect type mismatches from SQLite TEXT
        fields = {f["column"]: f for f in payload["fields"]}
        assert fields["reg_date"]["checks"]["type"]["status"] == "FAILED"
        assert fields["ts"]["checks"]["type"]["status"] == "FAILED"
