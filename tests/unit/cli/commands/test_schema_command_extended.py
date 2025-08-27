from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from click.testing import CliRunner

from cli.app import cli_app
from shared.enums import (
    ConnectionType,
    RuleAction,
    RuleCategory,
    RuleType,
    SeverityLevel,
)
from shared.schema.base import RuleTarget, TargetEntity
from shared.schema.rule_schema import RuleSchema
from tests.shared.builders import test_builders


def _write_tmp_file(tmp_path: Path, name: str, content: str) -> str:
    file_path = tmp_path / name
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


def _make_rule(
    *,
    name: str,
    rule_type: RuleType,
    column: str | None,
    parameters: Dict[str, Any],
    description: str | None = None,
) -> RuleSchema:
    target = RuleTarget(
        entities=[
            TargetEntity(
                database="", table="", column=column, connection_id=None, alias=None
            )
        ],
        relationship_type="single_table",
    )
    return RuleSchema(
        name=name,
        description=description,
        type=rule_type,
        target=target,
        parameters=parameters,
        cross_db_config=None,
        threshold=0.0,
        category=(
            RuleCategory.VALIDITY
            if rule_type in {RuleType.SCHEMA, RuleType.RANGE, RuleType.ENUM}
            else RuleCategory.COMPLETENESS
        ),
        severity=SeverityLevel.MEDIUM,
        action=RuleAction.ALERT,
        is_active=True,
        tags=[],
        template_id=None,
        validation_error=None,
    )


class TestSchemaDecompositionAndMapping:
    def test_map_type_names_are_case_insensitive_and_validated(
        self, tmp_path: Path
    ) -> None:
        from cli.commands.schema import _map_type_name_to_datatype

        assert _map_type_name_to_datatype("STRING").value == "STRING"
        assert _map_type_name_to_datatype("integer").value == "INTEGER"
        assert _map_type_name_to_datatype("DateTime").value == "DATETIME"

        with pytest.raises(Exception):
            _map_type_name_to_datatype("number")

    def test_decompose_to_atomic_rules_structure(self, tmp_path: Path) -> None:
        from cli.commands.schema import _decompose_schema_payload

        payload = {
            "strict_mode": True,
            "case_insensitive": True,
            "rules": [
                {"field": "id", "type": "integer", "required": True},
                {"field": "age", "min": 0, "max": 100},
                {"field": "status", "enum": ["A", "B"]},
            ],
        }
        # Create a mock ConnectionSchema for testing
        mock_source_config = (
            test_builders.TestDataBuilder.connection()
            .with_type(ConnectionType.CSV)
            .with_database("test_db")
            .with_available_tables("test_table")
            .with_parameters({})
            .build()
        )
        rules = _decompose_schema_payload(payload, mock_source_config)

        # First rule should be SCHEMA when any columns declared
        assert rules[0].type == RuleType.SCHEMA
        schema_params = rules[0].parameters or {}
        assert schema_params["columns"]["id"]["expected_type"] == "INTEGER"
        assert schema_params["strict_mode"] is True
        assert schema_params["case_insensitive"] is True

        types = [r.type for r in rules]
        # NOT_NULL created for required
        assert RuleType.NOT_NULL in types
        # RANGE created for min/max
        assert RuleType.RANGE in types
        # ENUM created when enum declared
        assert RuleType.ENUM in types


class TestSchemaPrioritizationAndOutputs:
    def test_prioritization_skip_map(self) -> None:
        from cli.commands.schema import _build_prioritized_atomic_status

        # Build atomic rules manually
        schema = _make_rule(
            name="schema",
            rule_type=RuleType.SCHEMA,
            column=None,
            parameters={
                "columns": {
                    "id": {"expected_type": "INTEGER"},
                    "email": {"expected_type": "STRING"},
                    "age": {"expected_type": "INTEGER"},
                }
            },
        )
        not_null_email = _make_rule(
            name="not_null_email",
            rule_type=RuleType.NOT_NULL,
            column="email",
            parameters={},
        )
        range_age = _make_rule(
            name="range_age",
            rule_type=RuleType.RANGE,
            column="age",
            parameters={"min_value": 0, "max_value": 120},
        )

        atomic_rules = [schema, not_null_email, range_age]

        # Simulate SCHEMA execution details
        schema_result = {
            "execution_plan": {
                "schema_details": {
                    "field_results": [
                        {"column": "email", "failure_code": "TYPE_MISMATCH"},
                        {"column": "age", "failure_code": "FIELD_MISSING"},
                        {"column": "id", "failure_code": "NONE"},
                    ]
                }
            }
        }

        skip_map = _build_prioritized_atomic_status(
            schema_result=schema_result, atomic_rules=atomic_rules
        )

        # email dependent rules should be skipped for TYPE_MISMATCH
        assert skip_map[str(not_null_email.id)]["status"] == "SKIPPED"
        assert skip_map[str(not_null_email.id)]["skip_reason"] == "TYPE_MISMATCH"
        # age dependent rules should be skipped for FIELD_MISSING
        assert skip_map[str(range_age.id)]["status"] == "SKIPPED"
        assert skip_map[str(range_age.id)]["skip_reason"] == "FIELD_MISSING"

    def test_json_output_aggregation_and_skip_semantics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Prepare known atomic rules and patch decomposition to return them
        schema = _make_rule(
            name="schema",
            rule_type=RuleType.SCHEMA,
            column=None,
            parameters={
                "columns": {
                    "email": {"expected_type": "STRING"},
                    "age": {"expected_type": "INTEGER"},
                }
            },
        )
        not_null_email = _make_rule(
            name="not_null_email",
            rule_type=RuleType.NOT_NULL,
            column="email",
            parameters={},
        )
        range_age = _make_rule(
            name="range_age",
            rule_type=RuleType.RANGE,
            column="age",
            parameters={"min_value": 0, "max_value": 150},
        )
        atomic_rules = [schema, not_null_email, range_age]

        # Patch decomposition
        monkeypatch.setattr(
            "cli.commands.schema._decompose_schema_payload",
            lambda payload, source_config: atomic_rules,
        )

        # Build SCHEMA and dependent rule results. Dependent rules are PASSED in raw
        # and should be overridden to SKIPPED in JSON when schema marks issues.
        schema_result = {
            "rule_id": str(schema.id),
            "status": "FAILED",
            "dataset_metrics": [
                {"entity_name": "x", "total_records": 2, "failed_records": 2}
            ],
            "execution_plan": {
                "schema_details": {
                    "field_results": [
                        {
                            "column": "age",
                            "existence": "FAILED",
                            "type": "SKIPPED",
                            "failure_code": "FIELD_MISSING",
                        },
                        {
                            "column": "email",
                            "existence": "PASSED",
                            "type": "FAILED",
                            "failure_code": "TYPE_MISMATCH",
                        },
                    ],
                    "extras": [],
                }
            },
        }
        not_null_email_result = {
            "rule_id": str(not_null_email.id),
            "status": "PASSED",
            "dataset_metrics": [
                {"entity_name": "x", "total_records": 10, "failed_records": 0}
            ],
        }
        range_age_result = {
            "rule_id": str(range_age.id),
            "status": "PASSED",
            "dataset_metrics": [
                {"entity_name": "x", "total_records": 10, "failed_records": 0}
            ],
        }

        # Patch DataValidator.validate to return our results
        class DummyValidator:
            def __init__(self, source_config, rules, core_config, cli_config):
                # Accept all required parameters but don't use them
                pass

            async def validate(self) -> List[Dict[str, Any]]:  # type: ignore[override]
                return [schema_result, not_null_email_result, range_age_result]

        monkeypatch.setattr("cli.commands.schema.DataValidator", DummyValidator)

        # Prepare inputs and run CLI in JSON output mode
        runner = CliRunner()
        data_path = _write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        rules_path = _write_tmp_file(
            tmp_path,
            "schema.json",
            json.dumps(
                {
                    "rules": [
                        {"field": "email", "type": "string"},
                        {"field": "age", "type": "integer"},
                    ]
                }
            ),
        )

        result = runner.invoke(
            cli_app,
            ["schema", "--conn", data_path, "--rules", rules_path, "--output", "json"],
        )

        assert result.exit_code == 1  # schema failed -> non-zero
        payload = json.loads(result.output)
        assert payload["status"] == "ok"
        assert payload["rules_count"] == len(atomic_rules)
        # Results should contain SKIPPED overrides for dependent rules
        results_map = {r["rule_id"]: r for r in payload["results"]}
        assert results_map[str(not_null_email.id)]["status"] == "SKIPPED"
        assert results_map[str(not_null_email.id)]["skip_reason"] == "TYPE_MISMATCH"
        assert results_map[str(range_age.id)]["status"] == "SKIPPED"
        assert results_map[str(range_age.id)]["skip_reason"] == "FIELD_MISSING"

        # Fields aggregate should include existence/type and dependent checks
        fields = {f["column"]: f for f in payload["fields"]}
        assert fields["age"]["checks"]["existence"]["status"] == "FAILED"
        assert fields["email"]["checks"]["type"]["status"] == "FAILED"
        assert fields["email"]["checks"]["not_null"]["status"] == "SKIPPED"
        assert fields["age"]["checks"]["range"]["status"] == "SKIPPED"

    def test_table_output_grouping_and_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Prepare known atomic rules and patch decomposition to return them
        schema = _make_rule(
            name="schema",
            rule_type=RuleType.SCHEMA,
            column=None,
            parameters={
                "columns": {
                    "email": {"expected_type": "STRING"},
                    "age": {"expected_type": "INTEGER"},
                }
            },
        )
        not_null_email = _make_rule(
            name="not_null_email",
            rule_type=RuleType.NOT_NULL,
            column="email",
            parameters={},
        )
        range_age = _make_rule(
            name="range_age",
            rule_type=RuleType.RANGE,
            column="age",
            parameters={"min_value": 0, "max_value": 150},
        )
        atomic_rules = [schema, not_null_email, range_age]

        monkeypatch.setattr(
            "cli.commands.schema._decompose_schema_payload",
            lambda payload, source_config: atomic_rules,
        )

        schema_result = {
            "rule_id": str(schema.id),
            "status": "FAILED",
            "dataset_metrics": [
                {"entity_name": "x", "total_records": 2, "failed_records": 2}
            ],
            "execution_plan": {
                "schema_details": {
                    "field_results": [
                        {
                            "column": "age",
                            "existence": "FAILED",
                            "type": "SKIPPED",
                            "failure_code": "FIELD_MISSING",
                        },
                        {
                            "column": "email",
                            "existence": "PASSED",
                            "type": "FAILED",
                            "failure_code": "TYPE_MISMATCH",
                        },
                    ],
                    "extras": [],
                }
            },
        }
        # Dependent rule raw statuses set to PASSED; should be skipped for display grouping
        not_null_email_result = {
            "rule_id": str(not_null_email.id),
            "status": "SKIPPED",
            "dataset_metrics": [
                {"entity_name": "x", "total_records": 10, "failed_records": 0}
            ],
            "skip_reason": "TYPE_MISMATCH",
        }
        range_age_result = {
            "rule_id": str(range_age.id),
            "status": "SKIPPED",
            "dataset_metrics": [
                {"entity_name": "x", "total_records": 10, "failed_records": 0}
            ],
            "skip_reason": "FIELD_MISSING",
        }

        class DummyValidator:
            def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
                pass

            async def validate(self) -> List[Dict[str, Any]]:  # type: ignore[override]
                return [schema_result, not_null_email_result, range_age_result]

        monkeypatch.setattr("cli.commands.schema.DataValidator", DummyValidator)

        runner = CliRunner()
        data_path = _write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        rules_path = _write_tmp_file(
            tmp_path,
            "schema.json",
            json.dumps(
                {
                    "rules": [
                        {"field": "email", "type": "string"},
                        {"field": "age", "type": "integer"},
                    ]
                }
            ),
        )

        result = runner.invoke(
            cli_app, ["schema", "--conn", data_path, "--rules", rules_path]
        )
        assert result.exit_code == 1
        output = result.output

        # Should show concise messages per column with skip semantics
        assert "✗ age: missing (skipped dependent checks)" in output
        assert "✗ email: type mismatch (skipped dependent checks)" in output
        # Should not render separate dependent issues since they are skipped
        assert "not_null" not in output
        assert "range" not in output


class TestSchemaValidationErrorsExtended:
    # def test_reject_tables_top_level(self, tmp_path: Path) -> None:
    #     runner = CliRunner()
    #     data_path = _write_tmp_file(tmp_path, "data.csv", "id\n1\n")
    #     rules_path = _write_tmp_file(
    #         tmp_path,
    #         "schema.json",
    #         json.dumps({"tables": {"users": []}, "rules": []}),
    #     )

    #     result = runner.invoke(cli_app, ["schema", "--conn", data_path, "--rules", rules_path])
    #     assert result.exit_code >= 2
    #     assert "not supported in v1" in result.output

    def test_enum_must_be_non_empty_array(self, tmp_path: Path) -> None:
        runner = CliRunner()
        data_path = _write_tmp_file(tmp_path, "data.csv", "id\n1\n")
        rules_path = _write_tmp_file(
            tmp_path,
            "schema.json",
            json.dumps({"rules": [{"field": "status", "enum": []}]}),
        )

        result = runner.invoke(
            cli_app, ["schema", "--conn", data_path, "--rules", rules_path]
        )
        assert result.exit_code >= 2
        assert "enum' must be a non-empty" in result.output
