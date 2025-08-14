"""
Integration tests: PostgreSQL schema drift behavior

Skips when PostgreSQL is not configured in the environment.
"""

from __future__ import annotations

from typing import cast

import pytest

from core.engine.rule_engine import RuleEngine
from shared.enums import RuleAction, RuleCategory, RuleType, SeverityLevel
from shared.enums.connection_types import ConnectionType
from shared.schema.base import RuleTarget, TargetEntity
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.rule_schema import RuleSchema
from tests.shared.utils.database_utils import (
    get_available_databases,
    get_postgresql_connection_params,
)

pytestmark = pytest.mark.asyncio


def _skip_if_pg_unavailable() -> None:
    if "postgresql" not in get_available_databases():
        pytest.skip("PostgreSQL not configured; skipping integration tests")


def _schema_rule_for(
    *,
    database: str,
    table: str,
    columns: dict[str, dict[str, str]],
    strict_mode: bool = False,
    case_insensitive: bool = False,
) -> RuleSchema:
    rule = RuleSchema(
        name="schema",
        type=RuleType.SCHEMA,
        category=RuleCategory.VALIDITY,
        severity=SeverityLevel.MEDIUM,
        action=RuleAction.LOG,
        target=RuleTarget(
            entities=[TargetEntity(database=database, table=table, column=None)],
            relationship_type="single_table",
        ),
        parameters={
            "columns": columns,
            "strict_mode": strict_mode,
            "case_insensitive": case_insensitive,
        },
    )
    rule.target.entities[0].column = None
    return rule


def _build_pg_connection() -> ConnectionSchema:
    params = get_postgresql_connection_params()
    return ConnectionSchema(
        name="pg_it_conn",
        description="PostgreSQL integration connection",
        connection_type=ConnectionType.POSTGRESQL,
        host=str(params["host"]),
        port=cast(int, params["port"]),
        db_name=str(params["database"]),
        username=str(params["username"]),
        password=str(params["password"]),
    )


class TestPostgreSQLSchemaDrift:
    async def test_existence_and_type_match(self) -> None:
        _skip_if_pg_unavailable()
        conn = _build_pg_connection()

        rule = _schema_rule_for(
            database=str(conn.db_name or "test_db"),
            table="customers",
            columns={
                "id": {"expected_type": "INTEGER"},
                "name": {"expected_type": "STRING"},
                "email": {"expected_type": "STRING"},
                "age": {"expected_type": "INTEGER"},
                "gender": {"expected_type": "INTEGER"},
                "created_at": {"expected_type": "DATETIME"},
            },
            case_insensitive=True,
        )

        engine = RuleEngine(connection=conn)
        results = await engine.execute(rules=[rule])
        res = results[0]

        assert res.status == "PASSED"
        assert res.dataset_metrics[0].total_records == 6
        assert res.dataset_metrics[0].failed_records == 0

    async def test_missing_and_type_mismatch(self) -> None:
        _skip_if_pg_unavailable()
        conn = _build_pg_connection()

        rule = _schema_rule_for(
            database=str(conn.db_name or "test_db"),
            table="customers",
            columns={
                "email": {"expected_type": "INTEGER"},  # mismatch on purpose
                "status": {"expected_type": "STRING"},  # missing
            },
        )

        engine = RuleEngine(connection=conn)
        results = await engine.execute(rules=[rule])
        res = results[0]

        assert res.status == "FAILED"
        assert res.dataset_metrics[0].total_records == 2
        assert res.dataset_metrics[0].failed_records == 2

        details = (res.execution_plan or {}).get("schema_details", {})
        field_results = {fr["column"]: fr for fr in details.get("field_results", [])}
        assert field_results["email"]["failure_code"] == "TYPE_MISMATCH"
        assert field_results["status"]["failure_code"] == "FIELD_MISSING"

    async def test_strict_mode_extras(self) -> None:
        _skip_if_pg_unavailable()
        conn = _build_pg_connection()

        rule = _schema_rule_for(
            database=str(conn.db_name or "test_db"),
            table="customers",
            columns={"id": {"expected_type": "INTEGER"}},
            strict_mode=True,
        )

        engine = RuleEngine(connection=conn)
        results = await engine.execute(rules=[rule])
        res = results[0]

        assert res.status == "FAILED"
        details = (res.execution_plan or {}).get("schema_details", {})
        extras = details.get("extras", [])
        assert len(extras) >= 1

    async def test_case_insensitive_column_matching(self) -> None:
        _skip_if_pg_unavailable()
        conn = _build_pg_connection()

        rule = _schema_rule_for(
            database=str(conn.db_name or "test_db"),
            table="customers",
            columns={"Name": {"expected_type": "STRING"}},
            case_insensitive=True,
        )

        engine = RuleEngine(connection=conn)
        results = await engine.execute(rules=[rule])
        res = results[0]

        assert res.status == "PASSED"
