from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from core.executors.schema_executor import SchemaExecutor
from shared.enums import ConnectionType, RuleType
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.rule_schema import RuleSchema
from tests.shared.builders.test_builders import TestDataBuilder


@pytest.fixture
def mock_connection() -> ConnectionSchema:
    return (
        TestDataBuilder.connection()
        .with_type(ConnectionType.SQLITE)
        .with_file_path("test.db")
        .build()
    )


def build_schema_rule(columns: dict) -> RuleSchema:
    builder = TestDataBuilder.rule()
    rule = (
        builder.with_name("schema_sqlite")
        .with_target("main", "data", "__table_level__")
        .with_type(RuleType.SCHEMA)
        .with_parameter("columns", columns)
        .build()
    )
    # Table-level rule
    rule.target.entities[0].column = None
    return rule


@pytest.mark.asyncio
async def test_sqlite_text_maps_to_string(mock_connection: ConnectionSchema) -> None:
    # TEXT should satisfy expected STRING
    rule = build_schema_rule({"name": {"expected_type": "STRING"}})

    executor = SchemaExecutor(mock_connection, test_mode=True)
    sqlite_cols = [{"name": "name", "type": "TEXT"}]

    with patch.object(executor, "get_engine") as mock_get_engine, patch(
        "shared.database.query_executor.QueryExecutor"
    ) as mock_qe_class:
        mock_engine = AsyncMock()
        mock_get_engine.return_value = mock_engine
        mock_qe = AsyncMock()
        mock_qe.get_column_list.return_value = sqlite_cols
        mock_qe_class.return_value = mock_qe

        result = await executor.execute_rule(rule)

    assert result.status == "PASSED"


@pytest.mark.asyncio
async def test_sqlite_integer_and_real_type_mapping(
    mock_connection: ConnectionSchema,
) -> None:
    # INTEGER should match INTEGER, REAL should map to FLOAT and mismatch if expecting INTEGER
    rule = build_schema_rule(
        {
            "id": {"expected_type": "INTEGER"},
            "value": {"expected_type": "INTEGER"},  # should mismatch (REAL)
        }
    )

    executor = SchemaExecutor(mock_connection, test_mode=True)
    sqlite_cols = [
        {"name": "id", "type": "INTEGER"},
        {"name": "value", "type": "REAL"},
    ]

    with patch.object(executor, "get_engine") as mock_get_engine, patch(
        "shared.database.query_executor.QueryExecutor"
    ) as mock_qe_class:
        mock_engine = AsyncMock()
        mock_get_engine.return_value = mock_engine
        mock_qe = AsyncMock()
        mock_qe.get_column_list.return_value = sqlite_cols
        mock_qe_class.return_value = mock_qe

        result = await executor.execute_rule(rule)

    assert result.status == "FAILED"
    # total declared = 2, one mismatch
    assert result.dataset_metrics[0].total_records == 2
    assert result.dataset_metrics[0].failed_records == 1


@pytest.mark.asyncio
async def test_sqlite_dates_are_text_unless_explicit_cast(
    mock_connection: ConnectionSchema,
) -> None:
    # In CSV/Excel→SQLite flow, dates are usually TEXT unless explicitly cast.
    # Expect mismatch when declaring DATE/DATETIME on TEXT columns.
    rule = build_schema_rule(
        {
            "reg_date": {"expected_type": "DATE"},
            "ts": {"expected_type": "DATETIME"},
        }
    )

    executor = SchemaExecutor(mock_connection, test_mode=True)
    sqlite_cols = [
        {"name": "reg_date", "type": "TEXT"},
        {"name": "ts", "type": "TEXT"},
    ]

    with patch.object(executor, "get_engine") as mock_get_engine, patch(
        "shared.database.query_executor.QueryExecutor"
    ) as mock_qe_class:
        mock_engine = AsyncMock()
        mock_get_engine.return_value = mock_engine
        mock_qe = AsyncMock()
        mock_qe.get_column_list.return_value = sqlite_cols
        mock_qe_class.return_value = mock_qe

        result = await executor.execute_rule(rule)

    assert result.status == "FAILED"
    assert result.dataset_metrics[0].total_records == 2
    assert result.dataset_metrics[0].failed_records == 2
