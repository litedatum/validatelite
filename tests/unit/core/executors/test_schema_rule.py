from unittest.mock import AsyncMock, patch

import pytest

from core.executors.schema_executor import SchemaExecutor
from shared.enums import RuleType
from shared.exceptions.exception_system import RuleExecutionError
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.rule_schema import RuleSchema
from tests.shared.builders.test_builders import TestDataBuilder


@pytest.fixture
def mock_connection() -> ConnectionSchema:
    # Use default MySQL-like connection to avoid sqlite file path requirements
    return TestDataBuilder.connection().build()


def build_schema_rule(
    columns: dict, strict_mode: bool = False, case_insensitive: bool = False
) -> RuleSchema:
    builder = TestDataBuilder.rule()
    rule = (
        builder.with_name("schema_users")
        .with_target("sales", "users", "id")
        .with_type(RuleType.SCHEMA)
        .with_parameter("columns", columns)
        .with_parameter("strict_mode", strict_mode)
        .with_parameter("case_insensitive", case_insensitive)
        .build()
    )
    # Make it table-level by clearing column
    rule.target.entities[0].column = None
    return rule


@pytest.mark.asyncio
async def test_schema_rule_pass(mock_connection: ConnectionSchema) -> None:
    rule = build_schema_rule(
        {
            "id": {"expected_type": "INTEGER"},
            "email": {"expected_type": "STRING"},
        }
    )

    executor = SchemaExecutor(mock_connection, test_mode=True)

    # Mock column list to match expected types
    columns = [
        {"name": "id", "type": "INTEGER"},
        {"name": "email", "type": "VARCHAR(255)"},
        {"name": "created_at", "type": "TEXT"},
    ]

    with patch.object(executor, "get_engine") as mock_get_engine, patch(
        "shared.database.query_executor.QueryExecutor"
    ) as mock_qe_class:
        mock_engine = AsyncMock()
        mock_get_engine.return_value = mock_engine
        mock_qe = AsyncMock()
        mock_qe.get_column_list.return_value = columns
        mock_qe_class.return_value = mock_qe

        result = await executor.execute_rule(rule)

    assert result.status == "PASSED"
    assert result.dataset_metrics[0].total_records == 2
    assert result.dataset_metrics[0].failed_records == 0


@pytest.mark.asyncio
async def test_schema_rule_missing_and_type_mismatch(
    mock_connection: ConnectionSchema,
) -> None:
    rule = build_schema_rule(
        {
            "id": {"expected_type": "INTEGER"},
            "email": {"expected_type": "STRING"},
            "created_at": {"expected_type": "DATETIME"},
        }
    )

    executor = SchemaExecutor(mock_connection, test_mode=True)

    # Actual has email wrong type and missing created_at
    columns = [
        {"name": "id", "type": "INTEGER"},
        {"name": "email", "type": "INT"},
    ]

    with patch.object(executor, "get_engine") as mock_get_engine, patch(
        "shared.database.query_executor.QueryExecutor"
    ) as mock_qe_class:
        mock_engine = AsyncMock()
        mock_get_engine.return_value = mock_engine
        mock_qe = AsyncMock()
        mock_qe.get_column_list.return_value = columns
        mock_qe_class.return_value = mock_qe

        result = await executor.execute_rule(rule)

    # Failures: email type mismatch + created_at missing = 2
    assert result.status == "FAILED"
    assert result.dataset_metrics[0].total_records == 3
    assert result.dataset_metrics[0].failed_records == 2


@pytest.mark.asyncio
async def test_schema_rule_strict_mode_counts_extras(
    mock_connection: ConnectionSchema,
) -> None:
    rule = build_schema_rule({"id": {"expected_type": "INTEGER"}}, strict_mode=True)
    executor = SchemaExecutor(mock_connection, test_mode=True)

    columns = [
        {"name": "id", "type": "INTEGER"},
        {"name": "extra_col", "type": "TEXT"},
        {"name": "another_extra", "type": "TEXT"},
    ]

    with patch.object(executor, "get_engine") as mock_get_engine, patch(
        "shared.database.query_executor.QueryExecutor"
    ) as mock_qe_class:
        mock_engine = AsyncMock()
        mock_get_engine.return_value = mock_engine
        mock_qe = AsyncMock()
        mock_qe.get_column_list.return_value = columns
        mock_qe_class.return_value = mock_qe

        result = await executor.execute_rule(rule)

    # Failures: 2 extra columns
    assert result.status == "FAILED"
    assert result.dataset_metrics[0].total_records == 1
    assert result.dataset_metrics[0].failed_records == 2


@pytest.mark.asyncio
async def test_schema_rule_case_insensitive_matching(
    mock_connection: ConnectionSchema,
) -> None:
    # Case-insensitive should match Email vs email and map VARCHAR to STRING
    rule = build_schema_rule(
        {"Email": {"expected_type": "STRING"}}, strict_mode=False, case_insensitive=True
    )

    executor = SchemaExecutor(mock_connection, test_mode=True)

    columns = [
        {"name": "email", "type": "VARCHAR(255)"},
    ]

    with patch.object(executor, "get_engine") as mock_get_engine, patch(
        "shared.database.query_executor.QueryExecutor"
    ) as mock_qe_class:
        mock_engine = AsyncMock()
        mock_get_engine.return_value = mock_engine
        mock_qe = AsyncMock()
        mock_qe.get_column_list.return_value = columns
        mock_qe_class.return_value = mock_qe

        result = await executor.execute_rule(rule)

    assert result.status == "PASSED"
    assert result.dataset_metrics[0].total_records == 1
    assert result.dataset_metrics[0].failed_records == 0


@pytest.mark.asyncio
async def test_schema_rule_invalid_expected_type_rejected_on_creation(
    mock_connection: ConnectionSchema,
) -> None:
    # Invalid expected_type should be rejected during RuleSchema validation
    with pytest.raises(RuleExecutionError, match="Unsupported expected_type"):
        build_schema_rule({"id": {"expected_type": "UNKNOWN_TYPE"}})
