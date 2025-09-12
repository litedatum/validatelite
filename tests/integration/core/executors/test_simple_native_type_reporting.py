"""
Simple integration test to verify native type reporting functionality works.

This is a minimal test to demonstrate that the native type reporting enhancements
work correctly with a real MySQL database.
"""

import pytest
from sqlalchemy import text

from core.executors.schema_executor import SchemaExecutor
from shared.enums import DataType, RuleType
from shared.enums.connection_types import ConnectionType
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.rule_schema import RuleSchema
from tests.shared.builders.test_builders import TestDataBuilder
from tests.shared.utils.database_utils import (
    get_available_databases,
    get_mysql_connection_params,
)

pytestmark = pytest.mark.asyncio


def _skip_if_mysql_unavailable() -> None:
    if "mysql" not in get_available_databases():
        pytest.skip("MySQL not configured; skipping integration tests")


def build_simple_schema_rule(columns: dict) -> RuleSchema:
    """Build a simple SCHEMA rule for testing."""
    builder = TestDataBuilder.rule()
    rule = (
        builder.with_name("test_native_reporting")
        .with_target("test_db", "native_test_table", None)  # Table-level rule
        .with_type(RuleType.SCHEMA)
        .with_parameter("columns", columns)
        .build()
    )
    return rule


@pytest.mark.integration
@pytest.mark.database
class TestSimpleNativeTypeReporting:
    """Simple test for native type reporting."""

    async def test_native_type_reporting_works(self):
        """Test that native type information is included in schema validation results."""
        _skip_if_mysql_unavailable()
        
        # Create connection
        params = get_mysql_connection_params()
        from typing import cast
        
        connection = ConnectionSchema(
            name="test_native_types",
            description="Test connection for native type reporting",
            connection_type=ConnectionType.MYSQL,
            host=str(params["host"]),
            port=cast(int, params["port"]),
            db_name=str(params["database"]),
            username=str(params["username"]),
            password=str(params["password"]),
        )
        
        # Create executor
        executor = SchemaExecutor(connection, test_mode=True)
        
        # Create and setup table
        engine = await executor.get_engine()
        
        # Use regular connection (not transaction) for DDL
        async with engine.connect() as conn:
            # Drop and create table
            await conn.execute(text("DROP TABLE IF EXISTS native_test_table"))
            await conn.execute(text("""
                CREATE TABLE native_test_table (
                    id INT PRIMARY KEY,
                    name VARCHAR(50) NOT NULL,
                    score DECIMAL(5,2)
                )
            """))
            await conn.commit()
        
        try:
            # Create schema rule
            columns = {
                "id": {"expected_type": DataType.INTEGER.value},
                "name": {"expected_type": DataType.STRING.value, "max_length": 50},
                "score": {"expected_type": DataType.FLOAT.value, "precision": 5, "scale": 2},
            }
            rule = build_simple_schema_rule(columns)
            
            # Execute rule
            result = await executor.execute_rule(rule)
            
            # Basic validation
            print(f"Rule execution status: {result.status}")
            print(f"Execution message: {result.execution_message}")
            
            # Check that we have schema details
            execution_plan = result.execution_plan
            assert "schema_details" in execution_plan
            
            schema_details = execution_plan["schema_details"]
            assert "field_results" in schema_details
            
            field_results = schema_details["field_results"]
            assert len(field_results) >= 1  # Should have at least one field result
            
            # Check that native type information is present
            for field_result in field_results:
                print(f"Field: {field_result.get('column')}")
                print(f"  - Native type: {field_result.get('native_type')}")
                print(f"  - Canonical type: {field_result.get('canonical_type')}")
                print(f"  - Native metadata: {field_result.get('native_metadata')}")
                
                # Verify enhanced fields are present
                assert "native_type" in field_result
                assert "canonical_type" in field_result
                assert "native_metadata" in field_result
                
                # Verify they have meaningful values
                assert field_result["native_type"] is not None
                assert field_result["canonical_type"] is not None
                assert isinstance(field_result["native_metadata"], dict)
            
            # Print overall result for debugging
            print(f"Test completed with result status: {result.status}")
            
        finally:
            # Clean up
            async with engine.connect() as conn:
                await conn.execute(text("DROP TABLE IF EXISTS native_test_table"))
                await conn.commit()
            
            # Close engine
            await engine.dispose()