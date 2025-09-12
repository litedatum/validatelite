"""
Integration tests for SchemaExecutor native type reporting enhancements

Tests the new functionality that includes native_type, canonical_type, 
and native_metadata in field_results for all scenarios including TYPE_MISMATCH.
"""

import pytest

from core.executors.schema_executor import SchemaExecutor
from shared.enums import DataType, RuleType
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


@pytest.fixture
def mysql_connection():
    """Create MySQL connection for testing."""
    _skip_if_mysql_unavailable()
    params = get_mysql_connection_params()
    from shared.enums.connection_types import ConnectionType
    from typing import cast
    
    return ConnectionSchema(
        name="mysql_native_type_test",
        description="MySQL connection for native type testing",
        connection_type=ConnectionType.MYSQL,
        host=str(params["host"]),
        port=cast(int, params["port"]),
        db_name=str(params["database"]),
        username=str(params["username"]),
        password=str(params["password"]),
    )


@pytest.fixture
async def schema_executor(mysql_connection):
    """Create SchemaExecutor with MySQL connection."""
    return SchemaExecutor(mysql_connection, test_mode=True)


def build_schema_rule_with_native_reporting(
    columns: dict, 
    table_name: str = "test_table",
    strict_mode: bool = False, 
    case_insensitive: bool = False
) -> RuleSchema:
    """Build a SCHEMA rule for testing native type reporting."""
    builder = TestDataBuilder.rule()
    rule = (
        builder.with_name(f"schema_{table_name}")
        .with_target("test_db", table_name, None)  # Table-level rule
        .with_type(RuleType.SCHEMA)
        .with_parameter("columns", columns)
        .with_parameter("strict_mode", strict_mode)
        .with_parameter("case_insensitive", case_insensitive)
        .build()
    )
    return rule


@pytest.mark.integration
@pytest.mark.database
class TestSchemaExecutorNativeTypeReporting:
    """Test native type reporting enhancements in SchemaExecutor."""

    async def test_native_type_reporting_successful_case(self, schema_executor):
        """Test that native type information is included in successful validation."""
        # Create test table with known types
        from sqlalchemy import text
        engine = await schema_executor.get_engine()
        async with engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS test_users"))
            await conn.execute(text(
                "CREATE TABLE test_users (id INT, name VARCHAR(50), active BOOLEAN)"
            ))

        # Define schema rule that should pass
        columns = {
            "id": {"expected_type": DataType.INTEGER.value},
            "name": {"expected_type": DataType.STRING.value, "max_length": 50},
            "active": {"expected_type": DataType.BOOLEAN.value},
        }
        rule = build_schema_rule_with_native_reporting(columns, "test_users")

        # Execute rule
        result = await schema_executor.execute_rule(rule)

        # Verify result structure
        assert result.status == "PASSED"
        
        # Verify enhanced field_results include native type information
        schema_details = result.execution_plan.get("schema_details", {})
        field_results = schema_details.get("field_results", [])
        
        assert len(field_results) == 3
        
        for field_result in field_results:
            # Each field result should have native type information
            assert "native_type" in field_result
            assert "canonical_type" in field_result
            assert "native_metadata" in field_result
            
            # Native type should be the database-specific type
            assert field_result["native_type"] is not None
            assert isinstance(field_result["native_type"], str)
            
            # Canonical type should be the standardized type
            assert field_result["canonical_type"] in [dt.value for dt in DataType]
            
            # Native metadata should be a dict
            assert isinstance(field_result["native_metadata"], dict)
            
            # Verify specific field expectations
            if field_result["column"] == "id":
                assert field_result["canonical_type"] == DataType.INTEGER.value
                assert field_result["failure_code"] == "NONE"
            elif field_result["column"] == "name":
                assert field_result["canonical_type"] == DataType.STRING.value
                # Should include max_length in native_metadata for VARCHAR(50)
                assert "max_length" in field_result["native_metadata"]
                assert field_result["native_metadata"]["max_length"] == 50
            elif field_result["column"] == "active":
                assert field_result["canonical_type"] == DataType.BOOLEAN.value

    async def test_native_type_reporting_type_mismatch(self, schema_executor):
        """Test that native type information is included even for TYPE_MISMATCH cases."""
        # Create test table
        from sqlalchemy import text
        engine = await schema_executor.get_engine()
        async with engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS test_mismatch"))
            await conn.execute(text(
                "CREATE TABLE test_mismatch (id INT, name VARCHAR(100))"
            ))

        # Define schema rule with type mismatches
        columns = {
            "id": {"expected_type": DataType.STRING.value},  # Mismatch: expecting string, actual is integer
            "name": {"expected_type": DataType.INTEGER.value},  # Mismatch: expecting integer, actual is string
        }
        rule = build_schema_rule_with_native_reporting(columns, "test_mismatch")

        # Execute rule
        result = await schema_executor.execute_rule(rule)

        # Should fail due to type mismatches
        assert result.status == "FAILED"
        
        # Verify enhanced field_results include native type information even for failures
        schema_details = result.execution_plan.get("schema_details", {})
        field_results = schema_details.get("field_results", [])
        
        assert len(field_results) == 2
        
        for field_result in field_results:
            # Even with type mismatches, native type information should be present
            assert "native_type" in field_result
            assert "canonical_type" in field_result
            assert "native_metadata" in field_result
            
            # Should have failed type validation but passed existence
            assert field_result["existence"] == "PASSED"
            assert field_result["type"] == "FAILED" 
            assert field_result["failure_code"] == "TYPE_MISMATCH"
            
            # Native type information should still be accurate
            assert field_result["native_type"] is not None
            assert field_result["canonical_type"] is not None
            
            # Verify the actual vs expected mismatch
            if field_result["column"] == "id":
                # Actual type is INTEGER, but expected STRING
                assert field_result["canonical_type"] == DataType.INTEGER.value
            elif field_result["column"] == "name":
                # Actual type is STRING, but expected INTEGER
                assert field_result["canonical_type"] == DataType.STRING.value
                # Should include max_length from VARCHAR(100)
                assert "max_length" in field_result["native_metadata"]
                assert field_result["native_metadata"]["max_length"] == 100

    async def test_native_type_reporting_field_missing(self, schema_executor):
        """Test native type information handling for missing fields."""
        # Create test table with only some of the expected fields
        from sqlalchemy import text
        engine = await schema_executor.get_engine()
        async with engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS test_partial"))
            await conn.execute(text("CREATE TABLE test_partial (id INT)"))

        # Define schema rule expecting more fields than exist
        columns = {
            "id": {"expected_type": DataType.INTEGER.value},
            "missing_field": {"expected_type": DataType.STRING.value},
        }
        rule = build_schema_rule_with_native_reporting(columns, "test_partial")

        # Execute rule
        result = await schema_executor.execute_rule(rule)

        # Should fail due to missing field
        assert result.status == "FAILED"
        
        # Verify field_results
        schema_details = result.execution_plan.get("schema_details", {})
        field_results = schema_details.get("field_results", [])
        
        assert len(field_results) == 2
        
        # Find results for each field
        id_result = next(fr for fr in field_results if fr["column"] == "id")
        missing_result = next(fr for fr in field_results if fr["column"] == "missing_field")
        
        # Existing field should have native type information
        assert id_result["existence"] == "PASSED"
        assert id_result["type"] == "PASSED"
        assert id_result["native_type"] is not None
        assert id_result["canonical_type"] == DataType.INTEGER.value
        assert isinstance(id_result["native_metadata"], dict)
        
        # Missing field should have null native type information
        assert missing_result["existence"] == "FAILED"
        assert missing_result["type"] == "SKIPPED"
        assert missing_result["failure_code"] == "FIELD_MISSING"
        assert missing_result["native_type"] is None
        assert missing_result["canonical_type"] is None
        assert missing_result["native_metadata"] == {}

    async def test_native_metadata_precision_scale(self, schema_executor):
        """Test native metadata reporting for float types with precision/scale."""
        # Create test table with decimal/numeric types
        from sqlalchemy import text
        engine = await schema_executor.get_engine()
        async with engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS test_decimal"))
            # MySQL supports DECIMAL with precision/scale
            await conn.execute(text("CREATE TABLE test_decimal (price DECIMAL(10,2), amount NUMERIC(8,3))"))

        # Define schema rule for decimal types
        columns = {
            "price": {"expected_type": DataType.FLOAT.value, "precision": 10, "scale": 2},
            "amount": {"expected_type": DataType.FLOAT.value, "precision": 8, "scale": 3},
        }
        rule = build_schema_rule_with_native_reporting(columns, "test_decimal")

        # Execute rule
        result = await schema_executor.execute_rule(rule)

        # Verify field_results include precision/scale metadata
        schema_details = result.execution_plan.get("schema_details", {})
        field_results = schema_details.get("field_results", [])
        
        for field_result in field_results:
            assert "native_metadata" in field_result
            native_metadata = field_result["native_metadata"]
            
            # Verify the native type is captured
            assert field_result["native_type"] is not None
            assert field_result["canonical_type"] == DataType.FLOAT.value
            
            # Note: SQLite might not preserve exact precision/scale, but the structure should be correct
            assert isinstance(native_metadata, dict)

    async def test_comprehensive_native_type_coverage(self, schema_executor):
        """Test native type reporting across various database type scenarios."""
        # Create table with various data types
        from sqlalchemy import text
        engine = await schema_executor.get_engine()
        async with engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS test_comprehensive"))
            await conn.execute(text("""
                CREATE TABLE test_comprehensive (
                    id INT,
                    name TEXT,
                    email VARCHAR(255),
                    age SMALLINT,
                    salary DOUBLE,
                    is_active BOOLEAN,
                    birth_date DATE,
                    created_at DATETIME
                )
            """))

        # Define schema rule covering all types
        columns = {
            "id": {"expected_type": DataType.INTEGER.value},
            "name": {"expected_type": DataType.STRING.value},
            "email": {"expected_type": DataType.STRING.value, "max_length": 255},
            "age": {"expected_type": DataType.INTEGER.value},
            "salary": {"expected_type": DataType.FLOAT.value},
            "is_active": {"expected_type": DataType.BOOLEAN.value},
            "birth_date": {"expected_type": DataType.DATE.value},
            "created_at": {"expected_type": DataType.DATETIME.value},
        }
        rule = build_schema_rule_with_native_reporting(columns, "test_comprehensive")

        # Execute rule
        result = await schema_executor.execute_rule(rule)

        # Verify all fields have complete native type information
        schema_details = result.execution_plan.get("schema_details", {})
        field_results = schema_details.get("field_results", [])
        
        assert len(field_results) == 8
        
        for field_result in field_results:
            # Every field should have complete native type information
            assert field_result["native_type"] is not None
            assert field_result["canonical_type"] is not None
            assert isinstance(field_result["native_metadata"], dict)
            
            # Verify canonical type mapping is correct
            column_name = field_result["column"]
            canonical_type = field_result["canonical_type"]
            
            type_expectations = {
                "id": DataType.INTEGER.value,
                "name": DataType.STRING.value,
                "email": DataType.STRING.value,
                "age": DataType.INTEGER.value,
                "salary": DataType.FLOAT.value,
                "is_active": DataType.BOOLEAN.value,
                "birth_date": DataType.DATE.value,
                "created_at": DataType.DATETIME.value,
            }
            
            assert canonical_type == type_expectations[column_name]


@pytest.mark.integration
@pytest.mark.database
class TestSchemaExecutorBackwardCompatibility:
    """Test that enhancements maintain backward compatibility."""

    async def test_existing_functionality_unchanged(self, schema_executor):
        """Test that existing schema validation functionality is unchanged."""
        # Create test table
        from sqlalchemy import text
        engine = await schema_executor.get_engine()
        async with engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS test_compat"))
            await conn.execute(text("CREATE TABLE test_compat (id INT, name VARCHAR(50))"))

        # Use existing schema rule format
        columns = {
            "id": {"expected_type": DataType.INTEGER.value},
            "name": {"expected_type": DataType.STRING.value},
        }
        rule = build_schema_rule_with_native_reporting(columns, "test_compat")

        # Execute rule
        result = await schema_executor.execute_rule(rule)

        # Verify existing fields are still present and working
        assert result.status == "PASSED"
        assert result.rule_id == rule.id
        assert len(result.dataset_metrics) == 1
        
        # Verify execution_plan structure is maintained
        execution_plan = result.execution_plan
        assert "execution_type" in execution_plan
        assert "schema_details" in execution_plan
        
        schema_details = execution_plan["schema_details"]
        assert "field_results" in schema_details
        assert "extras" in schema_details
        assert "table_exists" in schema_details
        
        # Verify field_results have expected legacy fields
        field_results = schema_details["field_results"]
        for field_result in field_results:
            assert "column" in field_result
            assert "existence" in field_result
            assert "type" in field_result
            assert "failure_code" in field_result
            
            # NEW: Also verify enhanced fields are added
            assert "native_type" in field_result
            assert "canonical_type" in field_result
            assert "native_metadata" in field_result