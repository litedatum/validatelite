"""
Integration test for native type reporting functionality using MySQL.

Based on the established pattern from test_mysql_integration.py.
Tests the enhanced SchemaExecutor that includes native_type, canonical_type,
and native_metadata in field_results.
"""

import pytest

from core.executors.schema_executor import SchemaExecutor
from shared.database.query_executor import QueryExecutor
from shared.enums import RuleAction, RuleCategory, RuleType, SeverityLevel
from shared.enums.connection_types import ConnectionType
from shared.enums.data_types import DataType
from shared.schema.base import RuleTarget, TargetEntity
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.rule_schema import RuleSchema
from shared.utils.logger import get_logger
from tests.shared.utils.database_utils import (
    get_available_databases,
    get_mysql_connection_params,
)

pytestmark = pytest.mark.asyncio

logger = get_logger(__name__)


def _skip_if_mysql_unavailable() -> None:
    if "mysql" not in get_available_databases():
        pytest.skip("MySQL not configured; skipping integration tests")


@pytest.mark.integration
@pytest.mark.database
class TestNativeTypeIntegration:
    """Test native type reporting functionality with real MySQL database."""

    async def _prepare_test_environment(self, mysql_connection_params):
        """Prepare MySQL test environment with test table."""
        from shared.database.connection import get_db_url, get_engine
        from typing import cast
        
        # Create engine for setup
        db_url = get_db_url(
            str(mysql_connection_params["db_type"]),
            str(mysql_connection_params["host"]),
            cast(int, mysql_connection_params["port"]),
            str(mysql_connection_params["database"]),
            str(mysql_connection_params["username"]),
            str(mysql_connection_params["password"]),
        )
        engine = await get_engine(db_url, pool_size=1, echo=False)
        executor = QueryExecutor(engine)

        # Clean up and create test table
        await executor.execute_query(
            "DROP TABLE IF EXISTS native_type_test", fetch=False
        )
        
        await executor.execute_query(
            """
            CREATE TABLE native_type_test (
                id INT PRIMARY KEY AUTO_INCREMENT,
                name VARCHAR(50) NOT NULL,
                email VARCHAR(100),
                age SMALLINT,
                score DECIMAL(5,2),
                is_active BOOLEAN DEFAULT TRUE,
                birth_date DATE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            fetch=False,
        )

        # Insert test data
        await executor.execute_query(
            """
            INSERT INTO native_type_test 
            (name, email, age, score, is_active, birth_date) VALUES 
            ('Alice', 'alice@example.com', 25, 85.50, TRUE, '1998-05-15'),
            ('Bob', 'bob@example.com', 30, 92.75, FALSE, '1993-08-20')
            """,
            fetch=False,
        )

        await engine.dispose()
        return executor

    async def test_native_type_reporting_comprehensive(self, mysql_connection_params):
        """Test that native type information is correctly reported for various MySQL types."""
        _skip_if_mysql_unavailable()
        
        # Prepare test environment
        await self._prepare_test_environment(mysql_connection_params)

        # Create connection schema
        connection = ConnectionSchema(
            name="native_type_test_connection",
            description="Connection for testing native type reporting",
            connection_type=ConnectionType.MYSQL,
            host=mysql_connection_params["host"],
            port=mysql_connection_params["port"],
            username=mysql_connection_params["username"],
            password=mysql_connection_params["password"],
            db_name=mysql_connection_params["database"],
        )

        # Create schema executor
        executor = SchemaExecutor(connection, test_mode=True)

        # Define schema rule with expected types
        columns = {
            "id": {"expected_type": DataType.INTEGER.value},
            "name": {"expected_type": DataType.STRING.value, "max_length": 50},
            "email": {"expected_type": DataType.STRING.value, "max_length": 100},
            "age": {"expected_type": DataType.INTEGER.value},
            "score": {"expected_type": DataType.FLOAT.value, "precision": 5, "scale": 2},
            "is_active": {"expected_type": DataType.INTEGER.value},  # MySQL BOOLEAN -> TINYINT(1) -> INTEGER
            "birth_date": {"expected_type": DataType.DATE.value},
            "created_at": {"expected_type": DataType.DATETIME.value},
            "description": {"expected_type": DataType.STRING.value},
        }

        rule = RuleSchema(
            id="native_type_test_rule",
            name="Native Type Reporting Test",
            description="Test rule for native type reporting",
            type=RuleType.SCHEMA,
            category=RuleCategory.VALIDITY,
            severity=SeverityLevel.MEDIUM,
            action=RuleAction.LOG,
            target=RuleTarget(
                entities=[TargetEntity(
                    database=mysql_connection_params["database"],
                    table="native_type_test",
                    column=None
                )],
                relationship_type="single_table",
            ),
            parameters={"columns": columns},
        )

        try:
            # Execute the schema rule
            result = await executor.execute_rule(rule)
            
            logger.info(f"Schema rule execution status: {result.status}")
            logger.info(f"Execution message: {result.execution_message}")

            # Debug: print detailed information
            execution_plan = result.execution_plan
            if "schema_details" in execution_plan:
                schema_details = execution_plan["schema_details"]
                if "field_results" in schema_details:
                    field_results = schema_details["field_results"]
                    logger.info(f"Number of field results: {len(field_results)}")
                    for fr in field_results:
                        logger.info(f"Field {fr.get('column')}: existence={fr.get('existence')}, type={fr.get('type')}, failure_code={fr.get('failure_code')}")
                        if fr.get('failure_code') != 'NONE':
                            logger.info(f"  Failure details: {fr.get('failure_details')}")

            # Verify basic execution - should pass now with corrected type expectations
            assert result.status == "PASSED", f"Expected PASSED, got {result.status}: {result.execution_message}"
            
            # Verify execution plan contains schema details
            assert "schema_details" in execution_plan
            
            schema_details = execution_plan["schema_details"]
            assert "field_results" in schema_details
            assert schema_details["table_exists"] is True
            
            field_results = schema_details["field_results"]
            assert len(field_results) == len(columns), f"Expected {len(columns)} field results, got {len(field_results)}"

            # Test native type information for each field
            field_map = {fr["column"]: fr for fr in field_results}
            
            # Test INTEGER type (id, age)
            for col in ["id", "age"]:
                field_result = field_map[col]
                assert "native_type" in field_result
                assert "canonical_type" in field_result  
                assert "native_metadata" in field_result
                
                assert field_result["canonical_type"] == DataType.INTEGER.value
                assert field_result["native_type"] is not None
                assert isinstance(field_result["native_metadata"], dict)
                
                logger.info(f"{col}: native_type={field_result['native_type']}, "
                          f"canonical_type={field_result['canonical_type']}")

            # Test STRING type with length (name, email)
            name_result = field_map["name"]
            assert name_result["canonical_type"] == DataType.STRING.value
            assert name_result["native_metadata"].get("max_length") == 50
            
            email_result = field_map["email"] 
            assert email_result["canonical_type"] == DataType.STRING.value
            assert email_result["native_metadata"].get("max_length") == 100

            # Test FLOAT type with precision/scale (score)
            score_result = field_map["score"]
            assert score_result["canonical_type"] == DataType.FLOAT.value
            # Note: MySQL may return precision/scale info in native_metadata
            logger.info(f"score native_metadata: {score_result['native_metadata']}")

            # Test BOOLEAN type (is_active) - Note: MySQL maps BOOLEAN to TINYINT(1) -> INTEGER
            boolean_result = field_map["is_active"]
            # In MySQL, BOOLEAN is actually stored as TINYINT(1) which maps to INTEGER
            assert boolean_result["canonical_type"] == DataType.INTEGER.value
            logger.info(f"is_active correctly identified as INTEGER (MySQL BOOLEAN -> TINYINT mapping)")

            # Test DATE type (birth_date)
            date_result = field_map["birth_date"]
            assert date_result["canonical_type"] == DataType.DATE.value

            # Test DATETIME type (created_at)
            datetime_result = field_map["created_at"]
            assert datetime_result["canonical_type"] == DataType.DATETIME.value

            # Test TEXT type (description) - should map to STRING
            desc_result = field_map["description"]
            assert desc_result["canonical_type"] == DataType.STRING.value

            # Verify all fields have the required enhanced information
            for field_result in field_results:
                assert field_result["existence"] == "PASSED"
                assert field_result["type"] == "PASSED" 
                assert field_result["failure_code"] == "NONE"
                
                # Verify enhanced fields exist and have meaningful values
                assert field_result["native_type"] is not None
                assert field_result["canonical_type"] is not None
                assert isinstance(field_result["native_metadata"], dict)
                
                logger.info(f"✓ {field_result['column']}: "
                          f"native='{field_result['native_type']}', "
                          f"canonical='{field_result['canonical_type']}', "
                          f"metadata={field_result['native_metadata']}")

            logger.info("✅ Native type reporting test completed successfully")

        finally:
            # Cleanup
            from shared.database.connection import get_db_url, get_engine
            from typing import cast
            
            db_url = get_db_url(
                str(mysql_connection_params["db_type"]),
                str(mysql_connection_params["host"]),
                cast(int, mysql_connection_params["port"]),
                str(mysql_connection_params["database"]),
                str(mysql_connection_params["username"]),
                str(mysql_connection_params["password"]),
            )
            cleanup_engine = await get_engine(db_url, pool_size=1, echo=False)
            cleanup_executor = QueryExecutor(cleanup_engine)
            
            await cleanup_executor.execute_query(
                "DROP TABLE IF EXISTS native_type_test", fetch=False
            )
            await cleanup_engine.dispose()

    async def test_native_type_reporting_with_type_mismatch(self, mysql_connection_params):
        """Test native type information is included even for TYPE_MISMATCH cases."""
        _skip_if_mysql_unavailable()
        
        # Prepare test environment  
        await self._prepare_test_environment(mysql_connection_params)

        # Create connection schema
        connection = ConnectionSchema(
            name="type_mismatch_test_connection", 
            description="Connection for testing type mismatch scenarios",
            connection_type=ConnectionType.MYSQL,
            host=mysql_connection_params["host"],
            port=mysql_connection_params["port"],
            username=mysql_connection_params["username"],
            password=mysql_connection_params["password"],
            db_name=mysql_connection_params["database"],
        )

        # Create schema executor
        executor = SchemaExecutor(connection, test_mode=True)

        # Define schema rule with intentional type mismatches
        columns = {
            "id": {"expected_type": DataType.STRING.value},  # Mismatch: actual is INT
            "name": {"expected_type": DataType.INTEGER.value},  # Mismatch: actual is VARCHAR
            "age": {"expected_type": DataType.FLOAT.value},  # Mismatch: actual is SMALLINT
        }

        rule = RuleSchema(
            id="type_mismatch_test_rule",
            name="Type Mismatch Test",
            description="Test rule for type mismatch scenarios",
            type=RuleType.SCHEMA,
            category=RuleCategory.VALIDITY,
            severity=SeverityLevel.MEDIUM,
            action=RuleAction.LOG,
            target=RuleTarget(
                entities=[TargetEntity(
                    database=mysql_connection_params["database"],
                    table="native_type_test",
                    column=None
                )],
                relationship_type="single_table",
            ),
            parameters={"columns": columns},
        )

        try:
            # Execute the schema rule
            result = await executor.execute_rule(rule)
            
            logger.info(f"Type mismatch test status: {result.status}")
            logger.info(f"Execution message: {result.execution_message}")

            # Should fail due to type mismatches
            assert result.status == "FAILED"
            
            # Verify schema details
            schema_details = result.execution_plan["schema_details"]
            field_results = schema_details["field_results"]
            assert len(field_results) == 3

            # Verify that native type information is provided even for failed cases
            for field_result in field_results:
                assert field_result["existence"] == "PASSED"
                assert field_result["type"] == "FAILED"
                assert field_result["failure_code"] == "TYPE_MISMATCH"
                
                # Critical: native type info should still be present for failed validations
                assert "native_type" in field_result
                assert "canonical_type" in field_result
                assert "native_metadata" in field_result
                
                assert field_result["native_type"] is not None
                assert field_result["canonical_type"] is not None
                assert isinstance(field_result["native_metadata"], dict)
                
                logger.info(f"❌ {field_result['column']}: TYPE_MISMATCH but still has "
                          f"native='{field_result['native_type']}', "
                          f"canonical='{field_result['canonical_type']}'")

            logger.info("✅ Type mismatch native type reporting test completed")

        finally:
            # Cleanup
            from shared.database.connection import get_db_url, get_engine
            from typing import cast
            
            db_url = get_db_url(
                str(mysql_connection_params["db_type"]),
                str(mysql_connection_params["host"]),
                cast(int, mysql_connection_params["port"]),
                str(mysql_connection_params["database"]),
                str(mysql_connection_params["username"]),
                str(mysql_connection_params["password"]),
            )
            cleanup_engine = await get_engine(db_url, pool_size=1, echo=False)
            cleanup_executor = QueryExecutor(cleanup_engine)
            
            await cleanup_executor.execute_query(
                "DROP TABLE IF EXISTS native_type_test", fetch=False
            )
            await cleanup_engine.dispose()