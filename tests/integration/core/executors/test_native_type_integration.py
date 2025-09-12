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
from tests.shared.builders.test_builders import TestDataBuilder
from tests.shared.utils.database_utils import get_available_databases

pytestmark = pytest.mark.asyncio

logger = get_logger(__name__)


def _skip_if_mysql_unavailable() -> None:
    if "mysql" not in get_available_databases():
        pytest.skip("MySQL not configured; skipping integration tests")


def build_schema_rule_with_native_reporting(
    columns: dict,
    table_name: str = "test_table",
    database_name: str = "test_db",
    strict_mode: bool = False,
    case_insensitive: bool = False,
) -> RuleSchema:
    """Build a SCHEMA rule for testing native type reporting."""
    builder = TestDataBuilder.rule()
    rule = (
        builder.with_name(f"schema_{table_name}")
        .with_target(database_name, table_name, "")  # Table-level rule
        .with_type(RuleType.SCHEMA)
        .with_parameter("columns", columns)
        .with_parameter("strict_mode", strict_mode)
        .with_parameter("case_insensitive", case_insensitive)
        .build()
    )
    return rule


@pytest.mark.integration
@pytest.mark.database
class TestNativeTypeIntegration:
    """Test native type reporting functionality with real MySQL database."""

    async def _prepare_test_environment(
        self, mysql_connection_params: dict
    ) -> QueryExecutor:
        """Prepare MySQL test environment with test table."""
        from typing import cast

        from shared.database.connection import get_db_url, get_engine

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

    async def test_native_type_reporting_comprehensive(
        self, mysql_connection_params: dict
    ) -> None:
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
            "score": {
                "expected_type": DataType.FLOAT.value,
                "precision": 5,
                "scale": 2,
            },
            "is_active": {
                "expected_type": DataType.INTEGER.value
            },  # MySQL BOOLEAN -> TINYINT(1) -> INTEGER
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
                entities=[
                    TargetEntity(
                        database=mysql_connection_params["database"],
                        table="native_type_test",
                        column=None,
                    )
                ],
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
            assert execution_plan is not None
            if "schema_details" in execution_plan:
                schema_details = execution_plan["schema_details"]
                if "field_results" in schema_details:
                    field_results = schema_details["field_results"]
                    logger.info(f"Number of field results: {len(field_results)}")
                    for fr in field_results:
                        logger.info(
                            f"Field {fr.get('column')}: existence={fr.get('existence')}, type={fr.get('type')}, failure_code={fr.get('failure_code')}"
                        )
                        if fr.get("failure_code") != "NONE":
                            logger.info(
                                f"  Failure details: {fr.get('failure_details')}"
                            )

            # Verify basic execution - should pass now with corrected type expectations
            assert (
                result.status == "PASSED"
            ), f"Expected PASSED, got {result.status}: {result.execution_message}"

            # Verify execution plan contains schema details
            assert execution_plan is not None
            assert "schema_details" in execution_plan

            schema_details = execution_plan["schema_details"]
            assert "field_results" in schema_details
            assert schema_details["table_exists"] is True

            field_results = schema_details["field_results"]
            assert len(field_results) == len(
                columns
            ), f"Expected {len(columns)} field results, got {len(field_results)}"

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

                logger.info(
                    f"{col}: native_type={field_result['native_type']}, "
                    f"canonical_type={field_result['canonical_type']}"
                )

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
            logger.info(
                f"is_active correctly identified as INTEGER (MySQL BOOLEAN -> TINYINT mapping)"
            )

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

                logger.info(
                    f"✓ {field_result['column']}: "
                    f"native='{field_result['native_type']}', "
                    f"canonical='{field_result['canonical_type']}', "
                    f"metadata={field_result['native_metadata']}"
                )

            logger.info("✅ Native type reporting test completed successfully")

        finally:
            # Cleanup
            from typing import cast

            from shared.database.connection import get_db_url, get_engine

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

    async def test_native_type_reporting_with_type_mismatch(
        self, mysql_connection_params: dict
    ) -> None:
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
            "name": {
                "expected_type": DataType.INTEGER.value
            },  # Mismatch: actual is VARCHAR
            "age": {
                "expected_type": DataType.FLOAT.value
            },  # Mismatch: actual is SMALLINT
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
                entities=[
                    TargetEntity(
                        database=mysql_connection_params["database"],
                        table="native_type_test",
                        column=None,
                    )
                ],
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
            assert result.execution_plan is not None
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

                logger.info(
                    f"❌ {field_result['column']}: TYPE_MISMATCH but still has "
                    f"native='{field_result['native_type']}', "
                    f"canonical='{field_result['canonical_type']}'"
                )

            logger.info("✅ Type mismatch native type reporting test completed")

        finally:
            # Cleanup
            from typing import cast

            from shared.database.connection import get_db_url, get_engine

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

    async def test_native_type_reporting_missing_field(
        self, mysql_connection_params: dict
    ) -> None:
        """Test native type information handling for missing fields."""
        _skip_if_mysql_unavailable()

        # Prepare test environment with limited fields
        await self._prepare_test_environment(mysql_connection_params)

        # Create connection schema
        connection = ConnectionSchema(
            name="missing_field_test_connection",
            description="Connection for testing missing field scenarios",
            connection_type=ConnectionType.MYSQL,
            host=mysql_connection_params["host"],
            port=mysql_connection_params["port"],
            username=mysql_connection_params["username"],
            password=mysql_connection_params["password"],
            db_name=mysql_connection_params["database"],
        )

        # Create schema executor
        executor = SchemaExecutor(connection, test_mode=True)

        # Define schema rule expecting more fields than exist in native_type_test
        columns = {
            "id": {"expected_type": DataType.INTEGER.value},
            "name": {"expected_type": DataType.STRING.value},
            "missing_field": {
                "expected_type": DataType.STRING.value
            },  # This field doesn't exist
        }

        rule = build_schema_rule_with_native_reporting(
            columns, "native_type_test", mysql_connection_params["database"]
        )

        try:
            # Execute the schema rule
            result = await executor.execute_rule(rule)

            logger.info(f"Missing field test status: {result.status}")
            logger.info(f"Execution message: {result.execution_message}")

            # Should fail due to missing field
            assert result.status == "FAILED"

            # Verify schema details
            assert result.execution_plan is not None
            schema_details = result.execution_plan["schema_details"]
            field_results = schema_details["field_results"]
            assert len(field_results) == 3

            # Find results for each field
            field_map = {fr["column"]: fr for fr in field_results}

            # Existing fields should have native type information
            for existing_field in ["id", "name"]:
                field_result = field_map[existing_field]
                assert field_result["existence"] == "PASSED"
                assert field_result["type"] == "PASSED"
                assert field_result["native_type"] is not None
                assert field_result["canonical_type"] is not None
                assert isinstance(field_result["native_metadata"], dict)
                logger.info(
                    f"✓ {existing_field}: native_type={field_result['native_type']}"
                )

            # Missing field should have null native type information
            missing_result = field_map["missing_field"]
            assert missing_result["existence"] == "FAILED"
            assert missing_result["type"] == "SKIPPED"
            assert missing_result["failure_code"] == "FIELD_MISSING"
            assert missing_result["native_type"] is None
            assert missing_result["canonical_type"] is None
            assert missing_result["native_metadata"] == {}
            logger.info("✓ missing_field: correctly handled as FIELD_MISSING")

            logger.info("✅ Missing field native type reporting test completed")

        finally:
            # Cleanup
            from typing import cast

            from shared.database.connection import get_db_url, get_engine

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

    async def test_native_metadata_precision_scale(
        self, mysql_connection_params: dict
    ) -> None:
        """Test native metadata reporting for decimal types with precision/scale."""
        _skip_if_mysql_unavailable()

        # Create test environment with decimal types
        from typing import cast

        from shared.database.connection import get_db_url, get_engine

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

        # Clean up and create test table with decimal types
        await executor.execute_query("DROP TABLE IF EXISTS precision_test", fetch=False)

        await executor.execute_query(
            """
            CREATE TABLE precision_test (
                price DECIMAL(10,2),
                amount NUMERIC(8,3),
                ratio FLOAT(7,4)
            ) ENGINE=InnoDB
            """,
            fetch=False,
        )

        await engine.dispose()

        # Create connection schema
        connection = ConnectionSchema(
            name="precision_test_connection",
            description="Connection for testing precision/scale metadata",
            connection_type=ConnectionType.MYSQL,
            host=mysql_connection_params["host"],
            port=mysql_connection_params["port"],
            username=mysql_connection_params["username"],
            password=mysql_connection_params["password"],
            db_name=mysql_connection_params["database"],
        )

        # Create schema executor
        schema_executor = SchemaExecutor(connection, test_mode=True)

        # Define schema rule for decimal types
        columns = {
            "price": {
                "expected_type": DataType.FLOAT.value,
                "precision": 10,
                "scale": 2,
            },
            "amount": {
                "expected_type": DataType.FLOAT.value,
                "precision": 8,
                "scale": 3,
            },
            "ratio": {"expected_type": DataType.FLOAT.value},
        }
        rule = build_schema_rule_with_native_reporting(
            columns, "precision_test", mysql_connection_params["database"]
        )

        try:
            # Execute rule
            result = await schema_executor.execute_rule(rule)

            logger.info(f"Precision/scale test status: {result.status}")

            # Verify field_results include precision/scale metadata
            assert result.execution_plan is not None
            schema_details = result.execution_plan["schema_details"]
            field_results = schema_details["field_results"]

            assert len(field_results) == 3

            for field_result in field_results:
                assert "native_metadata" in field_result
                native_metadata = field_result["native_metadata"]

                # Verify the native type is captured
                assert field_result["native_type"] is not None
                assert field_result["canonical_type"] == DataType.FLOAT.value

                # Verify structure (MySQL may provide precision/scale info)
                assert isinstance(native_metadata, dict)

                column_name = field_result["column"]
                logger.info(
                    f"✓ {column_name}: native_type={field_result['native_type']}, "
                    f"metadata={native_metadata}"
                )

            logger.info("✅ Precision/scale metadata test completed")

        finally:
            # Cleanup
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
                "DROP TABLE IF EXISTS precision_test", fetch=False
            )
            await cleanup_engine.dispose()

    async def test_comprehensive_type_coverage_extended(
        self, mysql_connection_params: dict
    ) -> None:
        """Test native type reporting across extended variety of database types."""
        _skip_if_mysql_unavailable()

        # Create test environment with comprehensive type coverage
        from typing import cast

        from shared.database.connection import get_db_url, get_engine

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

        # Clean up and create comprehensive test table
        await executor.execute_query(
            "DROP TABLE IF EXISTS comprehensive_test", fetch=False
        )

        await executor.execute_query(
            """
            CREATE TABLE comprehensive_test (
                id BIGINT PRIMARY KEY AUTO_INCREMENT,
                tiny_num TINYINT,
                small_num SMALLINT,
                medium_num MEDIUMINT,
                big_num BIGINT,
                float_num FLOAT,
                double_num DOUBLE,
                decimal_num DECIMAL(15,4),
                char_field CHAR(10),
                varchar_field VARCHAR(255),
                text_field TEXT,
                bool_field BOOLEAN,
                date_field DATE,
                datetime_field DATETIME,
                timestamp_field TIMESTAMP
            ) ENGINE=InnoDB
            """,
            fetch=False,
        )

        await engine.dispose()

        # Create connection schema
        connection = ConnectionSchema(
            name="comprehensive_test_connection",
            description="Connection for comprehensive type coverage testing",
            connection_type=ConnectionType.MYSQL,
            host=mysql_connection_params["host"],
            port=mysql_connection_params["port"],
            username=mysql_connection_params["username"],
            password=mysql_connection_params["password"],
            db_name=mysql_connection_params["database"],
        )

        # Create schema executor
        schema_executor = SchemaExecutor(connection, test_mode=True)

        # Define comprehensive schema rule
        columns = {
            "id": {"expected_type": DataType.INTEGER.value},
            "tiny_num": {"expected_type": DataType.INTEGER.value},
            "small_num": {"expected_type": DataType.INTEGER.value},
            "medium_num": {"expected_type": DataType.INTEGER.value},
            "big_num": {"expected_type": DataType.INTEGER.value},
            "float_num": {"expected_type": DataType.FLOAT.value},
            "double_num": {"expected_type": DataType.FLOAT.value},
            "decimal_num": {"expected_type": DataType.FLOAT.value},
            "char_field": {"expected_type": DataType.STRING.value},
            "varchar_field": {"expected_type": DataType.STRING.value},
            "text_field": {"expected_type": DataType.STRING.value},
            "bool_field": {
                "expected_type": DataType.INTEGER.value
            },  # MySQL BOOLEAN -> TINYINT
            "date_field": {"expected_type": DataType.DATE.value},
            "datetime_field": {"expected_type": DataType.DATETIME.value},
            "timestamp_field": {"expected_type": DataType.DATETIME.value},
        }

        rule = build_schema_rule_with_native_reporting(
            columns, "comprehensive_test", mysql_connection_params["database"]
        )

        try:
            # Execute rule
            result = await schema_executor.execute_rule(rule)

            logger.info(f"Comprehensive type coverage test status: {result.status}")
            logger.info(f"Execution message: {result.execution_message}")

            # Debug field-level failures before asserting
            if result.status == "FAILED":
                assert result.execution_plan is not None
                schema_details = result.execution_plan["schema_details"]
                field_results = schema_details["field_results"]

                for field_result in field_results:
                    if field_result["failure_code"] != "NONE":
                        logger.error(
                            f"❌ {field_result['column']}: {field_result['failure_code']} - "
                            f"native='{field_result.get('native_type')}', "
                            f"canonical='{field_result.get('canonical_type')}'"
                        )
                        if field_result.get("failure_details"):
                            logger.error(
                                f"   Details: {field_result['failure_details']}"
                            )

            # Should pass with correct type mappings
            assert result.status == "PASSED"

            # Verify all fields have complete native type information
            assert result.execution_plan is not None
            schema_details = result.execution_plan["schema_details"]
            field_results = schema_details["field_results"]

            assert len(field_results) == len(columns)

            for field_result in field_results:
                # Every field should have complete native type information
                assert field_result["native_type"] is not None
                assert field_result["canonical_type"] is not None
                assert isinstance(field_result["native_metadata"], dict)
                assert field_result["existence"] == "PASSED"
                assert field_result["type"] == "PASSED"
                assert field_result["failure_code"] == "NONE"

                column_name = field_result["column"]
                logger.info(
                    f"✓ {column_name}: native='{field_result['native_type']}', "
                    f"canonical='{field_result['canonical_type']}', "
                    f"metadata={field_result['native_metadata']}"
                )

            logger.info("✅ Comprehensive type coverage test completed successfully")

        finally:
            # Cleanup
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
                "DROP TABLE IF EXISTS comprehensive_test", fetch=False
            )
            await cleanup_engine.dispose()


@pytest.mark.integration
@pytest.mark.database
class TestNativeTypeReportingBackwardCompatibility:
    """Test that native type enhancements maintain backward compatibility."""

    async def _prepare_compatibility_test_environment(
        self, mysql_connection_params: dict
    ) -> QueryExecutor:
        """Prepare MySQL test environment for compatibility testing."""
        from typing import cast

        from shared.database.connection import get_db_url, get_engine

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
        await executor.execute_query("DROP TABLE IF EXISTS compat_test", fetch=False)

        await executor.execute_query(
            """
            CREATE TABLE compat_test (
                id INT PRIMARY KEY AUTO_INCREMENT,
                name VARCHAR(50) NOT NULL,
                status BOOLEAN DEFAULT TRUE
            ) ENGINE=InnoDB
            """,
            fetch=False,
        )

        await engine.dispose()
        return executor

    async def test_existing_functionality_unchanged(
        self, mysql_connection_params: dict
    ) -> None:
        """Test that existing schema validation functionality is unchanged."""
        _skip_if_mysql_unavailable()

        # Prepare test environment
        await self._prepare_compatibility_test_environment(mysql_connection_params)

        # Create connection schema
        connection = ConnectionSchema(
            name="compat_test_connection",
            description="Connection for backward compatibility testing",
            connection_type=ConnectionType.MYSQL,
            host=mysql_connection_params["host"],
            port=mysql_connection_params["port"],
            username=mysql_connection_params["username"],
            password=mysql_connection_params["password"],
            db_name=mysql_connection_params["database"],
        )

        # Create schema executor
        executor = SchemaExecutor(connection, test_mode=True)

        # Use existing schema rule format
        columns = {
            "id": {"expected_type": DataType.INTEGER.value},
            "name": {"expected_type": DataType.STRING.value},
            "status": {
                "expected_type": DataType.INTEGER.value
            },  # BOOLEAN -> INTEGER in MySQL
        }

        rule = build_schema_rule_with_native_reporting(
            columns, "compat_test", mysql_connection_params["database"]
        )

        try:
            # Execute rule
            result = await executor.execute_rule(rule)

            logger.info(f"Backward compatibility test status: {result.status}")

            # Verify existing fields are still present and working
            assert result.status == "PASSED"
            assert result.rule_id == rule.id
            assert len(result.dataset_metrics) == 1

            # Verify execution_plan structure is maintained
            execution_plan = result.execution_plan
            assert execution_plan is not None
            assert "execution_type" in execution_plan
            assert "schema_details" in execution_plan

            schema_details = execution_plan["schema_details"]
            assert "field_results" in schema_details
            assert "extras" in schema_details
            assert "table_exists" in schema_details

            # Verify field_results have expected legacy fields
            field_results = schema_details["field_results"]
            assert len(field_results) == 3

            for field_result in field_results:
                # Legacy fields must be present
                assert "column" in field_result
                assert "existence" in field_result
                assert "type" in field_result
                assert "failure_code" in field_result

                # Enhanced fields should also be present
                assert "native_type" in field_result
                assert "canonical_type" in field_result
                assert "native_metadata" in field_result

                # Values should be meaningful
                assert field_result["existence"] == "PASSED"
                assert field_result["type"] == "PASSED"
                assert field_result["failure_code"] == "NONE"
                assert field_result["native_type"] is not None
                assert field_result["canonical_type"] is not None
                assert isinstance(field_result["native_metadata"], dict)

                logger.info(
                    f"✓ {field_result['column']}: legacy + enhanced fields present"
                )

            logger.info("✅ Backward compatibility test completed successfully")

        finally:
            # Cleanup
            from typing import cast

            from shared.database.connection import get_db_url, get_engine

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
                "DROP TABLE IF EXISTS compat_test", fetch=False
            )
            await cleanup_engine.dispose()
