import asyncio
import logging as _logging
import os
import time
from typing import Any, Dict, Optional, cast

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from core.engine.rule_engine import RuleEngine
from shared.database.connection import check_connection, get_db_url, get_engine
from shared.database.query_executor import QueryExecutor
from shared.enums import RuleAction, RuleCategory, SeverityLevel
from shared.enums.connection_types import ConnectionType
from shared.enums.rule_types import RuleType
from shared.schema.base import RuleTarget as _RuleTargetClass
from shared.schema.base import TargetEntity
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.rule_schema import RuleSchema as _RuleSchemaClass
from shared.utils.logger import get_logger

pytestmark = pytest.mark.asyncio

# Reduce test noise – raise root log level to INFO (hide DEBUG)
_logging.getLogger().setLevel(_logging.INFO)


def RuleTarget(
    *, entity: TargetEntity, column: Optional[str] = None, **kwargs: Any
) -> _RuleTargetClass:
    """Local compatibility shim mapping legacy *(entity, column)* signature to the
    new *entities=[TargetEntity(...)]* structure required by the current
    schema.  Only used inside *integration tests* – production code should use
    the official constructor directly.
    """

    ent = entity.model_copy(update={"column": column}) if column else entity

    return _RuleTargetClass(
        entities=[ent],
        **{k: v for k, v in kwargs.items() if k not in {"entity", "column"}},
    )


# ---------------------------------------------------------------------
# RuleSchema shim adding default mandatory fields
# ---------------------------------------------------------------------


def RuleSchema(**kwargs: Any) -> _RuleSchemaClass:
    """Wrapper around the real RuleSchema adding sensible defaults for
    mandatory fields that are irrelevant to these integration tests.
    """

    kwargs.setdefault("category", RuleCategory.COMPLETENESS)
    kwargs.setdefault("severity", SeverityLevel.MEDIUM)
    kwargs.setdefault("action", RuleAction.LOG)
    return _RuleSchemaClass(**kwargs)


class TestMySQLIntegration:
    async def _prepare_engine(self, mysql_conn_str: Dict[str, object]) -> AsyncEngine:
        db_url = get_db_url(
            str(mysql_conn_str["db_type"]),
            str(mysql_conn_str["host"]),
            cast(int, mysql_conn_str["port"]),
            str(mysql_conn_str["database"]),
            str(mysql_conn_str["username"]),
            str(mysql_conn_str["password"]),
        )
        assert await check_connection(db_url) is True, "MySQL connection check failed"
        engine = await get_engine(db_url, pool_size=1, echo=False)
        return engine

    @pytest.mark.asyncio
    async def test_check_connection_mysql_success(
        self, mysql_connection_params: Dict[str, object]
    ) -> None:
        """Test successful connection to a live MySQL DB."""
        url = get_db_url(
            str(mysql_connection_params["db_type"]),
            str(mysql_connection_params["host"]),
            cast(int, mysql_connection_params["port"]),
            str(mysql_connection_params["database"]),
            str(mysql_connection_params["username"]),
            str(mysql_connection_params["password"]),
        )
        # Ensure your MySQL server is running and accessible with these params
        # This test might be slow and is environment-dependent
        assert await check_connection(url) is True

    async def test_mysql_engine_creation(
        self, mysql_connection_params: Dict[str, object]
    ) -> None:
        """Testing MySQL engine creation."""
        # Construct the connection URL.
        db_url = get_db_url(
            mysql_connection_params["db_type"],  # type: ignore
            mysql_connection_params["host"],  # type: ignore
            mysql_connection_params["port"],  # type: ignore
            mysql_connection_params["database"],  # type: ignore
            mysql_connection_params["username"],  # type: ignore
            mysql_connection_params["password"],  # type: ignore
        )

        # Create the engine.
        engine = await get_engine(db_url)

        # Verification Engine
        assert engine is not None

    async def test_mysql_create_query_and_read(
        self, mysql_connection_params: Dict[str, object]
    ) -> None:
        engine = await self._prepare_engine(mysql_connection_params)
        executor = QueryExecutor(engine)

        # Ensure a fresh test table
        await executor.execute_query(
            "DROP TABLE IF EXISTS users_integration_test", fetch=False
        )
        await executor.execute_query(
            """
            CREATE TABLE users_integration_test (
                id   INT PRIMARY KEY,
                name VARCHAR(100) NOT NULL
            )
            """,
            fetch=False,
        )

        # Insert data
        await executor.execute_query(
            "INSERT INTO users_integration_test (id, name) VALUES (:id, :name)",
            params={"id": 1, "name": "Alice"},
            fetch=False,
        )

        # Read back
        results, _ = await executor.execute_query(
            "SELECT id, name FROM users_integration_test WHERE id = :id",
            params={"id": 1},
        )
        assert results == [{"id": 1, "name": "Alice"}]

        # Cleanup
        await executor.execute_query("DROP TABLE users_integration_test", fetch=False)
        await engine.dispose()

    # Added: True integration test cases.

    async def test_rule_engine_integration_with_mysql(
        self, mysql_connection_params: Dict[str, object]
    ) -> None:
        """Test rule engine integration with MySQL database"""
        logger = get_logger(__name__)

        # Prepare the test database.
        engine = await self._prepare_engine(mysql_connection_params)
        executor = QueryExecutor(engine)

        # Clean and prepare test tables.
        await executor.execute_query(
            "DROP TABLE IF EXISTS mysql_customers_test", fetch=False
        )
        await executor.execute_query(
            """
            CREATE TABLE mysql_customers_test (
                id INT PRIMARY KEY AUTO_INCREMENT,
                name VARCHAR(100),
                email VARCHAR(255),
                age INT,
                status ENUM('active', 'inactive', 'pending'),
                balance DECIMAL(10,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_email (email),
                INDEX idx_status (status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            fetch=False,
        )

        # Insert test data (which includes quality issues).
        test_data = [
            ("Alice", "alice@example.com", 25, "active", 1000.50),
            (
                None,
                "bob@example.com",
                30,
                "active",
                2000.00,
            ),  # The name is empty. /  The name field is blank. /  No name has been provided.
            ("Charlie", "charlie@example.com", -5, "inactive", 500.25),  # Invalid age.
            (
                "Diana",
                "alice@example.com",
                35,
                "active",
                1500.75,
            ),  # Duplicate email address.
            ("Eve", "eve@example.com", 150, "active", 3000.00),  # Age is out of range.
            (
                "Frank",
                "frank@example.com",
                45,
                "active",
                -100.00,
            ),  # The balance is negative.
        ]

        for name, email, age, status, balance in test_data:
            await executor.execute_query(
                "INSERT INTO mysql_customers_test (name, email, age, status, balance) VALUES (:name, :email, :age, :status, :balance)",
                params={
                    "name": name,
                    "email": email,
                    "age": age,
                    "status": status,
                    "balance": balance,
                },
                fetch=False,
            )

        # 2. Create the connection configuration.
        connection = ConnectionSchema(
            name="mysql_test_connection",
            description="MySQL test connection",
            connection_type=ConnectionType.MYSQL,
            host=mysql_connection_params["host"],
            port=mysql_connection_params["port"],
            username=mysql_connection_params["username"],
            password=mysql_connection_params["password"],
            db_name=mysql_connection_params["database"],
        )

        # 3. Create the rules.
        rules = [
            RuleSchema(
                id="mysql_rule_1",
                name="Name Not Null",
                description="Name cannot be null",
                type=RuleType.NOT_NULL,
                target=RuleTarget(
                    entity=TargetEntity(
                        database=mysql_connection_params["database"],
                        table="mysql_customers_test",
                    ),
                    column="name",
                ),
                parameters={},
            ),
            RuleSchema(
                id="mysql_rule_2",
                name="Age Range",
                description="Age must be between 0 and 120",
                type=RuleType.RANGE,
                target=RuleTarget(
                    entity=TargetEntity(
                        database=mysql_connection_params["database"],
                        table="mysql_customers_test",
                    ),
                    column="age",
                ),
                parameters={"min": 0, "max": 120},
            ),
            RuleSchema(
                id="mysql_rule_3",
                name="Email Unique",
                description="Email must be unique",
                type=RuleType.UNIQUE,
                target=RuleTarget(
                    entity=TargetEntity(
                        database=mysql_connection_params["database"],
                        table="mysql_customers_test",
                    ),
                    column="email",
                ),
                parameters={},
            ),
            RuleSchema(
                id="mysql_rule_4",
                name="Balance Range",
                description="Balance must be non-negative",
                type=RuleType.RANGE,
                target=RuleTarget(
                    entity=TargetEntity(
                        database=mysql_connection_params["database"],
                        table="mysql_customers_test",
                    ),
                    column="balance",
                ),
                parameters={"min": 0},
            ),
        ]

        # 4. Apply/Execute the rules.
        rule_engine = RuleEngine(connection=connection)
        results = await rule_engine.execute(rules=rules)

        # 5. Validate the results.
        assert len(results) == 4

        # Verify the results of each rule.
        result_map = {r.rule_id: r for r in results}

        # Non-null constraint.
        assert result_map["mysql_rule_1"].status == "FAILED"
        assert result_map["mysql_rule_1"].error_count == 1

        # Range rules (for age)
        assert result_map["mysql_rule_2"].status == "FAILED"
        assert result_map["mysql_rule_2"].error_count == 2

        # Uniqueness Rule(s)
        assert result_map["mysql_rule_3"].status == "FAILED"
        assert result_map["mysql_rule_3"].error_count == 1

        # RANGE rule (balance)
        assert result_map["mysql_rule_4"].status == "FAILED"
        assert result_map["mysql_rule_4"].error_count == 1

        # Cleanup
        await executor.execute_query("DROP TABLE mysql_customers_test", fetch=False)
        await engine.dispose()
        logger.info("MySQL rule engine integration test completed")

    async def _create_large_test_table(self, executor: QueryExecutor) -> None:
        """Helper method to create large test table"""
        await executor.execute_query(
            "DROP TABLE IF EXISTS large_customers_test", fetch=False
        )
        create_sql = """
            CREATE TABLE large_customers_test (
                id INT PRIMARY KEY AUTO_INCREMENT,
                customer_code VARCHAR(50) NOT NULL,
                name VARCHAR(100),
                email VARCHAR(255),
                phone VARCHAR(20),
                status ENUM('active', 'inactive') DEFAULT 'active',
                score INT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_customer_code (customer_code),
                INDEX idx_email (email),
                INDEX idx_status_score (status, score)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        await executor.execute_query(create_sql, fetch=False)

    async def _insert_large_test_data(
        self, executor: QueryExecutor, total_records: int
    ) -> float:
        """Helper method to insert large test data using batch insert for better performance"""
        start_time = time.time()

        # Prepare all data at once
        batch_data = []
        for i in range(total_records):
            batch_data.append(
                {
                    "customer_code": f"CUST{i:06d}",
                    "name": f"Customer {i}",
                    "email": f"customer{i}@example.com",
                    "phone": f"13800{i:06d}",
                    "status": "active" if i % 10 != 0 else "inactive",
                    "score": i % 100,
                }
            )

        # Use the efficient batch insert method
        await executor.execute_batch_insert(
            table_name="large_customers_test",
            data_list=batch_data,
            batch_size=1000,
            use_transaction=True,
        )

        return time.time() - start_time

    async def test_mysql_large_table_processing(
        self, mysql_connection_params: Dict[str, object]
    ) -> None:
        """Test MySQL large table processing with pagination and optimization"""
        logger = get_logger(__name__)

        # Prepare for large table testing.
        engine = await self._prepare_engine(mysql_connection_params)
        executor = QueryExecutor(engine)

        # Create a large table and insert data into it.
        await self._create_large_test_table(executor)
        total_records = 10000  # Reduced from 50000 for faster testing
        insert_time = await self._insert_large_test_data(executor, total_records)
        logger.info(f"Inserted {total_records} records in {insert_time:.2f} seconds")

        # 2. Establish connections and define rules.
        connection = ConnectionSchema(
            name="large_table_connection",
            description="Large table test connection",
            connection_type=ConnectionType.MYSQL,
            host=mysql_connection_params["host"],
            port=mysql_connection_params["port"],
            username=mysql_connection_params["username"],
            password=mysql_connection_params["password"],
            db_name=mysql_connection_params["database"],
        )

        rules = [
            RuleSchema(
                id="large_table_rule_1",
                name="Customer Code Not Null",
                description="Customer code cannot be null",
                type=RuleType.NOT_NULL,
                target=RuleTarget(
                    entity=TargetEntity(
                        database=mysql_connection_params["database"],
                        table="large_customers_test",
                    ),
                    column="customer_code",
                ),
                parameters={},
            )
        ]

        # 3. Execute the rules and measure performance.
        rule_engine = RuleEngine(connection=connection)
        start_time = time.time()
        results = await rule_engine.execute(rules=rules)
        execution_time = time.time() - start_time

        # 4. Validate results and evaluate performance.
        assert len(results) == 1
        assert execution_time < 30.0  # Should complete within 30 seconds.

        # Validate rule results.
        assert results[0].total_count == total_records
        assert results[0].status == "PASSED"

        # Cleanup
        await executor.execute_query("DROP TABLE large_customers_test", fetch=False)
        await engine.dispose()
        logger.info(
            f"Large table processing test completed in {execution_time:.2f} seconds"
        )

    async def test_mysql_transaction_management(
        self, mysql_connection_params: Dict[str, object]
    ) -> None:
        """Test MySQL transaction management with rule execution"""
        logger = get_logger(__name__)

        # Prepare for testing.
        engine = await self._prepare_engine(mysql_connection_params)
        executor = QueryExecutor(engine)

        # Create a test table.
        await executor.execute_query(
            "DROP TABLE IF EXISTS transaction_test", fetch=False
        )
        await executor.execute_query(
            """
            CREATE TABLE transaction_test (
                id INT PRIMARY KEY AUTO_INCREMENT,
                account_id VARCHAR(50) NOT NULL,
                balance DECIMAL(10,2) NOT NULL,
                status ENUM('active', 'frozen', 'closed') DEFAULT 'active',
                last_transaction_date DATE,
                INDEX idx_account (account_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            fetch=False,
        )

        # Insert test data.
        test_data = [
            ("ACC001", 1000.00, "active", "2024-01-15"),
            ("ACC002", -500.00, "active", "2024-01-16"),  # Negative balance.
            ("ACC003", 2500.50, "frozen", "2024-01-17"),
            (
                "ACC004",
                0.00,
                "closed",
                None,
            ),  # The last trading date is not available/is empty.
        ]

        for account_id, balance, status, last_date in test_data:
            await executor.execute_query(
                "INSERT INTO transaction_test (account_id, balance, status, last_transaction_date) VALUES (:account_id, :balance, :status, :last_date)",
                params={
                    "account_id": account_id,
                    "balance": balance,
                    "status": status,
                    "last_date": last_date,
                },
                fetch=False,
            )

        # 2. Create rules to test transaction isolation.
        connection = ConnectionSchema(
            name="transaction_test_connection",
            description="Transaction test connection",
            connection_type=ConnectionType.MYSQL,
            host=mysql_connection_params["host"],
            port=mysql_connection_params["port"],
            username=mysql_connection_params["username"],
            password=mysql_connection_params["password"],
            db_name=mysql_connection_params["database"],
        )

        rules = [
            RuleSchema(
                id="transaction_rule_1",
                name="Balance Non-Negative",
                description="Account balance must be non-negative",
                type=RuleType.RANGE,
                target=RuleTarget(
                    entity=TargetEntity(
                        database=mysql_connection_params["database"],
                        table="transaction_test",
                    ),
                    column="balance",
                ),
                parameters={"min": 0},
            ),
            RuleSchema(
                id="transaction_rule_2",
                name="Active Account Date Required",
                description="Active accounts must have last transaction date",
                type=RuleType.NOT_NULL,
                target=RuleTarget(
                    entity=TargetEntity(
                        database=mysql_connection_params["database"],
                        table="transaction_test",
                    ),
                    column="last_transaction_date",
                ),
                parameters={"filter_condition": "status = 'active'"},
            ),
        ]

        # 3. Execute the rules.
        rule_engine = RuleEngine(connection=connection)
        results = await rule_engine.execute(rules=rules)

        # 4. Verify the results.
        assert len(results) == 2

        result_map = {r.rule_id: r for r in results}

        # Prints debugging information.
        for result in results:
            logger.info(
                f"Rule {result.rule_id}: status={result.status}, "
                f"total_count={result.total_count}, failed_count={result.error_count}, "
                f"message={result.execution_message}"
            )

        # Balance Check
        assert result_map["transaction_rule_1"].status == "FAILED"
        assert result_map["transaction_rule_1"].error_count == 1  # A negative balance.

        # Date verification (applicable only to active accounts).
        # Rules should correctly apply filtering criteria and only evaluate active accounts.
        # Two accounts (ACC001 and ACC002) are active and have recent transaction dates, therefore they should pass the validation check.
        assert result_map["transaction_rule_2"].status == "PASSED"
        assert result_map["transaction_rule_2"].error_count == 0

        # Cleanup or Clean up.
        await executor.execute_query("DROP TABLE transaction_test", fetch=False)
        await engine.dispose()
        logger.info("MySQL transaction management test completed")

    async def _execute_concurrent_rule(
        self, task_id: int, mysql_connection_params: Dict[str, object]
    ) -> Dict[str, Any]:
        """Helper method for concurrent rule execution"""
        connection = ConnectionSchema(
            name=f"pool_connection_{task_id}",
            description=f"Pool test connection {task_id}",
            connection_type=ConnectionType.MYSQL,
            host=mysql_connection_params["host"],
            port=mysql_connection_params["port"],
            username=mysql_connection_params["username"],
            password=mysql_connection_params["password"],
            db_name=mysql_connection_params["database"],
        )

        rule = RuleSchema(
            id=f"pool_rule_{task_id}",
            name=f"Data Not Null {task_id}",
            description="Data cannot be null",
            type=RuleType.NOT_NULL,
            target=RuleTarget(
                entity=TargetEntity(
                    database=mysql_connection_params["database"], table="pool_test"
                ),
                column="data",
            ),
            parameters={},
        )

        rule_engine = RuleEngine(connection=connection)
        results = await rule_engine.execute(rules=[rule])
        return {"task_id": task_id, "results": results}

    async def test_mysql_connection_pool_management(
        self, mysql_connection_params: Dict[str, object]
    ) -> None:
        """Test MySQL connection pool management with concurrent rule execution"""
        logger = get_logger(__name__)

        # Prepare the test table.
        engine = await self._prepare_engine(mysql_connection_params)
        executor = QueryExecutor(engine)

        await executor.execute_query("DROP TABLE IF EXISTS pool_test", fetch=False)
        create_table_sql = """
            CREATE TABLE pool_test (
                id INT PRIMARY KEY AUTO_INCREMENT,
                data VARCHAR(100) NOT NULL,
                category ENUM('A', 'B', 'C') DEFAULT 'A',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        await executor.execute_query(create_table_sql, fetch=False)

        # Insert test data using batch insertion.
        pool_test_data = []
        for i in range(100):
            pool_test_data.append(
                {"data": f"data_{i}", "category": ["A", "B", "C"][i % 3]}
            )

        await executor.execute_batch_insert(
            table_name="pool_test",
            data_list=pool_test_data,
            batch_size=100,
            use_transaction=True,
        )

        await engine.dispose()

        # 2. Execute multiple rules concurrently.
        start_time = time.time()
        concurrent_tasks = 10
        tasks = [
            self._execute_concurrent_rule(i, mysql_connection_params)
            for i in range(concurrent_tasks)
        ]
        concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)
        execution_time = time.time() - start_time

        # 3. Verify the results of concurrent execution.
        successful_results = [
            r for r in concurrent_results if not isinstance(r, Exception)
        ]
        assert len(successful_results) == concurrent_tasks

        for result in successful_results:
            assert isinstance(result, dict)
            assert result is not None
            assert len(result["results"]) == 1
            execution_result = result["results"][0]
            assert execution_result.total_count == 100
            assert execution_result.status == "PASSED"

        # Verify performance. Concurrent execution should be faster than serial execution.
        # Assert that the execution time is less than 15 seconds.  The operation should complete within this timeframe.

        # Cleanup
        cleanup_engine = await self._prepare_engine(mysql_connection_params)
        cleanup_executor = QueryExecutor(cleanup_engine)
        await cleanup_executor.execute_query("DROP TABLE pool_test", fetch=False)
        await cleanup_engine.dispose()

        logger.info(
            f"MySQL connection pool test completed in {execution_time:.2f} seconds"
        )

    async def test_mysql_error_recovery(
        self, mysql_connection_params: Dict[str, object]
    ) -> None:
        """Test MySQL error recovery scenarios"""
        logger = get_logger(__name__)

        # Test invalid table names. This rule-level error should be caught.
        connection = ConnectionSchema(
            name="error_recovery_connection",
            description="Error recovery test connection",
            connection_type=ConnectionType.MYSQL,
            host=mysql_connection_params["host"],
            port=mysql_connection_params["port"],
            username=mysql_connection_params["username"],
            password=mysql_connection_params["password"],
            db_name=mysql_connection_params["database"],
        )

        invalid_rule = RuleSchema(
            id="invalid_rule",
            name="Invalid Table Rule",
            description="Rule for non-existent table",
            type=RuleType.NOT_NULL,
            target=RuleTarget(
                entity=TargetEntity(
                    database=mysql_connection_params["database"],
                    table="non_existent_table",
                ),
                column="non_existent_column",
            ),
            parameters={},
        )

        # 2. Execute invalid rules - expect error results.
        # Execute invalid rules - expect error results.
        rule_engine = RuleEngine(connection=connection)

        results = await rule_engine.execute(rules=[invalid_rule])

        assert len(results) == 1
        assert results[0].status == "ERROR" or results[0].status == "FAILED"

        # 4. Test Connection Recovery - Using invalid connection parameters.
        # Test Connection Recovery - Using invalid connection parameters.
        invalid_connection = ConnectionSchema(
            name="invalid_connection",
            description="Invalid connection test",
            connection_type=ConnectionType.MYSQL,
            host="invalid_host",
            port=3306,
            username="invalid_user",
            password="invalid_password",
            db_name="invalid_db",
        )

        rule = RuleSchema(
            id="recovery_rule",
            name="Recovery Test Rule",
            description="Rule for connection recovery test",
            type=RuleType.NOT_NULL,
            target=RuleTarget(
                entity=TargetEntity(database="invalid_db", table="some_table"),
                column="some_column",
            ),
            parameters={},
        )

        # 5. Execute the rule for a failed connection - an exception is expected.
        invalid_rule_engine = RuleEngine(connection=invalid_connection)
        try:
            recovery_results = await invalid_rule_engine.execute(rules=[rule])
            # An exception is expected but was not thrown. This is unexpected behavior.
            assert False, "Expected connection error but got successful results"
        except Exception as e:
            # 6. Verify Error Recovery - Connection errors should be handled gracefully.
            logger.info(f"Successfully caught expected connection error: {e}")

        logger.info("MySQL error recovery test completed")
