import asyncio
import logging as _logging
import os
from pathlib import Path
from typing import Any, Dict, List, cast

import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from cli.core.config import CliConfig
from cli.core.data_validator import DataValidator
from core.config import CoreConfig
from core.engine.rule_engine import RuleEngine
from shared.database.connection import check_connection, get_db_url, get_engine
from shared.database.query_executor import QueryExecutor
from shared.enums import RuleAction, RuleCategory, SeverityLevel
from shared.enums.connection_types import ConnectionType
from shared.enums.rule_types import RuleType
from shared.schema.base import RuleTarget, TargetEntity
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.rule_schema import RuleSchema
from shared.utils.logger import get_logger

# Helper utilities must be defined *before* they are used --------------------

# Reduce test noise – elevate global log level to INFO to hide debug logs
_logging.getLogger().setLevel(_logging.INFO)


def make_target(
    table: str,
    column: str,
    database: str = "main",
) -> RuleTarget:
    """Create a RuleTarget for a **single-table** rule following the new schema."""

    return RuleTarget(
        entities=[TargetEntity(database=database, table=table, column=column)]
    )


from shared.utils.logger import get_logger

pytestmark = pytest.mark.asyncio


class TestSQLiteIntegration:
    """Integration tests against a real SQLite database file."""

    async def _prepare_engine(self, db_file: Path) -> AsyncEngine:
        db_url = get_db_url("sqlite", file_path=str(db_file))
        assert await check_connection(db_url) is True, "SQLite connection check failed"
        engine = await get_engine(db_url, echo=False)
        return engine

    async def test_sqlite_basic_crud(self, tmp_path: Path) -> None:
        """Create table, insert rows and query them back using QueryExecutor."""
        db_file = tmp_path / "integration_test.db"
        engine = await self._prepare_engine(db_file)

        executor = QueryExecutor(engine)

        # 1. Create table
        await executor.execute_query(
            """
            CREATE TABLE users (
                id   INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            )
            """,
            fetch=False,
        )

        # 2. Insert sample data (use params to exercise param binding)
        sample_data = [(1, "Alice"), (2, "Bob"), (3, "Charlie")]
        for user_id, name in sample_data:
            await executor.execute_query(
                "INSERT INTO users (id, name) VALUES (:id, :name)",
                params={"id": user_id, "name": name},
                fetch=False,
            )

        # 3. Query all rows back
        results, _ = await executor.execute_query(
            "SELECT id, name FROM users ORDER BY id"
        )
        assert len(results) == 3
        assert results[0] == {"id": 1, "name": "Alice"}
        assert results[2] == {"id": 3, "name": "Charlie"}

        # 4. Count helper
        results, _ = await executor.execute_query("SELECT COUNT(*) FROM users")
        assert results[0]["COUNT(*)"] == 3

        # 5. Parameterised SELECT with LIMIT
        limited_results, _ = await executor.execute_query(
            "SELECT id, name FROM users WHERE id > :min_id",
            params={"min_id": 1},
            sample_limit=1,
        )
        assert len(limited_results) == 1
        assert limited_results[0]["id"] > 1

        # 6. Cleanup
        await engine.dispose()

    # New: Authentic Integration Test Cases

    async def test_rule_engine_integration_with_sqlite(self, tmp_path: Path) -> None:
        """Test rule engine integration with SQLite database"""
        logger = get_logger(__name__)

        # Prepare the test database.
        db_file = tmp_path / "rule_integration_test.db"
        engine = await self._prepare_engine(db_file)
        executor = QueryExecutor(engine)

        # Create a test table.
        await executor.execute_query(
            """
            CREATE TABLE customers_test (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT,
                age INTEGER,
                status TEXT
            )
            """,
            fetch=False,
        )

        # Insert test data (including data with quality issues).
        test_data = [
            (1, "Alice", "alice@example.com", 25, "active"),
            (2, None, "bob@example.com", 30, "active"),  # The name is empty.
            (3, "Charlie", "charlie@example.com", -5, "inactive"),  # Invalid age.
            (4, "Diana", "alice@example.com", 35, "active"),  # Duplicate email address.
            (
                5,
                "Eve",
                "eve@example.com",
                150,
                "active",
            ),  # The age is outside the allowed/valid range.
        ]

        for customer_id, name, email, age, status in test_data:
            await executor.execute_query(
                "INSERT INTO customers_test (id, name, email, age, status) VALUES (:id, :name, :email, :age, :status)",
                params={
                    "id": customer_id,
                    "name": name,
                    "email": email,
                    "age": age,
                    "status": status,
                },
                fetch=False,
            )

        # 2. Create the connection configuration.
        connection = ConnectionSchema(
            name="sqlite_test_connection",
            description="SQLite test connection",
            connection_type=ConnectionType.SQLITE,
            file_path=str(db_file),
            db_name="main",
        )

        # 3. Create the rules.
        rules = [
            RuleSchema(
                id="rule_1",
                name="Name Not Null",
                description="Name cannot be null",
                type=RuleType.NOT_NULL,
                target=make_target("customers_test", "name"),
                parameters={},
                category=RuleCategory.COMPLETENESS,
                severity=SeverityLevel.MEDIUM,
                action=RuleAction.LOG,
            ),
            RuleSchema(
                id="rule_2",
                name="Age Range",
                description="Age must be between 0 and 120",
                type=RuleType.RANGE,
                target=make_target("customers_test", "age"),
                parameters={"min": 0, "max": 120},
                category=RuleCategory.COMPLETENESS,
                severity=SeverityLevel.MEDIUM,
                action=RuleAction.LOG,
            ),
            RuleSchema(
                id="rule_3",
                name="Email Unique",
                description="Email must be unique",
                type=RuleType.UNIQUE,
                target=make_target("customers_test", "email"),
                parameters={},
                category=RuleCategory.COMPLETENESS,
                severity=SeverityLevel.MEDIUM,
                action=RuleAction.LOG,
            ),
        ]

        # 4. Execute rules.
        rule_engine = RuleEngine(connection=connection)
        results = await rule_engine.execute(rules=rules)

        # 5. Validate the results.
        assert len(results) == 3

        # Add debugging information.
        for i, result in enumerate(results):
            logger.info(
                f"Result {i}: rule_id={result.rule_id}, status={result.status}, total_count={result.total_count}, error_count={result.error_count}"
            )

        # Verify the results of the NOT NULL constraint.
        not_null_result = next(r for r in results if r.rule_id == "rule_1")
        assert not_null_result.status == "FAILED"
        assert not_null_result.total_count == 5
        assert (
            not_null_result.error_count == 1
        )  # A record has a blank/empty name field.

        # Verify the results of the RANGE rule.
        range_result = next(r for r in results if r.rule_id == "rule_2")
        assert range_result.status == "FAILED"
        assert range_result.error_count == 2  # Two records have invalid ages.

        # Verify the results of the uniqueness rule (the `rule_id` may be either "rule_3" or "Email Unique").
        unique_result = next(
            (r for r in results if r.rule_id in ("rule_3", "Email Unique")), None
        )
        assert (
            unique_result is not None
        ), f"Not found UNIQUE rule result, actual rule_id list: {[r.rule_id for r in results]}"
        assert unique_result.status == "FAILED"
        assert unique_result.error_count == 1  # A duplicate email address.

        await engine.dispose()
        logger.info("SQLite rule engine integration test completed")

    async def test_csv_to_sqlite_integration(self, tmp_path: Path) -> None:
        """Test CSV import to SQLite and rule validation"""
        logger = get_logger(__name__)

        # Create a test CSV file.
        csv_file = tmp_path / "test_data.csv"
        csv_data = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "", "Charlie", "Diana", "Eve"],
                "email": [
                    "alice@test.com",
                    "bob@test.com",
                    "charlie@test.com",
                    "alice@test.com",
                    "eve@test.com",
                ],
                "age": [25, 30, -5, 35, 150],
            }
        )
        csv_data.to_csv(csv_file, index=False)

        # 2. Create a CSV connection configuration.
        csv_connection = ConnectionSchema(
            name="csv_test_connection",
            description="CSV test connection",
            connection_type=ConnectionType.CSV,
            file_path=str(csv_file),
        )

        # 3. Create the rules.
        rules = [
            RuleSchema(
                id="csv_rule_1",
                name="Name Length",
                description="Name must have at least 1 character",
                type=RuleType.LENGTH,
                target=make_target("test_data", "name"),
                parameters={"min_length": 1},
                category=RuleCategory.VALIDITY,
                severity=SeverityLevel.MEDIUM,
                action=RuleAction.LOG,
            )
        ]

        # 4. Perform integration testing using DataValidator.
        cli_config = CliConfig()
        core_config = CoreConfig()

        validator = DataValidator(
            source_config=csv_connection,
            rules=cast(list, rules),
            cli_config=cli_config,
            core_config=core_config,
        )

        # 5. Perform validation.
        results = await validator.validate()

        # 6. Validate the results.
        assert len(results) == 1
        result = results[0]
        assert result.rule_id == "csv_rule_1"
        assert result.status in ("FAILED", "ERROR")
        assert (
            result.error_count == 1
        )  # A record has a null name (empty strings in the CSV file are imported as NULL).

        logger.info("CSV to SQLite integration test completed")

    async def test_sqlite_large_dataset_performance(self, tmp_path: Path) -> None:
        """Test SQLite performance with large dataset using batch insert"""
        logger = get_logger(__name__)

        # Prepare the large dataset.
        db_file = tmp_path / "large_dataset_test.db"
        engine = await self._prepare_engine(db_file)
        executor = QueryExecutor(engine)

        # Create the table.
        await executor.execute_query(
            """
            CREATE TABLE large_table (
                id INTEGER PRIMARY KEY,
                value TEXT,
                status INTEGER
            )
            """,
            fetch=False,
        )

        # Prepare a large dataset (10,000 records).
        import time

        # Build/Construct all data.  Or, more contextually:  Assemble all data.
        all_data = []
        for i in range(10000):
            all_data.append({"id": i + 1, "value": f"value_{i + 1}", "status": i % 3})

        # Use batch insertion.
        start_time = time.time()
        inserted_count = await executor.execute_batch_insert(
            table_name="large_table",
            data_list=all_data,
            batch_size=1000,
            use_transaction=True,
        )
        insert_time = time.time() - start_time

        logger.info(
            f"Batch inserted {inserted_count} records in {insert_time:.2f} seconds"
        )

        # Verify successful insertion.
        assert inserted_count == 10000

        # 2. Evaluate rule performance.
        connection = ConnectionSchema(
            name="large_dataset_connection",
            description="Large dataset test connection",
            connection_type=ConnectionType.SQLITE,
            file_path=str(db_file),
            db_name="main",
        )

        rule = RuleSchema(
            id="performance_rule",
            name="Status Range",
            description="Status must be 0, 1, or 2",
            type=RuleType.RANGE,
            target=make_target("large_table", "status"),
            parameters={"min": 0, "max": 2},
            category=RuleCategory.VALIDITY,
            severity=SeverityLevel.MEDIUM,
            action=RuleAction.LOG,
        )

        # 3. Execute the rules and measure performance.
        rule_engine = RuleEngine(connection=connection)
        start_time = time.time()
        results = await rule_engine.execute(rules=[rule])
        execution_time = time.time() - start_time

        # 4. Validate performance.
        assert execution_time < 5.0  # Should complete within five seconds.
        assert len(results) == 1
        assert results[0].total_count == 10000
        assert results[0].status == "PASSED"

        await engine.dispose()
        logger.info(
            f"Large dataset rule execution completed in {execution_time:.2f} seconds"
        )

    async def test_sqlite_batch_insert_performance_comparison(
        self, tmp_path: Path
    ) -> None:
        """Compare performance of different insert strategies"""
        logger = get_logger(__name__)

        # Test data size
        test_size = 5000

        # Prepare test data
        test_data = []
        for i in range(test_size):
            test_data.append(
                {
                    "id": i + 1,
                    "name": f"user_{i + 1}",
                    "email": f"user{i + 1}@test.com",
                    "age": 20 + (i % 50),
                }
            )

        # Test 1: Traditional single insert (small sample)
        db_file1 = tmp_path / "single_insert_test.db"
        engine1 = await self._prepare_engine(db_file1)
        executor1 = QueryExecutor(engine1)

        await executor1.execute_query(
            """
            CREATE TABLE users_single (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT,
                age INTEGER
            )
            """,
            fetch=False,
        )

        # Test only first 100 records for single insert to avoid long test time
        single_test_data = test_data[:100]

        import time

        start_time = time.time()
        for record in single_test_data:
            await executor1.execute_query(
                "INSERT INTO users_single (id, name, email, age) VALUES (:id, :name, :email, :age)",
                params=record,
                fetch=False,
            )
        single_insert_time = time.time() - start_time
        logger.info(
            f"Single insert: {len(single_test_data)} records in {single_insert_time:.3f}s ({len(single_test_data)/single_insert_time:.1f} records/sec)"
        )
        await engine1.dispose()

        # Test 2: Batch insert with executemany
        db_file2 = tmp_path / "batch_insert_test.db"
        engine2 = await self._prepare_engine(db_file2)
        executor2 = QueryExecutor(engine2)

        await executor2.execute_query(
            """
            CREATE TABLE users_batch (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT,
                age INTEGER
            )
            """,
            fetch=False,
        )

        start_time = time.time()
        batch_count = await executor2.execute_batch_insert(
            table_name="users_batch",
            data_list=test_data,
            batch_size=1000,
            use_transaction=True,
        )
        batch_insert_time = time.time() - start_time
        logger.info(
            f"Batch insert: {batch_count} records in {batch_insert_time:.3f}s ({batch_count/batch_insert_time:.1f} records/sec)"
        )
        await engine2.dispose()

        # Test 3: Bulk VALUES insert
        db_file3 = tmp_path / "bulk_values_test.db"
        engine3 = await self._prepare_engine(db_file3)
        executor3 = QueryExecutor(engine3)

        await executor3.execute_query(
            """
            CREATE TABLE users_bulk (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT,
                age INTEGER
            )
            """,
            fetch=False,
        )

        start_time = time.time()
        bulk_count = await executor3.execute_bulk_insert_values(
            table_name="users_bulk",
            data_list=test_data,
            batch_size=500,  # Smaller batch size for VALUES method to avoid too large SQL
        )
        bulk_insert_time = time.time() - start_time
        logger.info(
            f"Bulk VALUES insert: {bulk_count} records in {bulk_insert_time:.3f}s ({bulk_count/bulk_insert_time:.1f} records/sec)"
        )
        await engine3.dispose()

        # Verify all methods inserted the correct number of records
        assert batch_count == test_size
        assert bulk_count == test_size

        # Performance assertions
        # Batch and bulk inserts should be significantly faster than single inserts
        # (when scaled to the same number of records)
        estimated_single_time_for_full_dataset = single_insert_time * (
            test_size / len(single_test_data)
        )

        logger.info(f"Performance comparison:")
        logger.info(
            f"  Estimated single insert time for {test_size} records: {estimated_single_time_for_full_dataset:.1f}s"
        )
        logger.info(
            f"  Batch insert time: {batch_insert_time:.3f}s (improvement: {estimated_single_time_for_full_dataset/batch_insert_time:.1f}x)"
        )
        logger.info(
            f"  Bulk VALUES insert time: {bulk_insert_time:.3f}s (improvement: {estimated_single_time_for_full_dataset/bulk_insert_time:.1f}x)"
        )

        # Both batch methods should be at least 10x faster than single insert method
        assert batch_insert_time < estimated_single_time_for_full_dataset / 10
        assert bulk_insert_time < estimated_single_time_for_full_dataset / 10

    async def test_sqlite_concurrent_access(self, tmp_path: Path) -> None:
        """Test SQLite concurrent access handling"""
        logger = get_logger(__name__)

        # Prepare the shared database.
        db_file = tmp_path / "concurrent_test.db"
        engine = await self._prepare_engine(db_file)
        executor = QueryExecutor(engine)

        # Create a test table.
        await executor.execute_query(
            """
            CREATE TABLE concurrent_table (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            fetch=False,
        )

        # Insert test data.
        for i in range(100):
            await executor.execute_query(
                "INSERT INTO concurrent_table (id, name) VALUES (:id, :name)",
                params={"id": i + 1, "name": f"record_{i + 1}"},
                fetch=False,
            )

        await engine.dispose()

        # 2. Create concurrent rule execution tasks.
        async def execute_rule_task(task_id: int) -> Dict[str, Any]:
            connection = ConnectionSchema(
                name=f"concurrent_connection_{task_id}",
                description=f"Concurrent test connection {task_id}",
                connection_type=ConnectionType.SQLITE,
                file_path=str(db_file),
                db_name="main",
            )

            rule = RuleSchema(
                id=f"concurrent_rule_{task_id}",
                name=f"Name Not Null {task_id}",
                description="Name cannot be null",
                type=RuleType.NOT_NULL,
                target=make_target("concurrent_table", "name"),
                parameters={},
                category=RuleCategory.COMPLETENESS,
                severity=SeverityLevel.MEDIUM,
                action=RuleAction.LOG,
            )

            rule_engine = RuleEngine(connection=connection)
            results = await rule_engine.execute(rules=[rule])
            return {"task_id": task_id, "results": results}

        # 3. Execute multiple rules concurrently.
        import asyncio

        tasks = [execute_rule_task(i) for i in range(5)]
        concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 4. Verify the results of concurrent execution.
        successful_results = [
            r for r in concurrent_results if not isinstance(r, Exception)
        ]
        assert len(successful_results) == 5  # All tasks must complete successfully.

        for result in successful_results:
            assert isinstance(result, dict)
            assert result is not None
            assert len(result["results"]) == 1
            assert result["results"][0].total_count == 100
            assert result["results"][0].status == "PASSED"

        logger.info("SQLite concurrent access test completed")

    async def test_sqlite_error_handling(self, tmp_path: Path) -> None:
        """Test SQLite error handling scenarios"""
        logger = get_logger(__name__)

        # Verify the database file does not exist.
        nonexistent_file = tmp_path / "nonexistent.db"

        connection = ConnectionSchema(
            name="nonexistent_connection",
            description="Nonexistent database connection",
            connection_type=ConnectionType.SQLITE,
            file_path=str(nonexistent_file),
            db_name="main",
        )

        rule = RuleSchema(
            id="error_rule",
            name="Test Rule",
            description="Test rule for error handling",
            type=RuleType.NOT_NULL,
            target=make_target("nonexistent_table", "nonexistent_column"),
            parameters={},
            category=RuleCategory.COMPLETENESS,
            severity=SeverityLevel.MEDIUM,
            action=RuleAction.LOG,
        )

        # The rules engine should handle errors gracefully.
        rule_engine = RuleEngine(connection=connection)

        # Execution should complete without crashing, but will return incorrect results.
        try:
            results = await rule_engine.execute(rules=[rule])
            # Verify the results for errors if no exception is thrown.
            assert len(results) == 1
            result = results[0]
            assert result.status in ("ERROR", "FAILED")
            assert result.error_message is not None
        except Exception as e:
            # This is the expected error handling behavior, even if an exception is raised.
            assert "not found" in str(e) or "failed" in str(e).lower()

        logger.info("SQLite error handling test completed")

    async def test_csv_to_sqlite_performance_optimization(self, tmp_path: Path) -> None:
        """Test CSV to SQLite conversion performance with batch insert optimization"""
        logger = get_logger(__name__)

        # 1. Create a large CSV file for testing purposes.
        csv_file = tmp_path / "large_performance_test.csv"

        # Generate a larger test dataset consisting of 10,000 records.
        import time

        import pandas as pd

        data_size = 10000
        test_data = {
            "id": range(1, data_size + 1),
            "name": [f"customer_{i}" for i in range(1, data_size + 1)],
            "email": [f"customer{i}@test.com" for i in range(1, data_size + 1)],
            "age": [(20 + (i % 60)) for i in range(data_size)],
            "city": [f"city_{i % 100}" for i in range(data_size)],
            "salary": [(30000 + (i * 100)) for i in range(data_size)],
            "department": [f"dept_{i % 10}" for i in range(data_size)],
            "join_date": [
                f"2020-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(data_size)
            ],
        }

        df = pd.DataFrame(test_data)
        df.to_csv(csv_file, index=False)

        logger.info(f"Created test CSV file with {data_size} records")

        # 2. Evaluate the performance of the CSV to SQLite conversion.
        from cli.core.config import CliConfig
        from cli.core.data_validator import DataValidator
        from core.config import CoreConfig
        from shared.enums.connection_types import ConnectionType
        from shared.schema.connection_schema import ConnectionSchema

        # Create a CSV connection configuration.
        csv_connection = ConnectionSchema(
            name="performance_test_csv",
            description="Performance test CSV connection",
            connection_type=ConnectionType.CSV,
            file_path=str(csv_file),
        )

        # Create a minimal set of rules (for testing transformation performance, not rule performance).
        rules: List[RuleSchema] = []

        # Create a DataValidator (class/object/instance).
        cli_config = CliConfig()
        core_config = CoreConfig()

        validator = DataValidator(
            source_config=csv_connection,
            rules=cast(list, rules),
            cli_config=cli_config,
            core_config=core_config,
        )

        # 3. Measure CSV conversion performance.
        conversion_start = time.time()

        # Load CSV data.
        df_loaded = validator._load_file_data()
        assert len(df_loaded) == data_size

        # Migrating to SQLite (using our optimized bulk insertion method).
        sqlite_config = await validator._convert_file_to_sqlite(df_loaded)

        conversion_time = time.time() - conversion_start

        logger.info(f"CSV to SQLite conversion completed:")
        logger.info(f"  - Records: {data_size}")
        logger.info(f"  - Conversion time: {conversion_time:.3f}s")
        logger.info(f"  - Records per second: {data_size/conversion_time:.1f}")

        # 4. Validate the conversion results.
        # Connect to the generated SQLite database and verify the data.
        from shared.database.connection import get_engine
        from shared.database.query_executor import QueryExecutor

        sqlite_engine = await get_engine(
            f"sqlite+aiosqlite:///{sqlite_config.file_path}"
        )
        executor = QueryExecutor(sqlite_engine)

        # Retrieve the table name.
        table_name = sqlite_config.parameters.get("table", "large_performance_test")

        # Verify the record count.
        results, _ = await executor.execute_query(f"SELECT COUNT(*) FROM {table_name}")
        assert results[0]["COUNT(*)"] == data_size

        # Verify data integrity (using a sample check).
        sample_results, _ = await executor.execute_query(
            f"SELECT * FROM {table_name} WHERE id IN (1, 5000, 10000) ORDER BY id"
        )

        assert len(sample_results) == 3
        assert sample_results[0]["name"] == "customer_1"
        assert sample_results[1]["id"] == 5000
        assert sample_results[2]["email"] == "customer10000@test.com"

        await sqlite_engine.dispose()

        # Performance Assertions
        # Processing 10,000 records should complete within a reasonable timeframe, given our batch insertion optimizations.
        # Batch insertion should now complete within 10 seconds (previously, this operation could take several minutes).
        max_expected_time = 10.0  # Seconds.
        assert (
            conversion_time < max_expected_time
        ), f"Conversion took {conversion_time:.3f}s, expected < {max_expected_time}s"

        # The records processed per second should be significantly increased.
        min_records_per_sec = (
            500  # A minimum throughput of 500 records per second is expected.
        )
        actual_records_per_sec = data_size / conversion_time
        assert (
            actual_records_per_sec >= min_records_per_sec
        ), f"Performance: {actual_records_per_sec:.1f} records/sec, expected >= {min_records_per_sec}"

        logger.info(
            "CSV to SQLite performance test passed - batch insert optimization verified"
        )

        # 6. Clean up temporary files.
        import os

        assert sqlite_config.file_path is not None
        if os.path.exists(sqlite_config.file_path):
            os.unlink(sqlite_config.file_path)
