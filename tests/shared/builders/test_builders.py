"""
🏗️ Test Data Builders - Schema Builder Pattern Implementation

As the Testing Ghost 👻, I eliminate fixture duplication and provide fluent interfaces
for creating test data. This module can be imported and used across all test files.
"""

import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from shared.enums import RuleAction, RuleCategory, RuleType, SeverityLevel
from shared.enums.connection_types import ConnectionType
from shared.schema.base import RuleTarget, TargetEntity
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.result_schema import ExecutionResultSchema
from shared.schema.rule_schema import RuleSchema


class TestDataBuilder:
    """
    Master builder class following Builder Pattern for test data creation.

    Benefits:
    1. Eliminates repetitive fixture code
    2. Provides fluent interface for readable test setup
    3. Ensures valid default values
    4. Reduces maintenance when schemas evolve
    """

    class RuleBuilder:
        """
        Builder for RuleSchema with fluent interface
        """

        def __init__(self) -> None:
            self._reset()

        def _reset(self) -> None:
            """Reset to default values for reuse"""
            self._id = str(uuid.uuid4())
            self._name = "test_rule"
            self._description = "Test rule description"
            self._connection_id = uuid.uuid4()
            self._type = RuleType.ENUM
            self._category = RuleCategory.VALIDITY
            self._severity = SeverityLevel.MEDIUM
            self._action = RuleAction.ALERT
            self._threshold = 0.0
            self._template_id: Optional[str] = None
            self._is_active = True
            self._tags: List[str] = []
            self._database = "test_db"
            self._table = "test_table"
            self._column = "test_column"
            self._relationship_type = "single_table"
            self._parameters: Dict[str, Any] = {}

        def with_name(self, name: str) -> "TestDataBuilder.RuleBuilder":
            self._name = name
            self._id = name  # Set ID to name for consistent lookup in tests
            return self

        def with_type(self, rule_type: RuleType) -> "TestDataBuilder.RuleBuilder":
            self._type = rule_type
            return self

        def with_severity(
            self, severity: SeverityLevel
        ) -> "TestDataBuilder.RuleBuilder":
            self._severity = severity
            return self

        def with_target(
            self, database: str, table: str, column: str
        ) -> "TestDataBuilder.RuleBuilder":
            self._database = database
            self._table = table
            self._column = column
            return self

        def as_enum_rule(
            self, allowed_values: List[str]
        ) -> "TestDataBuilder.RuleBuilder":
            """Configure as ENUM rule"""
            self._type = RuleType.ENUM
            self._parameters["allowed_values"] = allowed_values
            return self

        def as_not_null_rule(self) -> "TestDataBuilder.RuleBuilder":
            """Configure as NOT NULL rule"""
            self._type = RuleType.NOT_NULL
            self._category = RuleCategory.COMPLETENESS
            return self

        def as_unique_rule(self) -> "TestDataBuilder.RuleBuilder":
            """Configure as UNIQUE rule"""
            self._type = RuleType.UNIQUE
            self._category = RuleCategory.UNIQUENESS
            return self

        def as_range_rule(
            self, min_val: Optional[float] = None, max_val: Optional[float] = None
        ) -> "TestDataBuilder.RuleBuilder":
            """Configure as RANGE rule"""
            self._type = RuleType.RANGE
            self._category = RuleCategory.VALIDITY
            # Ensure at least one parameter is set for RANGE rules
            if min_val is None and max_val is None:
                min_val = 0  # Default minimum value
                max_val = 100  # Default maximum value
            if min_val is not None:
                self._parameters["min"] = min_val
            if max_val is not None:
                self._parameters["max"] = max_val
            return self

        def as_regex_rule(self, pattern: str) -> "TestDataBuilder.RuleBuilder":
            """Configure as REGEX rule"""
            self._type = RuleType.REGEX
            self._category = RuleCategory.VALIDITY
            self._parameters["pattern"] = pattern
            return self

        def as_length_rule(
            self, min_length: Optional[int] = None, max_length: Optional[int] = None
        ) -> "TestDataBuilder.RuleBuilder":
            """Configure as LENGTH rule"""
            self._type = RuleType.LENGTH
            self._category = RuleCategory.COMPLETENESS
            if min_length is not None:
                self._parameters["min_length"] = min_length
            if max_length is not None:
                self._parameters["max_length"] = max_length
            return self

        def as_date_format_rule(
            self, date_format: str
        ) -> "TestDataBuilder.RuleBuilder":
            """Configure as DATE_FORMAT rule"""
            self._type = RuleType.DATE_FORMAT
            self._category = RuleCategory.VALIDITY
            self._parameters["format"] = date_format
            return self

        def with_filter(self, filter_condition: str) -> "TestDataBuilder.RuleBuilder":
            self._parameters["filter_condition"] = filter_condition
            return self

        def with_email_domain_extraction(self) -> "TestDataBuilder.RuleBuilder":
            """For ENUM rules that extract email domains"""
            self._parameters["extract_domain"] = True
            return self

        def with_parameter(self, key: str, value: Any) -> "TestDataBuilder.RuleBuilder":
            """Add custom parameter"""
            self._parameters[key] = value
            return self

        def build(self) -> RuleSchema:
            """Build the final RuleSchema"""
            # Ensure mandatory parameters for certain rule types are present
            self._finalise_length_parameters()

            rule = RuleSchema(
                id=self._id,
                name=self._name,
                description=self._description,
                connection_id=self._connection_id,
                type=self._type,
                category=self._category,
                severity=self._severity,
                action=self._action,
                threshold=self._threshold,
                template_id=self._template_id,
                is_active=self._is_active,
                tags=self._tags,
                target=RuleTarget(
                    entities=[
                        TargetEntity(
                            database=self._database,
                            table=self._table,
                            column=self._column,
                        )
                    ],
                    relationship_type=self._relationship_type,
                ),
                parameters=self._parameters,
            )
            self._reset()  # Reset for reuse
            return rule

        def _finalise_length_parameters(self) -> None:
            if self._type == RuleType.LENGTH and not any(
                k in self._parameters
                for k in ("min_length", "max_length", "exact_length")
            ):
                # Provide a sensible lower bound as default.
                self._parameters["min_length"] = 1

    class ConnectionBuilder:
        """Builder for ConnectionSchema"""

        def __init__(self) -> None:
            self._name = "test_connection"
            self._description = "Test database connection"
            self._connection_type = ConnectionType.MYSQL
            self._host = "localhost"
            self._port = 3306
            self._db_name = "test_db"
            self._username = "test_user"
            self._password = "test_pass"
            self._db_schema = "test_schema"
            self._available_tables: Optional[List[str]] = None
            self._file_path: Optional[str] = None
            self._parameters: Dict[str, Any] = {}

        def with_name(self, name: str) -> "TestDataBuilder.ConnectionBuilder":
            self._name = name
            return self

        def with_type(
            self, conn_type: ConnectionType
        ) -> "TestDataBuilder.ConnectionBuilder":
            self._connection_type = conn_type
            return self

        def with_host(self, host: str) -> "TestDataBuilder.ConnectionBuilder":
            self._host = host
            return self

        def with_port(self, port: int) -> "TestDataBuilder.ConnectionBuilder":
            self._port = port
            return self

        def with_database(self, db_name: str) -> "TestDataBuilder.ConnectionBuilder":
            self._db_name = db_name
            return self

        def with_available_tables(
            self, table_name: str
        ) -> "TestDataBuilder.ConnectionBuilder":
            self._available_tables = [table_name]
            return self

        def with_credentials(
            self, username: str, password: str
        ) -> "TestDataBuilder.ConnectionBuilder":
            self._username = username
            self._password = password
            return self

        def with_file_path(self, file_path: str) -> "TestDataBuilder.ConnectionBuilder":
            self._file_path = file_path
            return self

        def with_parameters(
            self, parameters: Dict[str, Any]
        ) -> "TestDataBuilder.ConnectionBuilder":
            """Set connection parameters"""
            self._parameters = parameters
            return self

        def build(self) -> ConnectionSchema:
            """Build the final ConnectionSchema"""
            from shared.schema.base import DataSourceCapability

            # CSV / Excel / JSON connections **may** omit a physical file path
            # during unit-testing where the validator subsequently injects a
            # temporary path (e.g. converting to SQLite).  Therefore we no
            # longer fail hard when the caller does not provide one.

            return ConnectionSchema(
                name=self._name,
                description=self._description,
                connection_type=self._connection_type,
                host=self._host,
                port=self._port,
                db_name=self._db_name,
                username=self._username,
                password=self._password,
                db_schema=self._db_schema,
                file_path=self._file_path,
                parameters=self._parameters,
                available_tables=self._available_tables,
                capabilities=DataSourceCapability(supports_sql=True),
                cross_db_settings=None,
            )

    class ResultBuilder:
        """Builder for ExecutionResultSchema"""

        def __init__(self) -> None:
            self._rule_id = str(uuid.uuid4())
            self._rule_name = "test_rule"
            self._entity_name = "test_db.test_table"
            self._total_records = 100
            self._failed_records = 0
            self._execution_time = 1.5
            self._status = "PASSED"
            self._execution_message: Optional[str] = None
            self._error_message: Optional[str] = None
            self._sample_data: Optional[List[Any]] = None

        def with_rule(
            self, rule_id: str, rule_name: Optional[str] = None
        ) -> "TestDataBuilder.ResultBuilder":
            """Set rule ID and name"""
            self._rule_id = rule_id
            if rule_name:
                self._rule_name = rule_name
            return self

        def with_entity(self, entity_name: str) -> "TestDataBuilder.ResultBuilder":
            self._entity_name = entity_name
            return self

        def with_counts(
            self, failed_records: int, total_records: Optional[int] = None
        ) -> "TestDataBuilder.ResultBuilder":
            self._failed_records = failed_records
            if total_records is not None:
                self._total_records = total_records
            return self

        def with_timing(self, execution_time: float) -> "TestDataBuilder.ResultBuilder":
            self._execution_time = execution_time
            return self

        def with_error(self, error_message: str) -> "TestDataBuilder.ResultBuilder":
            self._error_message = error_message
            self._status = "ERROR"
            return self

        def with_status(self, status: str) -> "TestDataBuilder.ResultBuilder":
            self._status = status
            return self

        def with_message(self, message: str) -> "TestDataBuilder.ResultBuilder":
            self._execution_message = message
            return self

        def build(self) -> ExecutionResultSchema:
            """Build the final ExecutionResultSchema and attach extra metadata needed by CLI formatter tests."""
            # Parses `entity_name` to extract the database and table names.
            parts = self._entity_name.split(".")
            if len(parts) == 2:
                database, table = parts
            else:
                database, table = "test_db", self._entity_name

            # Generate the result object.
            result = ExecutionResultSchema.create_from_legacy(
                rule_id=self._rule_id,
                status=self._status,
                total_count=self._total_records,
                error_count=self._failed_records,
                execution_time=self._execution_time,
                database=database,
                table=table,
                execution_message=self._execution_message,
                error_message=self._error_message,
                sample_data=self._sample_data,
            )

            # Important: The `rule_name` is dynamically added to the result object for use by the CLI's OutputFormatter.
            # Pydantic ignores unknown fields by default, but we can use the `model_extra` configuration option to capture and store them.
            try:
                # Pydantic version 2 supports `model_extra` as a container for storing unknown or extra fields.
                result.__pydantic_extra__ = result.__pydantic_extra__ or {}
                result.__pydantic_extra__["rule_name"] = self._rule_name
            except AttributeError:
                # Fallback solution: Directly set the attribute (sufficient for the test environment).
                setattr(result, "rule_name", self._rule_name)

            return result

    # Factory methods
    @classmethod
    def rule(cls) -> "TestDataBuilder.RuleBuilder":
        """Create a new rule builder"""
        return cls.RuleBuilder()

    @classmethod
    def connection(cls) -> "TestDataBuilder.ConnectionBuilder":
        """Create a new connection builder"""
        return cls.ConnectionBuilder()

    @classmethod
    def result(cls) -> "TestDataBuilder.ResultBuilder":
        """Create a new result builder"""
        return cls.ResultBuilder()

    # Quick builders for common scenarios
    @classmethod
    def basic_enum_rule(
        cls, values: List[str], table: str = "test_table", column: str = "test_column"
    ) -> RuleSchema:
        """Quick builder for basic enum rule"""
        return (
            cls.rule()
            .with_target("test_db", table, column)
            .as_enum_rule(values)
            .build()
        )

    @classmethod
    def basic_not_null_rule(
        cls, table: str = "test_table", column: str = "test_column"
    ) -> RuleSchema:
        """Quick builder for NOT NULL rule"""
        return (
            cls.rule().with_target("test_db", table, column).as_not_null_rule().build()
        )

    @classmethod
    def basic_unique_rule(
        cls, table: str = "test_table", column: str = "id_column"
    ) -> RuleSchema:
        """Quick builder for UNIQUE rule"""
        return cls.rule().with_target("test_db", table, column).as_unique_rule().build()

    @classmethod
    def basic_range_rule(
        cls,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        table: str = "test_table",
        column: str = "value_column",
    ) -> RuleSchema:
        """Quick builder for RANGE rule"""
        return (
            cls.rule()
            .with_target("test_db", table, column)
            .as_range_rule(min_val, max_val)
            .build()
        )

    @classmethod
    def basic_regex_rule(
        cls, pattern: str, table: str = "test_table", column: str = "text_column"
    ) -> RuleSchema:
        """Quick builder for REGEX rule"""
        return (
            cls.rule()
            .with_target("test_db", table, column)
            .as_regex_rule(pattern)
            .build()
        )

    @classmethod
    def basic_length_rule(
        cls,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        table: str = "test_table",
        column: str = "text_column",
    ) -> RuleSchema:
        """Quick builder for LENGTH rule"""
        return (
            cls.rule()
            .with_target("test_db", table, column)
            .as_length_rule(min_length, max_length)
            .build()
        )

    @classmethod
    def basic_date_format_rule(
        cls, date_format: str, table: str = "test_table", column: str = "date_column"
    ) -> RuleSchema:
        """Quick builder for DATE_FORMAT rule"""
        return (
            cls.rule()
            .with_target("test_db", table, column)
            .as_date_format_rule(date_format)
            .build()
        )

    @classmethod
    def mysql_connection(cls, database: str = "test_db") -> ConnectionSchema:
        """Quick builder for MySQL connection"""
        return (
            cls.connection()
            .with_type(ConnectionType.MYSQL)
            .with_database(database)
            .build()
        )

    @classmethod
    def success_result(
        cls, rule_id: str, anomaly_count: int = 0
    ) -> ExecutionResultSchema:
        """Shorthand for creating a success result"""
        return (
            cls.result()
            .with_rule(rule_id, f"test_rule_{rule_id}")
            .with_counts(anomaly_count)
            .build()
        )

    class CsvDataBuilder:
        """Builder for CSV test data"""

        def __init__(self) -> None:
            self._headers = ["id", "name", "email"]
            self._rows: List[List[Any]] = []

        def with_headers(self, headers: List[str]) -> "TestDataBuilder.CsvDataBuilder":
            """Set CSV headers"""
            self._headers = headers
            return self

        def with_rows(self, rows: List[List[Any]]) -> "TestDataBuilder.CsvDataBuilder":
            """Add data rows"""
            self._rows = rows
            return self

        def with_random_rows(
            self, count: int, generators: Dict[str, Callable[[int], Any]]
        ) -> "TestDataBuilder.CsvDataBuilder":
            """Generate random rows using provided generator functions"""
            import random

            rows = []
            for i in range(count):
                row = []
                for header in self._headers:
                    if header in generators:
                        row.append(generators[header](i))
                    else:
                        row.append(f"value_{i}_{header}")
                rows.append(row)

            self._rows = rows
            return self

        def build_file(self) -> str:
            """Build and save CSV data to a temporary file"""
            import csv
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                writer = csv.writer(f)
                writer.writerow(self._headers)
                for row in self._rows:
                    writer.writerow(row)
                temp_file = f.name

            return temp_file

    @classmethod
    def csv_data(cls) -> "TestDataBuilder.CsvDataBuilder":
        """Create a CSV data builder"""
        return cls.CsvDataBuilder()
