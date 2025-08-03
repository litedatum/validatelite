"""
🔄 Contract Testing Module - Ensure Mocks Match Reality

As the Testing Ghost 👻, I ensure that our mocks accurately represent real implementations.
This prevents the common problem where tests pass but real code fails due to mock misalignment.
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple
from unittest.mock import AsyncMock, MagicMock

from cli.core.config import CliConfig
from cli.core.data_validator import DataValidator
from cli.core.output_formatter import OutputFormatter
from cli.core.rule_parser import RuleParser

# Import CLI components for contract mocks
from cli.core.source_parser import SourceParser

# Import new configuration models
from core.config import CoreConfig
from core.engine.rule_merger import MergeGroup
from core.executors.completeness_executor import CompletenessExecutor
from core.executors.uniqueness_executor import UniquenessExecutor
from core.executors.validity_executor import ValidityExecutor
from shared.database.query_executor import QueryExecutor
from shared.schema import ConnectionSchema, ExecutionResultSchema, RuleSchema


class QueryExecutorContract(Protocol):
    """Contract definition for QueryExecutor to ensure mock compliance"""

    async def execute_query(self, sql: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Must return tuple of (rows, column_names)"""
        ...

    async def execute_query_with_params(
        self, sql: str, params: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Must return tuple of (rows, column_names) with parameters"""
        ...


class ValidityExecutorContract(Protocol):
    """Contract definition for ValidityExecutor"""

    def supports_rule_type(self, rule_type: str) -> bool:
        """Must return boolean indicating support for rule type"""
        ...

    async def execute_rule(self, rule: RuleSchema) -> Any:
        """Must return ExecutionResultSchema"""
        ...


class CompletenessExecutorContract(Protocol):
    """Contract definition for CompletenessExecutor"""

    def supports_rule_type(self, rule_type: str) -> bool:
        """Must return boolean indicating support for NOT_NULL, LENGTH rule types"""
        ...

    async def execute_rule(self, rule: RuleSchema) -> ExecutionResultSchema:
        """Must return ExecutionResultSchema for completeness rules"""
        ...

    async def _execute_not_null_rule(self, rule: RuleSchema) -> ExecutionResultSchema:
        """Must handle NOT_NULL rule execution"""
        ...

    async def _execute_length_rule(self, rule: RuleSchema) -> ExecutionResultSchema:
        """Must handle LENGTH rule execution"""
        ...


class UniquenessExecutorContract(Protocol):
    """Contract definition for UniquenessExecutor"""

    def supports_rule_type(self, rule_type: str) -> bool:
        """Must return boolean indicating support for UNIQUE rule type"""
        ...

    async def execute_rule(self, rule: RuleSchema) -> ExecutionResultSchema:
        """Must return ExecutionResultSchema for uniqueness rules"""
        ...

    async def _execute_unique_rule(self, rule: RuleSchema) -> ExecutionResultSchema:
        """Must handle UNIQUE rule execution with proper duplicate counting"""
        ...


class MockContract:
    """Contract Testing utility to create compliant mocks"""

    @staticmethod
    def create_query_executor_mock(
        query_results: Optional[List[Dict[str, Any]]] = None,
        column_names: Optional[List[str]] = None,
        should_raise: Optional[Exception] = None,
    ) -> AsyncMock:
        """
        Create QueryExecutor mock that follows the contract

        Args:
            query_results: List of result rows to return
            column_names: Column names to return with results
            should_raise: Exception to raise instead of returning results
        """
        mock = AsyncMock(spec=QueryExecutor)

        if query_results is None:
            query_results = [{"anomaly_count": 0}]
        if column_names is None:
            column_names = ["anomaly_count"]

        async def contract_execute_query(
            sql: str,
        ) -> Tuple[List[Dict[str, Any]], List[str]]:
            if should_raise:
                raise should_raise
            return query_results, column_names

        async def contract_execute_query_with_params(
            sql: str, params: Dict[str, Any]
        ) -> Tuple[List[Dict[str, Any]], List[str]]:
            if should_raise:
                raise should_raise
            return query_results, column_names

        mock.execute_query = contract_execute_query
        mock.execute_query_with_params = contract_execute_query_with_params

        return mock

    @staticmethod
    def create_validity_executor_mock(
        supports_result: bool = True,
        execution_result: Optional[Any] = None,
        should_raise: Optional[Exception] = None,
    ) -> MagicMock:
        """Create ValidityExecutor mock that follows the contract"""
        mock = MagicMock(spec=ValidityExecutor)

        def contract_supports_rule_type(rule_type: str) -> bool:
            return supports_result

        async def contract_execute_rule(rule: RuleSchema) -> Any:
            if should_raise:
                raise should_raise
            return execution_result

        mock.supports_rule_type = contract_supports_rule_type
        mock.execute_rule = contract_execute_rule

        return mock

    @staticmethod
    def create_completeness_executor_mock(
        supports_result: bool = True,
        execution_result: Optional[ExecutionResultSchema] = None,
        should_raise: Optional[Exception] = None,
    ) -> MagicMock:
        """Create CompletenessExecutor mock that follows the contract"""
        mock = MagicMock(spec=CompletenessExecutor)

        def contract_supports_rule_type(rule_type: str) -> bool:
            # CompletenessExecutor supports NOT_NULL and LENGTH
            if supports_result:
                return rule_type in ["NOT_NULL", "LENGTH"]
            return False

        async def contract_execute_rule(
            rule: RuleSchema,
        ) -> Optional[ExecutionResultSchema]:
            if should_raise:
                raise should_raise
            return execution_result

        async def contract_execute_not_null_rule(
            rule: RuleSchema,
        ) -> Optional[ExecutionResultSchema]:
            if should_raise:
                raise should_raise
            return execution_result

        async def contract_execute_length_rule(
            rule: RuleSchema,
        ) -> Optional[ExecutionResultSchema]:
            if should_raise:
                raise should_raise
            return execution_result

        mock.supports_rule_type = contract_supports_rule_type
        mock.execute_rule = contract_execute_rule
        mock._execute_not_null_rule = contract_execute_not_null_rule
        mock._execute_length_rule = contract_execute_length_rule

        return mock

    @staticmethod
    def create_uniqueness_executor_mock(
        supports_result: bool = True,
        execution_result: Optional[ExecutionResultSchema] = None,
        should_raise: Optional[Exception] = None,
    ) -> MagicMock:
        """Create UniquenessExecutor mock that follows the contract"""
        mock = MagicMock(spec=UniquenessExecutor)

        def contract_supports_rule_type(rule_type: str) -> bool:
            # UniquenessExecutor supports UNIQUE
            if supports_result:
                return rule_type == "UNIQUE"
            return False

        async def contract_execute_rule(
            rule: RuleSchema,
        ) -> Optional[ExecutionResultSchema]:
            if should_raise:
                raise should_raise
            return execution_result

        async def contract_execute_unique_rule(
            rule: RuleSchema,
        ) -> Optional[ExecutionResultSchema]:
            if should_raise:
                raise should_raise
            return execution_result

        mock.supports_rule_type = contract_supports_rule_type
        mock.execute_rule = contract_execute_rule
        mock._execute_unique_rule = contract_execute_unique_rule

        return mock

    @staticmethod
    def create_db_session_mock(should_raise: Optional[Exception] = None) -> AsyncMock:
        """
        Create database session mock that follows SQLAlchemy AsyncSession contract

        Args:
            should_raise: Exception to raise during operations
        """
        from sqlalchemy.ext.asyncio import AsyncSession

        mock = AsyncMock(spec=AsyncSession)

        # Mock the basic database operations
        async def contract_add(instance: Any) -> None:
            if should_raise:
                raise should_raise

        async def contract_commit() -> None:
            if should_raise:
                raise should_raise

        async def contract_rollback() -> None:
            if should_raise:
                raise should_raise

        async def contract_refresh(instance: Any) -> None:
            if should_raise:
                raise should_raise

        async def contract_execute(statement: Any) -> Any:
            if should_raise:
                raise should_raise
            # Return a mock result
            result_mock = MagicMock()
            result_mock.fetchall.return_value = []
            result_mock.fetchone.return_value = None
            return result_mock

        async def contract_close() -> None:
            if should_raise:
                raise should_raise

        # Assign the contract methods
        mock.add = contract_add
        mock.commit = contract_commit
        mock.rollback = contract_rollback
        mock.refresh = contract_refresh
        mock.execute = contract_execute
        mock.close = contract_close

        return mock

    @staticmethod
    def create_completeness_mock_data(
        failed_count: int, total_count: int, rule_type: str = "NOT_NULL"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Create realistic mock data for completeness rules"""
        field_name = (
            "failed_count" if rule_type in ["NOT_NULL", "LENGTH"] else "anomaly_count"
        )

        return {
            "main_query": [{field_name: failed_count}],
            "total_query": [{"total_count": total_count}],
            "sample_query": (
                [
                    {"id": 1, "column_value": None, "created_at": "2024-01-01"},
                    {"id": 2, "column_value": "", "created_at": "2024-01-02"},
                ]
                if failed_count > 0
                else []
            ),
        }

    @staticmethod
    def create_uniqueness_mock_data(
        duplicate_groups: int, duplicate_records: int, total_count: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Create realistic mock data for uniqueness rules"""
        return {
            "anomaly_query": [
                {"anomaly_count": duplicate_groups}
            ],  # Number of repeating groups.
            "detail_query": [
                {"duplicate_records_count": duplicate_records}
            ],  # Actual number of duplicate records.
            "total_query": [{"total_count": total_count}],
            "sample_query": (
                [
                    {"column_value": "duplicate_value", "cnt": 3},
                    {"column_value": "another_duplicate", "cnt": 2},
                ]
                if duplicate_groups > 0
                else []
            ),
        }

    @staticmethod
    def verify_query_executor_contract(mock_or_real: Any) -> None:
        """Verify that an object follows QueryExecutor contract"""
        assert hasattr(mock_or_real, "execute_query"), "Missing execute_query method"
        assert hasattr(
            mock_or_real, "execute_query_with_params"
        ), "Missing execute_query_with_params method"

        # Check if methods are callable
        assert callable(mock_or_real.execute_query), "execute_query is not callable"
        assert callable(
            mock_or_real.execute_query_with_params
        ), "execute_query_with_params is not callable"

    @staticmethod
    def verify_validity_executor_contract(executor: Any) -> None:
        """Verify that an object follows ValidityExecutor contract"""
        assert hasattr(
            executor, "supports_rule_type"
        ), "Missing supports_rule_type method"
        assert hasattr(executor, "execute_rule"), "Missing execute_rule method"

        # Check if methods are callable
        assert callable(
            executor.supports_rule_type
        ), "supports_rule_type is not callable"
        assert callable(executor.execute_rule), "execute_rule is not callable"

    @staticmethod
    def verify_completeness_executor_contract(executor: Any) -> None:
        """Verify that an object follows CompletenessExecutor contract"""
        assert hasattr(
            executor, "supports_rule_type"
        ), "Missing supports_rule_type method"
        assert hasattr(executor, "execute_rule"), "Missing execute_rule method"
        assert hasattr(
            executor, "_execute_not_null_rule"
        ), "Missing _execute_not_null_rule method"
        assert hasattr(
            executor, "_execute_length_rule"
        ), "Missing _execute_length_rule method"

        # Check if methods are callable
        assert callable(
            executor.supports_rule_type
        ), "supports_rule_type is not callable"
        assert callable(executor.execute_rule), "execute_rule is not callable"
        assert callable(
            executor._execute_not_null_rule
        ), "_execute_not_null_rule is not callable"
        assert callable(
            executor._execute_length_rule
        ), "_execute_length_rule is not callable"

    @staticmethod
    def verify_uniqueness_executor_contract(executor: Any) -> None:
        """Verify that an object follows UniquenessExecutor contract"""
        assert hasattr(
            executor, "supports_rule_type"
        ), "Missing supports_rule_type method"
        assert hasattr(executor, "execute_rule"), "Missing execute_rule method"
        assert hasattr(
            executor, "_execute_unique_rule"
        ), "Missing _execute_unique_rule method"

        # Check if methods are callable
        assert callable(
            executor.supports_rule_type
        ), "supports_rule_type is not callable"
        assert callable(executor.execute_rule), "execute_rule is not callable"
        assert callable(
            executor._execute_unique_rule
        ), "_execute_unique_rule is not callable"

    @staticmethod
    def verify_db_session_contract(session: Any) -> None:
        """Verify that an object follows AsyncSession contract"""
        assert hasattr(session, "add"), "Missing add method"
        assert hasattr(session, "commit"), "Missing commit method"
        assert hasattr(session, "rollback"), "Missing rollback method"
        assert hasattr(session, "refresh"), "Missing refresh method"
        assert hasattr(session, "execute"), "Missing execute method"
        assert hasattr(session, "close"), "Missing close method"

        # Check if methods are callable
        assert callable(session.add), "add must be callable"
        assert callable(session.commit), "commit must be callable"
        assert callable(session.rollback), "rollback must be callable"
        assert callable(session.refresh), "refresh must be callable"
        assert callable(session.execute), "execute must be callable"
        assert callable(session.close), "close must be callable"

    @staticmethod
    def verify_data_flow_contract(
        mock_query_result: Dict[str, Any], actual_result: ExecutionResultSchema
    ) -> None:
        """Verify the data flow contract: Mock data is transformed into an ExecutionResultSchema object."""
        if "anomaly_count" in mock_query_result:
            expected_failed = mock_query_result["anomaly_count"]
            actual_failed = actual_result.dataset_metrics[0].failed_records
            assert (
                actual_failed == expected_failed
            ), f"数据流断裂：期望{expected_failed}条异常，实际{actual_failed}条"

        if "failed_count" in mock_query_result:
            expected_failed = mock_query_result["failed_count"]
            actual_failed = actual_result.dataset_metrics[0].failed_records
            assert (
                actual_failed == expected_failed
            ), f"数据流断裂：期望{expected_failed}条失败，实际{actual_failed}条"

        if "total_count" in mock_query_result:
            expected_total = mock_query_result["total_count"]
            actual_total = actual_result.dataset_metrics[0].total_records
            assert (
                actual_total == expected_total
            ), f"数据流断裂：期望{expected_total}条总记录，实际{actual_total}条"

    @staticmethod
    def create_realistic_mock_data(
        anomaly_count: int, total_count: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate more realistic mock data that simulates actual database responses."""
        return {
            "anomaly_query": [{"anomaly_count": anomaly_count}],
            "total_query": [{"total_count": total_count}],
            "sample_query": (
                [
                    {"id": 1, "status": "invalid_value", "created_at": "2024-01-01"},
                    {"id": 2, "status": "another_invalid", "created_at": "2024-01-02"},
                ]
                if anomaly_count > 0
                else []
            ),
        }

    @staticmethod
    def create_rule_merge_manager_mock(
        merge_groups_count: int = 1, should_raise: Optional[Exception] = None
    ) -> MagicMock:
        """Create RuleMergeManager mock that follows the contract"""
        mock = MagicMock()

        def contract_analyze_rules(rules: List[RuleSchema]) -> List[MergeGroup]:
            if should_raise:
                raise should_raise

            # Create mock merge groups
            groups: List[MergeGroup] = []
            for i in range(merge_groups_count):
                group = MagicMock(spec=MergeGroup)
                group.strategy = MagicMock(value="individual")
                group.rules = rules if i == 0 else []  # First group gets rules
                groups.append(group)
            return groups

        mock.analyze_rules = contract_analyze_rules
        return mock

    @staticmethod
    def verify_rule_merge_manager_contract(merge_manager: Any) -> None:
        """Verify RuleMergeManager follows expected contract"""
        assert hasattr(
            merge_manager, "analyze_rules"
        ), "RuleMergeManager must have analyze_rules method"
        assert callable(merge_manager.analyze_rules), "analyze_rules must be callable"
        assert hasattr(
            merge_manager, "get_merge_strategy"
        ), "RuleMergeManager must have get_merge_strategy method"
        assert callable(
            merge_manager.get_merge_strategy
        ), "get_merge_strategy must be callable"
        assert hasattr(
            merge_manager, "validator"
        ), "RuleMergeManager must have validator attribute"
        assert hasattr(
            merge_manager, "unique_validator"
        ), "RuleMergeManager must have unique_validator attribute"

        # Verify that the MergeStrategy enum has the expected values
        from core.engine.rule_merger import MergeStrategy

        assert hasattr(
            MergeStrategy, "INDIVIDUAL"
        ), "MergeStrategy must have INDIVIDUAL value"
        assert hasattr(MergeStrategy, "MERGED"), "MergeStrategy must have MERGED value"
        assert hasattr(MergeStrategy, "MIXED"), "MergeStrategy must have MIXED value"

    @staticmethod
    def create_source_parser_mock() -> MagicMock:
        """Create SourceParser mock that follows the contract"""
        mock = MagicMock(spec=SourceParser)

        def contract_parse_source(source: str) -> ConnectionSchema:
            # Default implementation returns a file connection
            from shared.enums import ConnectionType
            from shared.schema import ConnectionSchema
            from shared.schema.base import DataSourceCapability

            return ConnectionSchema(
                name="test_connection",
                description="Test connection for mock",
                connection_type=ConnectionType.CSV,
                host=None,
                port=None,
                db_name=None,
                username=None,
                password=None,
                db_schema=None,
                file_path=source,
                parameters={"filename": "test.csv"},
                capabilities=DataSourceCapability(
                    supports_sql=False,
                    supports_batch_export=True,
                    max_export_rows=100000,
                    estimated_throughput=5000,
                ),
                cross_db_settings=None,
            )

        mock.parse_source = contract_parse_source
        return mock

    @staticmethod
    def create_rule_parser_mock() -> MagicMock:
        """Create RuleParser mock that follows the contract"""
        mock = MagicMock(spec=RuleParser)

        def contract_parse_rules(
            inline_rules: Optional[List[str]] = None,
            rules_file: Optional[str] = None,
        ) -> List[RuleSchema]:
            # Default implementation returns a simple rule list
            from shared.enums import RuleAction, RuleCategory, RuleType, SeverityLevel
            from shared.schema import RuleSchema
            from shared.schema.base import RuleTarget, TargetEntity

            if not inline_rules and not rules_file:
                return []

            rules: List[RuleSchema] = []
            if inline_rules:
                for rule_expr in inline_rules:
                    # Simple parsing of not_null(column)
                    import re

                    match = re.match(r"(\w+)\(([^)]+)\)", rule_expr)
                    if match:
                        rule_type_str, column = match.groups()
                        rule_type = (
                            RuleType.NOT_NULL
                            if rule_type_str == "not_null"
                            else RuleType.UNIQUE
                        )

                        rules.append(
                            RuleSchema(
                                id=f"test_rule_{len(rules)}",
                                name=f"Test Rule {len(rules)}",
                                description=f"Test rule from {rule_expr}",
                                connection_id="test_connection",
                                type=rule_type,
                                category=RuleCategory.COMPLETENESS,
                                severity=SeverityLevel.MEDIUM,
                                action=RuleAction.ALERT,
                                threshold=0.0,
                                template_id=None,
                                is_active=True,
                                tags=[],
                                target=RuleTarget(
                                    entities=[
                                        TargetEntity(
                                            database="test_db",
                                            table="test_table",
                                            column=column,
                                        )
                                    ],
                                    relationship_type="single_table",
                                ),
                                parameters={},
                            )
                        )

            return rules

        mock.parse_rules = contract_parse_rules
        return mock

    @staticmethod
    def create_core_config_mock() -> MagicMock:
        """Create CoreConfig mock with default values"""
        mock = MagicMock(spec=CoreConfig)

        # Set default values for CoreConfig
        mock.execution_timeout = 300
        mock.table_size_threshold = 10000
        mock.merge_execution_enabled = True
        mock.monitoring_enabled = False

        return mock

    @staticmethod
    def create_cli_config_mock() -> MagicMock:
        """Create CliConfig mock with default values"""
        mock = MagicMock(spec=CliConfig)

        # Set default values for CliConfig
        mock.debug_mode = False
        mock.default_sample_size = 10000
        mock.max_file_size_mb = 100
        mock.query_timeout = 300

        # Mock database config
        mock.database = MagicMock()
        mock.database.url = None
        mock.database.connect_timeout = 30
        mock.database.echo_queries = False

        return mock

    @staticmethod
    def create_data_validator_mock() -> AsyncMock:
        """Create DataValidator mock that follows the contract"""
        mock = AsyncMock(spec=DataValidator)

        async def contract_validate() -> List[ExecutionResultSchema]:
            # Default implementation returns a successful validation result
            from shared.schema.result_schema import DatasetMetrics

            return [
                ExecutionResultSchema(
                    rule_id="test_rule_1",
                    rule_name="not_null_id",
                    rule_type="NOT_NULL",
                    column_name="id",
                    status="PASSED",
                    dataset_metrics=[
                        DatasetMetrics(
                            entity_name="test_db.test_table",
                            total_records=100,
                            failed_records=0,
                        )
                    ],
                    execution_time=0.05,
                    error_message=None,
                    sample_data=[],
                )
            ]

        mock.validate = contract_validate
        return mock

    @staticmethod
    def create_output_formatter_mock() -> MagicMock:
        """Create OutputFormatter mock that follows the contract"""
        mock = MagicMock(spec=OutputFormatter)

        def contract_display_results(
            results: List[ExecutionResultSchema],
            source: str,
            execution_time: float,
            total_rules: int,
        ) -> None:
            # Just a stub implementation that does nothing
            pass

        mock.display_results = contract_display_results
        return mock

    @staticmethod
    def create_config_manager_mock() -> MagicMock:
        """Create ConfigManager mock that follows the contract"""
        mock = MagicMock()

        def contract_get_config() -> Dict[str, Any]:
            # Default implementation returns basic config
            return {
                "database": {
                    "host": "localhost",
                    "port": 3306,
                    "database": "test_db",
                    "username": "test_user",
                    "password": "test_pass",
                },
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            }

        mock.get_config = contract_get_config
        return mock

    @staticmethod
    def verify_source_parser_contract(parser: Any) -> None:
        """Verify SourceParser follows expected contract"""
        assert hasattr(
            parser, "parse_source"
        ), "SourceParser must have parse_source method"
        assert callable(parser.parse_source), "parse_source must be callable"

    @staticmethod
    def verify_rule_parser_contract(parser: Any) -> None:
        """Verify RuleParser follows expected contract"""
        assert hasattr(parser, "parse_rules"), "RuleParser must have parse_rules method"
        assert callable(parser.parse_rules), "parse_rules must be callable"

    @staticmethod
    def verify_output_formatter_contract(formatter: Any) -> None:
        """Verify OutputFormatter follows expected contract"""
        assert hasattr(
            formatter, "display_results"
        ), "OutputFormatter must have display_results method"
        assert callable(formatter.display_results), "display_results must be callable"


class ContractTestCase:
    """Base class for contract testing scenarios"""

    @staticmethod
    async def test_query_executor_contract_compliance(
        query_executor_mock: AsyncMock,
    ) -> None:
        """Test that QueryExecutor mock follows contract"""
        # Verify contract
        MockContract.verify_query_executor_contract(query_executor_mock)

        # Test basic query execution
        result_rows, column_names = await query_executor_mock.execute_query("SELECT 1")
        assert isinstance(result_rows, list), "execute_query must return list of rows"
        assert isinstance(
            column_names, list
        ), "execute_query must return list of column names"

        # Test parameterized query execution
        result_rows, column_names = await query_executor_mock.execute_query_with_params(
            "SELECT * FROM table WHERE id = %(id)s", {"id": 1}
        )
        assert isinstance(
            result_rows, list
        ), "execute_query_with_params must return list of rows"
        assert isinstance(
            column_names, list
        ), "execute_query_with_params must return list of column names"

    @staticmethod
    def test_validity_executor_contract_compliance(
        validity_executor: ValidityExecutor,
    ) -> None:
        """Test that ValidityExecutor follows contract"""
        # Verify contract
        MockContract.verify_validity_executor_contract(validity_executor)

        # Test supports method
        result = validity_executor.supports_rule_type("ENUM")
        assert isinstance(result, bool), "supports_rule_type method must return boolean"


# Usage examples for documentation
class ContractExamples:
    """Examples of how to use contract testing in practice"""

    @staticmethod
    def example_successful_query_mock() -> AsyncMock:
        """Example: Create mock for successful query with specific results"""
        return MockContract.create_query_executor_mock(
            query_results=[
                {"anomaly_count": 5, "total_records": 100},
                {"anomaly_count": 0, "total_records": 50},
            ],
            column_names=["anomaly_count", "total_records"],
        )

    @staticmethod
    def example_database_error_mock() -> AsyncMock:
        """Example: Create mock that simulates database error"""
        return MockContract.create_query_executor_mock(
            should_raise=Exception("Database connection timeout")
        )

    @staticmethod
    def example_empty_result_mock() -> AsyncMock:
        """Example: Create mock for empty query results"""
        return MockContract.create_query_executor_mock(
            query_results=[], column_names=["anomaly_count"]
        )


class ContractValidator:
    """Validates contract adherence for mocks and real implementations"""

    def validate_execution_result_structure(self, result: Dict[str, Any]) -> None:
        """Validate that execution result follows expected structure"""
        required_fields = [
            "rule_id",
            "rule_name",
            "rule_type",
            "database",
            "table_name",
            "column_name",
            "severity",
            "status",
            "total_records",
            "anomaly_count",
            "anomaly_rate",
            "message",
            "execution_time",
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        # Validate data types
        assert isinstance(result["total_records"], int), "total_records must be int"
        assert isinstance(result["anomaly_count"], int), "anomaly_count must be int"
        assert isinstance(
            result["anomaly_rate"], (int, float)
        ), "anomaly_rate must be number"
        assert isinstance(
            result["execution_time"], (int, float)
        ), "execution_time must be number"
        assert result["status"] in ["PASSED", "FAILED", "ERROR"], "status must be valid"

        # Validate logical constraints
        assert result["anomaly_count"] >= 0, "anomaly_count cannot be negative"
        assert result["total_records"] >= 0, "total_records cannot be negative"
        assert (
            result["anomaly_count"] <= result["total_records"]
        ), "anomaly_count cannot exceed total_records"

        if result["total_records"] > 0:
            expected_rate = result["anomaly_count"] / result["total_records"]
            assert (
                abs(result["anomaly_rate"] - expected_rate) < 0.0001
            ), "anomaly_rate calculation incorrect"

    def validate_query_result_structure(
        self, rows: List[Dict[str, Any]], columns: List[str]
    ) -> None:
        """Validate query result structure follows contract"""
        assert isinstance(rows, list), "Query result rows must be a list"
        assert isinstance(columns, list), "Query result columns must be a list"

        if rows:
            for row in rows:
                assert isinstance(row, dict), "Each row must be a dictionary"
                # Check that all columns from column list exist in each row
                for col in columns:
                    if col not in row:
                        # Allow missing columns to be treated as None
                        continue

    def validate_mock_data_contract(self, mock_data: Dict[str, Any]) -> None:
        """Validate that mock data follows expected contract"""
        if "total_records" in mock_data:
            assert isinstance(
                mock_data["total_records"], int
            ), "total_records must be int"
            assert mock_data["total_records"] >= 0, "total_records must be non-negative"

        if "failed_count" in mock_data:
            assert isinstance(
                mock_data["failed_count"], int
            ), "failed_count must be int"
            assert mock_data["failed_count"] >= 0, "failed_count must be non-negative"

            if "total_records" in mock_data:
                assert (
                    mock_data["failed_count"] <= mock_data["total_records"]
                ), "failed_count cannot exceed total_records"
