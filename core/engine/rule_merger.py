"""
Rule merger module - refactored version

Implements the core logic for merged rule execution, supporting:
- Merge strategies based on the new Executor architecture
- Correct merged SQL generation (COUNT(CASE WHEN...) format)
- Schema interface adaptation
- Performance optimization
- Rule prevalidation mechanism to prevent invalid rules from affecting merge groups
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

from sqlalchemy.ext.asyncio import AsyncEngine

from shared.database.connection import get_db_url, get_engine
from shared.database.database_dialect import get_dialect
from shared.enums import RuleType
from shared.exceptions import EngineError, RuleExecutionError
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.result_schema import ExecutionResultSchema
from shared.schema.rule_schema import RuleSchema
from shared.utils.logger import get_logger


class MergeStrategy(Enum):
    """Merge strategy enum"""

    INDIVIDUAL = "individual"  # Execute individually
    MERGED = "merged"  # Execute merged
    MIXED = "mixed"  # Mixed execution


@dataclass
class MergeGroup:
    """Merge group"""

    strategy: MergeStrategy
    rules: List[RuleSchema]
    target_database: str
    target_table: str

    def __post_init__(self) -> None:
        """Post-initialize MergeGroup"""

        self.rule_count = len(self.rules)
        # Handle rule types, support both enums and strings
        self.rule_types = {
            rule.type.value if hasattr(rule.type, "value") else str(rule.type)
            for rule in self.rules
        }


@dataclass
class MergeResult:
    """Merge result"""

    sql: str
    params: Dict[str, Any]
    rule_mapping: Dict[str, RuleSchema]  # Mapping from result field to rule
    execution_time: float = 0.0
    total_records: int = 0
    results: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        """Post-initialize MergeResult"""
        if self.results is None:
            self.results = []


class BaseRuleMerger(ABC):
    """Base class for rule mergers - based on new Executor architecture"""

    def __init__(self, connection: ConnectionSchema):
        """Initialize BaseRuleMerger"""
        self.connection = connection
        self.dialect = get_dialect(connection.connection_type.value)
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def can_merge(self, rules: List[RuleSchema]) -> bool:
        """Determine whether rules can be merged"""
        pass

    @abstractmethod
    def merge_rules(self, rules: List[RuleSchema]) -> MergeResult:
        """Generate merged SQL for rules - using correct COUNT(CASE WHEN...) format"""
        pass

    @abstractmethod
    async def parse_results(
        self, merge_result: MergeResult, raw_results: List[Dict[str, Any]]
    ) -> List[ExecutionResultSchema]:
        """Parse merged execution results into standard ExecutionResultSchema"""
        pass


class ValidationRuleMerger(BaseRuleMerger):
    """Validation rule merger - based on new Executor architecture

    Supports merging the following rule types:
    - NOT_NULL: Not null check
    - RANGE: Range check
    - ENUM: Enum value check
    - REGEX: Regular expression check
    - LENGTH: Length check
    - DATE_FORMAT: Date format check
    """

    MERGEABLE_TYPES = {"NOT_NULL", "RANGE", "ENUM", "REGEX", "LENGTH", "DATE_FORMAT"}

    def can_merge(self, rules: List[RuleSchema]) -> bool:
        """Determine whether rules can be merged"""
        if len(rules) < 2:
            return False

        # Check if all rules are mergeable types
        rule_types = {
            rule.type.value if hasattr(rule.type, "value") else str(rule.type)
            for rule in rules
        }
        if not rule_types.issubset(self.MERGEABLE_TYPES):
            return False

        # Check if all target the same table
        databases = {rule.target.primary_entity.database for rule in rules}
        tables = {rule.target.primary_entity.table for rule in rules}

        if len(databases) > 1 or len(tables) > 1:
            return False

        # Check if filter conditions are the same
        filter_conditions = {rule.get_filter_condition() for rule in rules}
        if len(filter_conditions) > 1:
            return False

        return True

    def merge_rules(self, rules: List[RuleSchema]) -> MergeResult:
        """Generate merged SQL for rules - using correct COUNT(CASE WHEN...) format"""
        if not self.can_merge(rules):
            raise ValueError("Rules cannot be merged")

        # Get target table info
        first_rule = rules[0]
        database = first_rule.target.primary_entity.database
        table = first_rule.target.primary_entity.table
        full_table = self.dialect.build_full_table_name(database, table)

        # Generate merged SQL - using COUNT(CASE WHEN...) format
        count_clauses = []
        rule_mapping = {}
        params = {}

        for rule in rules:
            count_clause, rule_params, field_name = self._generate_count_case_clause(
                rule
            )
            count_clauses.append(f"COUNT({count_clause}) AS {field_name}")
            rule_mapping[field_name] = rule
            params.update(rule_params)

        # Build complete SQL, including WHERE clause (if filter condition exists)
        sql = f"""
            SELECT
                {', '.join(count_clauses)}
            FROM {full_table}
        """.strip()

        # Handle filter condition
        filter_condition = first_rule.get_filter_condition()
        if filter_condition:
            sql += f"\n            WHERE {filter_condition}"

        return MergeResult(sql=sql, params=params, rule_mapping=rule_mapping)

    def _generate_count_case_clause(
        self, rule: RuleSchema
    ) -> Tuple[str, Dict[str, Any], str]:
        """Generate COUNT(CASE WHEN...) clause for a single rule"""
        # Replace hyphens in UUID with underscores for MySQL compatibility
        safe_rule_id = str(rule.id).replace("-", "_")
        field_name = f"rule_{safe_rule_id}_count"
        params = {}
        column = self.dialect.quote_identifier(str(rule.target.primary_entity.column))

        if rule.type.value == "NOT_NULL":
            case_clause = f"CASE WHEN {column} IS NULL THEN 1 END"

        elif rule.type.value == "RANGE":
            min_val = rule.parameters.get("min_value") or rule.parameters.get("min")
            max_val = rule.parameters.get("max_value") or rule.parameters.get("max")
            conditions = []

            # Add NULL value check
            conditions.append(f"{column} IS NULL")

            if min_val is not None:
                param_name = f"min_val_{safe_rule_id}"
                conditions.append(f"{column} < :{param_name}")
                params[param_name] = min_val

            if max_val is not None:
                param_name = f"max_val_{safe_rule_id}"
                conditions.append(f"{column} > :{param_name}")
                params[param_name] = max_val

            if conditions:
                case_clause = f"CASE WHEN {' OR '.join(conditions)} THEN 1 END"
            else:
                case_clause = "CASE WHEN 1=0 THEN 1 END"  # Never matches

        elif rule.type.value == "ENUM":
            allowed_values = rule.parameters.get("allowed_values", [])
            if allowed_values:
                placeholders = []
                for i, value in enumerate(allowed_values):
                    param_name = f"enum_val_{safe_rule_id}_{i}"
                    placeholders.append(f":{param_name}")
                    params[param_name] = value

                case_clause = (
                    f"CASE WHEN {column} NOT IN ({', '.join(placeholders)}) THEN 1 END"
                )
            else:
                case_clause = "CASE WHEN 1=0 THEN 1 END"

        elif rule.type.value == "REGEX":
            pattern = rule.parameters.get("pattern", "")
            if pattern:
                # Directly embed regex pattern, do not use parameterized query
                # Because MySQL's REGEXP operator does not support parameterized queries
                escaped_pattern = pattern.replace("'", "''")  # Escape single quotes
                regex_op = self.dialect.get_not_regex_operator()
                case_clause = (
                    f"CASE WHEN {column} {regex_op} '{escaped_pattern}' THEN 1 END"
                )
            else:
                case_clause = "CASE WHEN 1=0 THEN 1 END"

        elif rule.type.value == "LENGTH":
            min_length = rule.parameters.get("min_length")
            max_length = rule.parameters.get("max_length")
            length_func = self.dialect.get_string_length_function()
            conditions = []

            if min_length is not None:
                param_name = f"min_len_{safe_rule_id}"
                conditions.append(f"{length_func}({column}) < :{param_name}")
                params[param_name] = min_length

            if max_length is not None:
                param_name = f"max_len_{safe_rule_id}"
                conditions.append(f"{length_func}({column}) > :{param_name}")
                params[param_name] = max_length

            if conditions:
                case_clause = f"CASE WHEN {' OR '.join(conditions)} THEN 1 END"
            else:
                case_clause = "CASE WHEN 1=0 THEN 1 END"

        elif rule.type.value == "DATE_FORMAT":
            params = rule.parameters if hasattr(rule, "parameters") else {}
            format_pattern = params.get("format_pattern") or params.get("format")

            if not format_pattern:
                raise RuleExecutionError("DATE_FORMAT rule requires format_pattern")

            date_clause = self.dialect.get_date_clause(column, format_pattern)
            case_clause = f"CASE WHEN {date_clause} IS NULL THEN 1 END"

        else:
            # Unknown rule type
            case_clause = "CASE WHEN 1=0 THEN 1 END"

        return case_clause, params, field_name

    async def parse_results(
        self, merge_result: MergeResult, raw_results: List[Dict[str, Any]]
    ) -> List[ExecutionResultSchema]:
        """Parse merged execution results into standard ExecutionResultSchema"""
        if not raw_results:
            return []

        # Merged query only returns one row of results
        result_row = raw_results[0]
        execution_results = []

        for field_name, rule in merge_result.rule_mapping.items():
            failed_count = result_row.get(field_name, 0)

            # Calculate total record count (if needed, can add extra query)
            total_records = (
                merge_result.total_records or 1000
            )  # Temporarily use default value

            # success_rate = (
            #     1.0 - (failed_count / total_records) if total_records > 0 else 1.0
            # )

            # Create required dataset_metrics
            from shared.schema.base import DatasetMetrics

            dataset_metric = DatasetMetrics(
                entity_name=(
                    f"{rule.target.primary_entity.database}."
                    f"{rule.target.primary_entity.table}"
                ),
                total_records=total_records,
                failed_records=failed_count,
                processing_time=None,
            )

            # Determine status based on error count
            from shared.enums import ExecutionStatus

            status = (
                ExecutionStatus.PASSED.value
                if failed_count == 0
                else ExecutionStatus.FAILED.value
            )

            # Generate sample data (only on failure)
            sample_data = None
            if failed_count > 0:
                sample_data = await self._generate_sample_data_for_merged_rule(
                    rule, merge_result
                )

            result = ExecutionResultSchema(
                rule_id=rule.id,
                status=status,
                dataset_metrics=[dataset_metric],
                execution_time=merge_result.execution_time
                / len(merge_result.rule_mapping),  # Evenly distribute execution time
                execution_message=(
                    f"Merged execution: {failed_count} violations found"
                    if failed_count > 0
                    else "Merged execution: validation passed"
                ),
                error_message=None,
                sample_data=sample_data,
                cross_db_metrics=None,
                execution_plan=None,
                started_at=None,
                ended_at=None,
            )

            execution_results.append(result)

        return execution_results

    async def _generate_sample_data_for_merged_rule(
        self, rule: RuleSchema, merge_result: MergeResult
    ) -> Optional[List[Dict[str, Any]]]:
        """Generate sample data for merged rule execution"""
        try:
            from core.config import get_core_config
            from shared.database.query_executor import QueryExecutor

            try:
                core_config = get_core_config()
                sample_data_enabled = (
                    core_config.sample_data_enabled
                    if core_config.sample_data_enabled is not None
                    else True
                )
                if not sample_data_enabled:
                    return None

                max_samples = (
                    core_config.sample_data_max_records
                    if core_config.sample_data_max_records is not None
                    else 5
                )
            except Exception:
                sample_data_enabled = True
                max_samples = 5

            # Get database engine
            engine = await self._get_engine()
            query_executor = QueryExecutor(engine)

            # Generate independent SQL query for this rule to get sample data

            table_name = self.dialect.build_full_table_name(
                rule.target.primary_entity.database, rule.target.primary_entity.table
            )
            column = rule.target.primary_entity.column
            if not column:
                raise RuleExecutionError(
                    "Column is required for sample data generation"
                )

            # Generate different sample queries based on rule type
            sample_sql = self._generate_sample_sql_for_rule(
                rule, table_name, column, max_samples
            )

            if sample_sql:
                sample_result, _ = await query_executor.execute_query(sample_sql)
                return sample_result if sample_result else None

            return None

        except Exception as e:
            # Log warning if sample data generation fails but do not affect main flow
            self.logger.warning(
                f"Failed to generate sample data for merged rule {rule.id}: {str(e)}"
            )
            return None

    def _generate_sample_sql_for_rule(
        self, rule: RuleSchema, table_name: str, column: str, max_samples: int
    ) -> Optional[str]:
        """Generate sample query SQL based on rule type"""
        rule_type = (
            rule.type
        )  # .value if hasattr(rule.type, 'value') else str(rule.type)

        if rule_type == RuleType.NOT_NULL:
            return (
                f"SELECT * FROM {table_name} WHERE {column} IS NULL "
                f"LIMIT {max_samples}"
            )

        elif rule_type == RuleType.RANGE:
            min_val = rule.parameters.get("min_value")
            max_val = rule.parameters.get("max_value")
            conditions = []
            if min_val is not None:
                conditions.append(f"{column} < {min_val}")
            if max_val is not None:
                conditions.append(f"{column} > {max_val}")
            if conditions:
                return (
                    f"SELECT * FROM {table_name} WHERE {' OR '.join(conditions)} "
                    f"LIMIT {max_samples}"
                )

        elif rule_type == RuleType.ENUM:
            values = rule.parameters.get("allowed_values", [])
            if values:
                # Convert value list to SQL IN clause
                value_list = ",".join(
                    [f"'{v}'" if isinstance(v, str) else str(v) for v in values]
                )
                return (
                    f"SELECT * FROM {table_name} WHERE {column} NOT IN "
                    f"({value_list}) LIMIT {max_samples}"
                )

        elif rule_type == RuleType.REGEX:
            pattern = rule.parameters.get("pattern", "")
            if pattern:
                # Directly embed regex pattern, do not use parameterized query
                escaped_pattern = pattern.replace("'", "''")  # Escape single quotes
                regex_op = self.dialect.get_not_regex_operator()
                return (
                    f"SELECT * FROM {table_name} WHERE {column} {regex_op} "
                    f"'{escaped_pattern}' LIMIT {max_samples}"
                )

        elif rule_type == RuleType.LENGTH:
            min_length = rule.parameters.get("min")
            max_length = rule.parameters.get("max")
            conditions = []
            if min_length is not None:
                conditions.append(f"LENGTH({column}) < {min_length}")
            if max_length is not None:
                conditions.append(f"LENGTH({column}) > {max_length}")
            if conditions:
                return (
                    f"SELECT * FROM {table_name} WHERE {' OR '.join(conditions)} "
                    f"LIMIT {max_samples}"
                )
        elif rule_type == RuleType.DATE_FORMAT:
            params = rule.parameters if hasattr(rule, "parameters") else {}
            format_pattern = params.get("format_pattern") or params.get("format")

            if not format_pattern:
                raise RuleExecutionError("DATE_FORMAT rule requires format_pattern")

            date_clause = self.dialect.get_date_clause(column, format_pattern)
            return (
                f"SELECT * FROM {table_name} WHERE {date_clause} IS NULL "
                f"LIMIT {max_samples}"
            )

        return None

    async def _get_engine(self) -> AsyncEngine:
        """
        Get AsyncEngine - reuse existing connection management functionality

        Use get_engine function from shared.database.connection
        Engine-level error handling: connection failure will raise EngineError
        """
        try:
            # Build database URL
            db_url = get_db_url(
                db_type=self.connection.connection_type.value,  # Use enum value,
                # not enum object
                host=self.connection.host,
                port=self.connection.port,
                database=self.connection.db_name,
                username=self.connection.username,
                password=self.connection.password,
                file_path=self.connection.file_path,
            )

            # Use existing get_engine function, already includes connection pool
            # management
            return await get_engine(
                db_url=db_url,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600,
                echo=False,
            )
        except Exception as e:
            # Connection failure is an engine-level error, raise EngineError
            error_msg = f"Failed to connect to database: {str(e)}"
            self.logger.error(error_msg)
            raise EngineError(
                error_msg,
                connection_id=(
                    self.connection.name
                    if hasattr(self.connection, "name")
                    else str(self.connection.host)
                ),
            )


class UniqueRuleMerger(BaseRuleMerger):
    """Uniqueness rule merger

    Uniqueness rules require special handling and usually cannot be
    merged with other rules
    """

    def can_merge(self, rules: List[RuleSchema]) -> bool:
        """Uniqueness rules are usually not merged"""
        return False

    def merge_rules(self, rules: List[RuleSchema]) -> MergeResult:
        """Uniqueness rules do not support merged execution"""
        raise NotImplementedError("Uniqueness rules do not support merged execution")

    async def parse_results(
        self, merge_result: MergeResult, raw_results: List[Dict[str, Any]]
    ) -> List[ExecutionResultSchema]:
        """Uniqueness rules do not support merged execution"""
        raise NotImplementedError("Uniqueness rules do not support merged execution")


class RuleMergerFactory:
    """Rule merger factory - based on new architecture"""

    _mergers: Dict[str, Type[BaseRuleMerger]] = {
        "validation": ValidationRuleMerger,
        "unique": UniqueRuleMerger,
    }

    @classmethod
    def get_merger(
        cls, merger_type: str, connection: ConnectionSchema
    ) -> BaseRuleMerger:
        """Get rule merger"""
        if merger_type not in cls._mergers:
            raise ValueError(f"Unsupported merger type: {merger_type}")

        merger_class = cls._mergers[merger_type]
        return merger_class(connection)

    @classmethod
    def register_merger(
        cls, merger_type: str, merger_class: Type[BaseRuleMerger]
    ) -> None:
        """Register custom merger"""
        cls._mergers[merger_type] = merger_class

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get supported merger types"""
        return list(cls._mergers.keys())


class RuleMergeManager:
    """Rule merge manager - simplified version

    Core design principles:
    - Rule engine is independent, does not depend on external config system
    - Receives config values via constructor parameters
    - CLI layer is responsible for config reading and parameter passing
    - Focuses on rule merge strategy analysis
    """

    def __init__(self, connection: ConnectionSchema):
        """
        Initialize rule merge manager

        Args:
            connection: Database connection config
        """
        self.connection = connection

        # Get rule merge config parameters from config system
        from core.config import get_core_config

        core_config = get_core_config()

        self.merge_execution_enabled = core_config.merge_execution_enabled
        self.table_size_threshold = core_config.table_size_threshold
        self.rule_count_threshold = core_config.rule_count_threshold
        self.max_rules_per_merge = core_config.max_rules_per_merge
        self.independent_rule_types = set(core_config.independent_rule_types)
        # Add dialect attribute, get dialect from connection
        self.dialect = get_dialect(connection.connection_type.value)

        if not self.dialect.is_supported_date_format():
            self.independent_rule_types.add(RuleType.DATE_FORMAT)

        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.validator = ValidationRuleMerger(connection)
        self.unique_validator = UniqueRuleMerger(connection)

    def analyze_rules(self, rules: List[RuleSchema]) -> List[MergeGroup]:
        """Analyze rules and generate merge groups

        (original logic, for backward compatibility)
        """
        if not rules:
            return []

        # Group by database and table
        table_groups: Dict[str, List[RuleSchema]] = {}
        for rule in rules:
            database = rule.target.primary_entity.database
            table = rule.target.primary_entity.table
            key = f"{database}.{table}"

            if key not in table_groups:
                table_groups[key] = []
            table_groups[key].append(rule)

        # Generate merge groups for each table
        merge_groups = []
        for key, table_rules in table_groups.items():
            database, table = key.split(".", 1)
            groups = self._analyze_table_rules(table_rules, database, table)
            merge_groups.extend(groups)

        return merge_groups

    def _analyze_table_rules(
        self, rules: List[RuleSchema], database: str, table: str
    ) -> List[MergeGroup]:
        """Analyze rules for a single table"""
        merge_groups = []
        remaining_rules = rules.copy()

        # 1. Extract rules that need to be executed individually
        independent_rules = []
        mergeable_rules = []

        for rule in remaining_rules:
            rule_type = (
                rule.type.value if hasattr(rule.type, "value") else str(rule.type)
            )
            if rule_type in self.independent_rule_types:
                independent_rules.append(rule)
            else:
                mergeable_rules.append(rule)

        # 2. Create individual groups for independent rules
        for rule in independent_rules:
            merge_groups.append(
                MergeGroup(
                    strategy=MergeStrategy.INDIVIDUAL,
                    rules=[rule],
                    target_database=database,
                    target_table=table,
                )
            )

        # 3. Handle mergeable rules
        if mergeable_rules:
            strategy = self.get_merge_strategy(mergeable_rules)

            if strategy == MergeStrategy.MERGED:
                # Group by batch size and check if they can really be merged
                for i in range(0, len(mergeable_rules), self.max_rules_per_merge):
                    batch = mergeable_rules[i : i + self.max_rules_per_merge]

                    # Check if rules in batch can really be merged
                    if len(batch) > 1 and self.validator.can_merge(batch):
                        merge_groups.append(
                            MergeGroup(
                                strategy=MergeStrategy.MERGED,
                                rules=batch,
                                target_database=database,
                                target_table=table,
                            )
                        )
                    else:
                        # Cannot merge, switch to individual execution
                        for rule in batch:
                            merge_groups.append(
                                MergeGroup(
                                    strategy=MergeStrategy.INDIVIDUAL,
                                    rules=[rule],
                                    target_database=database,
                                    target_table=table,
                                )
                            )
            else:
                # Individual execution
                for rule in mergeable_rules:
                    merge_groups.append(
                        MergeGroup(
                            strategy=MergeStrategy.INDIVIDUAL,
                            rules=[rule],
                            target_database=database,
                            target_table=table,
                        )
                    )

        return merge_groups

    def get_merge_strategy(
        self, rules: List[RuleSchema], table_size: int = 0
    ) -> MergeStrategy:
        """Determine merge strategy"""
        # Check if merge execution is enabled
        if not self.merge_execution_enabled:
            return MergeStrategy.INDIVIDUAL

        # Check table size threshold (if provided)
        if table_size > 0 and table_size < self.table_size_threshold:
            return MergeStrategy.INDIVIDUAL

        # Check rule count threshold
        if len(rules) < self.rule_count_threshold:
            return MergeStrategy.INDIVIDUAL

        # Check if there are independent rule types
        has_independent_rules = False
        has_mergeable_rules = False

        for rule in rules:
            rule_type = (
                rule.type.value if hasattr(rule.type, "value") else str(rule.type)
            )
            if rule_type in self.independent_rule_types:
                has_independent_rules = True
            else:
                has_mergeable_rules = True

        # If both independent and mergeable rules exist, return MIXED strategy
        if has_independent_rules and has_mergeable_rules:
            return MergeStrategy.MIXED

        # If only independent rules exist, return INDIVIDUAL strategy
        if has_independent_rules and not has_mergeable_rules:
            return MergeStrategy.INDIVIDUAL

        # Otherwise return MERGED strategy
        return MergeStrategy.MERGED


def get_rule_merger(connection: ConnectionSchema) -> RuleMergeManager:
    """
    Factory function to create rule merge manager

    Args:
        connection: Database connection config

    Returns:
        RuleMergeManager: Rule merge manager instance
    """
    return RuleMergeManager(connection=connection)
