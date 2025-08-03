"""
Rule execution engine module

Responsible for executing data quality rules and generating result reports.
Supports dynamic rule execution based on the rule type registry.
"""

# import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, OperationalError, SQLAlchemyError, TimeoutError
from sqlalchemy.ext.asyncio import AsyncEngine

from core.config import CoreConfig
from core.engine.prevalidation import Prevalidator
from core.engine.rule_merger import MergeGroup, RuleMergeManager
from core.executors import executor_registry
from shared.database.connection import check_connection, get_engine, retry_connection
from shared.exceptions import EngineError, RuleExecutionError
from shared.schema.connection_schema import ConnectionSchema as Connection
from shared.schema.result_schema import ExecutionResultSchema as ExecutionResult
from shared.schema.rule_schema import RuleSchema as Rule
from shared.utils.logger import get_logger

# Configure logger
# logger = get_logger(__name__)


class RuleGroup:
    """
    Rule group class

    Used to manage multiple rules for the same table, optimizing execution efficiency.
    Supports merged rule execution.
    """

    _merge_manager: Optional["RuleMergeManager"] = None

    def __init__(
        self, table_name: str, database: str, connection: Optional[Connection] = None
    ) -> None:
        """Initialize RuleGroup"""
        self.table_name = table_name
        self.database = database
        self.connection = connection  # Save original connection config

        # Get rule merge config parameters from the config system
        from core.config import get_core_config

        core_config = get_core_config()

        self.merge_execution_enabled = core_config.merge_execution_enabled
        self.table_size_threshold = core_config.table_size_threshold
        self.rule_count_threshold = core_config.rule_count_threshold
        self.max_rules_per_merge = core_config.max_rules_per_merge
        self.independent_rule_types = set(core_config.independent_rule_types)

        self.rules: List[Rule] = []
        self.column_rules: Dict[str, List[Rule]] = {}
        self.logger = get_logger(f"{__name__}.RuleGroup")

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the group"""
        try:
            self.rules.append(rule)
            target_info = rule.get_target_info()
            column = target_info.get("column")

            if column:
                if column not in self.column_rules:
                    self.column_rules[column] = []
                self.column_rules[column].append(rule)
                self.logger.debug(
                    f"Added rule {rule.id} for column {column} "
                    f"in table {self.table_name}"
                )
        except Exception as e:
            self.logger.error(f"Failed to add rule: {str(e)}")
            raise RuleExecutionError(f"Failed to add rule: {str(e)}")

    def get_column_rules(self, column_name: str) -> List[Rule]:
        """Get all rules for the specified column"""
        return self.column_rules.get(column_name, [])

    def get_all_columns(self) -> Set[str]:
        """Get all involved column names"""
        return set(self.column_rules.keys())

    def get_rule_types(self) -> Set[str]:
        """Get all rule types"""
        return {str(rule.type) for rule in self.rules}

    def _get_merge_manager(self, engine: AsyncEngine) -> "RuleMergeManager":
        """Get the rule merge manager"""
        if self._merge_manager is not None:
            return self._merge_manager

        try:
            # Fix architecture debt: use the real connection object to ensure
            # consistency with _execute_individual_group
            if not self.connection:
                raise RuleExecutionError(
                    "No connection available for rule merger initialization"
                )

            # Use the real connection object, only update database name when necessary
            from shared.schema.connection_schema import ConnectionSchema

            # Create a copy of the connection object to ensure the database name is
            # correct
            merger_connection = ConnectionSchema(
                name="rule_merger_connection",
                description=getattr(
                    self.connection, "description", "Connection for rule merger"
                ),
                connection_type=self.connection.connection_type,
                host=self.connection.host,
                port=self.connection.port,
                db_name=self.database,  # Use the current database name being
                # operated on
                username=self.connection.username,
                password=self.connection.password,
                db_schema=getattr(self.connection, "db_schema", None),
                file_path=getattr(self.connection, "file_path", None),
                parameters=getattr(self.connection, "parameters", {}),
                capabilities=getattr(self.connection, "capabilities", None),
                cross_db_settings=getattr(self.connection, "cross_db_settings", None),
            )

            from core.engine.rule_merger import get_rule_merger

            self._merge_manager = get_rule_merger(connection=merger_connection)

            return self._merge_manager
        except Exception as e:
            self.logger.error(f"Failed to create rule merge manager: {str(e)}")
            raise RuleExecutionError(f"Failed to create rule merge manager: {str(e)}")

    def _create_error_results_for_rules(
        self,
        rules: List[Rule],
        error_message: str,
        execution_time: float,
        log_message: Optional[str] = None,
    ) -> List[ExecutionResult]:
        """
        General method to create error results for the specified list of rules

        Args:
            rules: List of rules to create error results for
            error_message: Error message
            execution_time: Execution time
            log_message: Optional log message, if not provided uses default format

        Returns:
            List[ExecutionResult]: List of error results
        """
        error_results = []

        for rule in rules:
            error_result = ExecutionResult.create_error_result(
                rule_id=rule.id,
                entity_name=f"{self.database}.{self.table_name}",
                error_message=error_message,
                execution_time=execution_time,
            )
            error_results.append(error_result)

        # Log
        if log_message:
            self.logger.warning(log_message)
        else:
            self.logger.warning(
                f"Created error results for {len(rules)} rules: {error_message}"
            )

        return error_results

    async def execute(self, engine: AsyncEngine) -> List[ExecutionResult]:
        """Execute all rules in the group"""
        start_time = time.time()
        results = []

        try:
            self.logger.info(
                f"Starting rule execution for table {self.table_name} "
                f"with {len(self.rules)} rules"
            )

            # Get merge manager
            merge_manager = self._get_merge_manager(engine)

            # Directly analyze rules, no need for prevalidation
            # (already done at RuleEngine level)
            merge_groups = merge_manager.analyze_rules(self.rules)
            self.logger.info(
                f"Generated {len(merge_groups)} merge groups for table "
                f"{self.table_name}"
            )

            # Execute each merge group
            for group in merge_groups:
                group_start_time = time.time()
                group_execution_time = 0.0  # Initialize variable
                try:
                    if group.strategy.value == "merged":
                        # Merged execution
                        group_results = await self._execute_merged_group(
                            engine, group, merge_manager
                        )
                    else:
                        # Individual execution
                        group_results = await self._execute_individual_group(
                            engine, group
                        )

                    results.extend(group_results)

                    group_execution_time = time.time() - group_start_time
                    self.logger.info(
                        f"Executed {group.strategy.value} group with "
                        f"{len(group.rules)} rules "
                        f"in {group_execution_time:.2f}s"
                    )

                except Exception as e:
                    group_execution_time = time.time() - group_start_time

                    # Use the new error classification system
                    if isinstance(e, EngineError):
                        # Engine-level error: log and re-raise
                        self.logger.error(
                            f"Engine error in {group.strategy.value} group: "
                            f"{str(e)}\n"
                            f"{traceback.format_exc()}"
                        )
                        raise e
                    else:
                        # Rule-level error: create error results for each rule,
                        # continue with other groups
                        self.logger.error(
                            f"Rule error in {group.strategy.value} group: {str(e)}\n"
                            f"{traceback.format_exc()}"
                        )
                        # Use the general method to create error results
                        group_error_results = self._create_error_results_for_rules(
                            rules=group.rules,
                            error_message=f"Rule execution failed: {str(e)}",
                            execution_time=group_execution_time,
                        )
                        results.extend(group_error_results)
                        continue

            execution_time = time.time() - start_time
            self.logger.info(
                f"Completed rule execution for table {self.table_name} "
                f"in {execution_time:.2f}s"
            )

            return results

        except Exception as e:
            # Use the new error classification system
            if isinstance(e, EngineError):
                # Engine-level error: log and re-raise, do not convert
                self.logger.error(
                    f"Engine error in rule execution for table {self.table_name}: "
                    f"{str(e)}\n"
                    f"{traceback.format_exc()}"
                )
                raise e
            else:
                # Rule-level error: convert to RuleExecutionError
                self.logger.error(
                    f"Rule error in rule execution for table {self.table_name}: "
                    f"{str(e)}\n"
                    f"{traceback.format_exc()}"
                )
                raise RuleExecutionError(f"Rule execution failed: {str(e)}")

    async def _execute_merged_group(
        self,
        engine: AsyncEngine,
        group: "MergeGroup",
        merge_manager: "RuleMergeManager",
    ) -> List[ExecutionResult]:
        """Execute merged group"""
        try:
            # Get validation rule merger
            validation_merger = merge_manager.validator

            # Generate merged SQL
            merge_result = validation_merger.merge_rules(group.rules)

            self.logger.debug(f"Generated merged SQL: {merge_result.sql}")

            # Get total record count
            total_records = await self._get_total_records(engine)

            # Execute merged SQL
            execution_start = time.time()
            async with engine.begin() as conn:
                result = await conn.execute(text(merge_result.sql), merge_result.params)
                # Fix SQLAlchemy result row conversion issue - fetchall is not
                # async
                rows = result.fetchall()
                raw_results = [dict(row._mapping) for row in rows]

            merge_result.execution_time = time.time() - execution_start
            merge_result.total_records = total_records
            merge_result.results = raw_results

            # Parse results
            parsed_results = await validation_merger.parse_results(
                merge_result, raw_results
            )

            self.logger.info(
                f"Merged execution completed: {len(parsed_results)} rules, "
                f"{merge_result.total_records} records, "
                f"{merge_result.execution_time:.3f}s"
            )

            return parsed_results

        except Exception as e:
            error_msg = str(e)
            execution_time = (
                time.time() - execution_start if "execution_start" in locals() else 0.0
            )

            # Use the new error classification system
            if isinstance(e, EngineError):
                # Engine-level error: log and re-raise
                self.logger.error(f"Engine error in merged execution: {error_msg}")
                raise e
            else:
                # Rule-level error: create error results for each rule
                self.logger.error(f"Rule error in merged execution: {error_msg}")
                # Use the general method to create error results
                error_results = self._create_error_results_for_rules(
                    rules=group.rules,
                    error_message=f"Merged execution failed: {error_msg}",
                    execution_time=execution_time,
                )
                return error_results

    async def _execute_individual_group(
        self, engine: AsyncEngine, group: "MergeGroup"
    ) -> List[ExecutionResult]:
        """Execute individual group (using original logic)"""
        results = []

        for rule in group.rules:
            rule_start_time = time.time()
            rule_execution_time = 0.0  # Initialize execution time
            try:
                # Get rule type
                rule_type = str(rule.type)

                # Get rule executor - use ExecutorRegistry as per design doc
                executor_class = executor_registry.get_executor_for_rule_type(rule_type)

                # Use the original connection config, just update the name field
                from shared.schema.connection_schema import ConnectionSchema

                # Copy the original connection config
                connection = ConnectionSchema(
                    name=f"temp_connection_{rule.id}",
                    description=getattr(
                        self.connection,
                        "description",
                        "Temporary connection for rule execution",
                    ),
                    connection_type=(
                        self.connection.connection_type if self.connection else None
                    ),
                    host=self.connection.host if self.connection else None,
                    port=self.connection.port if self.connection else None,
                    db_name=self.connection.db_name if self.connection else None,
                    username=self.connection.username if self.connection else None,
                    password=self.connection.password if self.connection else None,
                    db_schema=getattr(self.connection, "db_schema", None),
                    file_path=getattr(self.connection, "file_path", None),
                    parameters=getattr(self.connection, "parameters", {}),
                    capabilities=getattr(self.connection, "capabilities", None),
                    cross_db_settings=getattr(
                        self.connection, "cross_db_settings", None
                    ),
                )

                # Use the new executor architecture
                executor = executor_class(connection)
                rule_results = await executor.execute_rules([rule])

                # Directly add ExecutionResultSchema objects, do not convert to dict
                results.extend(rule_results)

                rule_execution_time = time.time() - rule_start_time
                self.logger.debug(
                    f"Executed individual rule {rule.id} in {rule_execution_time:.3f}s"
                )

            except Exception as e:
                rule_execution_time = (
                    time.time() - rule_start_time
                )  # Calculate actual execution time

                # Use the new error classification system
                if isinstance(e, EngineError):
                    # Engine-level error: log and re-raise
                    self.logger.error(
                        f"Engine error executing individual rule {rule.id}: {str(e)}"
                    )
                    raise e
                else:
                    # Rule-level error: create error result
                    self.logger.error(
                        f"Rule error executing individual rule {rule.id}: {str(e)}"
                    )
                    # Use the general method to create error results
                    error_results = self._create_error_results_for_rules(
                        rules=[rule],
                        error_message=f"Rule execution failed: {str(e)}",
                        execution_time=rule_execution_time,
                        log_message=(
                            f"Individual rule {rule.id} execution failed: {str(e)}"
                        ),
                    )
                    results.extend(error_results)

        return results

    async def _get_total_records(self, engine: AsyncEngine) -> int:
        """Get total record count"""
        try:
            # Check database type and construct correct query
            db_url = str(engine.url)

            if "postgresql" in db_url.lower():
                # PostgreSQL uses public schema
                query = text(f"SELECT COUNT(*) FROM public.{self.table_name}")
            elif "sqlite" in db_url.lower():
                # SQLite does not use schema
                query = text(f"SELECT COUNT(*) FROM {self.table_name}")
            else:
                # MySQL uses database.table format
                query = text(f"SELECT COUNT(*) FROM {self.database}.{self.table_name}")

            async with engine.begin() as conn:
                result = await conn.execute(query)
                row = result.fetchone()  # fetchone is not async
                if row:
                    # Handle possible coroutine object (in test environment)
                    if hasattr(row, "__await__"):
                        row = await row  # type: ignore[attr-defined]
                    return row[0] if row else 0
                return 0
        except (OperationalError, TimeoutError, DBAPIError) as e:
            # Connection-level error, raise exception
            error_msg = str(e)
            self.logger.error(
                f"Database connection error when getting total records: {error_msg}"
            )
            raise EngineError(f"Database connection failed: {error_msg}")
        except SQLAlchemyError as e:
            # Other SQLAlchemy errors (e.g. table does not exist), log but return 0
            error_msg = str(e)
            self.logger.warning(
                f"Failed to get total records (table likely does not exist): "
                f"{error_msg}"
            )
            return 0
        except Exception as e:
            # Unexpected error, raise exception
            error_msg = str(e)
            self.logger.error(f"Unexpected error getting total records: {error_msg}")
            raise EngineError(f"Unexpected database error: {error_msg}")


class RuleEngine:
    """
    Rule engine class

    Responsible for executing data quality rule validation,
    supporting multiple rule types:
    - NOT NULL check
    - Uniqueness check
    - Range check
    - Enum value check
    - Regular expression check
    - Date format check
    - Length check
    - Custom SQL check
    - And other rule types registered via the rule type registry
    """

    def __init__(
        self,
        connection: Connection,
        core_config: Optional["CoreConfig"] = None,
        prevalidator: "Prevalidator | None" = None,
    ):
        """
        Initialize rule engine - refactored interface

        Args:
            connection: Database connection info (execution context)
            core_config: Core config (optional, will be auto-fetched if not provided)
        """
        self.connection = connection

        # Get config
        if core_config is None:
            from core.config import get_core_config

            core_config = get_core_config()

        self.merge_execution_enabled = core_config.merge_execution_enabled
        self.table_size_threshold = core_config.table_size_threshold
        self.rule_count_threshold = core_config.rule_count_threshold
        self.max_rules_per_merge = core_config.max_rules_per_merge
        self.independent_rule_types = set(core_config.independent_rule_types)

        self.engine: Optional[AsyncEngine] = None
        self.logger = get_logger("rule_engine")

        # Injectable prevalidation strategy; tests can pass in NoopPrevalidator
        self.prevalidator = prevalidator

        # Check for extra logging config
        try:
            from shared.config import get_typed_config
            from shared.config.logging_config import LoggingConfig

            # Get logging config
            logging_config = get_typed_config("logging", LoggingConfig)
            if logging_config and logging_config.level.upper() == "DEBUG":
                self.logger.debug(
                    f"Rule engine initialized: connection="
                    f"{self.connection.connection_type}, merge execution: "
                    f"{self.merge_execution_enabled}"
                )
        except ImportError:
            # If config module is not available, continue using default log level
            pass

    async def execute(self, rules: List[Rule]) -> List[ExecutionResult]:
        """
        Execute rule list - refactored interface, includes prevalidation step

        Args:
            rules: List of rules to execute
            (pure rule definitions, not tied to connection)

        Returns:
            List[Dict[str, Any]]: List of rule execution results
        """
        start_time = time.time()

        try:
            self.logger.info(
                f"Starting rule engine execution, total {len(rules)} rules"
            )

            # Get database engine
            self.engine = await self._get_engine()
            if not self.engine:
                connection_id = self.connection.name if self.connection else None
                raise EngineError(
                    "Unable to connect to database", connection_id=connection_id
                )

            # If no prevalidator provided, use DatabasePrevalidator by default
            if self.prevalidator is None:
                # Select appropriate prevalidator based on engine type
                from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

                from core.engine.prevalidation import (
                    DatabasePrevalidator,
                    NoopPrevalidator,
                )

                if isinstance(self.engine, (AsyncEngine, AsyncConnection)):
                    self.prevalidator = DatabasePrevalidator(self.engine, self.logger)
                else:
                    # Unit test or sync engine scenario, use NoopPrevalidator to avoid
                    # real DB access
                    self.prevalidator = NoopPrevalidator()  # type: ignore[unreachable]

            # If externally injected prevalidator needs engine but not set,
            # inject dependency
            try:
                from core.engine.prevalidation import DatabasePrevalidator

                if isinstance(self.prevalidator, DatabasePrevalidator):
                    # Allow constructing DatabasePrevalidator(None) in tests, then
                    # assign engine
                    self.prevalidator.engine = self.engine
            except ImportError:
                pass

            # Stage 1: batch prevalidation (moved up logic)
            try:
                prevalidation_results = await self._batch_prevalidate_rules(rules)
            except EngineError as e:
                # System-level exception: stop all execution
                self.logger.error(f"Prevalidation system error: {e.get_context_info()}")
                raise e
            except Exception as e:
                # Prevalidation failed, convert to system-level exception
                self.logger.error(f"Prevalidation failed: {str(e)}")
                raise EngineError(
                    message=f"Rule prevalidation failed: {str(e)}",
                    connection_id=str(self.connection.id) if self.connection else None,
                    operation="prevalidation",
                    cause=e,
                )

            # Stage 2: group rules based on prevalidation results
            valid_groups, invalid_rules = self._group_rules_with_validation(
                rules, prevalidation_results
            )

            # Stage 3: create error results for invalid rules
            results = []
            execution_time = time.time() - start_time
            for invalid_rule in invalid_rules:
                error_result = self._create_validation_error_result(
                    invalid_rule, execution_time
                )
                results.append(error_result)

            # Stage 4: execute valid rule groups
            for group in valid_groups:
                try:
                    group_results = await group.execute(self.engine)
                    results.extend(group_results)

                except EngineError as e:
                    # System-level exception: stop all execution
                    self.logger.error(f"System error: {e.get_context_info()}")
                    raise e

                except Exception as e:
                    # Other exceptions: skip this group, continue with others
                    self.logger.warning(f"Group execution error: {str(e)}")
                    error_results = self._create_group_error_results(
                        group, e, execution_time
                    )
                    results.extend(error_results)
                    continue

            execution_time = time.time() - start_time
            self.logger.info(
                f"Rule engine execution completed, time elapsed: "
                f"{execution_time:.2f}s"
            )

            return results

        except EngineError as e:
            # System-level exception, propagate directly
            raise e
        except Exception as e:
            # Unexpected exception, convert to system-level exception
            self.logger.error(f"Unexpected error in rule execution: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise EngineError(
                message=f"Unexpected error in rule execution: {str(e)}",
                connection_id=str(self.connection.id) if self.connection else None,
                operation="rule_execution",
                cause=e,
            )

    async def _batch_prevalidate_rules(self, rules: List[Rule]) -> Dict[str, Any]:
        """Delegate prevalidation to the injected strategy object."""
        assert self.prevalidator is not None
        return await self.prevalidator.validate(rules)

    def _group_rules_with_validation(
        self, rules: List[Rule], validation_results: Dict[str, Any]
    ) -> Tuple[List[RuleGroup], List[Rule]]:
        """Group rules based on prevalidation results"""

        valid_rules = []
        invalid_rules = []

        for rule in rules:
            target_info = rule.get_target_info()
            database = target_info.get("database")
            table = target_info.get("table")
            column = target_info.get("column")

            if not (database and table):
                rule.validation_error = "Missing database or table information"
                invalid_rules.append(rule)
                continue

            entity_key = f"{database}.{table}"
            column_key = f"{entity_key}.{column}" if column else None

            # Check if table and column exist
            table_exists = validation_results["tables"].get(entity_key, False)
            column_exists = (
                validation_results["columns"].get(column_key, True)
                if column_key
                else True
            )

            if not table_exists:
                rule.validation_error = f"Table {entity_key} does not exist"
                invalid_rules.append(rule)
            elif column and not column_exists:
                rule.validation_error = f"Column {column_key} does not exist"
                invalid_rules.append(rule)
            else:
                valid_rules.append(rule)

        # Group valid rules (using existing logic)
        valid_groups = self._group_rules(valid_rules)

        return list(valid_groups.values()), invalid_rules

    def _create_validation_error_result(
        self, rule: Rule, execution_time: float
    ) -> ExecutionResult:
        """Create error result for a rule that failed validation"""
        target_info = rule.get_target_info()
        database = target_info.get("database", "")
        table = target_info.get("table", "")
        entity_name = f"{database}.{table}" if database and table else "unknown"

        error_message = rule.validation_error or "Validation failed"

        return ExecutionResult.create_error_result(
            rule_id=rule.id,
            entity_name=entity_name,
            error_message=error_message,
            execution_time=execution_time,
        )

    def _create_group_error_results(
        self, group: RuleGroup, error: Exception, execution_time: float
    ) -> List[ExecutionResult]:
        """Create error results for the entire rule group"""
        error_results = []
        error_message = f"Group execution failed: {str(error)}"

        for rule in group.rules:
            error_result = ExecutionResult.create_error_result(
                rule_id=rule.id,
                entity_name=f"{group.database}.{group.table_name}",
                error_message=error_message,
                execution_time=execution_time,
            )
            error_results.append(error_result)

        return error_results

    def _group_rules(self, rules: List[Rule]) -> Dict[str, RuleGroup]:
        """Group rules by table"""
        rule_groups: Dict[str, RuleGroup] = {}

        for rule in rules:
            try:
                # Get target info
                target_info = rule.get_target_info()
                database = target_info.get("database", "")
                table = target_info.get("table", "")

                if not database or not table:
                    self.logger.warning(f"Rule {rule.id} missing target info, skipping")
                    continue

                # Create group key
                key = f"{database}.{table}"

                # Add to rule group
                if key not in rule_groups:
                    rule_groups[key] = RuleGroup(
                        table_name=table,
                        database=database,
                        connection=self.connection,
                    )
                rule_groups[key].add_rule(rule)
            except Exception as e:
                self.logger.error(f"Failed to group rule {rule.id}: {str(e)}")
                continue

        return rule_groups

    async def _get_engine(self) -> Optional[AsyncEngine]:
        """
        Get database engine

        Fully reuses shared/database/connection.py functionality to avoid all
        duplicate code.
        Keeps business-layer error handling and translation logic.

        Returns:
            Optional[AsyncEngine]: Returns database engine on success, None on failure
        """
        try:
            # Use database name from connection config, not from rule
            database = self.connection.db_name or ""

            # Use shared get_db_url function to build connection URL
            from shared.database.connection import get_db_url

            # Ensure correct data type is passed to get_db_url
            # connection_type = self.connection.connection_type
            # if hasattr(connection_type, "value"):
            #     # If it's an enum object, use its value
            #     connection_type = connection_type.value

            db_url = get_db_url(
                db_type=self.connection.connection_type,
                host=self.connection.host,
                port=self.connection.port,
                database=database,
                username=self.connection.username,
                password=self.connection.password,
                file_path=getattr(self.connection, "file_path", None),
            )

            # First check connection
            if not await check_connection(db_url):
                # Use shared retry mechanism
                engine = await retry_connection(db_url)
                if not engine:
                    connection_id = self.connection.name if self.connection else None
                    raise EngineError(
                        "Unable to connect to database", connection_id=connection_id
                    )
                return engine

            # Use shared engine getter
            engine = await get_engine(db_url)

            # Validate engine health
            try:
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                self.logger.debug("Engine health check passed")
                return engine
            except Exception as health_error:
                self.logger.warning(
                    f"Engine health check failed: {health_error}, "
                    "attempting to recreate"
                )
                # If health check fails, try to recreate engine
                engine = await retry_connection(db_url)
                if not engine:
                    connection_id = self.connection.name if self.connection else None
                    raise EngineError(
                        "Unable to connect to database", connection_id=connection_id
                    )
                return engine

        except OperationalError as e:
            # Business-layer error translation remains here - these are all
            # engine-level errors
            error_msg = str(e).lower()
            connection_id = self.connection.name if self.connection else None
            if "unknown database" in error_msg:
                raise EngineError("Database not found", connection_id=connection_id)
            elif "access denied" in error_msg:
                raise EngineError("Access denied", connection_id=connection_id)
            elif "unknown mysql server host" in error_msg:
                raise EngineError(
                    "Cannot resolve hostname", connection_id=connection_id
                )
            elif "connection timed out" in error_msg:
                raise EngineError("Connection timeout", connection_id=connection_id)
            else:
                # Other operation errors, try retry
                try:
                    # Rebuild URL just in case
                    from shared.database.connection import get_db_url

                    # Ensure correct data type is passed to get_db_url
                    # connection_type = self.connection.connection_type
                    # if hasattr(connection_type, "value"):
                    #     connection_type = connection_type.value

                    db_url = get_db_url(
                        db_type=self.connection.connection_type,
                        host=self.connection.host,
                        port=self.connection.port,
                        database=database,
                        username=self.connection.username,
                        password=self.connection.password,
                        file_path=getattr(self.connection, "file_path", None),
                    )
                    engine = await retry_connection(db_url)
                    if not engine:
                        raise EngineError(
                            "Connection retry failed", connection_id=connection_id
                        )
                    return engine
                except Exception as retry_error:
                    raise EngineError(
                        f"Connection failed after retry: {str(retry_error)}",
                        connection_id=connection_id,
                    )
        except EngineError as e:
            # Engine-level errors possibly thrown by get_db_url
            connection_id = self.connection.name if self.connection else None
            raise EngineError(
                f"Database connection configuration error: {str(e)}",
                connection_id=connection_id,
            )
        except Exception as e:
            # Other unexpected errors - these are engine-level errors
            self.logger.error(f"Unexpected error getting database engine: {str(e)}")
            connection_id = self.connection.name if self.connection else None
            raise EngineError(
                f"Unexpected error getting database engine: {str(e)}",
                connection_id=connection_id,
            )
