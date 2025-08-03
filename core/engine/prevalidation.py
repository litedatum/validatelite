"""Prevalidation strategy objects used by RuleEngine.

This module decouples the rule-prevalidation step from RuleEngine so that:
1. Production code uses DatabasePrevalidator — batch check for table/column
   existence via QueryExecutor;
2. Unit tests can inject NoopPrevalidator — always returns "all exist",
   avoiding dependency on real database or async engine;

Both classes implement the same `validate()` coroutine interface.
"""

from __future__ import annotations

import time
from logging import Logger
from typing import Any, Dict, List, Optional, Protocol, Set

from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from shared.database.query_executor import QueryExecutor
from shared.schema.rule_schema import RuleSchema as Rule
from shared.utils.logger import get_logger

# --------------------------------------------------------------------------------------
# Public protocol
# --------------------------------------------------------------------------------------


class Prevalidator(Protocol):
    """The interface expected by RuleEngine for any prevalidation strategy."""

    async def validate(self, rules: List[Rule]) -> Dict[str, Any]:
        """Return {'tables': {...}, 'columns': {...}} for given rules."""
        ...


# --------------------------------------------------------------------------------------
# No-op strategy (used by tests)
# --------------------------------------------------------------------------------------


class NoopPrevalidator:  # pragma: no cover – trivial implementation
    """A prevalidator that *assumes* every referenced table and column exists.

    Useful in unit tests that mock away real database interaction.
    """

    async def validate(self, rules: List[Rule]) -> Dict[str, Any]:  # noqa: D401
        """Validate rules"""
        table_results: Dict[str, bool] = {}
        column_results: Dict[str, bool] = {}

        for rule in rules:
            target_info = rule.get_target_info()
            database = target_info.get("database")
            table = target_info.get("table")
            column = target_info.get("column")
            if not (database and table):
                continue
            entity_key = f"{database}.{table}"
            table_results[entity_key] = True
            if column:
                column_results[f"{entity_key}.{column}"] = True

        return {"tables": table_results, "columns": column_results}


# --------------------------------------------------------------------------------------
# Database-backed strategy (production)
# --------------------------------------------------------------------------------------


class DatabasePrevalidator:
    """Use QueryExecutor to perform real table/column existence checks."""

    def __init__(
        self, engine: AsyncEngine | AsyncConnection, logger: Optional[Logger] = None
    ) -> None:
        """Initialize DatabasePrevalidator"""
        self.engine = engine
        self.logger = logger or get_logger(__name__)

    async def validate(self, rules: List[Rule]) -> Dict[str, Any]:  # noqa: D401
        """Validate rules"""
        start_time = time.time()
        query_executor = QueryExecutor(self.engine, self.logger)

        # Step 1 – build inverted index of requirements --------------------------------
        table_requirements: Dict[str, Dict[str, Set[str]]] = {}
        all_column_keys: Set[str] = set()

        for rule in rules:
            target_info = rule.get_target_info()
            database = target_info.get("database")
            table = target_info.get("table")
            column = target_info.get("column")

            if not (database and table):
                continue

            table_key = f"{database}.{table}"
            if table_key not in table_requirements:
                table_requirements[table_key] = {"columns": set()}

            if column:
                table_requirements[table_key]["columns"].add(column)
                all_column_keys.add(f"{table_key}.{column}")

        # Step 2 – initialise result maps ---------------------------------------------
        table_results: Dict[str, bool] = {
            key: False for key in table_requirements.keys()
        }
        column_results: Dict[str, bool] = {key: False for key in all_column_keys}

        # Step 3 – query database for each table --------------------------------------
        for table_key, requirements in table_requirements.items():
            try:
                database, table = table_key.split(".", 1)
                required_columns = requirements["columns"]

                # first rule id for context (if any)
                rule_id: Optional[str] = next(
                    (
                        rule.id
                        for rule in rules
                        if rule.get_target_info().get("database") == database
                        and rule.get_target_info().get("table") == table
                    ),
                    None,
                )

                # Table existence ----------------------------------------------------
                exists = await query_executor.table_exists(
                    table,
                    database=database,
                    entity_name=table_key,
                    resource_type="table",
                    rule_id=rule_id,
                )
                table_results[table_key] = exists

                if not exists:
                    # Table missing → all related columns remain False
                    continue

                if not required_columns:
                    continue

                # Column list --------------------------------------------------------
                actual_columns_info = await query_executor.get_column_list(
                    table,
                    database=database,
                    entity_name=table_key,
                    resource_type="column",
                    rule_id=rule_id,
                )
                # Use the normalized 'name' field
                actual_columns = {col["name"] for col in actual_columns_info}

                for col_name in required_columns:
                    column_key = f"{table_key}.{col_name}"
                    if col_name in actual_columns:
                        column_results[column_key] = True

            except Exception as exc:  # noqa: BLE001
                self.logger.error(
                    "Prevalidation DB access failed for %s: %s", table_key, exc
                )
                self.logger.exception(exc)
                raise  # keep original exception semantics

        self.logger.debug("Prevalidation completed in %.3fs", time.time() - start_time)
        return {"tables": table_results, "columns": column_results}
