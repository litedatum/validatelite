"""
Data Validator

Core component that integrates CLI with the rule engine.
Handles data loading, rule execution, and result processing.
"""

from pathlib import Path
from typing import Any, Dict, List, cast

import pandas as pd

from cli.core.config import CliConfig

# Import new configuration models
from core.config import CoreConfig
from core.engine.rule_engine import RuleEngine
from shared.enums import (
    ConnectionType,
    RuleAction,
    RuleCategory,
    RuleType,
    SeverityLevel,
)
from shared.schema import ConnectionSchema, ExecutionResultSchema, RuleSchema
from shared.schema.base import RuleTarget, TargetEntity
from shared.utils.logger import get_logger


class DataValidator:
    """
    Data validator that coordinates data loading and rule execution.

    Supports both file-based and database-based validation.
    Refactored to remove connection ID management - rules are now connection-agnostic.
    """

    def __init__(
        self,
        source_config: "ConnectionSchema | Dict[str, Any]",
        rules: "List[RuleSchema | Dict[str, Any]]",
        core_config: CoreConfig,
        cli_config: CliConfig,
    ):
        """Create a new *DataValidator* instance.

        The public API must be flexible enough to accept either fully-fledged
        ``ConnectionSchema`` / ``RuleSchema`` objects **or** their plain dict
        equivalents because some unit-tests (and likely end-users of the CLI)
        find it more convenient to supply dictionaries.  We convert these
        dictionaries to the appropriate schema objects up-front so that the
        remainder of the implementation can operate on a well-defined type
        contract.
        """

        # ------------------------------------------------------------------
        # Source configuration – allow dicts for convenience during testing.
        # ------------------------------------------------------------------
        if isinstance(source_config, dict):
            self.source_config = self._convert_source_dict(source_config)
        else:
            self.source_config = source_config

        # ConnectionSchema now has its own auto-generated ID, no need to inject one

        # ------------------------------------------------------------------
        # Rules – normalise to ``RuleSchema`` instances.
        # ------------------------------------------------------------------
        self.rules: List[RuleSchema] = []
        for rule in rules:
            if isinstance(rule, RuleSchema):
                self.rules.append(rule)
            elif isinstance(rule, dict):
                self.rules.append(self._convert_rule_dict(rule))
            else:
                raise TypeError(
                    "Each rule must be a RuleSchema or a dict, got "
                    f"{type(rule).__name__}"
                )

        self.core_config = core_config
        self.cli_config = cli_config
        # Expose commonly used CLI-tunable parameters for easier discovery.
        self.sample_size: int = cli_config.default_sample_size
        self.logger = get_logger(__name__)

        # Check for additional logging configuration
        try:
            from shared.config import get_typed_config
            from shared.config.logging_config import LoggingConfig

            # Get logging configuration
            logging_config = get_typed_config("logging", LoggingConfig)
            if logging_config and logging_config.level.upper() == "DEBUG":
                self.logger.debug(
                    f"Data validator initialized: Source="
                    f"{self.source_config.connection_type}, "
                    f"Rules={len(self.rules)}"
                )
        except ImportError:
            # If the configuration module is not available, continue with the
            # default log level
            pass

        # Update database and table names in rules based on source config
        self._complete_target_info()

    def _complete_target_info(self) -> None:
        """
        Complete database and table names in rules based on source config.

        This replaces the old _update_rule_connections method.
        """
        # If the source is multi-table, targets are already set. Do not overwrite.
        if self.source_config.parameters.get("is_multi_table"):
            self.logger.debug(
                "Multi-table source detected, skipping target info completion."
            )
            return

        if not self.rules:
            return

        # Determine database name from source config
        if self.source_config.connection_type == ConnectionType.SQLITE:
            db_name = "main"  # SQLite main database
        elif self.source_config.connection_type in [
            ConnectionType.CSV,
            ConnectionType.EXCEL,
            ConnectionType.JSON,
        ]:
            db_name = "main"  # File-based sources use SQLite internally
        else:
            db_name = self.source_config.db_name or "default"

        # Determine table name from source config
        table_name = None
        if "table" in self.source_config.parameters:
            table_name = self.source_config.parameters["table"]
        elif self.source_config.connection_type in [
            ConnectionType.CSV,
            ConnectionType.EXCEL,
            ConnectionType.JSON,
        ]:
            if self.source_config.file_path:
                # Extract table name from file path
                file_path = Path(self.source_config.file_path)
                table_name = self._clean_table_name(file_path.stem)
            else:
                table_name = "data"  # Default for files without path
        else:
            table_name = "default_table"  # Default for database connections

        # Update all rules
        for rule in self.rules:
            for entity in rule.target.entities:
                entity.database = db_name
                entity.table = table_name

    async def validate(self) -> List[ExecutionResultSchema]:
        """
        Execute validation rules against the data source.

        Returns:
            List[ExecutionResultSchema]: Validation results
        """
        self.logger.info(f"Starting validation with {len(self.rules)} rules")
        from shared.exceptions.exception_system import EngineError

        try:
            if self.source_config.connection_type in [
                ConnectionType.CSV,
                ConnectionType.EXCEL,
                ConnectionType.JSON,
            ]:
                return await self._validate_file()
            else:
                return await self._validate_database()

        except EngineError as e:
            # Allow EngineError exceptions to propagate to the CLI layer
            self.logger.error(f"Engine error during validation: {str(e)}")
            print(f"DEBUG: Raising EngineError: {str(e)}")
            raise e
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            # Return error results for all rules
            return self._create_error_results(str(e))

    async def _validate_file(self) -> List[ExecutionResultSchema]:
        """Validate file-based data source"""
        self.logger.info(f"Validating file: {self.source_config.file_path}")

        # Check if this is a multi-table Excel file
        is_multi_table = self.source_config.parameters.get("is_multi_table", False)
        self.logger.info(
            f"Multi-table detection: is_multi_table={is_multi_table}, "
            f"connection_type={self.source_config.connection_type}"
        )
        self.logger.info(f"Source config parameters: {self.source_config.parameters}")

        if (
            is_multi_table
            and self.source_config.connection_type == ConnectionType.EXCEL
        ):
            # Handle multi-table Excel file
            self.logger.info("Processing multi-table Excel file")
            sqlite_config = await self._convert_multi_table_excel_to_sqlite()
        else:
            # Handle single-table file (existing logic)
            self.logger.info("Processing single-table file")
            try:
                df = self._load_file_data()
                self.logger.info(f"Loaded {len(df)} records from file")
            except Exception as e:
                raise ValueError(f"Failed to load file data: {str(e)}")

            # Convert to SQLite for rule engine processing
            sqlite_config = await self._convert_file_to_sqlite(df)

        # Execute rules using rule engine with new interface
        rule_engine = RuleEngine(connection=sqlite_config, core_config=self.core_config)
        results = await rule_engine.execute(rules=self.rules)

        # Convert results to list of dicts before passing to _process_results
        return self._process_results([r.model_dump() for r in results])

    async def _validate_database(self) -> List[ExecutionResultSchema]:
        """Validate database-based data source"""
        self.logger.info(f"Validating database: {self.source_config.connection_type}")

        # Execute rules using rule engine with new interface
        rule_engine = RuleEngine(
            connection=self.source_config, core_config=self.core_config
        )
        results = await rule_engine.execute(rules=self.rules)

        return self._process_results([r.model_dump() for r in results])

    def _load_file_data(self) -> pd.DataFrame:
        """Load data from file into pandas DataFrame"""
        original_path_str = self.source_config.file_path
        file_path = Path(original_path_str if original_path_str else ".")

        if not file_path.exists():
            # Preserve the *exact* path string supplied by the caller to avoid
            # surprises in an OS-agnostic unit-test environment.
            raise FileNotFoundError(f"File not found: {original_path_str}")

        # Check file size using CLI configuration
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        max_size_mb = getattr(self.cli_config, "max_file_size_mb", 100)
        try:
            max_size_mb = float(max_size_mb)
        except (TypeError, ValueError):
            # If the config provides a non-numeric (e.g., MagicMock) just skip
            # the size check
            max_size_mb = None

        if max_size_mb is not None and file_size_mb > max_size_mb:
            raise ValueError(
                f"File size ({file_size_mb:.1f} MB) exceeds maximum allowed size "
                f"({max_size_mb} MB)"
            )

        # Load based on file type
        conn_type = self.source_config.connection_type
        encoding = self.source_config.parameters.get("encoding", "utf-8")

        try:
            if conn_type == ConnectionType.CSV:
                # Try different separators
                for sep in [",", ";", "\t"]:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                        if len(df.columns) > 1:  # Successfully parsed multiple columns
                            break
                    except Exception:
                        continue
                else:
                    # Fallback to default comma separator
                    df = pd.read_csv(file_path, encoding=encoding)

            elif conn_type == ConnectionType.EXCEL:
                try:
                    df = pd.read_excel(file_path, engine="openpyxl")
                except ModuleNotFoundError:
                    # Gracefully downgrade if openpyxl is missing so that the
                    # overall test-suite can still run.  We fall back to the
                    # older *xlrd* engine for legacy *.xls files.
                    try:
                        df = pd.read_excel(file_path, engine="xlrd")
                    except Exception as e:
                        raise ValueError(f"Failed to parse file: {str(e)}") from e

            elif conn_type == ConnectionType.JSON:
                df = pd.read_json(file_path, lines=file_path.suffix == ".jsonl")

            else:
                raise ValueError(f"Unsupported file type: {conn_type}")

            if df.empty:
                raise ValueError("File contains no data")

            self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df

        except pd.errors.EmptyDataError:
            raise ValueError("File contains no data")
        except pd.errors.ParserError as e:
            # Try a second time skipping bad lines – this allows the validator
            # to recover from *minor* corruption such as inconsistent column
            # counts whilst still surfacing fatal issues.
            try:
                df = pd.read_csv(file_path, encoding=encoding, on_bad_lines="skip")
                if not df.empty:
                    self.logger.warning(
                        "Parser error encountered – loaded file with bad lines skipped"
                    )
                    return df
            except Exception:
                pass

            if "No columns to parse from file" in str(e):
                raise ValueError("File contains no data")

            raise ValueError(f"Failed to parse file: {str(e)}")

        except PermissionError as e:
            # Surface permission issues as a unified parse failure so that the
            # caller can handle it uniformly.
            raise ValueError(f"Failed to parse file: {str(e)}") from e

        except Exception as e:
            raise ValueError(f"Failed to parse file: {str(e)}")

    async def _convert_multi_table_excel_to_sqlite(self) -> ConnectionSchema:
        """
        Convert multi-table Excel file to SQLite database.

        Returns:
            ConnectionSchema: SQLite connection configuration
        """
        import os
        import tempfile
        import time

        from sqlalchemy import create_engine

        temp_db_file = None
        temp_db_path = None
        start_time = time.time()

        try:
            # Create a temporary SQLite file
            temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            temp_db_path = temp_db_file.name
            temp_db_file.close()

            # Create SQLite engine
            engine = create_engine(f"sqlite:///{temp_db_path}")

            # Load all sheets into SQLite
            await self._load_multi_table_excel_to_sqlite(engine, temp_db_path)

            # Get table mapping for connection config
            table_mapping = self.source_config.parameters.get("table_mapping", {})

            # Create connection config with multi-table information
            sqlite_config = ConnectionSchema(
                name="temp_sqlite_multi_table",
                description="Temporary SQLite for multi-table Excel validation",
                connection_type=ConnectionType.SQLITE,
                file_path=temp_db_path,
                parameters={
                    "is_multi_table": True,
                    "table_mapping": table_mapping,
                    "temp_file": True,  # Mark as temporary file for cleanup
                },
            )

            # Log performance metrics
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"Created temporary SQLite database at {temp_db_path} with "
                f"{len(table_mapping)} tables in {elapsed_time:.2f} seconds"
            )

            return sqlite_config

        except Exception as e:
            # Clean up temporary file if it exists
            if temp_db_path and os.path.exists(temp_db_path):
                try:
                    os.unlink(temp_db_path)
                except Exception as cleanup_error:
                    self.logger.warning(
                        f"Failed to cleanup temporary file {temp_db_path}: "
                        f"{cleanup_error}"
                    )
            raise ValueError(f"Failed to create multi-table SQLite database: {str(e)}")

    async def _load_multi_table_excel_to_sqlite(
        self, engine: Any, temp_db_path: str
    ) -> None:
        """
        Load multiple sheets from Excel file into SQLite database.

        Args:
            engine: SQLAlchemy engine for SQLite
            temp_db_path: Path to temporary SQLite database
        """
        import pandas as pd

        file_path = self.source_config.file_path
        sheets_info = self.source_config.parameters.get("sheets", {})

        if not sheets_info:
            raise ValueError(
                "Multi-table Excel file but no sheets information available"
            )

        self.logger.info(
            f"Loading {len(sheets_info)} sheets into SQLite: {list(sheets_info.keys())}"
        )

        # Store table name mapping for later use
        table_mapping = {}

        # Load each sheet into a separate table
        for sheet_name, columns in sheets_info.items():
            try:
                # Read the specific sheet
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")

                # Validate that the sheet has the expected columns
                expected_columns = set(columns)
                actual_columns = set(df.columns)

                if not expected_columns.issubset(actual_columns):
                    missing_columns = expected_columns - actual_columns
                    self.logger.warning(
                        f"Sheet '{sheet_name}' missing expected columns: "
                        f"{missing_columns}"
                    )

                # Write to SQLite with sheet name as table name
                # Clean table name for SQLite (remove special characters)
                clean_table_name = "".join(
                    c for c in sheet_name if c.isalnum() or c == "_"
                )
                if not clean_table_name or clean_table_name[0].isdigit():
                    clean_table_name = f"sheet_{clean_table_name}"

                # Store the mapping from original sheet name to clean table name
                table_mapping[sheet_name] = clean_table_name

                df.to_sql(clean_table_name, engine, if_exists="replace", index=False)
                self.logger.info(
                    f"Loaded sheet '{sheet_name}' as table '{clean_table_name}' "
                    f"with {len(df)} rows"
                )

            except Exception as e:
                self.logger.error(f"Failed to load sheet '{sheet_name}': {str(e)}")
                # Continue with other sheets
                continue

        # Store the table mapping in the source config for later use
        if hasattr(self, "source_config") and hasattr(self.source_config, "parameters"):
            self.source_config.parameters["table_mapping"] = table_mapping
            self.logger.info(f"Stored table mapping: {table_mapping}")

    async def _convert_file_to_sqlite(self, df: pd.DataFrame) -> ConnectionSchema:
        """
        Convert pandas DataFrame to SQLite in-memory database

        Args:
            df: Pandas DataFrame with data

        Returns:
            ConnectionSchema: SQLite connection configuration
        """
        import os
        import sqlite3
        import tempfile
        import time

        from sqlalchemy import create_engine

        temp_db_file = None
        temp_db_path = None
        start_time = time.time()

        try:
            # Create a temporary SQLite file
            # We use a file instead of :memory: to avoid potential threading issues
            temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            temp_db_path = temp_db_file.name
            temp_db_file.close()

            # Test SQLite connection directly before creating SQLAlchemy engine
            try:
                # This will raise an exception if SQLite connection fails
                conn = sqlite3.connect(temp_db_path)
                conn.close()
            except sqlite3.Error as e:
                self.logger.error(f"Failed to connect to SQLite database: {str(e)}")
                raise ValueError(f"Failed to create temporary database: {str(e)}")

            # Create SQLite engine
            try:
                engine = create_engine(f"sqlite:///{temp_db_path}")
            except Exception as e:
                self.logger.error(f"Failed to create SQLite engine: {str(e)}")
                raise ValueError(f"Failed to create temporary database: {str(e)}")

            # Clean column names and handle duplicates
            original_columns = list(df.columns)
            cleaned_columns = []
            column_counts: Dict[str, int] = {}

            for col in original_columns:
                cleaned = self._clean_column_name(col)

                # Handle duplicate column names
                if cleaned in cleaned_columns:
                    # If we've seen this cleaned name before, add a suffix
                    column_counts[cleaned] = column_counts.get(cleaned, 0) + 1
                    cleaned = f"{cleaned}_{column_counts[cleaned]}"

                cleaned_columns.append(cleaned)

            # Update DataFrame columns
            df.columns = cleaned_columns

            # Determine table name from source config
            if (
                self.source_config.parameters
                and "table" in self.source_config.parameters
            ):
                # Use table name from parameters if available
                table_name = self.source_config.parameters["table"]
            elif self.source_config.file_path:
                # Extract table name from file path
                file_path = Path(self.source_config.file_path)
                table_name = self._clean_table_name(file_path.stem)
            else:
                # Default table name
                table_name = "data"

            # Write to SQLite - handle empty dataframes
            if len(df.columns) == 0:
                # Create a dummy column for empty dataframes to avoid SQL error
                df["dummy"] = None

            # Write to SQLite using efficient batch insert
            try:
                # Use our QueryExecutor for efficient batch insertion
                from sqlalchemy.ext.asyncio import create_async_engine

                from shared.database.query_executor import QueryExecutor

                # Create async engine for our QueryExecutor
                async_engine = create_async_engine(
                    f"sqlite+aiosqlite:///{temp_db_path}"
                )
                executor = QueryExecutor(async_engine)

                # Create table structure first
                df_sample = df.head(0)  # Empty DataFrame with columns
                df_sample.to_sql(table_name, engine, if_exists="replace", index=False)

                # Convert DataFrame to list of dictionaries for batch insert
                if len(df) > 0:
                    # Convert Timestamp types in DataFrame to ISO format strings
                    from shared.utils.datetime_utils import format_datetime

                    for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
                        # Use the project's unified datetime formatting tool
                        df[col] = df[col].apply(
                            lambda x: format_datetime(x) if pd.notna(x) else None
                        )

                    # Convert category types in DataFrame to strings
                    for col in df.select_dtypes(include=["category"]).columns:
                        df[col] = df[col].astype(str)

                    # Convert DataFrame to list[dict[str, Any]] before assignment
                    data_records = cast(list[dict[str, Any]], df.to_dict("records"))

                    # Use batch insert for better performance
                    batch_size = min(
                        1000, max(100, len(data_records) // 10)
                    )  # Dynamic batch size
                    insert_start = time.time()

                    inserted_count = await executor.execute_batch_insert(
                        table_name=table_name,
                        data_list=data_records,
                        batch_size=batch_size,
                        use_transaction=True,
                    )

                    insert_time = time.time() - insert_start
                    self.logger.info(
                        f"Batch inserted {inserted_count} records in "
                        f"{insert_time:.3f}s "
                        f"({inserted_count/insert_time:.1f} records/sec)"
                    )

                    # Cleanup async engine
                    await async_engine.dispose()
                else:
                    self.logger.info("Empty DataFrame, only creating table structure")

            except Exception as e:
                self.logger.error(f"Failed to write DataFrame to SQLite: {str(e)}")
                raise ValueError(f"Failed to create temporary database: {str(e)}")

            # Create connection config with table name in parameters
            sqlite_config = ConnectionSchema(
                name=f"temp_sqlite_{table_name}",
                description="Temporary SQLite for file validation",
                connection_type=ConnectionType.SQLITE,
                file_path=temp_db_path,
                parameters={
                    "table": table_name,
                    "temp_file": True,  # Mark as temporary file for cleanup
                },
            )

            # Log performance metrics
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"Created temporary SQLite database at {temp_db_path} with table "
                f"'{table_name}'"
            )
            self.logger.info(
                f"Converted file data to SQLite with {len(df)} rows in "
                f"{elapsed_time:.2f} seconds"
            )

            return sqlite_config

        except Exception as e:
            # Clean up temporary file if it exists
            if temp_db_path and os.path.exists(temp_db_path):
                try:
                    import os

                    os.unlink(temp_db_path)
                except Exception as cleanup_error:
                    self.logger.warning(
                        f"Failed to clean up temporary file: {str(cleanup_error)}"
                    )

            self.logger.error(f"Failed to convert file to SQLite: {str(e)}")
            raise ValueError(f"Failed to convert file to SQLite: {str(e)}")

    def _clean_column_name(self, column_name: str) -> str:
        """
        Clean column name to be SQLite compatible

        Args:
            column_name: Original column name

        Returns:
            str: Cleaned column name
        """
        import re

        # Handle None, empty string or whitespace only
        if column_name is None or not column_name.strip():
            return "unnamed_column"

        # Strip whitespace
        column_name = column_name.strip()

        # Replace non-alphanumeric characters with underscore
        cleaned = re.sub(r"[^a-zA-Z0-9]", "_", column_name)

        # Ensure it doesn't start with a number
        if cleaned and cleaned[0].isdigit():
            cleaned = f"col_{cleaned}"

        # Ensure it's not empty
        if not cleaned:
            cleaned = "unnamed_column"

        # Ensure it's not a SQLite keyword
        sqlite_keywords = {
            "abort",
            "action",
            "add",
            "after",
            "all",
            "alter",
            "analyze",
            "and",
            "as",
            "asc",
            "attach",
            "autoincrement",
            "before",
            "begin",
            "between",
            "by",
            "cascade",
            "case",
            "cast",
            "check",
            "collate",
            "column",
            "commit",
            "conflict",
            "constraint",
            "create",
            "cross",
            "current_date",
            "current_time",
            "current_timestamp",
            "database",
            "default",
            "deferrable",
            "deferred",
            "delete",
            "desc",
            "detach",
            "distinct",
            "drop",
            "each",
            "else",
            "end",
            "escape",
            "except",
            "exclusive",
            "exists",
            "explain",
            "fail",
            "for",
            "foreign",
            "from",
            "full",
            "glob",
            "group",
            "having",
            "if",
            "ignore",
            "immediate",
            "in",
            "index",
            "indexed",
            "initially",
            "inner",
            "insert",
            "instead",
            "intersect",
            "into",
            "is",
            "isnull",
            "join",
            "key",
            "left",
            "like",
            "limit",
            "match",
            "natural",
            "no",
            "not",
            "notnull",
            "null",
            "of",
            "offset",
            "on",
            "or",
            "order",
            "outer",
            "plan",
            "pragma",
            "primary",
            "query",
            "raise",
            "recursive",
            "references",
            "regexp",
            "reindex",
            "release",
            "rename",
            "replace",
            "restrict",
            "right",
            "rollback",
            "row",
            "savepoint",
            "select",
            "set",
            "table",
            "temp",
            "temporary",
            "then",
            "to",
            "transaction",
            "trigger",
            "union",
            "unique",
            "update",
            "using",
            "vacuum",
            "values",
            "view",
            "virtual",
            "when",
            "where",
            "with",
            "without",
        }

        if cleaned.lower() in sqlite_keywords:
            cleaned = f"{cleaned}_col"

        return cleaned

    def _clean_table_name(self, table_name: str) -> str:
        """
        Clean table name to be SQLite compatible

        Args:
            table_name: Original table name

        Returns:
            str: Cleaned table name
        """
        import re

        # Handle None, empty string or whitespace only
        if table_name is None or not table_name.strip():
            return "data"

        # Strip whitespace
        table_name = table_name.strip()

        # Replace non-alphanumeric characters with underscore
        cleaned = re.sub(r"[^a-zA-Z0-9]", "_", table_name)

        # Ensure it doesn't start with a number
        if cleaned and cleaned[0].isdigit():
            cleaned = f"table_{cleaned}"

        # Ensure it's not empty
        if not cleaned:
            cleaned = "data"

        # Ensure it's not a SQLite keyword
        sqlite_keywords = {
            "abort",
            "action",
            "add",
            "after",
            "all",
            "alter",
            "analyze",
            "and",
            "as",
            "asc",
            "attach",
            "autoincrement",
            "before",
            "begin",
            "between",
            "by",
            "cascade",
            "case",
            "cast",
            "check",
            "collate",
            "column",
            "commit",
            "conflict",
            "constraint",
            "create",
            "cross",
            "current_date",
            "current_time",
            "current_timestamp",
            "database",
            "default",
            "deferrable",
            "deferred",
            "delete",
            "desc",
            "detach",
            "distinct",
            "drop",
            "each",
            "else",
            "end",
            "escape",
            "except",
            "exclusive",
            "exists",
            "explain",
            "fail",
            "for",
            "foreign",
            "from",
            "full",
            "glob",
            "group",
            "having",
            "if",
            "ignore",
            "immediate",
            "in",
            "index",
            "indexed",
            "initially",
            "inner",
            "insert",
            "instead",
            "intersect",
            "into",
            "is",
            "isnull",
            "join",
            "key",
            "left",
            "like",
            "limit",
            "match",
            "natural",
            "no",
            "not",
            "notnull",
            "null",
            "of",
            "offset",
            "on",
            "or",
            "order",
            "outer",
            "plan",
            "pragma",
            "primary",
            "query",
            "raise",
            "recursive",
            "references",
            "regexp",
            "reindex",
            "release",
            "rename",
            "replace",
            "restrict",
            "right",
            "rollback",
            "row",
            "savepoint",
            "select",
            "set",
            "table",
            "temp",
            "temporary",
            "then",
            "to",
            "transaction",
            "trigger",
            "union",
            "unique",
            "update",
            "using",
            "vacuum",
            "values",
            "view",
            "virtual",
            "when",
            "where",
            "with",
            "without",
        }

        if cleaned.lower() in sqlite_keywords:
            cleaned = f"{cleaned}_table"

        return cleaned

    def _process_results(
        self, raw_results: List[Dict[str, Any]]
    ) -> List[ExecutionResultSchema]:
        """
        Process raw results from rule engine

        Args:
            raw_results: Raw results from rule engine

        Returns:
            List[ExecutionResultSchema]: Processed results
        """
        processed_results = []

        for result in raw_results:
            # Convert to ExecutionResultSchema
            # Enhance sample data if available
            if "sample_data" in result and result["sample_data"]:
                result["sample_data"] = self._enhance_sample_data(
                    result["sample_data"], result.get("rule_type", "")
                )
            processed_results.append(ExecutionResultSchema.model_validate(result))

        return processed_results

    def _enhance_sample_data(
        self, sample_data: List[Dict[str, Any]], rule_type: str
    ) -> List[Dict[str, Any]]:
        """
        Enhance sample data with additional context

        Args:
            sample_data: Sample data from rule engine
            rule_type: Rule type

        Returns:
            List[Dict[str, Any]]: Enhanced sample data
        """
        # Limit sample size based on CLI config
        max_samples = min(self.sample_size, len(sample_data))
        limited_samples = sample_data[:max_samples]

        # Add context based on rule type
        for sample in limited_samples:
            if rule_type == "NOT_NULL" and "value" in sample:
                sample["is_null"] = sample["value"] is None
            elif rule_type == "RANGE" and "value" in sample:
                if "min" in sample and "max" in sample:
                    sample["in_range"] = (
                        sample["min"] <= sample["value"] <= sample["max"]
                        if sample["value"] is not None
                        else False
                    )

        return limited_samples

    def _create_error_results(self, error_message: str) -> List[ExecutionResultSchema]:
        """
        Create error results for all rules

        Args:
            error_message: Error message

        Returns:
            List[ExecutionResultSchema]: Error results
        """
        results = []

        for rule in self.rules:
            # Get target info
            target_info = rule.get_target_info()

            # Create error result
            result = ExecutionResultSchema.create_error_result(
                rule_id=str(rule.id) if rule.id else rule.name,
                entity_name=(target_info.get("database") or "")
                + "."
                + (target_info.get("table") or "")
                + "."
                + (target_info.get("column") or ""),
                error_message=error_message,
                execution_time=0.0,
            )

            results.append(result)

        return results

    def _convert_source_dict(self, data: Dict[str, Any]) -> ConnectionSchema:
        """
        Convert source dict to ConnectionSchema

        Args:
            data: Source dict

        Returns:
            ConnectionSchema: Connection schema
        """
        # Extract connection type - check both "type" and "connection_type" fields
        conn_type_str = data.get("type", data.get("connection_type", "")).lower()
        if not conn_type_str:
            raise ValueError("source_config dict must contain a 'type' key")

        try:
            conn_type = ConnectionType(conn_type_str)
        except ValueError:
            raise ValueError(f"Invalid connection type: {conn_type_str}")

        # Create connection schema based on type
        if conn_type in [ConnectionType.CSV, ConnectionType.EXCEL, ConnectionType.JSON]:
            return ConnectionSchema(
                name=data.get("name", f"{conn_type.value}_source"),
                description=data.get("description", f"{conn_type.value} data source"),
                connection_type=conn_type,
                file_path=data.get("path", data.get("file_path")),
                parameters=data.get("parameters", {}),
            )
        elif conn_type == ConnectionType.SQLITE:
            return ConnectionSchema(
                name=data.get("name", "sqlite_source"),
                description=data.get("description", "SQLite data source"),
                connection_type=conn_type,
                file_path=data.get("path", data.get("file_path")),
                parameters=data.get("parameters", {}),
            )
        else:
            # Database connection - add default ports for database types
            default_ports = {
                ConnectionType.MYSQL: 3306,
                ConnectionType.POSTGRESQL: 5432,
                ConnectionType.MSSQL: 1433,
                ConnectionType.ORACLE: 1521,
            }

            # Get default port for this connection type
            default_port = default_ports.get(conn_type)

            return ConnectionSchema(
                name=data.get("name", f"{conn_type.value}_source"),
                description=data.get("description", f"{conn_type.value} data source"),
                connection_type=conn_type,
                host=data.get("host"),
                port=data.get("port", default_port),
                db_name=data.get("database", data.get("db_name")),
                username=data.get("username"),
                password=data.get("password"),
                db_schema=data.get("schema"),
                parameters=data.get("parameters", {}),
            )

    def _convert_rule_dict(self, data: Dict[str, Any]) -> RuleSchema:
        """
        Convert rule dict to RuleSchema

        Args:
            data: Rule dict

        Returns:
            RuleSchema: Rule schema
        """
        # Extract rule type - check both "type" and "rule_type" fields
        rule_type_str = data.get("type", data.get("rule_type", "")).upper()
        if not rule_type_str:
            raise ValueError("Rule type is required")

        try:
            rule_type = RuleType(rule_type_str)
        except ValueError:
            raise ValueError(f"Invalid rule type: {rule_type_str}")

        # Extract target - check both "target" and "column" fields
        target_str = data.get("target", data.get("column", ""))
        if not target_str:
            raise ValueError("Rule target is required")

        # Parse target string (format: [database.]table.column)
        parts = target_str.split(".")
        if len(parts) == 1:
            # Just column name
            database = "main"
            table = "data"
            column = parts[0]
        elif len(parts) == 2:
            # table.column
            database = "main"
            table, column = parts
        elif len(parts) == 3:
            # database.table.column
            database, table, column = parts
        else:
            raise ValueError(f"Invalid target format: {target_str}")

        # Create target entity
        target = RuleTarget(
            entities=[TargetEntity(database=database, table=table, column=column)]
        )

        # Extract parameters - ensure it's a dictionary
        parameters = data.get("parameters", {})
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                f"Rule parameters must be a dictionary, got {type(parameters).__name__}"
            )

        # Create rule schema
        return RuleSchema(
            name=data.get("name", f"{rule_type.value}_{target_str}"),
            description=data.get("description"),
            type=rule_type,
            target=target,
            parameters=parameters or {},  # Ensure parameters is a dict, not None
            category=RuleCategory.COMPLETENESS,  # Default
            severity=SeverityLevel.MEDIUM,  # Default
            action=RuleAction.LOG,  # Default
        )
