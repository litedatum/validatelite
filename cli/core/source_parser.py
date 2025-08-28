"""
Source Parser

Intelligent source identification and parsing.
Supports files (CSV, Excel, JSON) and database URLs.
"""

import re
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from cli.exceptions import ValidationError
from shared.enums import ConnectionType
from shared.schema import ConnectionSchema
from shared.schema.base import DataSourceCapability
from shared.utils.logger import get_logger


class SourceParser:
    """
    Smart source parser for files and database connections.

    Supports:
    - Files: CSV, Excel, JSON
    - Database URLs: MySQL, PostgreSQL, SQLite
    """

    def __init__(self) -> None:
        """Initialize SourceParser"""
        self.logger = get_logger(__name__)

        # URL patterns for database recognition
        self.db_url_patterns = {
            ConnectionType.MYSQL: [r"^mysql://.*", r"^mysql\+pymysql://.*"],
            ConnectionType.POSTGRESQL: [
                r"^postgres://.*",
                r"^postgresql://.*",
                r"^postgresql\+psycopg2://.*",
            ],
            ConnectionType.SQLITE: [r"^sqlite://.*", r"^sqlite:///.*"],
        }

        # File extensions mapping
        self.file_extensions = {
            ".csv": ConnectionType.CSV,
            ".tsv": ConnectionType.CSV,
            ".xlsx": ConnectionType.EXCEL,
            ".xls": ConnectionType.EXCEL,
            ".json": ConnectionType.JSON,
            ".jsonl": ConnectionType.JSON,
        }

    def parse_source(
        self, source: str, table_name: Optional[str] = None
    ) -> ConnectionSchema:
        """
        Parse source string into ConnectionSchema.

        Args:
            source: Source string (file path or database URL)
            table_name: Optional table name (overrides table from URL if provided)

        Returns:
            ConnectionSchema: Parsed connection configuration

        Raises:
            ValueError: If source format is not recognized
            FileNotFoundError: If file not found
            ValueError: If path is a directory
        """
        self.logger.info(f"Parsing source: {source}")

        try:
            # Check for empty string or string with only whitespace
            if not source or source.strip() == "":
                raise ValidationError("Unrecognized source format: Empty source")

            if self._is_database_url(source):
                return self._parse_database_url(source, table_name)
            elif source.startswith("file://"):
                # Handle file:// protocol
                file_path = source[7:]  # Remove file:// prefix
                return self._parse_file_path(file_path)
            elif self._is_file_path(source):
                return self._parse_file_path(source)
            else:
                # Check if it is a directory
                path = Path(source)
                if path.exists() and path.is_dir():
                    raise ValidationError(f"Path is not a file: {source}")
                raise ValidationError(f"Unrecognized source format: {source}")
        except Exception as e:
            self.logger.error(f"{str(e)}")
            raise

    def get_excel_sheets(self, file_path: str) -> Dict[str, List[str]]:
        """
        Get sheet names from Excel file.

        Args:
            file_path: Path to Excel file

        Returns:
            Dict with sheet names as keys and column lists as values

        Raises:
            ImportError: If pandas/openpyxl not available
            FileNotFoundError: If file not found
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required to read Excel files")

        try:
            excel_file = pd.ExcelFile(file_path)
            sheets_info = {}

            for sheet_name in excel_file.sheet_names:
                # Read first few rows to get column names
                df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=0)
                sheets_info[str(sheet_name)] = list(df.columns)

            return sheets_info
        except Exception as e:
            self.logger.error(f"Error reading Excel file {file_path}: {str(e)}")
            raise

    def is_multi_table_excel(self, file_path: str) -> bool:
        """
        Check if Excel file contains multiple sheets that could represent
          multiple tables.

        Args:
            file_path: Path to Excel file

        Returns:
            True if file has multiple sheets, False otherwise
        """
        try:
            import pandas as pd

            excel_file = pd.ExcelFile(file_path)
            return len(excel_file.sheet_names) > 1
        except ImportError:
            # If pandas not available, assume single table
            return False
        except Exception:
            # If any error occurs, assume single table
            return False

    def _is_database_url(self, source: str) -> bool:
        """Check if source is a database URL"""
        for patterns in self.db_url_patterns.values():
            for pattern in patterns:
                if re.match(pattern, source, re.IGNORECASE):
                    return True
        return False

    def _is_file_path(self, source: str) -> bool:
        """Check if source is a file path"""
        path = Path(source)

        # Check if it has a recognized file extension
        if path.suffix.lower() in self.file_extensions:
            return True

        # Check if it's an existing file (not a directory)
        if path.exists():
            if path.is_file():
                return True
            elif path.is_dir():
                # Do not process directories
                return False

        return False

    def _parse_database_url(
        self, url: str, table_name: Optional[str] = None
    ) -> ConnectionSchema:
        """
        Parse database URL into connection configuration.

        Supports formats:
        - mysql://user:pass@host:port/database.table
        - postgres://user:pass@host:port/database.table
        - sqlite:///path/to/database.db.table

        Args:
            url: Database connection URL
            table_name: Optional table name (overrides table from URL if provided)
        """
        self.logger.debug(f"Parsing database URL: {url}")

        # Determine connection type
        conn_type = self._detect_database_type(url)

        # Parse URL components
        parsed = urllib.parse.urlparse(url)

        # Extract database and table from path
        database, table_from_url = self._extract_db_table_from_path(parsed.path)

        # Use provided table_name if available, otherwise use table from URL
        table = table_name if table_name is not None else table_from_url

        # Handle SQLite special case
        if conn_type == ConnectionType.SQLITE:
            return self._create_sqlite_connection(url, database, table)

        # Handle other database types
        return ConnectionSchema(
            name=f"{conn_type.value}_connection_{uuid4().hex[:8]}",
            description=f"{conn_type.value.upper()} connection from CLI",
            connection_type=conn_type,
            host=parsed.hostname,
            port=parsed.port or ConnectionType.get_default_port(conn_type),
            db_name=database,
            username=parsed.username,
            password=parsed.password,
            db_schema=None,  # Will be inferred if needed
            file_path=None,
            parameters={"table": table} if table else {},
            capabilities=DataSourceCapability(
                supports_sql=True,
                supports_batch_export=True,
                max_export_rows=1000000,
                estimated_throughput=10000,
            ),
            cross_db_settings=None,
        )

    def _parse_file_path(self, file_path: str) -> ConnectionSchema:
        """Parse file path into connection configuration"""
        self.logger.debug(f"Parsing file path: {file_path}")

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")

        file_ext = path.suffix.lower()
        conn_type = self.file_extensions.get(file_ext)

        if not conn_type:
            conn_type = ConnectionType.CSV
            self.logger.warning(
                f"Unknown file extension {file_ext}, assuming CSV format"
            )

        is_multi_table = False
        sheets_info = {}
        if conn_type == ConnectionType.EXCEL:
            try:
                sheets_info = self.get_excel_sheets(file_path)
                if len(sheets_info) > 1:
                    is_multi_table = True
                    self.logger.info(
                        f"Multi-table Excel file detected with {len(sheets_info)} "
                        "sheets: {list(sheets_info.keys())}"
                    )
            except Exception as e:
                self.logger.warning(
                    f"Could not read Excel sheets, treating as single-table: {str(e)}"
                )
                is_multi_table = False

        parameters = {
            "filename": path.name,
            "file_size": path.stat().st_size,
            "encoding": "utf-8",
        }

        if is_multi_table and sheets_info:
            parameters["is_multi_table"] = True
            parameters["sheets"] = sheets_info
            available_tables = list(sheets_info.keys())
        else:
            parameters["is_multi_table"] = False
            available_tables = [path.stem]

        return ConnectionSchema(
            name=f"file_connection_{uuid4().hex[:8]}",
            description=f"File connection: {path.name}"
            + (" (multi-table)" if is_multi_table else ""),
            connection_type=conn_type,
            file_path=str(path.absolute()),
            parameters=parameters,
            available_tables=available_tables,
            capabilities=DataSourceCapability(
                supports_sql=False,
                supports_batch_export=True,
                max_export_rows=100000 if not is_multi_table else 50000,
                estimated_throughput=5000 if not is_multi_table else 2000,
            ),
        )

    def _detect_database_type(self, url: str) -> ConnectionType:
        """Detect database type from URL"""
        for conn_type, patterns in self.db_url_patterns.items():
            for pattern in patterns:
                if re.match(pattern, url, re.IGNORECASE):
                    return conn_type

        raise ValidationError(f"Unsupported database URL format: {url}")

    def _extract_db_table_from_path(self, path: str) -> Tuple[str, Optional[str]]:
        """
        Extract database and table name from URL path.

        Formats:
        - /database -> (database, None)
        - /database.table -> (database, table)
        - /path/to/database.db.table -> (database.db, table)
        """
        if not path or path == "/":
            raise ValidationError("Database path cannot be empty")

        # Remove leading slash
        path = path.lstrip("/")

        # Handle SQLite file paths
        if path.endswith(".db") or "/" in path:
            # For SQLite, the whole path is the database
            if "." in Path(path).name:
                # Extract table name if present after .db
                parts = path.split(".")
                if len(parts) >= 3 and parts[-2] == "db":
                    database = ".".join(parts[:-1])
                    table = parts[-1]
                    return database, table
            return path, None

        # Handle database.table format
        if "." in path:
            parts = path.split(".")
            if len(parts) == 2:
                return parts[0], parts[1]
            else:
                # Multiple dots - take last as table, rest as database
                database = ".".join(parts[:-1])
                table = parts[-1]
                return database, table

        # Just database name
        return path, None

    def _create_sqlite_connection(
        self, url: str, database: str, table: Optional[str]
    ) -> ConnectionSchema:
        """Create SQLite connection configuration"""

        # For SQLite, extract file path from URL
        if url.startswith("sqlite:///"):
            file_path = url[10:]  # Remove 'sqlite:///'
        elif url.startswith("sqlite://"):
            file_path = url[9:]  # Remove 'sqlite://'
        else:
            file_path = database

        # Handle table extraction
        if table:
            parameters = {"table": table}
        else:
            parameters = {}

        return ConnectionSchema(
            name=f"sqlite_connection_{uuid4().hex[:8]}",
            description=f"SQLite connection: {Path(file_path).name}",
            connection_type=ConnectionType.SQLITE,
            file_path=file_path,
            parameters=parameters,
            available_tables=[table] if table else [],
            capabilities=DataSourceCapability(
                supports_sql=True,
                supports_batch_export=True,
                max_export_rows=1000000,
                estimated_throughput=15000,
            ),
            cross_db_settings=None,
        )
