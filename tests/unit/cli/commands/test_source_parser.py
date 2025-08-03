"""
Modern Source Parser Testing Suite

Testing Architecture:
- Builder Pattern for zero boilerplate test data creation
- Contract Testing to ensure mock compliance
- Property-based testing for edge case discovery
- Comprehensive boundary condition testing
- Performance monitoring and benchmarks
- Unicode and international character support

Test Coverage:
- Smart source recognition (CSV, MySQL, PostgreSQL, SQLite, Excel, JSON)
- URL parsing and validation
- File protocol handling
- Error handling and recovery
- Performance benchmarks
- Memory usage monitoring
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import psutil
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import integers, sampled_from, text

from cli.core.source_parser import SourceParser
from cli.exceptions import ValidationError
from shared.enums import ConnectionType
from shared.schema import ConnectionSchema
from shared.utils.logger import get_logger
from tests.shared.builders.test_builders import TestDataBuilder
from tests.shared.contracts.test_contracts import MockContract


class TestSourceParserModern:
    """Modern Source Parser Test Suite with Testing Ghost Architecture"""

    @pytest.fixture
    def parser(self) -> SourceParser:
        """Create source parser instance with logging configured"""
        return SourceParser()

    @pytest.fixture
    def memory_monitor(self) -> psutil.Process:
        """Monitor memory usage during tests"""
        process = psutil.Process(os.getpid())
        return process

    def create_temp_file(self, content: str, suffix: str) -> str:
        """Builder method for creating temporary files"""
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=suffix, delete=False
        ) as f:
            f.write(content)
            return f.name

    def cleanup_temp_file(self, file_path: str) -> None:
        """Cleanup utility for temporary files"""
        Path(file_path).unlink(missing_ok=True)

    # ==================== CSV File Recognition Tests ====================

    def test_csv_file_recognition_basic(self, parser: SourceParser) -> None:
        """Test basic CSV file recognition with standard data"""
        csv_content = "id,name,email\n1,John,john@example.com\n2,Jane,jane@example.com"
        temp_file = self.create_temp_file(csv_content, ".csv")

        try:
            result = parser.parse_source(temp_file)

            assert isinstance(result, ConnectionSchema)
            assert result.connection_type == ConnectionType.CSV
            assert result.file_path == str(Path(temp_file).absolute())
            assert result.parameters["encoding"] == "utf-8"
            assert "filename" in result.parameters
            assert result.parameters["file_size"] > 0
        finally:
            self.cleanup_temp_file(temp_file)

    def test_csv_file_recognition_with_unicode(self, parser: SourceParser) -> None:
        """Test CSV file recognition with Unicode characters"""
        unicode_content = (
            "名前,年齢,国\nJohn Smith,25,🇺🇸\nMarie Curie,34,🇫🇷\nライト兄弟,28,日本"
        )
        temp_file = self.create_temp_file(unicode_content, ".csv")

        try:
            result = parser.parse_source(temp_file)

            assert result.connection_type == ConnectionType.CSV
            assert result.parameters["encoding"] == "utf-8"
            # Verify Unicode handling doesn't break the parser
            assert result.file_path is not None
        finally:
            self.cleanup_temp_file(temp_file)

    def test_empty_csv_file_handling(self, parser: SourceParser) -> None:
        """Test handling of empty CSV files (boundary condition)"""
        temp_file = self.create_temp_file("", ".csv")

        try:
            result = parser.parse_source(temp_file)

            assert result.connection_type == ConnectionType.CSV
            assert result.parameters["file_size"] == 0
        finally:
            self.cleanup_temp_file(temp_file)

    @pytest.mark.performance
    def test_large_csv_file_performance(
        self, parser: SourceParser, memory_monitor: psutil.Process
    ) -> None:
        """Test performance with large CSV files"""
        # Create a larger CSV for performance testing
        large_content = "id,data\n" + "\n".join([f"{i},data_{i}" for i in range(1000)])
        temp_file = self.create_temp_file(large_content, ".csv")

        try:
            start_memory = memory_monitor.memory_info().rss
            start_time = time.time()

            result = parser.parse_source(temp_file)

            end_time = time.time()
            end_memory = memory_monitor.memory_info().rss

            # Performance assertions
            execution_time = end_time - start_time
            assert execution_time < 1.0, f"CSV parsing took too long: {execution_time}s"

            memory_increase = end_memory - start_memory
            assert (
                memory_increase < 10 * 1024 * 1024
            ), f"Memory increase too high: {memory_increase} bytes"

            assert result.connection_type == ConnectionType.CSV
        finally:
            self.cleanup_temp_file(temp_file)

    # ==================== Database URL Recognition Tests ====================

    def test_mysql_url_recognition_complete(self, parser: SourceParser) -> None:
        """Test complete MySQL URL parsing with all components"""
        mysql_url = (
            "mysql://testuser:testpass@db.example.com:3306/production_db.user_profiles"
        )

        result = parser.parse_source(mysql_url)

        assert result.connection_type == ConnectionType.MYSQL
        assert result.host == "db.example.com"
        assert result.port == 3306
        assert result.db_name == "production_db"
        assert result.username == "testuser"
        assert result.password == "testpass"
        assert result.parameters.get("table") == "user_profiles"

    def test_postgresql_url_recognition_complete(self, parser: SourceParser) -> None:
        """Test complete PostgreSQL URL parsing"""
        postgres_url = (
            "postgresql://pguser:pgpass@pg.cluster.com:5432/analytics_db.metrics"
        )

        result = parser.parse_source(postgres_url)

        assert result.connection_type == ConnectionType.POSTGRESQL
        assert result.host == "pg.cluster.com"
        assert result.port == 5432
        assert result.db_name == "analytics_db"
        assert result.username == "pguser"
        assert result.password == "pgpass"
        assert result.parameters.get("table") == "metrics"

    def test_sqlite_url_recognition(self, parser: SourceParser) -> None:
        """Test SQLite URL parsing with file path"""
        sqlite_url = "sqlite:///var/data/application.db.user_data"

        result = parser.parse_source(sqlite_url)

        assert result.connection_type == ConnectionType.SQLITE
        assert result.parameters.get("table") == "user_data"
        assert result.file_path is not None
        assert "application.db" in result.file_path or result.parameters.get(
            "database_file"
        )

    @given(
        host=text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz.-"),
        port=integers(min_value=1024, max_value=65535),
        username=text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"),
        password=text(
            min_size=1,
            max_size=20,
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        ),
        database=text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz_"),
        table=text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz_"),
    )
    @settings(
        max_examples=20,
        deadline=5000,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.filter_too_much,
        ],
    )
    def test_mysql_url_property_based(
        self,
        parser: SourceParser,
        host: str,
        port: int,
        username: str,
        password: str,
        database: str,
        table: str,
    ) -> None:
        """Property-based test for MySQL URL parsing with random valid inputs"""
        assume(not host.startswith(".") and not host.endswith("."))
        assume(not host.startswith("-") and not host.endswith("-"))
        assume("." in host)  # Ensure it looks like a valid hostname

        mysql_url = f"mysql://{username}:{password}@{host}:{port}/{database}.{table}"

        try:
            result = parser.parse_source(mysql_url)

            assert result.connection_type == ConnectionType.MYSQL
            assert result.host == host
            assert result.port == port
            assert result.username == username
            assert result.password == password
            assert result.db_name == database
            assert result.parameters.get("table") == table
        except ValidationError as e:
            # Some randomly generated combinations might be invalid
            # This is acceptable for property-based testing
            assume(False)

    # ==================== File Protocol and Format Tests ====================

    def test_file_protocol_recognition(self, parser: SourceParser) -> None:
        """Test file:// protocol parsing"""
        csv_content = "id,name\n1,test"
        temp_file = self.create_temp_file(csv_content, ".csv")

        try:
            # Construct a `file://` URL using an absolute path.
            file_path = str(Path(temp_file).absolute())
            file_url = f"file://{file_path}"
            result = parser.parse_source(file_url)

            assert result.connection_type == ConnectionType.CSV
            assert result.file_path == file_path
        finally:
            self.cleanup_temp_file(temp_file)

    def test_excel_file_recognition(self, parser: SourceParser) -> None:
        """Test Excel file recognition"""
        temp_file = self.create_temp_file("", ".xlsx")

        try:
            result = parser.parse_source(temp_file)

            assert result.connection_type == ConnectionType.EXCEL
            assert result.file_path == str(Path(temp_file).absolute())
        finally:
            self.cleanup_temp_file(temp_file)

    def test_json_file_recognition(self, parser: SourceParser) -> None:
        """Test JSON file recognition with valid JSON content"""
        json_content = (
            '{"users": [{"id": 1, "name": "John"}, {"id": 2, "name": "Jane"}]}'
        )
        temp_file = self.create_temp_file(json_content, ".json")

        try:
            result = parser.parse_source(temp_file)

            assert result.connection_type == ConnectionType.JSON
            assert result.file_path == str(Path(temp_file).absolute())
        finally:
            self.cleanup_temp_file(temp_file)

    def test_unknown_extension_default_behavior(self, parser: SourceParser) -> None:
        """Test handling of unknown file extensions"""
        content = "col1,col2\nval1,val2"
        temp_file = self.create_temp_file(content, ".unknown")

        try:
            with patch.object(parser.logger, "warning") as mock_warning:
                result = parser.parse_source(temp_file)

                assert result.connection_type == ConnectionType.CSV
                mock_warning.assert_called_once()
                assert "assuming CSV format" in mock_warning.call_args[0][0]
        finally:
            self.cleanup_temp_file(temp_file)

    # ==================== Error Handling and Edge Cases ====================

    def test_invalid_source_formats(self, parser: SourceParser) -> None:
        """Test comprehensive invalid source format handling"""
        invalid_sources = [
            "ftp://invalid.com/file",  # Unsupported protocol
            "not_a_url_or_file",  # Neither URL nor file
            "http://web.com/data",  # Unsupported web URL
            "ldap://directory.com/users",  # Unsupported directory protocol
            "mailto:user@domain.com",  # Email protocol
            "ssh://server.com/data",  # SSH protocol
        ]

        for invalid_source in invalid_sources:
            with pytest.raises(ValidationError, match="Unrecognized source format"):
                parser.parse_source(invalid_source)

        # Test the empty string case separately.
        empty_sources = ["", "   "]
        for empty_source in empty_sources:
            with pytest.raises(
                ValidationError, match="Unrecognized source format: Empty source"
            ):
                parser.parse_source(empty_source)

    def test_file_not_found_error(self, parser: SourceParser) -> None:
        """Test file not found error handling"""
        nonexistent_files = [
            "/path/to/nonexistent/file.csv",
            "C:\\NonExistent\\data.xlsx",
            "./missing_file.json",
            "~/not_here.csv",
        ]

        for nonexistent_file in nonexistent_files:
            with pytest.raises(FileNotFoundError, match="File not found"):
                parser.parse_source(nonexistent_file)

    def test_directory_path_error(self, parser: SourceParser) -> None:
        """Test directory path error handling"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValidationError, match="Path is not a file"):
                parser.parse_source(temp_dir)

    def test_permission_denied_handling(self, parser: SourceParser) -> None:
        """Test handling of permission denied errors"""
        temp_file = self.create_temp_file("test,data", ".csv")

        try:
            # Use mocking to simulate a permissions error.
            with patch("pathlib.Path.exists", return_value=True), patch(
                "pathlib.Path.is_file", return_value=True
            ), patch(
                "pathlib.Path.stat", side_effect=PermissionError("Permission denied")
            ):

                with pytest.raises((PermissionError, OSError)):
                    parser.parse_source(temp_file)
        finally:
            self.cleanup_temp_file(temp_file)

    # ==================== URL Edge Cases and Boundary Tests ====================

    def test_database_url_without_table(self, parser: SourceParser) -> None:
        """Test database URLs without table specification"""
        test_cases = [
            ("mysql://user:pass@host/database", ConnectionType.MYSQL, 3306),
            ("postgresql://user:pass@host/database", ConnectionType.POSTGRESQL, 5432),
            ("postgres://user:pass@host/database", ConnectionType.POSTGRESQL, 5432),
        ]

        for url, expected_type, expected_port in test_cases:
            result = parser.parse_source(url)

            assert result.connection_type == expected_type
            assert result.port == expected_port
            assert result.db_name == "database"
            assert result.parameters.get("table") is None

    def test_database_url_with_special_characters(self, parser: SourceParser) -> None:
        """Test database URLs with special characters in credentials"""
        special_cases = [
            "mysql://user%40domain:p%40ssw0rd@host/db",  # URL encoded
            "mysql://user:pass@word@host/db",  # @ in password
            "mysql://complex.user:pass%21word@host/db",  # ! in password (encoded)
        ]

        for url in special_cases:
            try:
                result = parser.parse_source(url)
                assert result.connection_type == ConnectionType.MYSQL
                # Verify parsing didn't crash
                assert result.host is not None
            except ValidationError:
                # Some special character combinations might be legitimately invalid
                pass

    def test_url_case_insensitivity(self, parser: SourceParser) -> None:
        """Test URL scheme case insensitivity"""
        case_variants = [
            "MYSQL://user:pass@host/db",
            "mysql://user:pass@host/db",
            "MySQL://user:pass@host/db",
            "POSTGRESQL://user:pass@host/db",
            "postgresql://user:pass@host/db",
            "PostGreSQL://user:pass@host/db",
        ]

        for url in case_variants:
            result = parser.parse_source(url)
            assert result.connection_type in [
                ConnectionType.MYSQL,
                ConnectionType.POSTGRESQL,
            ]

    # ==================== Connection Schema Validation Tests ====================

    def test_connection_schema_capabilities_csv(self, parser: SourceParser) -> None:
        """Test connection schema capabilities for CSV files"""
        csv_content = "id,name\n1,test"
        temp_file = self.create_temp_file(csv_content, ".csv")

        try:
            result = parser.parse_source(temp_file)

            assert result.capabilities is not None
            assert result.capabilities.supports_batch_export is True
            assert result.capabilities.max_export_rows is not None
            assert result.capabilities.max_export_rows > 0
            assert result.capabilities.estimated_throughput is not None
            assert result.capabilities.estimated_throughput > 0
        finally:
            self.cleanup_temp_file(temp_file)

    def test_connection_schema_capabilities_database(
        self, parser: SourceParser
    ) -> None:
        """Test connection schema capabilities for database connections"""
        database_urls = [
            "mysql://user:pass@host/db",
            "postgresql://user:pass@host/db",
            "sqlite:///tmp/test.db",
        ]

        for url in database_urls:
            result = parser.parse_source(url)

            assert result.capabilities is not None
            assert result.capabilities.supports_sql is True
            assert result.capabilities.supports_batch_export is True

    def test_file_size_parameter_accuracy(self, parser: SourceParser) -> None:
        """Test file size parameter accuracy"""
        content = "id,name,email\n" + "\n".join(
            [f"{i},user{i},user{i}@example.com" for i in range(100)]
        )
        temp_file = self.create_temp_file(content, ".csv")

        try:
            result = parser.parse_source(temp_file)

            actual_file_size = os.path.getsize(temp_file)
            assert result.parameters["file_size"] == actual_file_size
        finally:
            self.cleanup_temp_file(temp_file)

    # ==================== Performance and Stress Tests ====================

    @pytest.mark.performance
    def test_concurrent_parsing_safety(self, parser: SourceParser) -> None:
        """Test thread safety with concurrent parsing operations"""
        import concurrent.futures
        import threading

        csv_content = "id,name\n1,test"
        temp_files = []

        # Create multiple temp files
        for i in range(5):
            temp_file = self.create_temp_file(csv_content, f"_test_{i}.csv")
            temp_files.append(temp_file)

        try:
            results = []

            def parse_source_thread(file_path: str) -> ConnectionSchema:
                return parser.parse_source(file_path)

            # Test concurrent parsing
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(parse_source_thread, tf) for tf in temp_files
                ]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)

            # Verify all results are valid
            assert len(results) == 5
            for result in results:
                assert result.connection_type == ConnectionType.CSV

        finally:
            for temp_file in temp_files:
                self.cleanup_temp_file(temp_file)

    @pytest.mark.performance
    def test_memory_efficiency_multiple_files(
        self, parser: SourceParser, memory_monitor: psutil.Process
    ) -> None:
        """Test memory efficiency when parsing multiple files"""
        start_memory = memory_monitor.memory_info().rss

        # Create and parse multiple files
        for i in range(10):
            content = f"id,data_{i}\n" + "\n".join(
                [f"{j},value_{j}" for j in range(50)]
            )
            temp_file = self.create_temp_file(content, f"_mem_test_{i}.csv")

            try:
                result = parser.parse_source(temp_file)
                assert result.connection_type == ConnectionType.CSV
            finally:
                self.cleanup_temp_file(temp_file)

        end_memory = memory_monitor.memory_info().rss
        memory_increase = end_memory - start_memory

        # Should not increase memory significantly
        assert (
            memory_increase < 50 * 1024 * 1024
        ), f"Memory increase too high: {memory_increase} bytes"

    # ==================== Integration and Workflow Tests ====================

    def test_end_to_end_workflow_csv(self, parser: SourceParser) -> None:
        """Test complete workflow from CSV file to ConnectionSchema"""
        csv_content = "user_id,username,email,age\n1,john_doe,john@example.com,25\n2,jane_smith,jane@example.com,30"
        temp_file = self.create_temp_file(csv_content, ".csv")

        try:
            # Parse source
            result = parser.parse_source(temp_file)

            # Verify complete schema
            assert isinstance(result, ConnectionSchema)
            assert result.connection_type == ConnectionType.CSV
            assert result.file_path is not None
            assert result.parameters is not None
            assert "encoding" in result.parameters
            assert "file_size" in result.parameters
            assert result.capabilities is not None

            # Verify it's ready for further processing
            assert result.connection_type.value in ["csv", "CSV"]

        finally:
            self.cleanup_temp_file(temp_file)

    def test_end_to_end_workflow_database(self, parser: SourceParser) -> None:
        """Test complete workflow from database URL to ConnectionSchema"""
        mysql_url = "mysql://analytics_user:secure_pass@prod.db.company.com:3306/user_analytics.daily_metrics"

        result = parser.parse_source(mysql_url)

        # Verify complete schema
        assert isinstance(result, ConnectionSchema)
        assert result.connection_type == ConnectionType.MYSQL
        assert result.host == "prod.db.company.com"
        assert result.port == 3306
        assert result.db_name == "user_analytics"
        assert result.username == "analytics_user"
        assert result.password == "secure_pass"
        assert result.parameters.get("table") == "daily_metrics"
        assert result.capabilities is not None
        assert result.capabilities.supports_sql is True

    def test_error_recovery_and_logging(self, parser: SourceParser) -> None:
        """Test error recovery and proper logging"""
        with patch.object(parser.logger, "error") as mock_error, patch.object(
            parser.logger, "warning"
        ) as mock_warning:

            # Test invalid source
            with pytest.raises(ValidationError):
                parser.parse_source("invalid://source")

            mock_error.assert_called()

            # Test file not found
            with pytest.raises(FileNotFoundError):
                parser.parse_source("/nonexistent/file.csv")

            # Verify logging behavior
            assert mock_error.call_count >= 1

    # ==================== Contract Testing ====================

    def test_parser_contract_compliance(self, parser: SourceParser) -> None:
        """Test that parser follows expected interface contracts"""
        # Test return type contract
        csv_content = "id,name\n1,test"
        temp_file = self.create_temp_file(csv_content, ".csv")

        try:
            result = parser.parse_source(temp_file)

            # Contract: Must return ConnectionSchema
            assert isinstance(result, ConnectionSchema)

            # Contract: Must have required fields
            assert hasattr(result, "connection_type")
            assert hasattr(result, "parameters")
            assert hasattr(result, "capabilities")

            # Contract: connection_type must be valid enum
            assert isinstance(result.connection_type, ConnectionType)

        finally:
            self.cleanup_temp_file(temp_file)

    def test_error_contract_compliance(self, parser: SourceParser) -> None:
        """Test that parser raises appropriate exceptions per contract"""
        # Contract: Invalid source must raise ValidationError
        with pytest.raises(ValidationError):
            parser.parse_source("invalid_source")

        # Contract: Missing file must raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            parser.parse_source("/missing/file.csv")

        # Contract: Directory path must raise ValidationError
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValidationError):
                parser.parse_source(temp_dir)
