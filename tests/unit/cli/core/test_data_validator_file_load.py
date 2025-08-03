"""
🧙‍♂️ Test Data Validator File Loading Logic - Testing Ghost's File Processing Suite

This module tests the critical file loading and parsing logic in DataValidator:
- CSV parsing with automatic separator detection and fallback
- File existence and size validation
- Empty file and no-data error handling
- Unsupported file type detection
- DataFrame validation and encoding handling

Modern Testing Strategies Applied:
✅ Schema Builder Pattern - Consistent test data creation
✅ Property-based Testing - File size and encoding edge cases
✅ Contract Testing - Mock pandas behavior verification
✅ Boundary Testing - File size limits and edge conditions
✅ Error State Testing - All failure modes covered
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import Mock, mock_open, patch

import pandas as pd
import pytest

from cli.core.data_validator import DataValidator
from cli.exceptions import ValidationError
from shared.enums import ConnectionType

# Import our modern testing infrastructure
from tests.shared.builders.test_builders import TestDataBuilder
from tests.shared.contracts.test_contracts import MockContract


class TestDataValidatorFileLoad:
    """
    📁 Comprehensive test suite for DataValidator file loading logic

    Focus Areas:
    1. CSV parsing with separator auto-detection
    2. File size validation and limits
    3. Empty file and no-data handling
    4. Encoding and special character support
    5. File type validation and error handling
    6. Performance with large files
    """

    @pytest.fixture
    def mock_configs(self) -> Dict[str, Any]:
        """Provide mock configurations using Contract Testing"""
        cli_config = MockContract.create_cli_config_mock()
        cli_config.max_file_size_mb = 50  # Set reasonable limit for testing
        return {
            "core_config": MockContract.create_core_config_mock(),
            "cli_config": cli_config,
        }

    @pytest.fixture
    def temp_csv_file(self) -> Generator[str, None, None]:
        """Create temporary CSV file for testing"""
        content = """id,name,email,age
1,Alice,alice@test.com,25
2,Bob,bob@test.com,30
3,Charlie,charlie@test.com,35"""

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        temp_file.write(content)
        temp_file.close()

        yield temp_file.name

        # Cleanup
        try:
            os.unlink(temp_file.name)
        except:
            pass

    @pytest.fixture
    def temp_excel_file(self) -> Generator[str, None, None]:
        """Create temporary Excel file for testing"""
        temp_file = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        temp_file.close()

        # Create Excel file with pandas
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "value": [100, 200, 300],
            }
        )
        df.to_excel(temp_file.name, index=False)

        yield temp_file.name

        # Cleanup
        try:
            os.unlink(temp_file.name)
        except:
            pass

    # ============================================================================
    # CSV Parsing and Separator Detection Tests
    # ============================================================================

    @pytest.mark.parametrize(
        "csv_content,expected_columns,test_description",
        [
            # Standard comma-separated
            (
                "id,name,email\n1,Alice,alice@test.com\n2,Bob,bob@test.com",
                ["id", "name", "email"],
                "standard_comma_csv",
            ),
            # Semicolon-separated (European format)
            (
                "id;name;email\n1;Alice;alice@test.com\n2;Bob;bob@test.com",
                ["id", "name", "email"],
                "semicolon_separated_csv",
            ),
            # Tab-separated
            (
                "id\tname\temail\n1\tAlice\talice@test.com\n2\tBob\tbob@test.com",
                ["id", "name", "email"],
                "tab_separated_csv",
            ),
            # Mixed content with commas in values (should use semicolon)
            (
                'id;name;description\n1;Alice;"Great person, works hard"\n2;Bob;"Good employee, reliable"',
                ["id", "name", "description"],
                "semicolon_with_comma_in_values",
            ),
        ],
    )
    def test_csv_separator_auto_detection(
        self,
        mock_configs: Dict[str, Any],
        csv_content: str,
        expected_columns: list,
        test_description: str,
    ) -> None:
        """Test automatic CSV separator detection and fallback"""
        # Arrange
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        temp_file.write(csv_content)
        temp_file.close()

        try:
            source_config = (
                TestDataBuilder.connection()
                .with_type(ConnectionType.CSV)
                .with_file_path(temp_file.name)
                .build()
            )
            validator = DataValidator(
                source_config=source_config,
                rules=[],
                core_config=mock_configs["core_config"],
                cli_config=mock_configs["cli_config"],
            )

            # Act
            df = validator._load_file_data()

            # Assert
            assert (
                list(df.columns) == expected_columns
            ), f"Test case '{test_description}' failed"
            assert len(df) >= 2, "Should have at least 2 data rows"

        finally:
            # Cleanup
            try:
                os.unlink(temp_file.name)
            except:
                pass

    def test_csv_separator_fallback_to_comma(
        self, mock_configs: Dict[str, Any]
    ) -> None:
        """Test fallback to comma separator when auto-detection fails"""
        # Arrange - Create CSV that doesn't parse well with any separator
        csv_content = "single_column_no_separators\nvalue1\nvalue2\nvalue3"

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        temp_file.write(csv_content)
        temp_file.close()

        try:
            source_config = (
                TestDataBuilder.connection()
                .with_type(ConnectionType.CSV)
                .with_file_path(temp_file.name)
                .build()
            )
            validator = DataValidator(
                source_config=source_config,
                rules=[],
                core_config=mock_configs["core_config"],
                cli_config=mock_configs["cli_config"],
            )

            # Act
            df = validator._load_file_data()

            # Assert
            assert len(df.columns) == 1  # Should have single column
            assert "single_column_no_separators" in df.columns[0]
            assert len(df) == 3  # Should have 3 data rows

        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass

    def test_csv_encoding_handling(self, mock_configs: Dict[str, Any]) -> None:
        """Test CSV loading with different encodings"""
        # Arrange - Create CSV with UTF-8 characters
        csv_content = "id,name,description\n1,José,café\n2,François,naïve\n3,北京,测试"

        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        )
        temp_file.write(csv_content)
        temp_file.close()

        try:
            source_config = (
                TestDataBuilder.connection()
                .with_type(ConnectionType.CSV)
                .with_file_path(temp_file.name)
                .build()
            )
            source_config.parameters = {"encoding": "utf-8"}

            validator = DataValidator(
                source_config=source_config,
                rules=[],
                core_config=mock_configs["core_config"],
                cli_config=mock_configs["cli_config"],
            )

            # Act
            df = validator._load_file_data()

            # Assert
            assert len(df) == 3
            assert "José" in df["name"].values
            assert "François" in df["name"].values
            assert "北京" in df["name"].values

        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass

    # ============================================================================
    # File Size Validation Tests
    # ============================================================================

    def test_file_size_limit_enforcement(self, mock_configs: Dict[str, Any]) -> None:
        """Test file size limit enforcement"""
        # Arrange - Mock a file that's too large
        large_file_path = "/fake/large_file.csv"

        source_config = (
            TestDataBuilder.connection()
            .with_type(ConnectionType.CSV)
            .with_file_path(large_file_path)
            .build()
        )
        validator = DataValidator(
            source_config=source_config,
            rules=[],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Mock file operations
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_stat = Mock()
        mock_stat.st_size = 100 * 1024 * 1024  # 100 MB (exceeds 50 MB limit)
        mock_path.stat.return_value = mock_stat

        with patch.object(Path, "__new__", return_value=mock_path):
            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                validator._load_file_data()

            assert "exceeds maximum allowed size" in str(exc_info.value)
            assert "100.0 MB" in str(exc_info.value)
            assert "50" in str(exc_info.value)  # Max size from config

    def test_file_size_within_limit(
        self, mock_configs: Dict[str, Any], temp_csv_file: str
    ) -> None:
        """Test file loading when size is within limit"""
        # Arrange
        source_config = (
            TestDataBuilder.connection()
            .with_type(ConnectionType.CSV)
            .with_file_path(temp_csv_file)
            .build()
        )
        validator = DataValidator(
            source_config=source_config,
            rules=[],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Act
        df = validator._load_file_data()

        # Assert
        assert len(df) == 3  # Should load successfully
        assert len(df.columns) == 4  # id, name, email, age

    def test_file_size_check_with_non_numeric_config(
        self, mock_configs: Dict[str, Any], temp_csv_file: str
    ) -> None:
        """Test file size check when config has non-numeric value"""
        # Arrange - Simulate config with non-numeric max_file_size_mb
        mock_configs["cli_config"].max_file_size_mb = "invalid"  # Non-numeric

        source_config = (
            TestDataBuilder.connection()
            .with_type(ConnectionType.CSV)
            .with_file_path(temp_csv_file)
            .build()
        )
        validator = DataValidator(
            source_config=source_config,
            rules=[],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Act - Should skip size check gracefully
        df = validator._load_file_data()

        # Assert
        assert len(df) == 3  # Should load successfully despite invalid config

    # ============================================================================
    # Empty File and No-Data Handling Tests
    # ============================================================================

    def test_empty_file_handling(self, mock_configs: Dict[str, Any]) -> None:
        """Test handling of completely empty files"""
        # Arrange
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        temp_file.write("")  # Empty file
        temp_file.close()

        try:
            source_config = (
                TestDataBuilder.connection()
                .with_type(ConnectionType.CSV)
                .with_file_path(temp_file.name)
                .build()
            )
            validator = DataValidator(
                source_config=source_config,
                rules=[],
                core_config=mock_configs["core_config"],
                cli_config=mock_configs["cli_config"],
            )

            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                validator._load_file_data()

            assert "File contains no data" in str(exc_info.value)

        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass

    def test_header_only_file_handling(self, mock_configs: Dict[str, Any]) -> None:
        """Test handling of files with headers but no data"""
        # Arrange
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        temp_file.write("id,name,email\n")  # Header only, no data rows
        temp_file.close()

        try:
            source_config = (
                TestDataBuilder.connection()
                .with_type(ConnectionType.CSV)
                .with_file_path(temp_file.name)
                .build()
            )
            validator = DataValidator(
                source_config=source_config,
                rules=[],
                core_config=mock_configs["core_config"],
                cli_config=mock_configs["cli_config"],
            )

            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                validator._load_file_data()

            assert "File contains no data" in str(exc_info.value)

        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass

    def test_whitespace_only_file_handling(self, mock_configs: Dict[str, Any]) -> None:
        """Test handling of files with only whitespace"""
        # Arrange
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        temp_file.write("   \n  \n \t \n")  # Only whitespace
        temp_file.close()

        try:
            source_config = (
                TestDataBuilder.connection()
                .with_type(ConnectionType.CSV)
                .with_file_path(temp_file.name)
                .build()
            )
            validator = DataValidator(
                source_config=source_config,
                rules=[],
                core_config=mock_configs["core_config"],
                cli_config=mock_configs["cli_config"],
            )

            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                validator._load_file_data()

            assert "File contains no data" in str(exc_info.value)

        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass

    # ============================================================================
    # File Type Support Tests
    # ============================================================================

    def test_excel_file_loading(
        self, mock_configs: Dict[str, Any], temp_excel_file: str
    ) -> None:
        """Test Excel file loading"""
        # Arrange
        source_config = (
            TestDataBuilder.connection()
            .with_type(ConnectionType.EXCEL)
            .with_file_path(temp_excel_file)
            .build()
        )
        validator = DataValidator(
            source_config=source_config,
            rules=[],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Act
        df = validator._load_file_data()

        # Assert
        assert len(df) == 3
        assert "id" in df.columns
        assert "name" in df.columns
        assert "value" in df.columns

    def test_json_file_loading(self, mock_configs: Dict[str, Any]) -> None:
        """Test JSON file loading"""
        # Arrange
        json_content = """[
            {"id": 1, "name": "Alice", "email": "alice@test.com"},
            {"id": 2, "name": "Bob", "email": "bob@test.com"},
            {"id": 3, "name": "Charlie", "email": "charlie@test.com"}
        ]"""

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        temp_file.write(json_content)
        temp_file.close()

        try:
            source_config = (
                TestDataBuilder.connection()
                .with_type(ConnectionType.JSON)
                .with_file_path(temp_file.name)
                .build()
            )
            validator = DataValidator(
                source_config=source_config,
                rules=[],
                core_config=mock_configs["core_config"],
                cli_config=mock_configs["cli_config"],
            )

            # Act
            df = validator._load_file_data()

            # Assert
            assert len(df) == 3
            assert "id" in df.columns
            assert "name" in df.columns
            assert "email" in df.columns

        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass

    def test_jsonl_file_loading(self, mock_configs: Dict[str, Any]) -> None:
        """Test JSON Lines file loading"""
        # Arrange
        jsonl_content = """{"id": 1, "name": "Alice", "email": "alice@test.com"}
{"id": 2, "name": "Bob", "email": "bob@test.com"}
{"id": 3, "name": "Charlie", "email": "charlie@test.com"}"""

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        temp_file.write(jsonl_content)
        temp_file.close()

        try:
            source_config = (
                TestDataBuilder.connection()
                .with_type(ConnectionType.JSON)
                .with_file_path(temp_file.name)
                .build()
            )
            validator = DataValidator(
                source_config=source_config,
                rules=[],
                core_config=mock_configs["core_config"],
                cli_config=mock_configs["cli_config"],
            )

            # Act
            df = validator._load_file_data()

            # Assert
            assert len(df) == 3
            assert "id" in df.columns
            assert "name" in df.columns
            assert "email" in df.columns

        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass

    def test_unsupported_file_type_handling(self, mock_configs: Dict[str, Any]) -> None:
        """Test handling of unsupported file types"""
        # Arrange
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        temp_file.write("some text content")
        temp_file.close()

        try:
            # Force unsupported connection type
            source_config = (
                TestDataBuilder.connection().with_file_path(temp_file.name).build()
            )
            source_config.connection_type = "UNSUPPORTED"  # type: ignore[assignment]

            validator = DataValidator(
                source_config=source_config,
                rules=[],
                core_config=mock_configs["core_config"],
                cli_config=mock_configs["cli_config"],
            )

            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                validator._load_file_data()

            assert "Unsupported file type" in str(exc_info.value)

        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass

    # ============================================================================
    # File Existence and Access Tests
    # ============================================================================

    def test_file_not_found_handling(self, mock_configs: Dict[str, Any]) -> None:
        """Test handling of non-existent files"""
        # Arrange
        non_existent_file = "/path/that/does/not/exist.csv"
        source_config = (
            TestDataBuilder.connection()
            .with_type(ConnectionType.CSV)
            .with_file_path(non_existent_file)
            .build()
        )
        validator = DataValidator(
            source_config=source_config,
            rules=[],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Act & Assert
        with pytest.raises(FileNotFoundError) as exc_info:
            validator._load_file_data()

        assert "File not found" in str(exc_info.value)
        assert non_existent_file in str(exc_info.value)

    def test_file_access_permission_error(
        self, mock_configs: Dict[str, Any], temp_csv_file: str
    ) -> None:
        """Test handling of file access permission errors"""
        # Arrange
        source_config = (
            TestDataBuilder.connection()
            .with_type(ConnectionType.CSV)
            .with_file_path(temp_csv_file)
            .build()
        )
        validator = DataValidator(
            source_config=source_config,
            rules=[],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Mock pandas to raise PermissionError
        with patch("pandas.read_csv", side_effect=PermissionError("Access denied")):
            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                validator._load_file_data()

            assert "Failed to parse file" in str(exc_info.value)

    # ============================================================================
    # Data Quality and Validation Tests
    # ============================================================================

    def test_corrupted_csv_handling(self, mock_configs: Dict[str, Any]) -> None:
        """Test handling of corrupted CSV files"""
        # Arrange - Create malformed CSV
        corrupted_content = """id,name,email
1,Alice,alice@test.com
2,Bob,bob@test.com,extra_field,another_field
3,Charlie"""  # Inconsistent columns

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        temp_file.write(corrupted_content)
        temp_file.close()

        try:
            source_config = (
                TestDataBuilder.connection()
                .with_type(ConnectionType.CSV)
                .with_file_path(temp_file.name)
                .build()
            )
            validator = DataValidator(
                source_config=source_config,
                rules=[],
                core_config=mock_configs["core_config"],
                cli_config=mock_configs["cli_config"],
            )

            # Act - pandas should handle this gracefully
            df = validator._load_file_data()

            # Assert - Should load but with some NaN values
            assert len(df) >= 2  # Should have at least some rows

        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass

    def test_extremely_wide_csv_handling(self, mock_configs: Dict[str, Any]) -> None:
        """Test handling of CSV with many columns"""
        # Arrange - Create CSV with many columns
        headers = [f"col_{i}" for i in range(100)]  # 100 columns
        values = [str(i) for i in range(100)]

        csv_content = ",".join(headers) + "\n" + ",".join(values) + "\n"

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        temp_file.write(csv_content)
        temp_file.close()

        try:
            source_config = (
                TestDataBuilder.connection()
                .with_type(ConnectionType.CSV)
                .with_file_path(temp_file.name)
                .build()
            )
            validator = DataValidator(
                source_config=source_config,
                rules=[],
                core_config=mock_configs["core_config"],
                cli_config=mock_configs["cli_config"],
            )

            # Act
            df = validator._load_file_data()

            # Assert
            assert len(df.columns) == 100
            assert len(df) == 1

        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass

    # ============================================================================
    # Performance and Memory Tests
    # ============================================================================

    def test_loading_logging_behavior(
        self, mock_configs: Dict[str, Any], temp_csv_file: str
    ) -> None:
        """Test that file loading generates appropriate log messages"""
        # Arrange
        source_config = (
            TestDataBuilder.connection()
            .with_type(ConnectionType.CSV)
            .with_file_path(temp_csv_file)
            .build()
        )
        validator = DataValidator(
            source_config=source_config,
            rules=[],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Act
        with patch.object(validator.logger, "info") as mock_logger:
            df = validator._load_file_data()

            # Assert
            assert mock_logger.called
            # Check that logging includes row and column counts
            log_calls = [call.args[0] for call in mock_logger.call_args_list]
            assert any(
                "rows" in log_msg and "columns" in log_msg for log_msg in log_calls
            )

    # ============================================================================
    # Property-Based Testing for Edge Cases
    # ============================================================================

    @pytest.mark.parametrize(
        "special_content,test_description",
        [
            # Files with special characters
            (
                'id,name\n1,"value with\nnewline"\n2,"value,with,commas"',
                "newlines_and_commas_in_values",
            ),
            # Files with quotes
            (
                'id,name\n1,"quoted value"\n2,\'single quoted\'\n3,"""triple quoted"""',
                "various_quote_types",
            ),
            # Files with Unicode
            ("id,name\n1,测试\n2,🎯emoji\n3,αβγ", "unicode_and_emoji"),
            # Files with very long values
            (f"id,description\n1,{'x' * 1000}\n2,normal", "very_long_values"),
        ],
    )
    def test_special_content_handling(
        self, mock_configs: Dict[str, Any], special_content: str, test_description: str
    ) -> None:
        """Property-based testing for special content scenarios"""
        # Arrange
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        )
        temp_file.write(special_content)
        temp_file.close()

        try:
            source_config = (
                TestDataBuilder.connection()
                .with_type(ConnectionType.CSV)
                .with_file_path(temp_file.name)
                .build()
            )
            validator = DataValidator(
                source_config=source_config,
                rules=[],
                core_config=mock_configs["core_config"],
                cli_config=mock_configs["cli_config"],
            )

            # Act - Should handle gracefully
            df = validator._load_file_data()

            # Assert basic properties
            assert (
                len(df) >= 1
            ), f"Test case '{test_description}' should have at least 1 row"
            assert (
                len(df.columns) >= 1
            ), f"Test case '{test_description}' should have at least 1 column"

        except Exception as e:
            # If it fails, it should be with a clear error message
            assert "Failed to parse file" in str(
                e
            ), f"Test case '{test_description}' should fail gracefully"

        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass

    # ============================================================================
    # Integration Tests
    # ============================================================================

    def test_full_file_loading_integration(self, mock_configs: Dict[str, Any]) -> None:
        """Test complete file loading integration with realistic data"""
        # Arrange - Create realistic CSV data
        csv_content = """customer_id,first_name,last_name,email,phone,registration_date,status
1001,John,Doe,john.doe@example.com,+1-555-0123,2023-01-15,active
1002,Jane,Smith,jane.smith@example.com,+1-555-0124,2023-01-16,active
1003,Bob,Johnson,bob.johnson@example.com,+1-555-0125,2023-01-17,inactive
1004,Alice,Williams,alice.williams@example.com,+1-555-0126,2023-01-18,pending"""

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        temp_file.write(csv_content)
        temp_file.close()

        try:
            source_config = (
                TestDataBuilder.connection()
                .with_type(ConnectionType.CSV)
                .with_file_path(temp_file.name)
                .build()
            )
            validator = DataValidator(
                source_config=source_config,
                rules=[],
                core_config=mock_configs["core_config"],
                cli_config=mock_configs["cli_config"],
            )

            # Act
            df = validator._load_file_data()

            # Assert comprehensive data quality
            assert len(df) == 4, "Should have 4 customer records"
            assert len(df.columns) == 7, "Should have 7 columns"

            # Verify column names
            expected_columns = [
                "customer_id",
                "first_name",
                "last_name",
                "email",
                "phone",
                "registration_date",
                "status",
            ]
            assert list(df.columns) == expected_columns

            # Verify data types can be inferred
            assert df["customer_id"].dtype in ["int64", "object"]
            assert df["email"].dtype == "object"

            # Verify no missing data in this clean dataset
            assert (
                not df.isnull().any().any()
            ), "Clean dataset should have no missing values"

        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass
