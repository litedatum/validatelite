"""
Test for issue #4: DATE_FORMAT validation support for PostgreSQL and SQLite

This test verifies:
1. PostgreSQL two-stage validation (regex + Python)
2. SQLite custom function validation
3. Support for flexible date format patterns (YYYY/yyyy, MM/mm, etc.)
4. Rule merger correctly identifies DATE_FORMAT rules as independent for PostgreSQL/SQLite
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from shared.database.database_dialect import PostgreSQLDialect, SQLiteDialect, MySQLDialect, DatabaseType
from shared.enums import RuleType
from shared.schema.connection_schema import ConnectionSchema
from shared.enums.connection_types import ConnectionType
from core.engine.rule_merger import RuleMergeManager


class TestDateFormatPatternSupport:
    """Test flexible date format pattern support"""

    def test_postgresql_format_pattern_to_regex(self):
        """Test PostgreSQL format pattern conversion to regex"""
        dialect = PostgreSQLDialect()

        # Test various format patterns with case variations
        test_cases = [
            ("YYYY-MM-DD", r"^\\d{4}-\\d{2}-\\d{2}$"),
            ("yyyy-mm-dd", r"^\\d{4}-\\d{2}-\\d{2}$"),
            ("MM/DD/YYYY", r"^\\d{2}/\\d{2}/\\d{4}$"),
            ("DD.MM.yyyy", r"^\\d{2}.\\d{2}.\\d{4}$"),
            ("YYYY-MM-DD HH:MI:SS", r"^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}$"),
        ]

        for format_pattern, expected_regex in test_cases:
            result = dialect._format_pattern_to_regex(format_pattern)
            assert result == expected_regex, f"Format {format_pattern} should generate regex {expected_regex}, got {result}"

    def test_postgresql_normalize_format_pattern(self):
        """Test PostgreSQL format pattern normalization for Python"""
        dialect = PostgreSQLDialect()

        test_cases = [
            ("YYYY-MM-DD", "%Y-%m-%d"),
            ("yyyy-mm-dd", "%Y-%m-%d"),
            ("MM/DD/YYYY", "%m/%d/%Y"),
            ("DD.MM.yyyy", "%d.%m.%Y"),
            ("YYYY-MM-DD HH:MI:SS", "%Y-%m-%d %H:%M:%S"),
        ]

        for format_pattern, expected_python in test_cases:
            result = dialect._normalize_format_pattern(format_pattern)
            assert result == expected_python, f"Format {format_pattern} should normalize to {expected_python}, got {result}"

    def test_sqlite_normalize_format_pattern(self):
        """Test SQLite format pattern normalization"""
        dialect = SQLiteDialect()

        test_cases = [
            ("YYYY-MM-DD", "%Y-%m-%d"),
            ("yyyy-mm-dd", "%Y-%m-%d"),
            ("MM/DD/YYYY", "%m/%d/%Y"),
            ("DD.MM.yyyy", "%d.%m.%Y"),
            ("YYYY-MM-DD HH:MI:SS", "%Y-%m-%d %H:%M:%S"),
        ]

        for format_pattern, expected_python in test_cases:
            result = dialect._normalize_format_pattern(format_pattern)
            assert result == expected_python, f"Format {format_pattern} should normalize to {expected_python}, got {result}"


class TestDateFormatSupportStatus:
    """Test that databases report correct date format support status"""

    def test_mysql_supports_date_format(self):
        """MySQL should support date formats"""
        dialect = MySQLDialect()
        assert dialect.is_supported_date_format() == True

    def test_postgresql_supports_date_format(self):
        """PostgreSQL should now support date formats with two-stage validation"""
        dialect = PostgreSQLDialect()
        assert dialect.is_supported_date_format() == True

    def test_sqlite_supports_date_format(self):
        """SQLite should now support date formats with custom functions"""
        dialect = SQLiteDialect()
        assert dialect.is_supported_date_format() == True


class TestPostgreSQLTwoStageValidation:
    """Test PostgreSQL two-stage date validation SQL generation"""

    def test_two_stage_sql_generation(self):
        """Test PostgreSQL two-stage SQL generation"""
        dialect = PostgreSQLDialect()

        column = "birth_date"
        format_pattern = "YYYY-MM-DD"
        table_name = "users"
        filter_condition = "active = true"

        stage1_sql, stage2_sql = dialect.get_two_stage_date_validation_sql(
            column, format_pattern, table_name, filter_condition
        )

        # Stage 1 should count regex failures
        assert "regex_failed_count" in stage1_sql
        assert "!~" in stage1_sql  # PostgreSQL regex operator
        assert "WHERE birth_date IS NOT NULL" in stage1_sql
        assert "active = true" in stage1_sql

        # Stage 2 should get candidates for Python validation
        assert "DISTINCT birth_date" in stage2_sql
        assert "~" in stage2_sql  # PostgreSQL regex operator (positive match)
        assert "LIMIT 10000" in stage2_sql
        assert "active = true" in stage2_sql


class TestSQLiteCustomFunction:
    """Test SQLite custom function setup"""

    def test_sqlite_date_validation_function(self):
        """Test SQLite date validation custom function"""
        from shared.database.sqlite_functions import is_valid_date

        # Test valid dates
        assert is_valid_date("2023-12-25", "%Y-%m-%d") == True
        assert is_valid_date("12/25/2023", "%m/%d/%Y") == True
        assert is_valid_date("", "%Y-%m-%d") == True  # Empty should be valid

        # Test invalid dates
        assert is_valid_date("2023-02-31", "%Y-%m-%d") == False  # Invalid date
        assert is_valid_date("not-a-date", "%Y-%m-%d") == False  # Invalid format
        assert is_valid_date("2023-13-01", "%Y-%m-%d") == False  # Invalid month

    def test_sqlite_get_date_clause(self):
        """Test SQLite get_date_clause uses custom function"""
        dialect = SQLiteDialect()

        result = dialect.get_date_clause("birth_date", "YYYY-MM-DD")

        assert "IS_VALID_DATE(birth_date, 'YYYY-MM-DD')" in result
        assert "CASE WHEN" in result
        assert "THEN 'valid' ELSE NULL END" in result


class TestRuleMergerDateFormatHandling:
    """Test that rule merger correctly handles DATE_FORMAT rules"""

    def test_postgresql_date_format_rules_are_independent(self):
        """PostgreSQL DATE_FORMAT rules should be marked as independent"""
        # Mock PostgreSQL connection
        connection = Mock(spec=ConnectionSchema)
        connection.connection_type = ConnectionType.POSTGRESQL

        with patch('core.engine.rule_merger.get_dialect') as mock_get_dialect:
            mock_dialect = Mock()
            mock_dialect.database_type = DatabaseType.POSTGRESQL
            mock_dialect.is_supported_date_format.return_value = True
            mock_get_dialect.return_value = mock_dialect

            merger = RuleMergeManager(connection)

            # DATE_FORMAT should be in independent rule types for PostgreSQL
            assert RuleType.DATE_FORMAT in merger.independent_rule_types

    def test_sqlite_date_format_rules_are_independent(self):
        """SQLite DATE_FORMAT rules should be marked as independent"""
        # Mock SQLite connection
        connection = Mock(spec=ConnectionSchema)
        connection.connection_type = ConnectionType.SQLITE

        with patch('core.engine.rule_merger.get_dialect') as mock_get_dialect:
            mock_dialect = Mock()
            mock_dialect.database_type = DatabaseType.SQLITE
            mock_dialect.is_supported_date_format.return_value = True
            mock_get_dialect.return_value = mock_dialect

            merger = RuleMergeManager(connection)

            # DATE_FORMAT should be in independent rule types for SQLite
            assert RuleType.DATE_FORMAT in merger.independent_rule_types

    def test_mysql_date_format_rules_can_be_merged(self):
        """MySQL DATE_FORMAT rules should be mergeable"""
        # Mock MySQL connection
        connection = Mock(spec=ConnectionSchema)
        connection.connection_type = ConnectionType.MYSQL

        with patch('core.engine.rule_merger.get_dialect') as mock_get_dialect:
            mock_dialect = Mock()
            mock_dialect.database_type = DatabaseType.MYSQL
            mock_dialect.is_supported_date_format.return_value = True
            mock_get_dialect.return_value = mock_dialect

            merger = RuleMergeManager(connection)

            # DATE_FORMAT should NOT be in independent rule types for MySQL
            assert RuleType.DATE_FORMAT not in merger.independent_rule_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])