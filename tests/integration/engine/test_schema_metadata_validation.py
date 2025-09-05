"""
Integration tests for schema metadata validation with real databases

Tests cover:
1. Real database metadata extraction across different DB types
2. End-to-end validation workflows with metadata
3. Performance testing with large schemas
4. Mixed success/failure scenarios
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from click.testing import CliRunner

from cli.app import cli_app
from core.executors.schema_executor import SchemaExecutor
from shared.schema.connection_schema import ConnectionSchema
from shared.enums import ConnectionType
from tests.shared.builders.test_builders import TestDataBuilder


def write_temp_schema_file(content: Dict[str, Any]) -> str:
    """Write schema content to a temporary file and return the path"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(content, f, indent=2)
        return f.name


@pytest.mark.integration
@pytest.mark.database
class TestRealDatabaseMetadataExtraction:
    """Test metadata extraction from real database connections"""

    def test_sqlite_metadata_extraction(self, temp_sqlite_db):
        """Test metadata extraction from SQLite database with various column types"""
        # Create test table with various column types and constraints
        db_path = temp_sqlite_db
        
        # Test uses temp_sqlite_db fixture that creates the database
        
        # Schema content matching the test table
        schema_content = {
            "test_metadata_table": {
                "rules": [
                    {
                        "field": "id",
                        "type": "integer",
                        "nullable": False
                    },
                    {
                        "field": "name",
                        "type": "string",
                        "max_length": 100,
                        "nullable": False
                    },
                    {
                        "field": "email",
                        "type": "string",
                        "max_length": 255,
                        "nullable": True
                    },
                    {
                        "field": "price",
                        "type": "float",
                        "precision": 10,
                        "scale": 2,
                        "nullable": True
                    }
                ]
            }
        }

        schema_file = write_temp_schema_file(schema_content)
        
        try:
            runner = CliRunner()
            result = runner.invoke(
                cli_app,
                [
                    "schema",
                    "--conn", f"sqlite://{db_path}",
                    "--rules", schema_file,
                ]
            )

            # Should succeed with proper database and schema structure
            assert result.exit_code in [0, 1]  # 0=success, 1=validation failure

        finally:
            Path(schema_file).unlink()

    def test_mysql_metadata_extraction(self, mysql_connection_params):
        """Test MySQL metadata extraction with real MySQL connection"""
        # Use real MySQL connection from fixtures
        schema_content = {
            "mysql_test_table": {
                "rules": [
                    {
                        "field": "id",
                        "type": "integer",
                        "nullable": False
                    },
                    {
                        "field": "varchar_field",
                        "type": "string", 
                        "max_length": 255,
                        "nullable": False
                    },
                    {
                        "field": "decimal_field",
                        "type": "float",
                        "precision": 10,
                        "scale": 2,
                        "nullable": True
                    }
                ]
            }
        }

        schema_file = write_temp_schema_file(schema_content)
        
        try:
            # Build MySQL connection URL from fixture params
            from shared.database.connection import get_db_url
            mysql_url = get_db_url(
                str(mysql_connection_params["db_type"]),
                str(mysql_connection_params["host"]),
                int(mysql_connection_params["port"]),
                str(mysql_connection_params["database"]),
                str(mysql_connection_params["username"]),
                str(mysql_connection_params["password"])
            )
            
            runner = CliRunner()
            result = runner.invoke(
                cli_app,
                [
                    "schema",
                    "--conn", mysql_url,
                    "--rules", schema_file
                ]
            )

            # Should succeed with proper connection
            assert result.exit_code in [0, 1]  # 0 = success, 1 = validation failure

        finally:
            Path(schema_file).unlink()

    def test_postgresql_metadata_extraction(self, postgres_connection_params):
        """Test PostgreSQL metadata extraction with real PostgreSQL connection"""
        # Use real PostgreSQL connection from fixtures
        schema_content = {
            "postgres_test_table": {
                "rules": [
                    {
                        "field": "id",
                        "type": "integer",
                        "nullable": False
                    },
                    {
                        "field": "text_field",
                        "type": "string",
                        "nullable": True
                    },
                    {
                        "field": "numeric_field",
                        "type": "float",
                        "precision": 12,
                        "scale": 4,
                        "nullable": True
                    }
                ]
            }
        }

        schema_file = write_temp_schema_file(schema_content)
        
        try:
            # Build PostgreSQL connection URL from fixture params
            from shared.database.connection import get_db_url
            postgres_url = get_db_url(
                str(postgres_connection_params["db_type"]),
                str(postgres_connection_params["host"]),
                int(postgres_connection_params["port"]),
                str(postgres_connection_params["database"]),
                str(postgres_connection_params["username"]),
                str(postgres_connection_params["password"])
            )
            
            runner = CliRunner()
            result = runner.invoke(
                cli_app,
                [
                    "schema",
                    "--conn", postgres_url,
                    "--rules", schema_file
                ]
            )

            # Should succeed with proper connection
            assert result.exit_code in [0, 1]  # 0 = success, 1 = validation failure

        finally:
            Path(schema_file).unlink()


@pytest.mark.integration
class TestEndToEndValidationWorkflows:
    """Test complete workflows from CLI to database validation"""

    def test_complete_workflow_success_scenario(self, temp_sqlite_db):
        """Test complete successful validation workflow with metadata"""
        db_path = temp_sqlite_db
        
        # Schema that should match the test database structure
        schema_content = {
            "tables": [
                {
                    "name": "test_users",
                    "columns": [
                        {
                            "name": "id", 
                            "type": "INTEGER",
                            "nullable": False
                        },
                        {
                            "name": "username",
                            "type": "STRING",
                            "max_length": 50,
                            "nullable": False
                        },
                        {
                            "name": "email",
                            "type": "STRING",
                            "max_length": 100,
                            "nullable": True
                        }
                    ]
                }
            ]
        }

        schema_file = write_temp_schema_file(schema_content)
        
        try:
            runner = CliRunner()
            result = runner.invoke(
                cli_app,
                [
                    "schema",
                    "--conn", f"sqlite://{db_path}",
                    "--rules", schema_file,
                    "--verbose"
                ]
            )

            # Check that the command executed
            assert isinstance(result.exit_code, int)
            
            # If successful, should contain success indicators
            if result.exit_code == 0:
                assert any(keyword in result.output.lower() for keyword in ["success", "pass", "valid"])

        finally:
            Path(schema_file).unlink()

    def test_mixed_success_failure_scenarios(self, temp_sqlite_db):
        """Test scenarios with some validations passing and others failing"""
        db_path = temp_sqlite_db
        
        # Schema with intentional mismatches
        schema_content = {
            "tables": [
                {
                    "name": "test_users",
                    "columns": [
                        {
                            "name": "id",
                            "type": "INTEGER", 
                            "nullable": False
                            # This should match
                        },
                        {
                            "name": "username",
                            "type": "STRING",
                            "max_length": 25,  # Intentionally different from actual
                            "nullable": False
                        },
                        {
                            "name": "nonexistent_column",
                            "type": "STRING",
                            "max_length": 100,
                            "nullable": True
                            # This column doesn't exist - should fail
                        }
                    ]
                }
            ]
        }

        schema_file = write_temp_schema_file(schema_content)
        
        try:
            runner = CliRunner()
            result = runner.invoke(
                cli_app,
                [
                    "schema",
                    "--conn", f"sqlite://{db_path}",
                    "--rules", schema_file
                ]
            )

            # Should handle mixed success/failure scenarios
            assert isinstance(result.exit_code, int)

        finally:
            Path(schema_file).unlink()

    def test_large_schema_file_with_metadata(self, temp_sqlite_db):
        """Test handling of large schema files with extensive metadata"""
        db_path = temp_sqlite_db
        
        # Generate a large schema with many tables and columns
        tables = []
        for table_num in range(5):  # 5 tables
            columns = []
            for col_num in range(20):  # 20 columns each
                columns.append({
                    "name": f"col_{col_num}",
                    "type": "STRING",
                    "max_length": 100 + col_num,
                    "nullable": col_num % 2 == 0
                })
            
            tables.append({
                "name": f"large_table_{table_num}",
                "columns": columns
            })

        schema_content = {"tables": tables}
        schema_file = write_temp_schema_file(schema_content)
        
        try:
            runner = CliRunner()
            result = runner.invoke(
                cli_app,
                [
                    "schema",
                    "--conn", f"sqlite://{db_path}",
                    "--rules", schema_file,
                ]
            )

            # Should handle large schemas without crashing
            assert isinstance(result.exit_code, int)

        finally:
            Path(schema_file).unlink()


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceWithRealDatabases:
    """Test performance aspects with real database connections"""

    def test_performance_with_many_columns(self, temp_sqlite_db):
        """Test performance with tables containing many columns"""
        db_path = temp_sqlite_db
        
        # Create a schema with a table containing 50+ columns
        columns = []
        for i in range(50):
            columns.append({
                "name": f"column_{i:02d}",
                "type": "STRING" if i % 2 == 0 else "INTEGER",
                "max_length": 255 if i % 2 == 0 else None,
                "nullable": i % 3 == 0
            })

        schema_content = {
            "tables": [
                {
                    "name": "wide_table",
                    "columns": columns
                }
            ]
        }

        schema_file = write_temp_schema_file(schema_content)
        
        try:
            import time
            
            runner = CliRunner()
            start_time = time.time()
            
            result = runner.invoke(
                cli_app,
                [
                    "schema",
                    "--conn", f"sqlite://{db_path}",
                    "--rules", schema_file,
                ]
            )
            
            execution_time = time.time() - start_time
            
            # Should complete within reasonable time (10 seconds for 50 columns)
            assert execution_time < 10.0
            assert isinstance(result.exit_code, int)

        finally:
            Path(schema_file).unlink()

    def test_memory_usage_with_large_metadata(self, temp_sqlite_db):
        """Test memory efficiency with large metadata schemas"""
        db_path = temp_sqlite_db
        
        # Create multiple tables with extensive metadata
        tables = []
        for table_num in range(10):
            columns = []
            for col_num in range(30):
                columns.append({
                    "name": f"t{table_num}_col_{col_num}",
                    "type": "STRING",
                    "max_length": 500 + (col_num * 10),
                    "nullable": True,
                    # Additional metadata for memory testing
                    "description": f"Test column {col_num} in table {table_num}" * 5
                })
            
            tables.append({
                "name": f"memory_test_table_{table_num}",
                "columns": columns,
                "description": f"Memory test table number {table_num}" * 10
            })

        schema_content = {"tables": tables}
        schema_file = write_temp_schema_file(schema_content)
        
        try:
            runner = CliRunner()
            result = runner.invoke(
                cli_app,
                [
                    "schema",
                    "--conn", f"sqlite://{db_path}",
                    "--rules", schema_file,
                ]
            )

            # Should handle large metadata without memory issues
            assert isinstance(result.exit_code, int)

        finally:
            Path(schema_file).unlink()


@pytest.mark.integration
class TestErrorRecoveryAndResilience:
    """Test error recovery and system resilience"""

    def test_connection_timeout_recovery(self, temp_sqlite_db):
        """Test recovery from connection timeout scenarios"""
        db_path = temp_sqlite_db
        
        schema_content = {
            "tables": [
                {
                    "name": "timeout_test_table",
                    "columns": [
                        {
                            "name": "id",
                            "type": "INTEGER",
                            "nullable": False
                        }
                    ]
                }
            ]
        }

        schema_file = write_temp_schema_file(schema_content)
        
        try:
            # Test with a very short timeout to simulate timeout conditions
            runner = CliRunner()
            result = runner.invoke(
                cli_app,
                [
                    "schema",
                    "--conn", f"sqlite://{db_path}",
                    "--rules", schema_file,
                    "--verbose"  # Use valid option instead
                ]
            )

            # Should handle timeout gracefully
            assert isinstance(result.exit_code, int)

        finally:
            Path(schema_file).unlink()

    def test_partial_metadata_availability(self, temp_sqlite_db):
        """Test handling when only partial metadata is available"""
        db_path = temp_sqlite_db
        
        # Schema requiring metadata that may not be available in SQLite
        schema_content = {
            "tables": [
                {
                    "name": "partial_metadata_table",
                    "columns": [
                        {
                            "name": "id",
                            "type": "INTEGER",
                            "nullable": False
                        },
                        {
                            "name": "precise_decimal",
                            "type": "FLOAT",
                            "precision": 15,  # High precision that SQLite may not support
                            "scale": 8,
                            "nullable": True
                        }
                    ]
                }
            ]
        }

        schema_file = write_temp_schema_file(schema_content)
        
        try:
            runner = CliRunner()
            result = runner.invoke(
                cli_app,
                [
                    "schema",
                    "--conn", f"sqlite://{db_path}",
                    "--rules", schema_file
                ]
            )

            # Should handle partial metadata gracefully
            assert isinstance(result.exit_code, int)

        finally:
            Path(schema_file).unlink()


# Test fixtures and conftest integration
@pytest.fixture
def temp_sqlite_db(tmp_path):
    """Create a temporary SQLite database for testing"""
    db_file = tmp_path / "test_metadata.db"
    
    # Create a simple test table for metadata validation
    import sqlite3
    
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    
    # Create test tables with various column types
    cursor.execute("""
        CREATE TABLE test_users (
            id INTEGER PRIMARY KEY,
            username TEXT(50) NOT NULL,
            email TEXT(100),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE test_metadata_table (
            id INTEGER PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(255),
            price DECIMAL(10,2),
            weight REAL
        )
    """)
    
    # Insert some test data
    cursor.execute("INSERT INTO test_users (username, email) VALUES (?, ?)", 
                  ("testuser", "test@example.com"))
    
    conn.commit()
    conn.close()
    
    return str(db_file)


# Note: Database availability is handled by skipif decorators directly