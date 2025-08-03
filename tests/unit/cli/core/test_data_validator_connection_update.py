"""
🧙‍♂️ Test Data Validator Connection Update Logic - Testing Ghost's Connection Suite

This module tests the critical connection update and injection logic in DataValidator:
- Database/table name completion based on source configuration
- Multi-rule scenarios with shared connection information
- Entity relationship management
- Connection-agnostic rule handling (after refactoring)

Modern Testing Strategies Applied:
✅ Schema Builder Pattern - Fluent test data construction
✅ Contract Testing - Mock/implementation consistency
✅ State Machine Testing - Connection update state transitions
✅ Invariant Testing - Connection consistency guarantees
"""

import uuid
from typing import Any, Dict, List, Sequence, cast
from unittest.mock import Mock

import pytest

from cli.core.data_validator import DataValidator
from shared.enums import ConnectionType, RuleType
from shared.schema import ConnectionSchema, RuleSchema
from shared.schema.base import RuleTarget, TargetEntity

# Import our modern testing infrastructure
from tests.shared.builders.test_builders import TestDataBuilder
from tests.shared.contracts.test_contracts import MockContract


class TestDataValidatorConnectionUpdate:
    """
    🔗 Comprehensive test suite for DataValidator connection handling

    Focus Areas:
    1. Database name completion from source config
    2. Table name completion for different source types
    3. Multi-rule connection sharing
    4. Entity relationship consistency
    """

    @pytest.fixture
    def mock_configs(self) -> Dict[str, Any]:
        """Provide mock configurations using Contract Testing"""
        return {
            "core_config": MockContract.create_core_config_mock(),
            "cli_config": MockContract.create_cli_config_mock(),
        }

    # ============================================================================
    # Database Name Completion Tests
    # ============================================================================

    @pytest.mark.parametrize(
        "source_config,expected_database",
        [
            # SQLite - should use "main"
            (
                TestDataBuilder.connection()
                .with_type(ConnectionType.SQLITE)
                .with_file_path("/tmp/test.db")
                .build(),
                "main",
            ),
            # MySQL with explicit database
            (
                TestDataBuilder.connection()
                .with_type(ConnectionType.MYSQL)
                .with_database("analytics")
                .build(),
                "analytics",
            ),
            # PostgreSQL with explicit database
            (
                TestDataBuilder.connection()
                .with_type(ConnectionType.POSTGRESQL)
                .with_database("warehouse")
                .build(),
                "warehouse",
            ),
            # CSV file - should use "main" (SQLite conversion)
            (
                TestDataBuilder.connection()
                .with_type(ConnectionType.CSV)
                .with_file_path("/data/users.csv")
                .build(),
                "main",
            ),
        ],
    )
    def test_database_name_completion(
        self,
        mock_configs: Dict[str, Any],
        source_config: ConnectionSchema,
        expected_database: str,
    ) -> None:
        """Test database name completion based on source configuration"""
        # Arrange
        rule = (
            TestDataBuilder.rule()
            .as_not_null_rule()
            .with_target("default", "users", "id")
            .build()
        )

        # Act
        validator = DataValidator(
            source_config=source_config,
            rules=[rule],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Assert
        updated_rule = validator.rules[0]
        assert updated_rule.target.entities[0].database == expected_database

    def test_database_name_fallback_to_default(
        self, mock_configs: Dict[str, Any]
    ) -> None:
        """Test database name falls back to 'default' when not specified"""
        # Arrange
        source_config = (
            TestDataBuilder.connection().with_type(ConnectionType.MYSQL).build()
        )
        source_config.db_name = None  # No database specified
        rule = (
            TestDataBuilder.rule()
            .as_not_null_rule()
            .with_target("original", "users", "id")
            .build()
        )

        # Act
        validator = DataValidator(
            source_config=source_config,
            rules=[rule],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Assert
        updated_rule = validator.rules[0]
        assert updated_rule.target.entities[0].database == "default"

    # ============================================================================
    # Table Name Completion Tests
    # ============================================================================

    @pytest.mark.parametrize(
        "source_type,source_params,expected_table",
        [
            # Table specified in parameters
            (ConnectionType.MYSQL, {"table": "user_profiles"}, "user_profiles"),
            (ConnectionType.POSTGRESQL, {"table": "orders"}, "orders"),
            # CSV file - table name from filename
            (ConnectionType.CSV, {}, None),  # Will test separately with file path
            # Excel file - table name from filename
            (ConnectionType.EXCEL, {}, None),  # Will test separately with file path
            # No table specified - default
            (ConnectionType.MYSQL, {}, "default_table"),
            (ConnectionType.POSTGRESQL, {}, "default_table"),
        ],
    )
    def test_table_name_completion_from_parameters(
        self,
        mock_configs: Dict[str, Any],
        source_type: ConnectionType,
        source_params: Dict[str, Any],
        expected_table: str,
    ) -> None:
        """Test table name completion from source parameters"""
        # Arrange
        source_config = (
            TestDataBuilder.connection()
            .with_type(source_type)
            .with_parameters(source_params)
            .build()
        )
        rule = (
            TestDataBuilder.rule()
            .as_not_null_rule()
            .with_target("test_db", "original_table", "id")
            .build()
        )

        # Act
        validator = DataValidator(
            source_config=source_config,
            rules=[rule],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )
        # Assert
        updated_rule = validator.rules[0]
        if expected_table:
            assert updated_rule.target.entities[0].table == expected_table
        else:
            # For file types without explicit table, should preserve original
            assert updated_rule.target.entities[0].table == "data"

    @pytest.mark.parametrize(
        "file_path,expected_clean_table",
        [
            ("/data/users.csv", "users"),
            ("/reports/sales-2023.csv", "sales_2023"),
            ("/temp/user profiles.xlsx", "user_profiles"),
            ("/data/2023-orders.json", "table_2023_orders"),  # Starts with number
            ("/path/complex-file@name.csv", "complex_file_name"),
            ("simple.csv", "simple"),
        ],
    )
    def test_table_name_completion_from_file_path(
        self, mock_configs: Dict[str, Any], file_path: str, expected_clean_table: str
    ) -> None:
        """Test table name completion from file path for file-based sources"""
        # Arrange
        source_config = (
            TestDataBuilder.connection()
            .with_type(ConnectionType.CSV)
            .with_file_path(file_path)
            .build()
        )
        rule = (
            TestDataBuilder.rule()
            .as_not_null_rule()
            .with_target("test_db", "original_table", "id")
            .build()
        )

        # Act
        validator = DataValidator(
            source_config=source_config,
            rules=[rule],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Assert
        updated_rule = validator.rules[0]
        assert updated_rule.target.entities[0].table == expected_clean_table

    def test_table_name_missing_file_path_fallback(
        self, mock_configs: Dict[str, Any]
    ) -> None:
        """Test table name fallback when file path is missing"""
        # Arrange
        source_config = (
            TestDataBuilder.connection()
            .with_type(ConnectionType.CSV)
            .build()  # No file path
        )
        rule = (
            TestDataBuilder.rule()
            .as_not_null_rule()
            .with_target("test_db", "original_table", "id")
            .build()
        )
        # Act
        validator = DataValidator(
            source_config=source_config,
            rules=[rule],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Assert
        updated_rule = validator.rules[0]
        assert updated_rule.target.entities[0].table == "data"

    # ============================================================================
    # Multi-Rule Connection Sharing Tests
    # ============================================================================

    def test_multiple_rules_same_table_different_columns(
        self, mock_configs: Dict[str, Any]
    ) -> None:
        """Test multiple rules targeting the same table get consistent connection info"""
        # Arrange
        source_config = (
            TestDataBuilder.connection()
            .with_type(ConnectionType.MYSQL)
            .with_database("analytics")
            .with_parameters({"table": "user_profiles"})
            .build()
        )

        rules = [
            TestDataBuilder.rule()
            .as_not_null_rule()
            .with_target("original_db", "original_table", "id")
            .build(),
            TestDataBuilder.rule()
            .as_unique_rule()
            .with_target("different_db", "different_table", "email")
            .build(),
            TestDataBuilder.rule()
            .as_range_rule(0, 120)
            .with_target("another_db", "another_table", "age")
            .build(),
        ]

        # Act
        validator = DataValidator(
            source_config=source_config,
            rules=cast(List[RuleSchema | Dict[str, Any]], rules),
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Assert
        for rule in validator.rules:
            assert rule.target.entities[0].database == "analytics"
            assert rule.target.entities[0].table == "user_profiles"

    def test_rules_with_multiple_entities_connection_update(
        self, mock_configs: Dict[str, Any]
    ) -> None:
        """Test rules with multiple target entities get consistent connection info"""
        # Arrange
        source_config = (
            TestDataBuilder.connection()
            .with_type(ConnectionType.POSTGRESQL)
            .with_database("warehouse")
            .with_parameters({"table": "orders"})
            .build()
        )

        # Create rule with multiple entities
        rule = (
            TestDataBuilder.rule()
            .as_not_null_rule()
            .with_target("original_db", "original_table", "order_id")
            .build()
        )
        # Add second entity to the same rule
        second_entity = TargetEntity(
            database="different_db", table="different_table", column="customer_id"
        )
        rule.target.entities.append(second_entity)

        # Act
        validator = DataValidator(
            source_config=source_config,
            rules=[rule],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Assert
        updated_rule = validator.rules[0]
        for entity in updated_rule.target.entities:
            assert entity.database == "warehouse"
            assert entity.table == "orders"

    # ============================================================================
    # Connection Update Preservation Tests
    # ============================================================================

    def test_connection_update_preserves_original_columns(
        self, mock_configs: Dict[str, Any]
    ) -> None:
        """Test that connection update preserves original column names"""
        # Arrange
        source_config = (
            TestDataBuilder.connection()
            .with_type(ConnectionType.SQLITE)
            .with_file_path("/tmp/test.db")
            .build()
        )

        original_columns = ["id", "name", "email", "age"]
        rules = []
        for col in original_columns:
            rule = (
                TestDataBuilder.rule()
                .as_not_null_rule()
                .with_target("test_db", "users", col)
                .build()
            )
            rules.append(rule)

        # Act
        validator = DataValidator(
            source_config=source_config,
            rules=cast(List[RuleSchema | Dict[str, Any]], rules),
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Assert
        for i, rule in enumerate(validator.rules):
            assert rule.target.entities[0].column == original_columns[i]

    # ============================================================================
    # Edge Cases and Error Handling
    # ============================================================================

    def test_connection_update_with_empty_rules_list(
        self, mock_configs: Dict[str, Any]
    ) -> None:
        """Test connection update with empty rules list"""
        # Arrange
        source_config = (
            TestDataBuilder.connection()
            .with_type(ConnectionType.CSV)
            .with_file_path("/data/test.csv")
            .build()
        )

        # Act
        validator = DataValidator(
            source_config=source_config,
            rules=[],
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Assert
        assert len(validator.rules) == 0

    def test_connection_update_with_rule_missing_entities(
        self, mock_configs: Dict[str, Any]
    ) -> None:
        """Test connection update with rule that has no target entities"""
        # Arrange
        source_config = (
            TestDataBuilder.connection()
            .with_type(ConnectionType.MYSQL)
            .with_database("test_db")
            .build()
        )

        # Create rule with empty target entities
        rule = (
            TestDataBuilder.rule()
            .as_not_null_rule()
            .with_target("original_db", "original_table", "id")
            .build()
        )
        rule.target.entities = []  # Clear entities

        # Act
        validator = DataValidator(
            source_config=source_config,
            rules=cast(List[RuleSchema | Dict[str, Any]], [rule]),
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Assert
        updated_rule = validator.rules[0]
        assert len(updated_rule.target.entities) == 0

    # ============================================================================
    # Invariant Testing
    # ============================================================================

    def test_connection_consistency_invariants(
        self, mock_configs: Dict[str, Any]
    ) -> None:
        """Test connection consistency invariants across all rules"""
        # Arrange
        source_config = (
            TestDataBuilder.connection()
            .with_type(ConnectionType.POSTGRESQL)
            .with_database("analytics")
            .with_parameters({"table": "user_metrics"})
            .build()
        )

        rules = [
            TestDataBuilder.rule()
            .as_not_null_rule()
            .with_target("db1", "table1", "col1")
            .build(),
            TestDataBuilder.rule()
            .as_unique_rule()
            .with_target("db2", "table2", "col2")
            .build(),
            TestDataBuilder.rule()
            .as_range_rule(0, 100)
            .with_target("db3", "table3", "col3")
            .build(),
        ]

        # Act
        validator = DataValidator(
            source_config=source_config,
            rules=cast(List[RuleSchema | Dict[str, Any]], rules),
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Assert invariants
        databases = {rule.target.entities[0].database for rule in validator.rules}
        tables = {rule.target.entities[0].table for rule in validator.rules}

        # All rules should have the same database and table
        assert len(databases) == 1
        assert len(tables) == 1
        assert "analytics" in databases
        assert "user_metrics" in tables

    # ============================================================================
    # Performance Testing
    # ============================================================================

    def test_connection_update_performance_with_many_rules(
        self, mock_configs: Dict[str, Any]
    ) -> None:
        """Test connection update performance with many rules"""
        # Arrange
        source_config = (
            TestDataBuilder.connection()
            .with_type(ConnectionType.MYSQL)
            .with_database("large_dataset")
            .with_parameters({"table": "customer_data"})
            .build()
        )

        # Create many rules
        rules = []
        for i in range(100):
            rule = (
                TestDataBuilder.rule()
                .as_not_null_rule()
                .with_target(f"db_{i}", f"table_{i}", f"col_{i}")
                .build()
            )
            rules.append(rule)

        # Act - Should complete quickly
        import time

        start_time = time.time()

        validator = DataValidator(
            source_config=source_config,
            rules=cast(List[RuleSchema | Dict[str, Any]], rules),
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        end_time = time.time()

        # Assert
        assert len(validator.rules) == 100
        assert (end_time - start_time) < 1.0  # Should complete in under 1 second
        for rule in validator.rules:
            assert rule.target.entities[0].database == "large_dataset"
            assert rule.target.entities[0].table == "customer_data"

    # ============================================================================
    # Real-World Scenarios
    # ============================================================================

    def test_real_world_csv_validation_scenario(
        self, mock_configs: Dict[str, Any]
    ) -> None:
        """Test real-world CSV validation scenario with multiple rules"""
        # Arrange
        source_config = (
            TestDataBuilder.connection()
            .with_type(ConnectionType.CSV)
            .with_file_path("/data/customer_data.csv")
            .build()
        )

        rules = [
            TestDataBuilder.rule()
            .as_not_null_rule()
            .with_target("main", "data", "customer_id")
            .build(),
            TestDataBuilder.rule()
            .as_unique_rule()
            .with_target("main", "data", "email")
            .build(),
            TestDataBuilder.rule()
            .as_range_rule(18, 120)
            .with_target("main", "data", "age")
            .build(),
            TestDataBuilder.rule()
            .as_length_rule(2, 50)
            .with_target("main", "data", "name")
            .build(),
        ]

        # Act
        validator = DataValidator(
            source_config=source_config,
            rules=cast(List[RuleSchema | Dict[str, Any]], rules),
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Assert
        assert len(validator.rules) == 4
        for rule in validator.rules:
            assert rule.target.entities[0].table == "customer_data"
            assert rule.target.entities[0].database == "main"  # SQLite default

        # Column names should be preserved
        expected_columns = [
            "customer_id",
            "email",
            "age",
            "name",
        ]
        actual_columns = [rule.target.entities[0].column for rule in validator.rules]
        assert actual_columns == expected_columns

    def test_database_connection_with_table_override(
        self, mock_configs: Dict[str, Any]
    ) -> None:
        """Test database connection with table override in parameters"""
        # Arrange
        source_config = (
            TestDataBuilder.connection()
            .with_type(ConnectionType.POSTGRESQL)
            .with_database("production")
            .with_parameters({"table": "override_table"})
            .build()
        )

        rule = (
            TestDataBuilder.rule()
            .as_not_null_rule()
            .with_target("original_db", "original_table", "id")
            .build()
        )

        # Act
        validator = DataValidator(
            source_config=source_config,
            rules=cast(List[RuleSchema | Dict[str, Any]], [rule]),
            core_config=mock_configs["core_config"],
            cli_config=mock_configs["cli_config"],
        )

        # Assert
        updated_rule = validator.rules[0]
        assert updated_rule.target.entities[0].database == "production"
        assert updated_rule.target.entities[0].table == "override_table"
