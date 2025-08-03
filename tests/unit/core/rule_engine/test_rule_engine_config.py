"""
Tests for rule engine configuration integration.

Verifies that the rule engine correctly uses the configuration system.
"""

import uuid
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import CoreConfig
from core.engine.rule_engine import RuleEngine
from shared.enums import RuleAction, RuleCategory, RuleType, SeverityLevel
from shared.schema import ConnectionSchema, RuleSchema
from shared.schema.base import RuleTarget, TargetEntity
from tests.shared.builders.test_builders import TestDataBuilder


@pytest.fixture
def builder() -> TestDataBuilder:
    """Schema builder for tests"""
    return TestDataBuilder()


@pytest.fixture
def connection(builder: TestDataBuilder) -> ConnectionSchema:
    """Test connection"""
    from shared.enums.connection_types import ConnectionType

    return builder.connection().with_type(ConnectionType.MYSQL).build()


@pytest.fixture
def rules(builder: TestDataBuilder) -> List[RuleSchema]:
    """Test rules"""
    rules = []
    for i in range(5):
        rule = (
            builder.rule()
            .as_not_null_rule()
            .with_target("test_db", "test_table", f"column_{i}")
            .build()
        )
        rules.append(rule)
    return rules


class TestRuleEngineConfig:
    """Tests for rule engine configuration integration"""

    def test_rule_engine_uses_config(
        self, rules: List[RuleSchema], connection: ConnectionSchema
    ) -> None:
        """Test that RuleEngine uses the configuration system"""
        # Create a mock config
        mock_config = CoreConfig(
            merge_execution_enabled=True,
            table_size_threshold=5000,
            rule_count_threshold=3,
            max_rules_per_merge=8,
            independent_rule_types=["UNIQUE", "CUSTOM_SQL"],
        )

        # Mock the get_core_config function to return our mock config
        with patch("core.config.get_core_config", return_value=mock_config):
            # Initialize rule engine
            engine = RuleEngine(connection, core_config=mock_config)

            # Verify that rule engine uses the config values
            assert engine.merge_execution_enabled is True
            assert engine.table_size_threshold == 5000
            assert engine.rule_count_threshold == 3
            assert engine.max_rules_per_merge == 8
            assert "UNIQUE" in engine.independent_rule_types
            assert "CUSTOM_SQL" in engine.independent_rule_types

    def test_rule_engine_with_disabled_merge(
        self, rules: List[RuleSchema], connection: ConnectionSchema
    ) -> None:
        """Test rule engine behavior when merge is disabled in config"""
        # Create a mock config with merge disabled
        mock_config = CoreConfig(
            merge_execution_enabled=False,
            table_size_threshold=5000,
            rule_count_threshold=3,
        )

        # Mock the get_core_config function
        with patch("core.config.get_core_config", return_value=mock_config):
            # Initialize rule engine
            engine = RuleEngine(connection, core_config=mock_config)

            # Verify that merge is disabled
            assert engine.merge_execution_enabled is False

    def test_rule_engine_with_custom_independent_rule_types(
        self, builder: TestDataBuilder, connection: ConnectionSchema
    ) -> None:
        """Test rule engine with custom independent rule types configuration"""
        # Create rules of different types
        rules = [
            builder.rule()
            .with_type(RuleType.NOT_NULL)
            .with_target("test_db", "test_table", "column_1")
            .build(),
            builder.rule()
            .with_type(RuleType.UNIQUE)
            .with_target("test_db", "test_table", "column_2")
            .build(),
            builder.rule()
            .as_range_rule(min_val=0, max_val=100)
            .with_target("test_db", "test_table", "column_3")
            .build(),
        ]

        # Create a mock config with custom independent rule types
        mock_config = CoreConfig(
            merge_execution_enabled=True,
            independent_rule_types=["RANGE", "UNIQUE"],  # NOT_NULL can be merged
        )

        # Mock the get_core_config function
        with patch("core.config.get_core_config", return_value=mock_config):
            # Initialize rule engine
            engine = RuleEngine(connection, core_config=mock_config)

            # Verify that independent rule types are set correctly
            assert "RANGE" in engine.independent_rule_types
            assert "UNIQUE" in engine.independent_rule_types
            assert "NOT_NULL" not in engine.independent_rule_types

    @pytest.mark.asyncio
    async def test_rule_engine_execution_uses_config(
        self, rules: List[RuleSchema], connection: ConnectionSchema
    ) -> None:
        """Test that rule engine execution uses configuration values"""
        # Create a mock config
        mock_config = CoreConfig(
            merge_execution_enabled=True,
            table_size_threshold=100,  # Set low to ensure merge is enabled
            rule_count_threshold=2,  # Set low to ensure merge is enabled
            max_rules_per_merge=8,
        )

        # Mock the get_core_config function
        with patch("core.config.get_core_config", return_value=mock_config):
            # Initialize rule engine
            engine = RuleEngine(connection, core_config=mock_config)

            # Create a mock async engine
            mock_async_engine = AsyncMock()

            # Mock the _get_engine method directly on the engine instance
            with patch.object(engine, "_get_engine", return_value=mock_async_engine):
                # Mock the RuleGroup's methods
                mock_rule_group = MagicMock()
                mock_rule_group.execute = AsyncMock(
                    return_value=[{"status": "PASSED"}] * len(rules)
                )

                # Mock the _group_rules method to return our mock rule group
                mock_rule_groups = {"test_db.test_table": mock_rule_group}
                with patch.object(
                    engine, "_group_rules", return_value=mock_rule_groups
                ):
                    # Execute the rules
                    await engine.execute(rules=rules)

                    # Verify that the rule group's execute method was called
                    mock_rule_group.execute.assert_called_once()

    def test_rule_engine_logging_uses_config(
        self, rules: List[RuleSchema], connection: ConnectionSchema
    ) -> None:
        """Test that rule engine logging uses configuration"""
        # Create a mock config
        mock_config = CoreConfig(monitoring_enabled=True)

        # Mock the get_core_config function
        with patch("core.config.get_core_config", return_value=mock_config):
            # Mock the logger
            mock_logger = MagicMock()

            # Initialize rule engine with the mock logger
            with patch("core.engine.rule_engine.get_logger", return_value=mock_logger):
                engine = RuleEngine(connection, core_config=mock_config)

                # Verify that monitoring is enabled
                assert engine.logger is mock_logger
