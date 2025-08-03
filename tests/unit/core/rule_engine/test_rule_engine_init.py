"""
🧙‍♂️ Modern Rule Engine Initialization Testing - Testing Ghost's Complete Init Coverage

This modernized test file demonstrates comprehensive rule engine initialization testing with:
1. Builder Pattern - Eliminates 450+ lines of duplicate Mock code
2. Contract Testing - Ensures schema accuracy
3. Property-based Testing - Edge case coverage with random data
4. Comprehensive Init Scenarios - All initialization modes covered
5. Boundary Testing - Engine limits and edge cases

As the Testing Ghost, I ensure every initialization path is bulletproof! 👻
"""

import asyncio
import uuid
from typing import Any, Dict, List, Set, Tuple
from unittest.mock import AsyncMock, Mock, patch

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

# Import core components
from core.engine.rule_engine import RuleEngine, RuleGroup
from shared.enums import (
    ConnectionType,
    RuleAction,
    RuleCategory,
    RuleType,
    SeverityLevel,
)
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.rule_schema import RuleSchema

# Import testing infrastructure
from tests.shared.builders.test_builders import TestDataBuilder
from tests.shared.contracts.test_contracts import MockContract


# 🧙‍♂️ Property-based testing strategies
@st.composite
def database_name_strategy(draw: st.DrawFn) -> str:
    """Generate valid database names"""
    return draw(
        st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(
                whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_"
            ),
        )
    )


@st.composite
def table_name_strategy(draw: st.DrawFn) -> str:
    """Generate valid table names"""
    return draw(
        st.text(
            min_size=1,
            max_size=30,
            alphabet=st.characters(
                whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_"
            ),
        )
    )


class TestRuleEngineInitialization:
    """🧙‍♂️ Modern Rule Engine Initialization Testing with Zero Redundancy"""

    # ====== Basic Initialization Tests ======

    def test_single_rule_initialization(self) -> None:
        """Test engine initialization with single rule"""
        # Using Builder Pattern - NO fixture duplication!
        builder = TestDataBuilder()
        connection = builder.connection().build()
        rule = builder.rule().as_not_null_rule().with_name("single_rule").build()

        engine = RuleEngine(connection=connection)

        # New flow: group rules at runtime
        rule_groups = engine._group_rules([rule])

        # Critical invariants that must hold
        assert len(rule_groups) == 1
        assert engine.connection == connection

        # Verify rule group structure
        group_key = (
            f"{rule.target.entities[0].database}.{rule.target.entities[0].table}"
        )
        assert group_key in rule_groups
        assert len(rule_groups[group_key].rules) == 1
        assert rule_groups[group_key].rules[0] == rule

    def test_multi_rule_same_table_initialization(self) -> None:
        """Test multiple rules for same table group correctly"""
        builder = TestDataBuilder()
        connection = builder.connection().build()

        # Create rules for same table
        rules = [
            builder.rule().as_not_null_rule().with_name("rule1").build(),
            builder.rule().as_unique_rule().with_name("rule2").build(),
            builder.rule().as_range_rule(0, 100).with_name("rule3").build(),
        ]

        engine = RuleEngine(connection=connection)

        rule_groups = engine._group_rules(rules)

        # Verify proper grouping
        assert len(rule_groups) == 1

        group_key = f"{rules[0].target.entities[0].database}.{rules[0].target.entities[0].table}"
        assert len(rule_groups[group_key].rules) == 3

    def test_multi_table_rule_grouping(self) -> None:
        """Test rules across multiple tables group correctly"""
        builder = TestDataBuilder()
        connection = builder.connection().build()

        # Create rules for different tables
        rules = [
            builder.rule()
            .with_target("test_db", "table1", "col1")
            .as_not_null_rule()
            .build(),
            builder.rule()
            .with_target("test_db", "table2", "col1")
            .as_unique_rule()
            .build(),
            builder.rule()
            .with_target("test_db", "table3", "col1")
            .as_range_rule(0, 100)
            .build(),
        ]

        engine = RuleEngine(connection=connection)

        rule_groups = engine._group_rules(rules)

        # Verify table-based grouping
        assert len(rule_groups) == 3  # 3 different tables

        # Verify each table has its own group
        expected_keys = ["test_db.table1", "test_db.table2", "test_db.table3"]
        for key in expected_keys:
            assert key in rule_groups
            assert len(rule_groups[key].rules) == 1

    def test_multi_database_rule_grouping(self) -> None:
        """Test rules across multiple databases group correctly"""
        builder = TestDataBuilder()
        connection = builder.connection().build()

        # Create rules for different databases
        rules = [
            builder.rule()
            .with_target("db1", "table1", "col1")
            .as_not_null_rule()
            .build(),
            builder.rule()
            .with_target("db2", "table1", "col1")
            .as_unique_rule()
            .build(),
            builder.rule()
            .with_target("db1", "table2", "col1")
            .as_range_rule(0, 100)
            .build(),
        ]

        engine = RuleEngine(connection=connection)

        rule_groups = engine._group_rules(rules)

        # Verify database.table-based grouping
        assert len(rule_groups) == 3  # 3 different db.table combinations

        # Verify each db.table has its own group
        expected_keys = ["db1.table1", "db2.table1", "db1.table2"]
        for key in expected_keys:
            assert key in rule_groups
            assert len(rule_groups[key].rules) == 1

    # ====== Edge Cases and Boundary Tests ======

    def test_empty_rules_initialization(self) -> None:
        """Test engine initialization with empty rule list"""
        builder = TestDataBuilder()
        connection = builder.connection().build()

        engine = RuleEngine(connection=connection)

        rule_groups = engine._group_rules([])

        # Empty rules should create empty groups dict
        assert len(rule_groups) == 0
        assert engine.connection == connection

    def test_rule_type_diversity_in_groups(self) -> None:
        """Test groups can contain diverse rule types"""
        builder = TestDataBuilder()
        connection = builder.connection().build()

        # Create all different rule types for same table
        rules = [
            builder.rule().as_not_null_rule().build(),
            builder.rule().as_unique_rule().build(),
            builder.rule().as_range_rule(0, 100).build(),
            builder.rule().as_enum_rule(["A", "B", "C"]).build(),
            builder.rule().as_regex_rule(r"^test.*").build(),
            builder.rule().as_length_rule(1, 50).build(),
            builder.rule().as_date_format_rule("%Y-%m-%d").build(),
        ]

        engine = RuleEngine(connection=connection)

        rule_groups = engine._group_rules(rules)

        # All rules should be in same group (same table)
        assert len(rule_groups) == 1

        group_key = f"{rules[0].target.entities[0].database}.{rules[0].target.entities[0].table}"
        group_rules = rule_groups[group_key].rules

        # Verify rule type diversity
        rule_types = [rule.type for rule in group_rules]
        expected_types = [
            RuleType.NOT_NULL,
            RuleType.UNIQUE,
            RuleType.RANGE,
            RuleType.ENUM,
            RuleType.REGEX,
            RuleType.LENGTH,
            RuleType.DATE_FORMAT,
        ]

        for expected_type in expected_types:
            assert expected_type in rule_types

    # ====== Property-based Testing ======

    @given(
        num_rules=st.integers(min_value=1, max_value=20),
        num_tables=st.integers(min_value=1, max_value=5),
    )
    def test_rule_grouping_mathematical_properties(
        self, num_rules: int, num_tables: int
    ) -> None:
        """🎯 Property test: Rule grouping should satisfy mathematical properties"""
        assume(num_rules >= num_tables)  # Can't have more tables than rules

        builder = TestDataBuilder()
        connection = builder.connection().build()

        # Create rules distributed across tables
        rules = []
        for i in range(num_rules):
            table_idx = i % num_tables
            rule = (
                builder.rule()
                .with_target("test_db", f"table_{table_idx}", f"col_{i}")
                .as_not_null_rule()
                .with_name(f"rule_{i}")
                .build()
            )
            rules.append(rule)

        engine = RuleEngine(connection=connection)

        rule_groups = engine._group_rules(rules)

        # Verify number of groups equals num_tables
        assert len(rule_groups) == num_tables

    @given(
        database_names=st.lists(
            database_name_strategy(), min_size=1, max_size=3, unique=True
        ),
        table_names=st.lists(
            table_name_strategy(), min_size=1, max_size=3, unique=True
        ),
    )
    def test_database_table_naming_properties(
        self, database_names: List[str], table_names: List[str]
    ) -> None:
        """🎯 Property test: Engine should handle arbitrary valid database/table names"""
        builder = TestDataBuilder()
        connection = builder.connection().build()

        # Create rules for all db/table combinations
        rules = []
        for db_name in database_names:
            for table_name in table_names:
                rule = (
                    builder.rule()
                    .with_target(db_name, table_name, "test_column")
                    .as_not_null_rule()
                    .build()
                )
                rules.append(rule)

        engine = RuleEngine(connection=connection)

        rule_groups = engine._group_rules(rules)

        # Verify correct number of groups
        expected_groups = len(database_names) * len(table_names)
        assert len(rule_groups) == expected_groups

        # Verify group keys are correct
        for db_name in database_names:
            for table_name in table_names:
                expected_key = f"{db_name}.{table_name}"
                assert expected_key in rule_groups

    # ====== Contract Testing ======

    def test_rule_group_contract_compliance(self) -> None:
        """🔄 Contract test: RuleGroup objects should be properly formed"""
        builder = TestDataBuilder()
        connection = builder.connection().build()
        rule = builder.rule().as_not_null_rule().build()

        engine = RuleEngine(connection=connection)

        rule_groups = engine._group_rules([rule])

        # Verify RuleGroup contract compliance
        for group_key, rule_group in rule_groups.items():
            # RuleGroup must have these attributes
            assert hasattr(rule_group, "rules")
            assert hasattr(rule_group, "database")
            assert hasattr(rule_group, "table_name")  # 🧙‍♂️ Fixed: table_name, not table

            # Verify group attributes match key
            db_name, table_name = group_key.split(".", 1)
            assert rule_group.database == db_name
            assert (
                rule_group.table_name == table_name
            )  # 🧙‍♂️ Fixed: table_name, not table

            # Verify rules are properly typed
            assert isinstance(rule_group.rules, list)
            for rule in rule_group.rules:
                assert isinstance(rule, RuleSchema)

    def test_engine_state_invariants(self) -> None:
        """🎯 Test critical engine state invariants"""
        builder = TestDataBuilder()
        connection = builder.connection().build()
        rules = [
            builder.rule()
            .with_target("db1", "table1", "col1")
            .as_not_null_rule()
            .build(),
            builder.rule()
            .with_target("db1", "table1", "col2")
            .as_unique_rule()
            .build(),
            builder.rule()
            .with_target("db2", "table1", "col1")
            .as_range_rule(0, 100)
            .build(),
        ]

        engine = RuleEngine(connection=connection)

        rule_groups = engine._group_rules(rules)

        # State invariants that must always hold
        assert rule_groups is not None
        assert engine.connection is not None

        # Integrity constraints
        assert len(rules) > 0
        assert len(rule_groups) > 0

        # Consistency constraints
        all_group_rules = []
        for group in rule_groups.values():
            all_group_rules.extend(group.rules)

        # All original rules should be in some group
        assert len(all_group_rules) == len(rules)
        for original_rule in rules:
            assert original_rule in all_group_rules

    def test_large_rule_set_initialization(self) -> None:
        """Test engine can handle large rule sets efficiently"""
        builder = TestDataBuilder()
        connection = builder.connection().build()

        # Create 100 rules across 10 tables
        rules = []
        for i in range(100):
            table_idx = i % 10
            rule = (
                builder.rule()
                .with_target("large_db", f"table_{table_idx}", f"col_{i}")
                .as_not_null_rule()
                .with_name(f"rule_{i}")
                .build()
            )
            rules.append(rule)

        # This should complete without performance issues
        engine = RuleEngine(connection=connection)

        rule_groups = engine._group_rules(rules)

        # Verify correct grouping
        assert len(rule_groups) == 10  # 10 unique tables

        # Verify even distribution
        for group in rule_groups.values():
            assert len(group.rules) == 10  # 100 rules / 10 tables = 10 per table

    @given(
        st.lists(
            st.builds(
                lambda db, tbl, col, is_missing_table, missing_cols: (
                    db,
                    f"Nonexistent_{tbl}" if is_missing_table else tbl,
                    col,
                    missing_cols,
                ),
                db=st.sampled_from(["db1", "db2", "db3"]),
                tbl=st.text(
                    min_size=3,
                    max_size=10,
                    alphabet=st.characters(
                        whitelist_categories=("Ll", "Lu", "Nd"),
                        whitelist_characters="_",
                    ),
                ),
                col=st.text(
                    min_size=3,
                    max_size=10,
                    alphabet=st.characters(
                        whitelist_categories=("Ll", "Lu", "Nd"),
                        whitelist_characters="_",
                    ),
                ),
                is_missing_table=st.booleans(),
                missing_cols=st.lists(
                    st.text(
                        min_size=3,
                        max_size=10,
                        alphabet=st.characters(
                            whitelist_categories=("Ll", "Lu", "Nd"),
                            whitelist_characters="_",
                        ),
                    ).map(lambda c: f"Nonexistent_{c}"),
                    min_size=0,
                    max_size=3,
                ),
            ),
            min_size=1,
            max_size=10,
        ).filter(
            lambda rules: 0
            <= sum(1 for r in rules if r[1].startswith("Nonexistent_"))
            <= 3
        )
    )
    def test_prevalidation_strategy_property(
        self, rule_specs: List[Tuple[str, str, str, List[str]]]
    ) -> None:
        """Property-based test: prevalidation with random rules, 0-3 missing tables, any missing columns"""
        builder = TestDataBuilder()
        rules = []
        expected_tables: Dict[str, bool] = {}
        expected_columns: Dict[str, bool] = {}

        # First pass: collect all missing columns per table
        missing_columns_per_table: Dict[str, Set[str]] = {}
        for db, tbl, col, missing_cols in rule_specs:
            entity_key = f"{db}.{tbl}"
            if entity_key not in missing_columns_per_table:
                missing_columns_per_table[entity_key] = set()
            # Add missing columns (remove Nonexistent_ prefix)
            for mc in missing_cols:
                missing_columns_per_table[entity_key].add(
                    mc.replace("Nonexistent_", "")
                )

        # Second pass: build rules and expectations consistently
        for db, tbl, col, missing_cols in rule_specs:
            rule = builder.rule().with_target(db, tbl, col).as_not_null_rule().build()
            rules.append(rule)
            entity_key = f"{db}.{tbl}"
            expected_tables[entity_key] = not tbl.startswith("Nonexistent_")

            # The main column - check against all missing columns for this table
            col_key = f"{entity_key}.{col}"
            if tbl.startswith("Nonexistent_"):
                expected_columns[col_key] = False
            else:
                # Column exists if it's not in the missing set for this table
                expected_columns[col_key] = col not in missing_columns_per_table.get(
                    entity_key, set()
                )

        mock_engine = AsyncMock()

        def table_exists_side_effect(
            table: str,
            database: str,
            entity_name: str,
            resource_type: str,
            rule_id: str,
        ) -> bool:
            return not table.startswith("Nonexistent_")

        def get_column_list_side_effect(
            table: str,
            database: str,
            entity_name: str,
            resource_type: str,
            rule_id: str,
        ) -> List[Dict[str, str]]:
            # Get all missing columns for this table
            entity_key = f"{database}.{table}"
            missing_for_table = missing_columns_per_table.get(entity_key, set())

            # Return all columns referenced by rules for this table,
            # except missing ones
            cols = set()
            for db2, tbl2, col2, _ in rule_specs:
                if db2 == database and tbl2 == table:
                    if col2 not in missing_for_table:
                        cols.add(col2)
            return [{"name": c} for c in cols]

        from core.engine.prevalidation import DatabasePrevalidator
        from shared.database.query_executor import QueryExecutor

        mock_query_executor = AsyncMock(spec=QueryExecutor)
        mock_query_executor.table_exists.side_effect = table_exists_side_effect
        mock_query_executor.get_column_list.side_effect = get_column_list_side_effect
        with patch(
            "core.engine.prevalidation.QueryExecutor", return_value=mock_query_executor
        ):
            prevalidator = DatabasePrevalidator(mock_engine)
            result = asyncio.get_event_loop().run_until_complete(
                prevalidator.validate(rules)
            )
            # Check table existence results
            for table_key, should_exist in expected_tables.items():
                assert result["tables"].get(table_key) == should_exist

            # Check column existence results
            for col_key, should_exist in expected_columns.items():
                # If the column key is missing in the result, treat it as False (non-existent)
                assert result["columns"].get(col_key, False) == should_exist
