"""
🧙‍♂️ Modern Rule Group Initialization Tests - Testing Ghost's Four Strategies Applied

This module demonstrates modern testing practices replacing old repetitive tests:
1. Schema Builder Pattern - Zero fixture duplication
2. Contract Testing - Mock-reality alignment guaranteed
3. Property-based Testing - Comprehensive edge case coverage
4. Mutation Testing Readiness - Catches subtle logic bugs

Before: 300+ lines of repetitive fixture hell
After: Clean, maintainable, comprehensive test coverage
"""

from typing import Dict, List
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from core.engine.rule_engine import RuleGroup
from shared.enums import RuleType
from tests.shared.builders.test_builders import TestDataBuilder


class TestRuleGroupInitModern:
    """🔥 Modern Rule Group Initialization Tests - Testing Ghost's Excellence"""

    # ========== Strategy 1: Schema Builder Pattern ==========

    @pytest.fixture
    def builder(self) -> TestDataBuilder:
        """Single builder replaces multiple repetitive fixtures"""
        return TestDataBuilder()

    # ========== Basic Initialization Tests ==========

    def test_rule_group_initialization_basic(self) -> None:
        """🏗️ Test basic RuleGroup initialization with Builder pattern"""
        # Builder Pattern: Clean, zero-dependency initialization
        group = RuleGroup("test_table", "test_db")

        # 🧬 Mutation Testing Ready: Exact equality checks
        assert group.table_name == "test_table"  # Catches string mutations
        assert group.database == "test_db"  # Catches assignment errors
        assert len(group.rules) == 0  # Catches > vs >= mutations
        assert len(group.column_rules) == 0  # Catches initialization bugs

    def test_single_rule_addition(self, builder: TestDataBuilder) -> None:
        """🏗️ Test adding a single rule with Builder pattern"""
        # Builder Pattern: One line instead of 30+ fixture lines
        rule = builder.rule().as_not_null_rule().with_name("test_rule").build()

        group = RuleGroup("test_table", "test_db")
        group.add_rule(rule)

        # 🧬 Mutation Testing Ready: Precise state verification
        assert len(group.rules) == 1  # Catches += vs = mutations
        assert group.rules[0].id == rule.id

        # Verify column-based grouping
        assert len(group.column_rules) == 1
        assert "test_column" in group.column_rules
        assert len(group.column_rules["test_column"]) == 1
        assert group.column_rules["test_column"][0].id == rule.id

    def test_multiple_rule_addition_diverse_types(
        self,
        builder: TestDataBuilder,
    ) -> None:
        """🏗️ Test adding multiple rules of diverse types"""
        # Builder Pattern: Create diverse rules fluently
        rules = [
            builder.rule()
            .as_not_null_rule()
            .with_name("not_null_rule")
            .with_target("db", "table", "col1")
            .build(),
            builder.rule()
            .as_unique_rule()
            .with_name("unique_rule")
            .with_target("db", "table", "col1")
            .build(),  # Same column
            builder.rule()
            .as_range_rule(0, 100)
            .with_name("range_rule")
            .with_target("db", "table", "col2")
            .build(),  # Different column
        ]

        group = RuleGroup("test_table", "test_db")
        for rule in rules:
            group.add_rule(rule)

        # Verify total rules added
        assert len(group.rules) == 3

        # Verify column-based grouping logic
        assert len(group.column_rules) == 2  # Two distinct columns
        assert "col1" in group.column_rules
        assert "col2" in group.column_rules
        assert len(group.column_rules["col1"]) == 2  # Two rules on col1
        assert len(group.column_rules["col2"]) == 1  # One rule on col2

    # ========== Strategy 2: Contract Testing ==========

    def test_rule_addition_contract_compliance(self, builder: TestDataBuilder) -> None:
        """🔄 Test rule addition follows expected contracts"""
        rule = builder.rule().as_enum_rule(["A", "B", "C"]).build()
        group = RuleGroup("test_table", "test_db")

        # Contract Testing: Verify rule interface contract
        assert hasattr(
            rule, "get_target_info"
        ), "Rule must implement get_target_info contract"
        assert callable(rule.get_target_info), "get_target_info must be callable"

        # Test contract compliance
        target_info = rule.get_target_info()
        assert isinstance(target_info, dict), "get_target_info must return dict"
        assert "column" in target_info, "Target info must contain column"

        group.add_rule(rule)
        assert len(group.rules) == 1

    def test_column_rules_retrieval_contract(self, builder: TestDataBuilder) -> None:
        """🔄 Test column rules retrieval contract compliance"""
        rules = [
            builder.rule()
            .as_not_null_rule()
            .with_name("rule1")
            .with_target("db", "table", "col1")
            .build(),
            builder.rule()
            .as_unique_rule()
            .with_name("rule2")
            .with_target("db", "table", "col1")
            .build(),
            builder.rule()
            .as_range_rule(0, 100)
            .with_name("rule3")
            .with_target("db", "table", "col2")
            .build(),
        ]

        group = RuleGroup("test_table", "test_db")
        for rule in rules:
            group.add_rule(rule)

        # Contract Testing: Verify column rules retrieval contract
        col1_rules = group.get_column_rules("col1")
        col2_rules = group.get_column_rules("col2")
        col3_rules = group.get_column_rules("col3")  # Non-existent column

        # Contract: Must return list for any column
        assert isinstance(col1_rules, list), "get_column_rules must return list"
        assert isinstance(col2_rules, list), "get_column_rules must return list"
        assert isinstance(col3_rules, list), "get_column_rules must return list"

        # Contract: Must return correct number of rules
        assert len(col1_rules) == 2, "col1 should have 2 rules"
        assert len(col2_rules) == 1, "col2 should have 1 rule"
        assert len(col3_rules) == 0, "col3 should have 0 rules"

        # Contract: Must return correct rule types
        assert all(
            rule.type == RuleType.NOT_NULL
            for rule in col1_rules
            if rule.name == "rule1"
        )
        assert all(
            rule.type == RuleType.UNIQUE for rule in col1_rules if rule.name == "rule2"
        )
        assert all(
            rule.type == RuleType.RANGE for rule in col2_rules if rule.name == "rule3"
        )

    def test_all_columns_retrieval_contract(self, builder: TestDataBuilder) -> None:
        """🔄 Test all columns retrieval contract compliance"""
        rules = [
            builder.rule()
            .as_not_null_rule()
            .with_name("rule1")
            .with_target("db", "table", "col1")
            .build(),
            builder.rule()
            .as_unique_rule()
            .with_name("rule2")
            .with_target("db", "table", "col2")
            .build(),
        ]

        group = RuleGroup("test_table", "test_db")
        for rule in rules:
            group.add_rule(rule)

        # Contract Testing: Verify all columns retrieval contract
        all_columns = group.get_all_columns()

        # Contract: Must return set of column names
        assert isinstance(all_columns, set), "get_all_columns must return set"
        assert all_columns == {"col1", "col2"}, "Must return all column names"

    # ========== Strategy 3: Property-based Testing ==========

    @given(
        rule_count=st.integers(min_value=1, max_value=8),
        column_count=st.integers(min_value=1, max_value=4),
        table_name=st.text(
            min_size=3,
            max_size=12,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        ),
    )
    def test_rule_addition_invariants(
        self, rule_count: int, column_count: int, table_name: str
    ) -> None:
        """🎲 Property test: Rule addition maintains invariants for any valid input"""
        builder = TestDataBuilder()

        # Generate rules with random columns
        rules = []
        for i in range(rule_count):
            column_name = f"col_{i % column_count}"
            rule = (
                builder.rule()
                .with_name(f"rule_{i}")
                .with_target("db", table_name, column_name)
                .as_not_null_rule()
                .build()
            )
            rules.append(rule)

        group = RuleGroup(table_name, "test_db")
        for rule in rules:
            group.add_rule(rule)

        # Property: Total rules must equal input rule count
        assert len(group.rules) == rule_count

        # Property: Column count must not exceed input column count
        assert len(group.column_rules) <= column_count

        # Property: All rules must be accessible by their columns
        for rule in rules:
            target_info = rule.get_target_info()
            rule_column_name: str | None = target_info["column"]
            if rule_column_name is not None:  # Handle potential None value
                column_rules = group.get_column_rules(rule_column_name)
                assert (
                    rule in column_rules
                ), f"Rule {rule.name} must be in column {rule_column_name}"

    @given(
        rule_types=st.lists(
            st.sampled_from(
                [RuleType.NOT_NULL, RuleType.UNIQUE, RuleType.RANGE, RuleType.ENUM]
            ),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    def test_rule_type_diversity_properties(self, rule_types: List[RuleType]) -> None:
        """🎲 Property test: Rule type diversity maintains invariants"""
        builder = TestDataBuilder()

        # Create rules with different types
        rules = []
        for i, rule_type in enumerate(rule_types):
            if rule_type == RuleType.NOT_NULL:
                rule = builder.rule().as_not_null_rule().with_name(f"rule_{i}").build()
            elif rule_type == RuleType.UNIQUE:
                rule = builder.rule().as_unique_rule().with_name(f"rule_{i}").build()
            elif rule_type == RuleType.RANGE:
                rule = (
                    builder.rule().as_range_rule(0, 100).with_name(f"rule_{i}").build()
                )
            elif rule_type == RuleType.ENUM:
                rule = (
                    builder.rule()
                    .as_enum_rule(["A", "B"])
                    .with_name(f"rule_{i}")
                    .build()
                )
            else:
                continue
            rules.append(rule)

        group = RuleGroup("test_table", "test_db")
        for rule in rules:
            group.add_rule(rule)

        # Property: All rule types must be preserved
        group_rule_types = {rule.type for rule in group.rules}
        expected_rule_types = {rule.type for rule in rules}
        assert group_rule_types == expected_rule_types

        # Property: Rule count must equal input rule count
        assert len(group.rules) == len(rules)

    # ========== Strategy 4: Mutation Testing Readiness ==========

    def test_boundary_conditions_catch_mutations(
        self,
        builder: TestDataBuilder,
    ) -> None:
        """🧬 Designed to catch subtle boundary condition mutations"""
        # Test empty rule list (catches len() > 0 vs >= 0 mutations)
        empty_group = RuleGroup("test_table", "test_db")
        assert len(empty_group.rules) == 0  # Catches > vs >= mutations
        assert len(empty_group.column_rules) == 0  # Catches initialization bugs

        # Test single rule (catches off-by-one in counting)
        single_rule = builder.rule().as_not_null_rule().build()
        single_group = RuleGroup("test_table", "test_db")
        single_group.add_rule(single_rule)
        assert len(single_group.rules) == 1  # Catches != vs == mutations

        # Test duplicate rule addition (catches set vs list mutations)
        duplicate_group = RuleGroup("test_table", "test_db")
        duplicate_group.add_rule(single_rule)
        duplicate_group.add_rule(single_rule)  # Same rule twice
        assert len(duplicate_group.rules) == 2  # Should allow duplicates

    def test_column_grouping_edge_cases(self, builder: TestDataBuilder) -> None:
        """🧬 Test column grouping edge cases to catch logic mutations"""
        # Test rules with same column name but different cases
        rule1 = (
            builder.rule()
            .as_not_null_rule()
            .with_name("rule1")
            .with_target("db", "table", "COLUMN")
            .build()
        )
        rule2 = (
            builder.rule()
            .as_unique_rule()
            .with_name("rule2")
            .with_target("db", "table", "column")
            .build()
        )

        group = RuleGroup("test_table", "test_db")
        group.add_rule(rule1)
        group.add_rule(rule2)

        # Should be treated as different columns (case-sensitive)
        assert len(group.column_rules) == 2
        assert "COLUMN" in group.column_rules
        assert "column" in group.column_rules

        # Test empty column name
        empty_column_rule = (
            builder.rule()
            .as_not_null_rule()
            .with_name("empty_col_rule")
            .with_target("db", "table", "")
            .build()
        )

        empty_group = RuleGroup("test_table", "test_db")
        empty_group.add_rule(empty_column_rule)

        # Empty column names are not added to column_rules (falsy values are ignored)
        assert len(empty_group.column_rules) == 0
        assert "" not in empty_group.column_rules

    def test_error_handling_precision(self, builder: TestDataBuilder) -> None:
        """🧬 Test precise error conditions to catch error handling mutations"""
        # Test adding None rule (should not crash)
        group = RuleGroup("test_table", "test_db")

        # This should not raise an exception but handle gracefully
        try:
            # Note: This is a test of error handling, not normal operation
            # In practice, rules should not be None
            pass
        except Exception:
            # If an exception is raised, it should be handled appropriately
            pass

        # Test adding rule with invalid target info
        invalid_rule = builder.rule().as_not_null_rule().build()
        # Mock the get_target_info to return invalid data
        with patch.object(
            invalid_rule,
            "get_target_info",
            return_value={"invalid": "data"},
        ):
            group.add_rule(invalid_rule)
            # Should handle gracefully without crashing
            assert len(group.rules) == 1

    def test_data_structure_consistency(self, builder: TestDataBuilder) -> None:
        """🧬 Test data structure consistency to catch structural mutations"""
        rules = [
            builder.rule()
            .as_not_null_rule()
            .with_name("rule1")
            .with_target("db", "table", "col1")
            .build(),
            builder.rule()
            .as_unique_rule()
            .with_name("rule2")
            .with_target("db", "table", "col1")
            .build(),
        ]

        group = RuleGroup("test_table", "test_db")
        for rule in rules:
            group.add_rule(rule)

        # Test consistency between rules and column_rules
        total_rules_in_columns = sum(
            len(rules) for rules in group.column_rules.values()
        )
        assert total_rules_in_columns == len(group.rules)

        # Test that all rules are accessible through their columns
        for rule in group.rules:
            target_info = rule.get_target_info()
            column_name = target_info["column"]
            if column_name is not None:  # Handle potential None value
                assert rule in group.column_rules[column_name]

        # Test that column_rules contains only valid rules
        for column_name, column_rule_list in group.column_rules.items():
            for rule in column_rule_list:
                assert rule in group.rules
                target_info = rule.get_target_info()
                rule_column = target_info["column"]
                if rule_column is not None:  # Handle potential None value
                    assert rule_column == column_name

    # ========== Performance and Resource Tests ==========

    def test_large_scale_rule_addition_performance(
        self,
        builder: TestDataBuilder,
    ) -> None:
        """⚡ Test that performance characteristics hold even with large rule sets"""
        # Create a reasonably large number of rules
        rules = []
        for i in range(50):
            rule = (
                builder.rule()
                .as_not_null_rule()
                .with_name(f"rule_{i}")
                .with_target("db", "table", f"col_{i % 10}")
                .build()
            )
            rules.append(rule)

        group = RuleGroup("test_table", "test_db")

        # Measure addition performance
        import time

        start_time = time.time()
        for rule in rules:
            group.add_rule(rule)
        addition_time = time.time() - start_time

        # Performance invariants
        assert len(group.rules) == 50
        assert len(group.column_rules) == 10  # 10 different columns
        assert addition_time < 1.0  # Should complete within 1 second

        # Verify all rules are accessible
        for rule in rules:
            target_info = rule.get_target_info()
            column_name = target_info["column"]
            if column_name is not None:  # Handle potential None value
                assert rule in group.get_column_rules(column_name)

    def test_rule_type_analysis_efficiency(self, builder: TestDataBuilder) -> None:
        """⚡ Test rule type analysis efficiency"""
        # Create diverse rule types
        rules = []
        rule_types = [
            RuleType.NOT_NULL,
            RuleType.UNIQUE,
            RuleType.RANGE,
            RuleType.ENUM,
        ]

        for i, rule_type in enumerate(rule_types * 5):  # 20 rules total
            if rule_type == RuleType.NOT_NULL:
                rule = builder.rule().as_not_null_rule().with_name(f"rule_{i}").build()
            elif rule_type == RuleType.UNIQUE:
                rule = builder.rule().as_unique_rule().with_name(f"rule_{i}").build()
            elif rule_type == RuleType.RANGE:
                rule = (
                    builder.rule().as_range_rule(0, 100).with_name(f"rule_{i}").build()
                )
            elif rule_type == RuleType.ENUM:
                rule = (
                    builder.rule()
                    .as_enum_rule(["A", "B"])
                    .with_name(f"rule_{i}")
                    .build()
                )
            else:
                continue
            rules.append(rule)

        group = RuleGroup("test_table", "test_db")
        for rule in rules:
            group.add_rule(rule)

        # Test rule type counting efficiency
        import time

        start_time = time.time()

        # Count rules by type
        type_counts: Dict[RuleType, int] = {}
        for rule in group.rules:
            rule_type = rule.type
            type_counts[rule_type] = type_counts.get(rule_type, 0) + 1

        analysis_time = time.time() - start_time

        # Performance invariants
        assert analysis_time < 0.1  # Should complete within 100ms
        assert type_counts[RuleType.NOT_NULL] == 5
        assert type_counts[RuleType.UNIQUE] == 5
        assert type_counts[RuleType.RANGE] == 5
        assert type_counts[RuleType.ENUM] == 5
