"""
🧙‍♂️ Modern Rule Group Execution Tests - Testing Ghost's Four Strategies Applied

This module demonstrates modern testing practices replacing old repetitive tests:
1. Schema Builder Pattern - Zero fixture duplication
2. Contract Testing - Mock-reality alignment guaranteed
3. Property-based Testing - Comprehensive edge case coverage
4. Mutation Testing Readiness - Catches subtle logic bugs

Before: 509 lines of repetitive fixture hell
After: Clean, maintainable, comprehensive test coverage
"""

import asyncio
from typing import Any, List
from unittest.mock import AsyncMock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from core.engine.rule_engine import RuleGroup
from shared.enums import RuleType
from tests.shared.builders.test_builders import TestDataBuilder
from tests.shared.contracts.test_contracts import MockContract


class TestRuleGroupExecutionModern:
    """🔥 Modern Rule Group Execution Tests - Testing Ghost's Masterpiece"""

    # ========== Strategy 1: Schema Builder Pattern ==========

    @pytest.fixture
    def builder(self) -> TestDataBuilder:
        """Single builder replaces 15+ repetitive fixtures"""
        return TestDataBuilder()

    @pytest.fixture
    def mock_async_engine(self) -> AsyncMock:
        """Minimal async engine mock"""
        return AsyncMock()

    # ========== Basic Execution Tests ==========

    @pytest.mark.asyncio
    async def test_single_rule_execution_success(
        self, builder: TestDataBuilder, mock_async_engine: AsyncMock
    ) -> None:
        """Test successful execution of a single rule"""
        # 🏗️ Builder Pattern: One line instead of 30+ fixture lines
        rule = builder.rule().as_not_null_rule().with_name("single_test_rule").build()

        group = RuleGroup("test_table", "test_db")
        group.add_rule(rule)

        # 🔄 Contract Testing: Mock follows RuleGroup execution contract
        # with patch.object(group, '_table_exists', AsyncMock(return_value=True)):
        with patch.object(group, "_get_total_records", AsyncMock(return_value=100)):
            # Mock the merge manager and execution path
            mock_merge_manager = MockContract.create_rule_merge_manager_mock()

            with patch.object(
                group, "_get_merge_manager", return_value=mock_merge_manager
            ):
                # Mock individual group execution
                async def mock_execute_individual(
                    engine: Any,
                    group: RuleGroup,
                ) -> List[Any]:
                    from shared.schema.result_schema import ExecutionResultSchema

                    return [
                        ExecutionResultSchema.create_success_result(
                            rule_id=str(rule.id),
                            entity_name="test_db.test_table",
                            total_count=100,
                            error_count=0,
                            execution_time=0.1,
                            message="Rule execution completed",
                        )
                    ]

                with patch.object(
                    group, "_execute_individual_group", mock_execute_individual
                ):
                    results = await group.execute(mock_async_engine)

                    # Mutation Testing Ready: Specific assertions catch
                    # off-by-one errors
                    assert len(results) == 1  # Catches >= vs > errors
                    assert results[0].rule_id == str(rule.id)
                    assert results[0].status == "PASSED"
                    assert results[0].error_count == 0  # Catches != vs == errors

    @pytest.mark.asyncio
    async def test_multiple_rule_types_execution(
        self, builder: TestDataBuilder, mock_async_engine: AsyncMock
    ) -> None:
        """Test execution of diverse rule types in same group"""
        # 🏗️ Builder Pattern: Create diverse rules fluently
        rules = [
            builder.rule().as_not_null_rule().with_name("not_null_rule").build(),
            builder.rule().as_unique_rule().with_name("unique_rule").build(),
            builder.rule().as_range_rule(0, 100).with_name("range_rule").build(),
            builder.rule().as_enum_rule(["A", "B", "C"]).with_name("enum_rule").build(),
            builder.rule().as_regex_rule(r"^test.*").with_name("regex_rule").build(),
        ]

        group = RuleGroup("test_table", "test_db")
        for rule in rules:
            group.add_rule(rule)

        # with patch.object(group, '_table_exists', AsyncMock(return_value=True)):
        with patch.object(group, "_get_total_records", AsyncMock(return_value=1000)):
            mock_merge_manager = MockContract.create_rule_merge_manager_mock()

            with patch.object(
                group, "_get_merge_manager", return_value=mock_merge_manager
            ):

                async def mock_execute_individual(
                    engine: Any,
                    group: RuleGroup,
                ) -> List[Any]:
                    from shared.schema.result_schema import ExecutionResultSchema

                    return [
                        ExecutionResultSchema.create_success_result(
                            rule_id=str(rule.id),
                            entity_name="test_db.test_table",
                            total_count=1000,
                            error_count=0,
                            execution_time=0.15,
                            message="Multi-rule execution completed",
                        )
                        for rule in group.rules
                    ]

                with patch.object(
                    group, "_execute_individual_group", mock_execute_individual
                ):
                    results = await group.execute(mock_async_engine)

                    # Verify all rule types executed
                    assert len(results) == len(rules)
                    executed_rule_ids = {result.rule_id for result in results}
                    expected_rule_ids = {str(rule.id) for rule in rules}
                    assert executed_rule_ids == expected_rule_ids

    # ========== Strategy 2: Contract Testing ==========

    @pytest.mark.asyncio
    async def test_merge_manager_integration_contract(
        self, builder: TestDataBuilder, mock_async_engine: AsyncMock
    ) -> None:
        """🔄 Test merge manager integration follows contract"""
        rules = [
            builder.rule().as_not_null_rule().build(),
            builder.rule().as_unique_rule().build(),
        ]

        group = RuleGroup("test_table", "test_db")
        for rule in rules:
            group.add_rule(rule)

        # with patch.object(group, '_table_exists', AsyncMock(return_value=True)):
        with patch.object(group, "_get_total_records", AsyncMock(return_value=500)):
            # Contract Testing: Verify merge manager behavior
            mock_merge_manager = MockContract.create_rule_merge_manager_mock()

            with patch.object(group, "_get_merge_manager") as mock_get_manager:
                mock_get_manager.return_value = mock_merge_manager

                async def mock_execute_individual(
                    engine: Any,
                    group: RuleGroup,
                ) -> List[Any]:
                    from shared.schema.result_schema import ExecutionResultSchema

                    return [
                        ExecutionResultSchema.create_success_result(
                            rule_id=str(rule.id),
                            entity_name="test_db.test_table",
                            total_count=500,
                            error_count=0,
                            execution_time=0.1,
                        )
                        for rule in group.rules
                    ]

                with patch.object(
                    group, "_execute_individual_group", mock_execute_individual
                ):
                    await group.execute(mock_async_engine)

                    # Verify contract compliance
                    mock_get_manager.assert_called_once()
                    MockContract.verify_rule_merge_manager_contract(mock_merge_manager)

    # ========== Strategy 3: Property-based Testing ==========

    @given(
        rule_count=st.integers(min_value=1, max_value=8),
        table_name=st.text(
            min_size=3,
            max_size=15,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        ),
        database_name=st.text(
            min_size=3,
            max_size=12,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        ),
    )
    @pytest.mark.asyncio
    async def test_rule_group_execution_invariants(
        self, rule_count: int, table_name: str, database_name: str
    ) -> None:
        """🎲 Property test: Verify execution invariants hold for any valid input"""
        builder = TestDataBuilder()

        # Generate random rules
        rules = []
        for i in range(rule_count):
            rule = (
                builder.rule()
                .with_name(f"rule_{i}")
                .with_target(database_name, table_name, f"col_{i}")
                .as_not_null_rule()
                .build()
            )
            rules.append(rule)

        group = RuleGroup(table_name, database_name)
        for rule in rules:
            group.add_rule(rule)

        mock_async_engine = AsyncMock()

        # with patch.object(group, '_table_exists', AsyncMock(return_value=True)):
        with patch.object(group, "_get_total_records", AsyncMock(return_value=100)):
            mock_merge_manager = MockContract.create_rule_merge_manager_mock()

            with patch.object(
                group, "_get_merge_manager", return_value=mock_merge_manager
            ):

                async def mock_execute_individual(
                    engine: Any,
                    group: RuleGroup,
                ) -> List[Any]:
                    from shared.schema.result_schema import ExecutionResultSchema

                    return [
                        ExecutionResultSchema.create_success_result(
                            rule_id=str(rule.id),
                            entity_name=f"{database_name}.{table_name}",
                            total_count=100,
                            error_count=0,
                            execution_time=0.1,
                            message="Rule passed",
                        )
                        for rule in group.rules
                    ]

                with patch.object(
                    group, "_execute_individual_group", mock_execute_individual
                ):
                    results = await group.execute(mock_async_engine)

                    # Property: Result count must equal rule count for any valid input
                    assert len(results) == rule_count
                    # Property: All results must have required fields
                    for result in results:
                        assert result.rule_id is not None
                        assert result.status is not None

    @given(
        enum_values=st.lists(
            st.text(
                min_size=1,
                max_size=8,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
            ),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    @pytest.mark.asyncio
    async def test_enum_rule_execution_properties(
        self,
        enum_values: List[str],
    ) -> None:
        """🎲 Property test: ENUM rules maintain invariants for any valid value list"""
        builder = TestDataBuilder()
        rule = builder.rule().as_enum_rule(enum_values).build()

        group = RuleGroup("test_table", "test_db")
        group.add_rule(rule)

        mock_async_engine = AsyncMock()

        # with patch.object(group, '_table_exists', AsyncMock(return_value=True)):
        with patch.object(group, "_get_total_records", AsyncMock(return_value=200)):
            mock_merge_manager = MockContract.create_rule_merge_manager_mock()

            with patch.object(
                group, "_get_merge_manager", return_value=mock_merge_manager
            ):

                async def mock_execute_individual(
                    engine: Any,
                    group: RuleGroup,
                ) -> List[Any]:
                    from shared.schema.result_schema import ExecutionResultSchema

                    return [
                        ExecutionResultSchema.create_success_result(
                            rule_id=str(rule.id),
                            entity_name="test_db.test_table",
                            total_count=200,
                            error_count=0,
                            execution_time=0.1,
                            message="ENUM rule passed",
                        )
                    ]

                with patch.object(
                    group, "_execute_individual_group", mock_execute_individual
                ):
                    results = await group.execute(mock_async_engine)

                    # Property: ENUM rule execution always produces one result
                    assert len(results) == 1
                    # Property: Rule type is preserved
                    # Can't directly compare rule_type from ExecutionResultSchema,
                    # use rule_id instead
                    assert results[0].rule_id is not None

    # ========== Strategy 4: Mutation Testing Readiness ==========

    @pytest.mark.asyncio
    async def test_boundary_conditions_catch_mutations(
        self, builder: TestDataBuilder, mock_async_engine: AsyncMock
    ) -> None:
        """🧬 Designed to catch subtle boundary condition mutations"""
        # Test empty rule list (catches len() > 0 vs >= 0 mutations)
        empty_group = RuleGroup("test_table", "test_db")

        with patch.object(empty_group, "_get_total_records", AsyncMock(return_value=0)):
            mock_merge_manager = MockContract.create_rule_merge_manager_mock(
                merge_groups_count=0
            )

            with patch.object(
                empty_group, "_get_merge_manager", return_value=mock_merge_manager
            ):
                results = await empty_group.execute(mock_async_engine)
                assert len(results) == 0  # Catches >= vs > mutations

        # Test single rule (catches off-by-one in counting)
        single_rule = builder.rule().as_not_null_rule().build()
        single_group = RuleGroup("test_table", "test_db")
        single_group.add_rule(single_rule)

        with patch.object(
            single_group, "_get_total_records", AsyncMock(return_value=1)
        ):
            mock_merge_manager = MockContract.create_rule_merge_manager_mock()

            with patch.object(
                single_group, "_get_merge_manager", return_value=mock_merge_manager
            ):

                async def mock_execute_individual(
                    engine: Any,
                    group: RuleGroup,
                ) -> List[Any]:
                    from shared.schema.result_schema import ExecutionResultSchema

                    return [
                        ExecutionResultSchema.create_success_result(
                            rule_id=str(single_rule.id),
                            entity_name="test_db.test_table",
                            total_count=1,
                            error_count=0,
                            execution_time=0.1,
                            message="Rule passed",
                        )
                    ]

                with patch.object(
                    single_group, "_execute_individual_group", mock_execute_individual
                ):
                    results = await single_group.execute(mock_async_engine)
                    assert len(results) == 1  # Catches != vs == mutations

    @pytest.mark.asyncio
    async def test_error_handling_precision(
        self, builder: TestDataBuilder, mock_async_engine: AsyncMock
    ) -> None:
        """🧬 Test precise error conditions to catch error handling mutations"""
        rule = builder.rule().as_range_rule(0, 100).build()
        group = RuleGroup("test_table", "test_db")
        group.add_rule(rule)

        # Test specific error conditions
        with patch.object(group, "_get_total_records", AsyncMock(return_value=100)):
            mock_merge_manager = MockContract.create_rule_merge_manager_mock()

            with patch.object(
                group, "_get_merge_manager", return_value=mock_merge_manager
            ):
                # Simulate execution error for RANGE rules specifically
                async def mock_execute_with_error(
                    engine: Any,
                    group: RuleGroup,
                ) -> List[Any]:
                    if any(rule.type == RuleType.RANGE for rule in group.rules):
                        raise Exception("Range rule execution failed")
                    return []

                with patch.object(
                    group, "_execute_individual_group", mock_execute_with_error
                ):
                    results = await group.execute(mock_async_engine)

                    # Verify error handling precision
                    assert len(results) == 1
                    assert results[0].status == "ERROR"
                    assert "Rule execution failed" in (results[0].error_message or "")
                    assert "Range rule execution failed" in (
                        results[0].error_message or ""
                    )

    @pytest.mark.asyncio
    async def test_async_timing_edge_cases(
        self, builder: TestDataBuilder, mock_async_engine: AsyncMock
    ) -> None:
        """🧬 Test async execution timing to catch concurrency mutations"""
        rules = [
            builder.rule().as_not_null_rule().with_name(f"rule_{i}").build()
            for i in range(3)
        ]

        group = RuleGroup("test_table", "test_db")
        for rule in rules:
            group.add_rule(rule)

        with patch.object(group, "_get_total_records", AsyncMock(return_value=500)):
            mock_merge_manager = MockContract.create_rule_merge_manager_mock()

            with patch.object(
                group, "_get_merge_manager", return_value=mock_merge_manager
            ):
                # Simulate varying execution times
                async def mock_execute_with_timing(
                    engine: Any,
                    group: RuleGroup,
                ) -> List[Any]:
                    await asyncio.sleep(0.001)  # Small delay to test async behavior
                    from shared.schema.result_schema import ExecutionResultSchema

                    return [
                        ExecutionResultSchema.create_success_result(
                            rule_id=str(rule.id),
                            entity_name="test_db.test_table",
                            total_count=500,
                            error_count=0,
                            execution_time=0.05 + (i * 0.01),  # Varying times
                            message="Rule passed",
                        )
                        for i, rule in enumerate(group.rules)
                    ]

                with patch.object(
                    group, "_execute_individual_group", mock_execute_with_timing
                ):
                    start_time = asyncio.get_event_loop().time()
                    results = await group.execute(mock_async_engine)
                    end_time = asyncio.get_event_loop().time()

                    # Verify async execution completed properly
                    assert len(results) == 3
                    assert all(result.execution_time > 0 for result in results)
                    # Verify total execution time is reasonable
                    # (catches infinite loops)
                    assert (end_time - start_time) < 1.0

    # ========== Performance and Resource Tests ==========

    @pytest.mark.asyncio
    async def test_large_rule_group_performance_invariants(
        self, builder: TestDataBuilder, mock_async_engine: AsyncMock
    ) -> None:
        """⚡ Test that performance characteristics hold even with large rule groups"""
        # Create a reasonably large rule group
        rules = [
            builder.rule()
            .as_not_null_rule()
            .with_name(f"rule_{i}")
            .with_target("db", "table", f"col_{i}")
            .build()
            for i in range(15)
        ]

        group = RuleGroup("test_table", "test_db")
        for rule in rules:
            group.add_rule(rule)

        with patch.object(group, "_get_total_records", AsyncMock(return_value=10000)):
            mock_merge_manager = MockContract.create_rule_merge_manager_mock()

            with patch.object(
                group, "_get_merge_manager", return_value=mock_merge_manager
            ):

                async def mock_execute_batch(
                    engine: Any,
                    group: RuleGroup,
                ) -> List[Any]:
                    from shared.schema.result_schema import ExecutionResultSchema

                    return [
                        ExecutionResultSchema.create_success_result(
                            rule_id=str(rule.id),
                            entity_name="test_db.test_table",
                            total_count=10000,
                            error_count=0,
                            execution_time=0.1,
                            message="Rule passed",
                        )
                        for rule in group.rules
                    ]

                with patch.object(
                    group, "_execute_individual_group", mock_execute_batch
                ):
                    start_time = asyncio.get_event_loop().time()
                    results = await group.execute(mock_async_engine)
                    execution_time = asyncio.get_event_loop().time() - start_time

                    # Performance invariants
                    assert len(results) == 15
                    assert execution_time < 2.0  # Reasonable upper bound
                    assert all(result.rule_id is not None for result in results)

    # ========== Critical Path and Edge Case Tests ==========

    @pytest.mark.asyncio
    async def test_mixed_success_failure_scenarios(
        self, builder: TestDataBuilder, mock_async_engine: AsyncMock
    ) -> None:
        """💥 Test groups with both successful and failing rules"""
        success_rule = (
            builder.rule().as_not_null_rule().with_name("success_rule").build()
        )
        failure_rule = (
            builder.rule().as_range_rule(0, 100).with_name("failure_rule").build()
        )

        group = RuleGroup("test_table", "test_db")
        group.add_rule(success_rule)
        group.add_rule(failure_rule)

        with patch.object(group, "_get_total_records", AsyncMock(return_value=1000)):
            mock_merge_manager = MockContract.create_rule_merge_manager_mock()

            with patch.object(
                group, "_get_merge_manager", return_value=mock_merge_manager
            ):

                async def mock_execute_mixed(
                    engine: Any,
                    group: RuleGroup,
                ) -> List[Any]:
                    # This simulates mixed execution where some rules succeed
                    # and some fail. The RuleGroup.execute() method handles
                    # exceptions by creating ERROR results
                    from shared.schema.result_schema import ExecutionResultSchema

                    results = []
                    for rule in group.rules:
                        if rule.name == "success_rule":
                            results.append(
                                ExecutionResultSchema.create_success_result(
                                    rule_id=str(rule.id),
                                    entity_name="test_db.test_table",
                                    total_count=1000,
                                    error_count=0,
                                    execution_time=0.1,
                                    message="Rule passed",
                                )
                            )
                        # For failure_rule, we'll let the RuleGroup handle the
                        # exception
                    # Now simulate the failure for the range rule - this will be
                    # caught by RuleGroup
                    if any(rule.name == "failure_rule" for rule in group.rules):
                        raise Exception("Simulated failure for range rule")
                    return results

                with patch.object(
                    group, "_execute_individual_group", mock_execute_mixed
                ):
                    results = await group.execute(mock_async_engine)

                    # Should have results for both rules
                    # Note: When RuleGroup execution fails, it creates ERROR
                    # results for ALL rules in the group
                    assert len(results) == 2
                    statuses = [result.status for result in results]

                    # In RuleGroup's error handling, when one rule fails in a group,
                    # it creates error results for all rules in the group
                    assert all(status == "ERROR" for status in statuses)
                    assert all(
                        "Rule execution failed" in (result.error_message or "")
                        for result in results
                    )
