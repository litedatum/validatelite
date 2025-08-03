"""
Test rule type registry validation improvements

Tests the enhanced validation in the rule type registry to ensure
that invalid parameters are properly rejected.
"""

from typing import Any

import pytest

from core.registry.rule_type_registry import RuleTypeRegistry


class TestRuleTypeRegistryValidation:
    """Test validation improvements in rule type registry"""

    def test_executor_class_validation(self) -> None:
        """Test that executor_class must be a class"""
        registry = RuleTypeRegistry()

        # Test with invalid executor_class (string)
        with pytest.raises(ValueError, match="executor_class must be a class"):
            registry.register_rule_type(
                type_id="INVALID_EXECUTOR",
                name="Invalid Executor",
                description="Test invalid executor",
                executor_class="not_a_class",  # type: ignore[arg-type]
            )

        # Test with valid executor_class
        class ValidExecutor:
            def execute(self, rule: Any, connection: Any) -> dict:
                return {"status": "PASSED"}

        registry.register_rule_type(
            type_id="VALID_EXECUTOR",
            name="Valid Executor",
            description="Test valid executor",
            executor_class=ValidExecutor,
        )

        assert registry.has_rule_type("VALID_EXECUTOR")
        assert registry.get_executor_class("VALID_EXECUTOR") == ValidExecutor

        # Cleanup
        registry.unregister_rule_type("VALID_EXECUTOR")

    def test_form_class_validation(self) -> None:
        """Test that form_class must be a class"""
        registry = RuleTypeRegistry()

        # Test with invalid form_class (string)
        with pytest.raises(ValueError, match="form_class must be a class"):
            registry.register_rule_type(
                type_id="INVALID_FORM",
                name="Invalid Form",
                description="Test invalid form",
                form_class="not_a_class",  # type: ignore[arg-type]
            )

        # Test with valid form_class
        class ValidForm:
            def render(self) -> str:
                return "form"

        registry.register_rule_type(
            type_id="VALID_FORM",
            name="Valid Form",
            description="Test valid form",
            form_class=ValidForm,
        )

        assert registry.has_rule_type("VALID_FORM")
        assert registry.get_form_class("VALID_FORM") == ValidForm

        # Cleanup
        registry.unregister_rule_type("VALID_FORM")

    def test_validator_validation(self) -> None:
        """Test that validator must be callable"""
        registry = RuleTypeRegistry()

        # Test with invalid validator (string)
        with pytest.raises(ValueError, match="validator must be callable"):
            registry.register_rule_type(
                type_id="INVALID_VALIDATOR",
                name="Invalid Validator",
                description="Test invalid validator",
                validator="not_callable",  # type: ignore[arg-type]
            )

        # Test with valid validator (function)
        def valid_validator(params: Any) -> bool:
            return True

        registry.register_rule_type(
            type_id="VALID_VALIDATOR",
            name="Valid Validator",
            description="Test valid validator",
            validator=valid_validator,
        )

        assert registry.has_rule_type("VALID_VALIDATOR")
        assert registry.get_validator("VALID_VALIDATOR") == valid_validator

        # Cleanup
        registry.unregister_rule_type("VALID_VALIDATOR")

    def test_sql_generator_validation(self) -> None:
        """Test that sql_generator must be callable"""
        registry = RuleTypeRegistry()

        # Test with invalid sql_generator (string)
        with pytest.raises(ValueError, match="sql_generator must be callable"):
            registry.register_rule_type(
                type_id="INVALID_SQL_GENERATOR",
                name="Invalid SQL Generator",
                description="Test invalid sql generator",
                sql_generator="not_callable",  # type: ignore[arg-type]
            )

        # Test with valid sql_generator (function)
        def valid_sql_generator(rule: Any, table: str, column: str) -> str:
            return f"SELECT * FROM {table} WHERE {column} IS NOT NULL"

        registry.register_rule_type(
            type_id="VALID_SQL_GENERATOR",
            name="Valid SQL Generator",
            description="Test valid sql generator",
            sql_generator=valid_sql_generator,
        )

        assert registry.has_rule_type("VALID_SQL_GENERATOR")
        assert registry.get_sql_generator("VALID_SQL_GENERATOR") == valid_sql_generator

        # Cleanup
        registry.unregister_rule_type("VALID_SQL_GENERATOR")

    def test_multiple_validation_errors(self) -> None:
        """Test that multiple validation errors are caught"""
        registry = RuleTypeRegistry()

        # Test with multiple invalid parameters
        with pytest.raises(ValueError, match="executor_class must be a class"):
            registry.register_rule_type(
                type_id="MULTIPLE_INVALID",
                name="Multiple Invalid",
                description="Test multiple invalid parameters",
                executor_class="not_a_class",  # type: ignore[arg-type]
                form_class="not_a_class",  # type: ignore[arg-type]
                validator="not_callable",  # type: ignore[arg-type]
                sql_generator="not_callable",  # type: ignore[arg-type]
            )

        # Verify nothing was registered
        assert not registry.has_rule_type("MULTIPLE_INVALID")

    def test_valid_registration_with_all_components(self) -> None:
        """Test that valid registration with all components works"""
        registry = RuleTypeRegistry()

        class ValidExecutor:
            def execute(self, rule: Any, connection: Any) -> dict:
                return {"status": "PASSED"}

        class ValidForm:
            def render(self) -> str:
                return "form"

        def valid_validator(params: Any) -> bool:
            return True

        def valid_sql_generator(rule: Any, table: str, column: str) -> str:
            return f"SELECT * FROM {table} WHERE {column} IS NOT NULL"

        registry.register_rule_type(
            type_id="ALL_VALID",
            name="All Valid Components",
            description="Test all valid components",
            executor_class=ValidExecutor,
            form_class=ValidForm,
            validator=valid_validator,
            sql_generator=valid_sql_generator,
        )

        assert registry.has_rule_type("ALL_VALID")
        assert registry.get_executor_class("ALL_VALID") == ValidExecutor
        assert registry.get_form_class("ALL_VALID") == ValidForm
        assert registry.get_validator("ALL_VALID") == valid_validator
        assert registry.get_sql_generator("ALL_VALID") == valid_sql_generator

        # Cleanup
        registry.unregister_rule_type("ALL_VALID")
