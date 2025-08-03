"""
Test rule type registry.

This section tests the functionality of the rule type registry.
Register rule type.
2. Retrieve the rule type.
3. Deregister Rule Type
4. Decorator Registration
"""

from typing import Any, Callable, Dict, Optional

from core.registry.rule_type_registry import RuleTypeRegistry
from shared.schema.rule_schema import RuleSchema


class TestRuleTypeRegistry:
    """Test rule type registry."""

    def test_register_rule_type(self) -> None:
        """Testing registration rule types."""
        # Create an instance of the registry.
        registry = RuleTypeRegistry()

        # Registered rule types
        registry.register_rule_type(
            type_id="TEST_TYPE", name="Test type", description="Test rule type"
        )

        # Verify successful registration.
        assert registry.has_rule_type("TEST_TYPE")
        rule_type = registry.get_rule_type("TEST_TYPE")
        assert rule_type is not None
        assert rule_type["id"] == "TEST_TYPE"
        assert rule_type["name"] == "Test type"
        assert rule_type["description"] == "Test rule type"

    def test_register_with_components(self) -> None:
        """Testing the registration of rule types that include components."""
        # Create an instance of the registry.
        registry = RuleTypeRegistry()

        # Create test components.
        class TestForm:
            pass

        class TestExecutor:
            pass

        def test_validator(params: Dict[str, Any]) -> bool:
            return True

        def test_sql_generator(rule: RuleSchema, table: str, column: str) -> str:
            return f"SELECT * FROM {table} WHERE {column} IS NOT NULL"

        # Registered Rule Types
        registry.register_rule_type(
            type_id="TEST_TYPE",
            name="Test type",
            description="Test rule type",
            form_class=TestForm,
            executor_class=TestExecutor,
            validator=test_validator,
            sql_generator=test_sql_generator,
        )

        # Verify successful component registration.
        assert registry.get_form_class("TEST_TYPE") == TestForm
        assert registry.get_executor_class("TEST_TYPE") == TestExecutor
        assert registry.get_validator("TEST_TYPE") == test_validator
        assert registry.get_sql_generator("TEST_TYPE") == test_sql_generator

    def test_get_rule_types(self) -> None:
        """Test retrieval of all rule types."""
        # Create an instance of the registry.
        registry = RuleTypeRegistry()

        # Register multiple rule types.
        registry.register_rule_type(
            type_id="TYPE1", name="Type 1", description="Description 1"
        )

        registry.register_rule_type(
            type_id="TYPE2", name="Type 2", description="Description 2"
        )

        # Retrieve all rule types.
        rule_types = registry.get_rule_types()

        # Validate the results.
        assert len(rule_types) == 2
        assert {"id": "TYPE1", "name": "Type 1"} == {
            k: rule_types[0][k] for k in ["id", "name"]
        }
        assert {"id": "TYPE2", "name": "Type 2"} == {
            k: rule_types[1][k] for k in ["id", "name"]
        }

    def test_unregister_rule_type(self) -> None:
        """Test deregistration of rule types."""
        # Create an instance of the registry.
        registry = RuleTypeRegistry()

        # Register a rule type.
        registry.register_rule_type(
            type_id="TEST_TYPE", name="Test type", description="Test rule type"
        )

        # Verify successful registration.
        assert registry.has_rule_type("TEST_TYPE")

        # Deregister the rule type.
        registry.unregister_rule_type("TEST_TYPE")

        # Verify successful deregistration.
        assert not registry.has_rule_type("TEST_TYPE")
        assert registry.get_rule_type("TEST_TYPE") is None

    def test_decorator_registration(self) -> None:
        """Test registration using decorators."""
        # Create an instance of the registry.
        registry = RuleTypeRegistry()

        # Create a custom decorator that mimics the real register_rule_type decorator.
        def custom_decorator(
            type_id: str,
            name: str,
            description: str,
            **kwargs: Any,
        ) -> Callable:
            def inner_decorator(cls_or_func: Any) -> Any:
                # Decide registration method based on the type of the decorated object
                if isinstance(cls_or_func, type):
                    # Decorated object is a class, register as executor class
                    registry.register_rule_type(
                        type_id=type_id,
                        name=name,
                        description=description,
                        executor_class=cls_or_func,
                        **kwargs,
                    )
                else:
                    # Decorated object is a function, register as SQL generation
                    # function
                    registry.register_rule_type(
                        type_id=type_id,
                        name=name,
                        description=description,
                        sql_generator=cls_or_func,
                        **kwargs,
                    )
                return cls_or_func

            return inner_decorator

        # Register a class using the decorator.
        @custom_decorator(
            type_id="DECORATED_TYPE",
            name="Decorator type",
            description="Type registered using a decorator",
        )
        class TestExecutor:
            pass

        # Register a function using the decorator.
        @custom_decorator(
            type_id="FUNC_TYPE",
            name="Function type",
            description="Function registered using a decorator",
        )
        def test_sql_generator(
            database: str,
            table: str,
            column: str,
            rule_config: Optional[Dict[str, Any]] = None,
            filter_condition: Optional[str] = None,
        ) -> str:
            return f"SELECT * FROM {database}.{table} WHERE {column} IS NOT NULL"

        # Verify successful registration.
        assert registry.has_rule_type("DECORATED_TYPE")
        assert registry.has_rule_type("FUNC_TYPE")

        # Verify that the class was registered as executor_class
        assert registry.get_executor_class("DECORATED_TYPE") == TestExecutor

        # Verify that the function was registered as sql_generator
        assert registry.get_sql_generator("FUNC_TYPE") == test_sql_generator

    def test_builtin_rule_types(self) -> None:
        """Test that builtin rule types are properly registered."""
        # Use the global registry instance which has builtin types registered
        # Import builtin rule types to ensure they are registered
        import core.registry.builtin_rule_types
        from core.registry.rule_type_registry import rule_type_registry

        # Verify that builtin rule types are registered.
        builtin_types = ["NOT_NULL", "UNIQUE", "RANGE", "ENUM", "REGEX", "LENGTH"]
        for rule_type in builtin_types:
            assert rule_type_registry.has_rule_type(
                rule_type
            ), f"Builtin rule type {rule_type} not found"
