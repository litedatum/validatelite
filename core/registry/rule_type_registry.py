"""
Rule type registry

Provides registration, management, and query functions for rule types.
Supports dynamic registration of new rule types and provides rule type metadata.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Type

# Configure logger
logger = logging.getLogger(__name__)


class RuleTypeRegistry:
    """
    Rule type registry

    Manages the rule types supported by the system, provides registration, query,
    and execution functions.
    """

    def __init__(self) -> None:
        """Initialize rule type registry"""
        self.rule_types: Dict[str, Dict[str, Any]] = {}
        self.rule_forms: Dict[str, Type] = {}
        self.rule_executors: Dict[str, Type] = {}
        self.rule_validators: Dict[str, Callable] = {}
        self.rule_sql_generators: Dict[str, Callable] = {}

    def register_rule_type(
        self,
        type_id: str,
        name: str,
        description: str,
        form_class: Optional[Type] = None,
        executor_class: Optional[Type] = None,
        validator: Optional[Callable] = None,
        sql_generator: Optional[Callable] = None,
        category: str = "custom",
        icon: str = "rule",
        parameters_schema: Optional[Dict[str, Any]] = None,
        ui_schema: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Register a new rule type

        Args:
            type_id: Rule type ID, unique identifier
            name: Rule type name
            description: Rule type description
            form_class: Rule form class, used to generate UI forms
            executor_class: Rule executor class, used to execute rules
            validator: Rule parameter validation function
            sql_generator: SQL generation function
            category: Rule type category
            icon: Rule type icon
            parameters_schema: Parameter JSON Schema
            ui_schema: UI JSON Schema
            examples: Example configurations
        """
        if type_id in self.rule_types:
            logger.warning(
                f"Rule type {type_id} already exists and will be overwritten"
            )

        # Validate executor_class parameter
        if executor_class is not None and not isinstance(executor_class, type):
            raise ValueError(
                f"executor_class must be a class, got "
                f"{type(executor_class).__name__}: {executor_class}"
            )

        # Validate form_class parameter
        if form_class is not None and not isinstance(form_class, type):
            raise ValueError(
                f"form_class must be a class, got "
                f"{type(form_class).__name__}: {form_class}"
            )

        # Validate validator parameter
        if validator is not None and not callable(validator):
            raise ValueError(
                f"validator must be callable, got "
                f"{type(validator).__name__}: {validator}"
            )

        # Validate sql_generator parameter
        if sql_generator is not None and not callable(sql_generator):
            raise ValueError(
                f"sql_generator must be callable, got "
                f"{type(sql_generator).__name__}: {sql_generator}"
            )

        # Register rule type metadata
        self.rule_types[type_id] = {
            "id": type_id,
            "name": name,
            "description": description,
            "category": category,
            "icon": icon,
            "parameters_schema": parameters_schema or {},
            "ui_schema": ui_schema or {},
            "examples": examples or [],
        }

        # Register related components
        if form_class:
            self.rule_forms[type_id] = form_class

        if executor_class:
            self.rule_executors[type_id] = executor_class

        if validator:
            self.rule_validators[type_id] = validator

        if sql_generator:
            self.rule_sql_generators[type_id] = sql_generator

        logger.info(f"Successfully registered rule type: {type_id}")

    def get_rule_types(self) -> List[Dict[str, Any]]:
        """
        Get all registered rule types

        Returns:
            List[Dict[str, Any]]: List of rule types
        """
        return list(self.rule_types.values())

    def get_rule_type(self, type_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for the specified rule type

        Args:
            type_id: Rule type ID

        Returns:
            Optional[Dict[str, Any]]: Rule type metadata, None if not found
        """
        return self.rule_types.get(type_id)

    def get_form_class(self, type_id: str) -> Optional[Type]:
        """
        Get the form class corresponding to the rule type

        Args:
            type_id: Rule type ID

        Returns:
            Optional[Type]: Form class, None if not found
        """
        return self.rule_forms.get(type_id)

    def get_executor_class(self, type_id: str) -> Optional[Type]:
        """
        Get the executor class corresponding to the rule type

        Args:
            type_id: Rule type ID

        Returns:
            Optional[Type]: Executor class, None if not found
        """
        return self.rule_executors.get(type_id)

    def get_validator(self, type_id: str) -> Optional[Callable]:
        """
        Get the validation function corresponding to the rule type

        Args:
            type_id: Rule type ID

        Returns:
            Optional[Callable]: Validation function, None if not found
        """
        return self.rule_validators.get(type_id)

    def get_sql_generator(self, type_id: str) -> Optional[Callable]:
        """
        Get the SQL generation function corresponding to the rule type

        Args:
            type_id: Rule type ID

        Returns:
            Optional[Callable]: SQL generation function, None if not found
        """
        return self.rule_sql_generators.get(type_id)

    def has_rule_type(self, type_id: str) -> bool:
        """
        Check whether the rule type exists

        Args:
            type_id: Rule type ID

        Returns:
            bool: Whether it exists
        """
        return type_id in self.rule_types

    def unregister_rule_type(self, type_id: str) -> bool:
        """
        Unregister a rule type

        Args:
            type_id: Rule type ID

        Returns:
            bool: Whether unregistration was successful
        """
        if type_id not in self.rule_types:
            logger.warning(f"Rule type {type_id} does not exist, cannot unregister")
            return False

        self.rule_types.pop(type_id, None)
        self.rule_forms.pop(type_id, None)
        self.rule_executors.pop(type_id, None)
        self.rule_validators.pop(type_id, None)
        self.rule_sql_generators.pop(type_id, None)

        logger.info(f"Successfully unregistered rule type: {type_id}")
        return True


# Create global rule type registry instance
rule_type_registry = RuleTypeRegistry()


def register_rule_type(
    type_id: str, name: str, description: str, **kwargs: Any
) -> Callable:
    """
    Rule type registration decorator

    Used to simplify the rule type registration process, can be directly used to
    decorate executor classes

    Args:
        type_id: Rule type ID
        name: Rule type name
        description: Rule type description
        **kwargs: Other parameters, consistent with register_rule_type method

    Returns:
        Callable: Decorator function
    """

    def decorator(cls_or_func: Any) -> Any:
        # Decide registration method based on the type of the decorated object
        if isinstance(cls_or_func, type):
            # Decorated object is a class, register as executor class
            rule_type_registry.register_rule_type(
                type_id=type_id,
                name=name,
                description=description,
                executor_class=cls_or_func,
                **kwargs,
            )
        else:
            # Decorated object is a function, register as SQL generation function
            rule_type_registry.register_rule_type(
                type_id=type_id,
                name=name,
                description=description,
                sql_generator=cls_or_func,
                **kwargs,
            )

        return cls_or_func

    return decorator
