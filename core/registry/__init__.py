"""
Core module

Contains core functions of the application, such as the rule type registry, etc.
"""

# Import builtin rule types to ensure they are registered
import core.registry.builtin_rule_types  # noqa: F401

# Import rule type registry
from core.registry.rule_type_registry import register_rule_type, rule_type_registry

# Export public interfaces
__all__ = ["rule_type_registry", "register_rule_type"]
