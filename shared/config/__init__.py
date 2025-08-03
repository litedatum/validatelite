"""
Shared Configuration Package

Provides configuration models and utilities for loading and accessing
application configuration across modules.
"""

from typing import Any, Dict, Optional, Type, TypeVar

from .loader import load_config

# Import configuration models
from .logging_config import LoggingConfig

# Type variable for configuration models
T = TypeVar("T")

# Global configuration registry
_config_registry: Dict[str, Any] = {}


def register_config(name: str, config: Any) -> None:
    """
    Register a configuration object in the global registry.

    Args:
        name: Name to register the configuration under
        config: Configuration object to register
    """
    # global _config_registry
    _config_registry[name] = config


def get_config(name: str) -> Optional[Any]:
    """
    Get a configuration object from the global registry.

    Args:
        name: Name of the configuration to retrieve

    Returns:
        The configuration object or None if not found
    """
    return _config_registry.get(name)


def get_typed_config(name: str, config_type: Type[T]) -> Optional[T]:
    """
    Get a configuration object from the global registry with type checking.

    Args:
        name: Name of the configuration to retrieve
        config_type: Expected type of the configuration

    Returns:
        The configuration object or None if not found or type doesn't match
    """
    config = get_config(name)
    if config is not None and isinstance(config, config_type):
        return config  # type: ignore[no-any-return]
    return None


__all__ = [
    "LoggingConfig",
    "load_config",
    "register_config",
    "get_config",
    "get_typed_config",
]
