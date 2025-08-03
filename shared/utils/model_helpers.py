"""
Model helper utilities

Provides helper functions for working with Pydantic models:
1. Safe model serialization with error handling
2. Enum value extraction and validation
3. Centralized error handling for model operations
"""

from typing import Any, Dict, TypeVar

from pydantic import BaseModel

from shared.exceptions import OperationError
from shared.utils.logger import get_logger

# Get logger
logger = get_logger(__name__)

# Type variable for Pydantic models
T = TypeVar("T", bound=BaseModel)


def safe_model_dump(model: BaseModel, mode: str = "json") -> Dict[str, Any]:
    """
    Safely convert a model to a dictionary with error handling and logging

    Args:
        model: Pydantic model instance
        mode: Serialization mode, default is 'json'

    Returns:
        Dict[str, Any]: Serialized dictionary

    Raises:
        OperationError: When an enum field is None
    """
    try:
        return model.model_dump(mode=mode)
    except Exception as e:
        logger.error(f"Model serialization error: {str(e)}")
        # Try to identify if it's an enum serialization error
        if "None" in str(e) and "enum" in str(e).lower():
            raise OperationError(
                message=f"Enum serialization error in model {model.__class__.__name__}",
                operation="enum_serialization",
            )
        raise
