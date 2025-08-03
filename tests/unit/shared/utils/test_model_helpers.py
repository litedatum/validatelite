"""
Model helpers test module

Tests the model helper utilities, ensuring:
1. Safe model serialization works correctly
2. Error handling for model serialization is robust
3. Enum serialization errors are properly caught and reported
"""

from enum import Enum
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from shared.exceptions import OperationError
from shared.schema.base import BaseSchema
from shared.utils.model_helpers import safe_model_dump


class ClsTestEnum(str, Enum):
    """Test enum for serialization tests"""

    VALUE1 = "value1"
    VALUE2 = "value2"


class ClsTestModel(BaseSchema):
    """Test model for serialization tests"""

    test_enum: ClsTestEnum


def test_safe_model_dump_basic() -> None:
    """Test basic functionality of safe_model_dump"""
    # Create test object
    model = ClsTestModel(test_enum=ClsTestEnum.VALUE1)

    # Test safe_model_dump function with default mode
    data = safe_model_dump(model)
    assert data["test_enum"] == "value1"
    assert isinstance(data["test_enum"], str)

    # Test safe_model_dump function with explicit json mode
    json_data = safe_model_dump(model, mode="json")
    assert json_data["test_enum"] == "value1"
    assert isinstance(json_data["test_enum"], str)

    # Test safe_model_dump function with python mode
    python_data = safe_model_dump(model, mode="python")
    assert python_data["test_enum"] == "value1"
    assert isinstance(python_data["test_enum"], str)


@patch("shared.utils.model_helpers.logger")
def test_safe_model_dump_error_logging(mock_logger: MagicMock) -> None:
    """Test error logging in safe_model_dump"""
    # Create test object
    model = ClsTestModel(test_enum=ClsTestEnum.VALUE1)

    # Mock BaseModel.model_dump to raise an exception
    with patch("pydantic.BaseModel.model_dump", side_effect=Exception("Test error")):
        # Test that the exception is re-raised
        with pytest.raises(Exception) as exc_info:
            safe_model_dump(model)

        # Verify the exception message
        assert "Test error" in str(exc_info.value)

        # Verify that the error was logged
        mock_logger.error.assert_called_once()
        assert (
            "Model serialization error: Test error" in mock_logger.error.call_args[0][0]
        )


@patch("shared.utils.model_helpers.logger")
def test_safe_model_dump_enum_error(mock_logger: MagicMock) -> None:
    """Test enum serialization error handling in safe_model_dump"""
    # Create test object
    model = ClsTestModel(test_enum=ClsTestEnum.VALUE1)

    # Mock BaseModel.model_dump to raise an exception that looks like an enum serialization error
    with patch(
        "pydantic.BaseModel.model_dump",
        side_effect=Exception("None is not a valid ClsTestEnum"),
    ):
        # Test that OperationError is raised for enum serialization issues
        with pytest.raises(OperationError) as exc_info:
            safe_model_dump(model)

        # Verify the exception details
        assert model.__class__.__name__ in str(exc_info.value)
        assert "Enum serialization error" in str(exc_info.value)

        # Verify that the error was logged
        mock_logger.error.assert_called_once()
        assert (
            "Model serialization error: None is not a valid ClsTestEnum"
            in mock_logger.error.call_args[0][0]
        )


def test_safe_model_dump_with_complex_model() -> None:
    """Test safe_model_dump with a more complex model structure"""

    # Create a more complex model with nested structure
    class NestedModel(BaseSchema):
        nested_enum: ClsTestEnum

    class ComplexModel(BaseSchema):
        name: str
        enum_value: ClsTestEnum
        nested: NestedModel

    # Create test object
    nested = NestedModel(nested_enum=ClsTestEnum.VALUE2)
    model = ComplexModel(name="Test", enum_value=ClsTestEnum.VALUE1, nested=nested)

    # Test safe_model_dump function
    data = safe_model_dump(model)

    # Verify top-level enum serialization
    assert data["enum_value"] == "value1"
    assert isinstance(data["enum_value"], str)

    # Verify nested enum serialization
    assert data["nested"]["nested_enum"] == "value2"
    assert isinstance(data["nested"]["nested_enum"], str)
