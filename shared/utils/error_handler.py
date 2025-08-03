"""
Error handling utility module

Provides unified error handling functions:
1. Exception capture and handling
2. Error log recording
3. Error response formatting
4. Context information handling
5. Exception type mapping
6. Custom exception handling
7. Exception handling decorators (sync and async)
8. Internationalization support
"""

import functools
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Protocol, TypeVar, Union, overload

from sqlalchemy.exc import OperationalError, ProgrammingError, SQLAlchemyError
from typing_extensions import ParamSpec

from shared.exceptions import EngineError, OperationError, RuleExecutionError
from shared.utils.logger import get_logger

# Type variable definition
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])
AsyncF = TypeVar("AsyncF", bound=Callable[..., Any])


P = ParamSpec("P")
R = TypeVar("R", covariant=True)


class AsyncCallable(Protocol[P, R]):
    """Define a simple protocol that represents an asynchronous function"""

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the function"""
        ...


class SyncCallable(Protocol[P, R]):
    """Define a simple protocol that represents a synchronous functionthat"""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the function"""
        ...


# Exception mapping function
def map_exception(exception: Exception) -> Exception:
    """
    Map the original exception to a custom exception

    Args:
        exception: Original exception

    Returns:
        Mapped exception
    """
    # Map database-related exceptions to system custom exceptions
    if isinstance(exception, SQLAlchemyError):
        if isinstance(exception, OperationalError):
            # Operational error (e.g., connection timeout, connection failure, etc.)
            error_msg = (
                str(exception.orig) if hasattr(exception, "orig") else str(exception)
            )
            return EngineError(
                f"Database operation error: {error_msg}", cause=exception
            )
        elif isinstance(exception, ProgrammingError):
            # Programming error (e.g., SQL syntax error, etc.)
            error_msg = (
                str(exception.orig) if hasattr(exception, "orig") else str(exception)
            )
            return RuleExecutionError(
                f"Database programming error: {error_msg}", cause=exception
            )
        else:
            # Other SQLAlchemy errors
            return OperationError(f"Database error: {str(exception)}", cause=exception)

    # Other exceptions remain unchanged
    return exception


# Error log recording function
def log_error(
    exception: Exception,
    context: str = "",
    details: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Record exception log

    Args:
        exception: Exception object
        context: Context description
        details: Detailed information
        logger: Logger instance
    """
    # Get logger
    if logger is None:
        logger = get_logger(__name__)

    # Build log message
    if context:
        message = f"{context}: {str(exception)}"
    else:
        message = str(exception)

    # Add detailed information
    if details:
        detail_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
        message = f"{message} - Details: {detail_str}"

    # Record exception log
    logger.exception(message)


# Error response formatting function
def format_error_response(
    exception: Exception,
    context: str = "",
    details: Optional[Dict[str, str]] = None,
    error_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Format error response

    Args:
        exception: Exception object
        context: Context description
        details: Detailed information
        error_code: Error code

    Returns:
        Formatted error response
    """
    # Build error message
    if context:
        message = f"{context}: {str(exception)}"
    else:
        message = str(exception)

    # Build basic response
    response: Dict[str, Any] = {
        "status": "error",
        "message": message,
        "error_type": type(exception).__name__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Add error code
    if error_code:
        response["error_code"] = error_code

    # Add detailed information
    if details:
        response["details"] = details

    return response


# Internationalization support
def translate_message(message: str, lang: str = "en") -> str:
    """
    Translate error message

    Args:
        message: Error message
        lang: Language code (default: "en" for English)

    Returns:
        Translated message
    """
    # In actual projects, a real translation service should be called here
    # This is just a simple example
    if lang == "zh":
        # Simple English-Chinese mapping example
        translations = {
            "Test error": "测试错误",
            "When executing rule": "执行规则时",
            "Database error": "数据库错误",
            "Rule execution failed": "规则执行失败",
            "Division by zero is not allowed": "除数不能为零",
            "Even number error": "偶数错误",
        }

        for en, zh in translations.items():
            message = message.replace(en, zh)

    return message


# Exception handling function
def handle_exception(
    exception: Exception,
    context: str = "",
    details: Optional[Dict[str, str]] = None,
    error_code: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    lang: str = "en",
) -> Dict[str, Any]:
    """
    Handle exception

    Args:
        exception: Exception object
        context: Context description
        details: Detailed information
        error_code: Error code
        logger: Logger instance
        lang: Language code

    Returns:
        Formatted error response
    """
    # Get logger
    if logger is None:
        logger = get_logger(__name__)

    # Map exception
    mapped_exception = map_exception(exception)

    # Record error log
    log_error(mapped_exception, context, details, logger)

    # Format error response
    response = format_error_response(mapped_exception, context, details, error_code)

    # Internationalization processing
    if lang != "en":
        response["message"] = translate_message(response["message"], lang)

    return response


# The following implementation uses @overload to tell MyPy that the function
# has multiple "faces" to ensure that it passes the MyPy check


# Synchronous error handling decorator
@overload
def with_error_handling(
    *, logger: Optional[logging.Logger] = None
) -> Callable[[SyncCallable[P, R]], SyncCallable[P, Union[R, Dict[str, Any]]]]:
    """
    @overload 1: When the decorator is called with parameters
    @with_error_handling(logger=...)
    """
    ...


@overload
def with_error_handling(
    func: SyncCallable[P, R], *, logger: Optional[logging.Logger] = None
) -> SyncCallable[P, Union[R, Dict[str, Any]]]:
    """
    @overload 2: When the decorator is called directly without parameters
    @with_error_handling
    """
    ...


def with_error_handling(
    func: Optional[SyncCallable] = None, *, logger: Optional[logging.Logger] = None
) -> Union[Callable, SyncCallable]:
    """
    Error handling decorator for synchronous functions
    Returns a function with the same parameters, but return type
    Union[R, Dict[str, Any]]
    """

    def decorator(fn: SyncCallable[P, R]) -> SyncCallable[P, Union[R, Dict[str, Any]]]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Union[R, Dict[str, Any]]:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                return handle_exception(e, logger=logger)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


# Asynchronous error handling decorator


@overload
def async_with_error_handling(
    *, logger: Optional[logging.Logger] = None
) -> Callable[[AsyncCallable[P, R]], AsyncCallable[P, Union[R, Dict[str, Any]]]]:
    """
    @overload 1: When the decorator is called with parameters
    @async_with_error_handling(logger=...)
    It receives the logger parameter and returns a "decorator function"
    """
    ...


@overload
def async_with_error_handling(
    func: AsyncCallable[P, R], *, logger: Optional[logging.Logger] = None
) -> AsyncCallable[P, Union[R, Dict[str, Any]]]:
    """
    @overload 2: When the decorator is called directly without parameters
    @async_with_error_handling
    It directly receives the decorated function func
    """
    ...


def async_with_error_handling(
    func: Optional[AsyncCallable] = None, *, logger: Optional[logging.Logger] = None
) -> Union[Callable, AsyncCallable]:
    """
    Error handling decorator for asynchronous functions
    Returns a function with the same parameters, but return type
    Awaitable[Union[R, Dict[str, Any]]]
    """

    def decorator(fn: Callable[P, "R"]) -> AsyncCallable[P, Union[R, Dict[str, Any]]]:
        @functools.wraps(fn)
        async def wrapper(
            *args: P.args, **kwargs: P.kwargs
        ) -> Union["R", Dict[str, Any]]:
            try:
                return await fn(*args, **kwargs)  # type: ignore
            except Exception as e:
                return handle_exception(e, logger=logger)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
