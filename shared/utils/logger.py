"""
Unified logger manager

Provides application-level unified logging functions, supports:
- Configurable logging settings
- Multiple log handlers (file, console, performance, error, audit)
- Log rotation and filtering
- Structured log support
- Contextual log recording
"""

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union

# Global logger manager instance
_logger_manager: Optional["LoggerManager"] = None
_lock = threading.RLock()  # Use re-entrant lock to avoid deadlocks
_initialized = False


class ContextFilter(logging.Filter):
    """Context filter, adds context information to log records"""

    def __init__(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize context filter"""
        super().__init__()
        self.context = context or {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record"""
        # Add context information to log record
        for key, value in self.context.items():
            setattr(record, key, value)

        # Add timestamp
        record.timestamp = datetime.now().isoformat()

        # Add thread information
        record.thread_name = threading.current_thread().name

        return True


class ModuleFilter(logging.Filter):
    """Module filter, filters logs for specific modules"""

    def __init__(self, filtered_modules: list, filtered_level: str = "WARNING"):
        """Initialize module filter"""
        super().__init__()
        self.filtered_modules = filtered_modules or []
        self.filtered_level = getattr(logging, filtered_level.upper(), logging.WARNING)

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record"""
        # Apply higher log level to specified modules
        for module in self.filtered_modules:
            if record.name.startswith(module):
                return record.levelno >= self.filtered_level
        return True


class StructuredFormatter(logging.Formatter):
    """Structured log formatter, outputs logs in JSON format"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record"""
        log_data = {
            "timestamp": getattr(record, "timestamp", datetime.now().isoformat()),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": getattr(record, "thread_name", threading.current_thread().name),
        }

        # Add exception information
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra context information
        for key, value in record.__dict__.items():
            if key not in log_data and not key.startswith("_"):
                log_data[key] = value

        return json.dumps(log_data, ensure_ascii=False, default=str)


class PerformanceLogger:
    """Performance log recorder"""

    def __init__(self, logger: logging.Logger, threshold: float = 1.0) -> None:
        """Initialize performance logger"""
        self.logger = logger
        self.threshold = threshold

    def log_operation(
        self, operation_name: str, duration: float, **kwargs: Any
    ) -> None:
        """Record operation performance"""
        if duration >= self.threshold:
            extra: Dict[str, Any] = {
                "operation": operation_name,
                "duration": duration,
                "performance_log": True,
                **kwargs,
            }
            self.logger.warning(
                f"Slow operation: {operation_name} took {duration:.3f}s", extra=extra
            )
        else:
            extra = {
                "operation": operation_name,
                "duration": duration,
                "performance_log": True,
                **kwargs,
            }
            self.logger.debug(
                f"Operation: {operation_name} took {duration:.3f}s", extra=extra
            )


class LoggerManager:
    """Logger manager"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LoggerManager"""
        self.config = config or self._get_default_config()
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        self.performance_logger: Optional[PerformanceLogger] = None
        self._setup_complete = False
        self._setup_logging()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config"""
        return {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "date_format": "%Y-%m-%d %H:%M:%S",
            "to_file": True,
            "to_console": True,
            "log_file": "logs/app.log",
            "max_bytes": 10 * 1024 * 1024,  # 10MB
            "backup_count": 5,
            "structured": False,
            "context_enabled": True,
            "performance_enabled": True,
            "performance_threshold": 1.0,
            "performance_log_file": "logs/performance.log",
            "error_log_enabled": True,
            "error_log_file": "logs/error.log",
            "audit_log_enabled": True,
            "audit_log_file": "logs/audit.log",
            "module_levels": {},
            "filters": ["werkzeug", "urllib3.connectionpool"],
            "filtered_level": "WARNING",
        }

    def _setup_logging(self) -> None:
        """Set up logging system"""
        if self._setup_complete:
            return

        try:
            # Ensure log directories exist
            self._ensure_log_directories()

            # Set root log level
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)

            # Clear existing handlers (avoid duplication)
            if not self._setup_complete:
                root_logger.handlers.clear()

            # Create handlers
            self._create_handlers()

            # Set up performance logger
            if self.config.get("performance_enabled", True):
                try:
                    perf_logger = self.get_logger("performance")
                    self.performance_logger = PerformanceLogger(
                        perf_logger, self.config.get("performance_threshold", 1.0)
                    )
                except Exception:
                    # If performance logger creation fails, continue running
                    pass

            self._setup_complete = True

        except Exception as e:
            # If setup fails, use basic config
            print(f"Failed to set up logging system: {e}", file=sys.stderr)
            self._setup_basic_logging()

    def _setup_basic_logging(self) -> None:
        """Set up basic logging config (fail-safe)"""
        try:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)
            self.handlers["console"] = console_handler
            self._setup_complete = True
        except Exception as e:
            print(f"Basic logging setup also failed: {e}", file=sys.stderr)

    def _ensure_log_directories(self) -> None:
        """Ensure log directories exist"""
        try:
            log_files = [
                self.config.get("log_file"),
                self.config.get("performance_log_file"),
                self.config.get("error_log_file"),
                self.config.get("audit_log_file"),
            ]

            for log_file in log_files:
                if log_file:
                    log_dir = os.path.dirname(log_file)
                    if log_dir and not os.path.exists(log_dir):
                        os.makedirs(log_dir, exist_ok=True)
        except Exception:
            # If directory creation fails, disable file logging
            self.config["to_file"] = False
            self.config["error_log_enabled"] = False
            self.config["audit_log_enabled"] = False

    def _create_handlers(self) -> None:
        """Create log handlers"""
        try:
            # Console handler
            if self.config.get("to_console", True):
                console_handler = logging.StreamHandler(sys.stderr)
                # Convert string level to logging level constant
                level_str = self.config.get("level", "INFO")
                level = getattr(logging, level_str.upper(), logging.INFO)
                console_handler.setLevel(level)

                formatter: Union[StructuredFormatter, logging.Formatter]
                if self.config.get("structured", False):
                    formatter = StructuredFormatter()
                else:
                    formatter = logging.Formatter(
                        self.config.get("format"), self.config.get("date_format")
                    )
                console_handler.setFormatter(formatter)

                # Add module filter
                if self.config.get("filters"):
                    module_filter = ModuleFilter(
                        self.config.get("filters", []),
                        self.config.get("filtered_level", "WARNING"),
                    )
                    console_handler.addFilter(module_filter)

                self.handlers["console"] = console_handler

            # Main file handler
            if self.config.get("to_file", True):
                level_str = self.config.get("level", "INFO")
                level = getattr(logging, level_str.upper(), logging.INFO)
                self._create_file_handler("main", self.config.get("log_file"), level)

            # Performance log handler
            if self.config.get("performance_enabled", True):
                self._create_file_handler(
                    "performance", self.config.get("performance_log_file")
                )

            # Error log handler
            if self.config.get("error_log_enabled", True):
                self._create_file_handler(
                    "error", self.config.get("error_log_file"), logging.ERROR
                )

            # Audit log handler
            if self.config.get("audit_log_enabled", True):
                self._create_file_handler("audit", self.config.get("audit_log_file"))

        except Exception as e:
            print(f"Handler creation failed: {e}", file=sys.stderr)

    def _create_file_handler(
        self, name: str, log_file: Optional[str], min_level: int = logging.DEBUG
    ) -> Optional[logging.Handler]:
        """Create file handler"""
        if not log_file:
            return None

        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.get("max_bytes", 10 * 1024 * 1024),
                backupCount=self.config.get("backup_count", 5),
                encoding="utf-8",
            )
            file_handler.setLevel(min_level)

            formatter: Union[StructuredFormatter, logging.Formatter]
            if self.config.get("structured", False):
                formatter = StructuredFormatter()
            else:
                formatter = logging.Formatter(
                    self.config.get("format"), self.config.get("date_format")
                )
            file_handler.setFormatter(formatter)

            # Add context filter
            if self.config.get("context_enabled", True):
                context_filter = ContextFilter()
                file_handler.addFilter(context_filter)

            self.handlers[name] = file_handler
            return file_handler

        except Exception as e:
            print(f"Failed to create file handler {name}: {e}", file=sys.stderr)
            return None

    def get_logger(self, name: str) -> logging.Logger:
        """Get logger"""
        if name in self.loggers:
            return self.loggers[name]

        try:
            logger = logging.getLogger(name)

            # Set log level
            level = self.config.get("level", "INFO")
            if name in self.config.get("module_levels", {}):
                level = self.config["module_levels"][name]
            logger.setLevel(getattr(logging, level.upper(), logging.INFO))

            # Add handlers
            for handler_name, handler in self.handlers.items():
                if handler not in logger.handlers:
                    # Select appropriate handler by log type
                    if self._should_add_handler(name, handler_name):
                        logger.addHandler(handler)

            # Prevent duplicate logs
            logger.propagate = False

            self.loggers[name] = logger
            return logger

        except Exception as e:
            print(f"Failed to get logger: {e}", file=sys.stderr)
            # Return basic Python logger
            return logging.getLogger(name)

    def _should_add_handler(self, logger_name: str, handler_name: str) -> bool:
        """Determine whether to add a specific handler to logger"""
        # Console and main file handlers are added to all loggers
        if handler_name in ["console", "main"]:
            return True

        # Performance log handler is only added to performance-related loggers
        if handler_name == "performance":
            return logger_name in [
                "performance",
                "app.core.rule_engine",
                "app.services",
            ]

        # Error log handler is added to all loggers (controlled by level)
        if handler_name == "error":
            return True

        # Audit log handler is only added to audit-related loggers
        if handler_name == "audit":
            return logger_name in ["audit", "app.services", "app.api"]

        return False

    def log_performance(
        self, operation_name: str, duration: float, **kwargs: Any
    ) -> None:
        """Record performance log"""
        try:
            if self.performance_logger:
                self.performance_logger.log_operation(
                    operation_name, duration, **kwargs
                )
        except Exception:
            # Performance log failure should not affect main flow
            pass

    def log_audit(self, user: str, action: str, message: str, **kwargs: Any) -> None:
        """Record audit log"""
        try:
            audit_logger = self.get_logger("audit")
            extra: Dict[str, Any] = {
                "audit_log": True,
                "user": user,
                "action": action,
                "timestamp": datetime.now().isoformat(),
                **kwargs,
            }
            audit_logger.info(message, extra=extra)
        except Exception:
            # Audit log failure should not affect main flow
            pass

    def log_database_operation(
        self,
        operation: str,
        table: str,
        duration: float,
        success: bool = True,
        **kwargs: Any,
    ) -> None:
        """Record database operation log"""
        try:
            db_logger = self.get_logger("database")
            extra: Dict[str, Any] = {
                "database_log": True,
                "operation": operation,
                "table": table,
                "duration": duration,
                "success": success,
                **kwargs,
            }

            if success:
                db_logger.info(
                    f"Database operation: {operation} on {table} took {duration:.3f}s",
                    extra=extra,
                )
            else:
                db_logger.error(
                    f"Database operation failed: {operation} on {table} "
                    f"took {duration:.3f}s",
                    extra=extra,
                )
        except Exception:
            # Database log failure should not affect main flow
            pass

    def log_rule_execution(
        self,
        rule_id: str,
        rule_type: str,
        table: str,
        duration: float,
        result_count: int,
        **kwargs: Any,
    ) -> None:
        """Record rule execution log"""
        try:
            rule_logger = self.get_logger("rule_execution")
            extra: Dict[str, Any] = {
                "rule_execution_log": True,
                "rule_id": rule_id,
                "rule_type": rule_type,
                "table": table,
                "duration": duration,
                "result_count": result_count,
                **kwargs,
            }
            rule_logger.info(
                f"Rule execution: {rule_type}({rule_id}) on {table} took "
                f"{duration:.3f}s, result count: {result_count}",
                extra=extra,
            )
        except Exception:
            # Rule execution log failure should not affect main flow
            pass

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update config"""
        try:
            self.config.update(new_config)
            # Re-set logging system
            self._setup_complete = False
            self._setup_logging()
            # Clear cached loggers
            self.loggers.clear()
        except Exception as e:
            print(f"Failed to update config: {e}", file=sys.stderr)


def get_logger_manager() -> LoggerManager:
    """Get global logger manager instance"""
    global _logger_manager, _initialized

    if _logger_manager is None:
        with _lock:
            if _logger_manager is None and not _initialized:
                try:
                    _initialized = True
                    # Use default config to initialize, avoid circular dependency
                    _logger_manager = LoggerManager()
                except Exception as e:
                    print(f"Failed to initialize logger manager: {e}", file=sys.stderr)
                    # Create a basic manager
                    _logger_manager = LoggerManager(
                        {"to_file": False, "to_console": True}
                    )

    if _logger_manager is None:
        raise RuntimeError("Logger manager is not initialized")

    return _logger_manager


def get_logger(name: str) -> logging.Logger:
    """Get logger"""
    try:
        manager = get_logger_manager()
        return manager.get_logger(name)
    except Exception as e:
        print(f"Failed to get logger: {e}", file=sys.stderr)
        # Return basic Python logger
        return logging.getLogger(name)


def log_performance(operation_name: str, duration: float, **kwargs: Any) -> None:
    """Record performance log"""
    try:
        manager = get_logger_manager()
        manager.log_performance(operation_name, duration, **kwargs)
    except Exception:
        # Performance log failure should not affect main flow
        pass


def log_audit(user: str, action: str, message: str, **kwargs: Any) -> None:
    """Record audit log"""
    try:
        manager = get_logger_manager()
        manager.log_audit(user, action, message, **kwargs)
    except Exception:
        # Audit log failure should not affect main flow
        pass


def log_database_operation(
    operation: str, table: str, duration: float, success: bool = True, **kwargs: Any
) -> None:
    """Record database operation log"""
    try:
        manager = get_logger_manager()
        manager.log_database_operation(operation, table, duration, success, **kwargs)
    except Exception:
        # Database log failure should not affect main flow
        pass


def log_rule_execution(
    rule_id: str,
    rule_type: str,
    table: str,
    duration: float,
    result_count: int,
    **kwargs: Any,
) -> None:
    """Record rule execution log"""
    try:
        manager = get_logger_manager()
        manager.log_rule_execution(
            rule_id, rule_type, table, duration, result_count, **kwargs
        )
    except Exception:
        # Rule execution log failure should not affect main flow
        pass


def performance_monitor(operation_name: Optional[str] = None) -> Callable:
    """Performance monitoring decorator"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                log_performance(op_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                log_performance(op_name, duration, error=str(e))
                raise

        return wrapper

    return decorator


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """Set up logging system"""
    global _logger_manager, _initialized
    with _lock:
        try:
            _initialized = False  # Reset initialization flag
            if config:
                _logger_manager = LoggerManager(config)
            else:
                _logger_manager = LoggerManager()
            _initialized = True
        except Exception as e:
            print(f"Failed to set up logging system: {e}", file=sys.stderr)
            _logger_manager = LoggerManager({"to_file": False, "to_console": True})
            _initialized = True


# Backward compatible function
def configure_logging() -> None:
    """Configure logging system (backward compatible)"""
    setup_logging()


# Simplified direct function, avoid complex initialization
def get_simple_logger(name: str) -> logging.Logger:
    """Get simple logger, avoid complex initialization"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
