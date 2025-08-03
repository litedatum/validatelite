"""
Base schema module

Defines the base Pydantic model class and core business schema base classes.
All schema models should inherit from these base classes to ensure consistent behavior.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
)

from shared.enums import (
    ConnectionType,
    RuleAction,
    RuleCategory,
    RuleType,
    SeverityLevel,
)
from shared.exceptions.exception_system import OperationError
from shared.utils.datetime_utils import format_datetime
from shared.utils.logger import get_logger

logger = get_logger(__name__)


class BaseSchema(BaseModel):
    """
    Base Pydantic model

    All Pydantic models should inherit from this class to ensure consistent behavior:
    - Enums are preserved as enum objects internally but serialized to their values
    - Support for creating models from ORM objects
    - Datetime fields are serialized to ISO 8601 format with Z suffix
    """

    model_config = ConfigDict(
        from_attributes=True,  # Replaces the old orm_mode
        # Don't use use_enum_values=True to preserve enum objects internally
        arbitrary_types_allowed=True,
        extra="allow",  # Permit and store unknown fields (e.g., rule_name for tests)
    )

    @field_serializer("*")
    def serialize_fields(self, value: Any, _info: Any) -> Any:
        """
        Serialize all fields: datetime fields to ISO 8601 format,
        enum fields to their values

        Args:
            value: Field value
            _info: Field info

        Returns:
            Serialized value
        """
        if isinstance(value, datetime):
            return format_datetime(value)
        elif isinstance(value, Enum):
            return value.value
        return value


class ExecutionStrategy(str, Enum):
    """Execution strategy enumeration - full roadmap"""

    # Currently supported version
    SQL_NATIVE = "sql_native"

    # Reserved strategies - cross-database features
    MEMORY_DATAFRAME = "memory_dataframe"
    LOCAL_TEMP_DB = "local_temp_db"
    DISTRIBUTED_SPARK = "distributed_spark"
    STREAMING_FLINK = "streaming_flink"

    @classmethod
    def get_current_supported(cls) -> List["ExecutionStrategy"]:
        """Get currently supported strategies"""
        return [cls.SQL_NATIVE]

    @classmethod
    def is_cross_db_strategy(cls, strategy: "ExecutionStrategy") -> bool:
        """Determine if it is a cross-database strategy"""
        return strategy != cls.SQL_NATIVE


class DataSourceCapability(BaseSchema):
    """Data source capability declaration - reserved for hooks"""

    supports_sql: bool = True
    supports_batch_export: bool = False  # Reserved
    max_export_rows: Optional[int] = None  # Reserved
    supports_streaming: bool = False  # Reserved
    estimated_throughput: Optional[int] = None  # Reserved


class TargetEntity(BaseSchema):
    """Data entity definition - supports single and multiple tables"""

    database: str = Field(..., description="Database name")
    table: str = Field(..., description="Table name")
    column: Optional[str] = Field(
        None, description="Column name, empty for table-level rules"
    )

    # Reserved fields for hooks
    connection_id: Optional[str] = Field(
        None, description="Connection identifier, used for cross-database"
    )
    alias: Optional[str] = Field(None, description="Alias, used for multi-table rules")

    @property
    def full_name(self) -> str:
        """Full entity name"""
        if self.column:
            return f"{self.database}.{self.table}.{self.column}"
        return f"{self.database}.{self.table}"


class RuleTarget(BaseSchema):
    """Rule target definition - unified for single and multi-table"""

    entities: List[TargetEntity] = Field(..., description="Target entity list")

    # Reserved fields for hooks
    relationship_type: Optional[str] = Field(
        None, description="Relationship type: single_table/foreign_key/aggregation"
    )
    join_conditions: List[str] = Field(
        default_factory=list, description="Join conditions, used for multi-table rules"
    )

    @property
    def is_single_table(self) -> bool:
        """Whether it is a single-table rule"""
        if len(self.entities) == 1:
            return True
        # Multiple entities but same table (e.g., multi-column rules)
        first = self.entities[0]
        return all(
            e.database == first.database and e.table == first.table
            for e in self.entities
        )

    @property
    def is_multi_table(self) -> bool:
        """Whether it is a multi-table rule"""
        return not self.is_single_table

    @property
    def is_cross_database(self) -> bool:
        """Whether it is cross-database

        - reserved property, always False in current version
        """
        if self.is_single_table:
            return False
        # Check if there are different connection_ids
        connection_ids = set(e.connection_id for e in self.entities if e.connection_id)
        return len(connection_ids) > 1

    @property
    def primary_entity(self) -> TargetEntity:
        """Primary entity (the first one)"""
        return self.entities[0]


class CrossDbParameters(BaseSchema):
    """Cross-database parameters - reserved for hooks"""

    execution_strategy: ExecutionStrategy = ExecutionStrategy.MEMORY_DATAFRAME

    # Basic parameters
    sampling_enabled: bool = False
    sample_size: Optional[int] = None
    timeout_seconds: int = 300

    # Reserved parameters
    parallel_degree: Optional[int] = None
    memory_limit_mb: Optional[int] = None
    temp_storage_config: Optional[Dict[str, Any]] = None


class RuleBase(BaseSchema):
    """
    Rule base model - refactored rule base model

    Integrates target definition and parameters, prepared for future expansion
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Rule unique identifier, auto-generated by the system",
    )
    name: str = Field(..., min_length=1, max_length=100, description="Rule name")
    description: Optional[str] = Field(
        None, max_length=500, description="Rule description"
    )

    # Core definition
    type: RuleType = Field(..., description="Rule type")
    target: RuleTarget = Field(..., description="Rule target")

    # Rule parameters (integrated from original params field)
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Rule parameters"
    )

    # Reserved: cross-database configuration
    cross_db_config: Optional[CrossDbParameters] = Field(
        None, description="Cross-database configuration, currently null"
    )

    # Execution configuration
    threshold: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Success threshold"
    )

    # Category and action
    category: RuleCategory = Field(..., description="Rule category")
    severity: SeverityLevel = Field(..., description="Severity level")
    action: RuleAction = Field(..., description="Action to execute")

    # Metadata
    is_active: bool = Field(True, description="Whether active")
    tags: Optional[List[str]] = Field(None, description="Tags")
    template_id: Optional[UUID] = Field(None, description="Associated template ID")
    validation_error: Optional[str] = Field(
        None, description="Validation error message, used for execution result"
    )


class DatasetMetrics(BaseSchema):
    """Dataset metrics - supports multiple datasets"""

    entity_name: str = Field(..., description="Entity name, format: database.table")
    total_records: int = Field(..., description="Total record count")
    failed_records: int = Field(0, description="Failed record count")

    # Reserved fields for hooks
    processing_time: Optional[float] = Field(
        None, description="Processing time, used for cross-database"
    )

    @property
    def success_rate(self) -> float:
        """Success rate"""
        if self.total_records == 0:
            return 1.0
        return (self.total_records - self.failed_records) / self.total_records


class CrossDbMetrics(BaseSchema):
    """Cross-database metrics - reserved for hooks"""

    strategy_used: str = Field(..., description="Execution strategy used")
    data_transfer_time: Optional[float] = Field(None, description="Data transfer time")
    total_processing_time: Optional[float] = Field(
        None, description="Total processing time"
    )
    temp_data_size_mb: Optional[int] = Field(None, description="Temporary data size")


class ExecutionResultBase(BaseSchema):
    """Execution result base model - refactored to support multiple datasets"""

    rule_id: str = Field(..., description="Rule ID")
    status: str = Field(..., description="Execution status")

    # Refactored: use dataset metrics instead of single statistics
    dataset_metrics: List[DatasetMetrics] = Field(..., description="Dataset metrics")

    # Execution info
    execution_time: float = Field(..., description="Execution time (seconds)")
    execution_message: Optional[str] = Field(None, description="Execution message")
    error_message: Optional[str] = Field(None, description="Error message")

    # Sample data
    sample_data: Optional[List[Dict[str, Any]]] = Field(
        None, description="Failed sample data"
    )

    # Reserved: cross-database metrics
    cross_db_metrics: Optional[CrossDbMetrics] = Field(
        None, description="Cross-database metrics, currently null"
    )

    # Reserved: execution plan
    execution_plan: Optional[Dict[str, Any]] = Field(
        None, description="Execution plan details, for advanced features"
    )

    # Timestamps
    started_at: Optional[datetime] = Field(None, description="Start time")
    ended_at: Optional[datetime] = Field(None, description="End time")

    # Compatibility properties: calculated from primary dataset metrics
    @property
    def total_count(self) -> int:
        """Total record count - backward compatible"""
        return sum(dm.total_records for dm in self.dataset_metrics)

    @property
    def error_count(self) -> int:
        """Error record count - backward compatible"""
        return sum(dm.failed_records for dm in self.dataset_metrics)

    @property
    def error_rate(self) -> float:
        """Error rate - backward compatible"""
        total = self.total_count
        return self.error_count / total if total > 0 else 0.0

    @property
    def primary_dataset(self) -> Optional[DatasetMetrics]:
        """Primary dataset metrics"""
        return self.dataset_metrics[0] if self.dataset_metrics else None


class ConnectionBase(BaseSchema):
    """Connection base model - adds cross-database support hooks"""

    name: str = Field(..., min_length=1, max_length=100, description="Connection name")
    description: Optional[str] = Field(
        None, max_length=500, description="Connection description"
    )
    connection_type: ConnectionType = Field(..., description="Connection type")

    # Database connection parameters
    host: Optional[str] = Field(None, description="Host address")
    port: Optional[int] = Field(None, description="Port")
    db_name: Optional[str] = Field(None, description="Database name")
    username: Optional[str] = Field(None, description="Username")
    password: Optional[str] = Field(None, description="Password")
    db_schema: Optional[str] = Field(None, description="Database Schema")

    # File connection parameters
    file_path: Optional[str] = Field(None, description="File path")

    # Extension parameters
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional connection parameters"
    )

    # Reserved: data source capabilities
    capabilities: DataSourceCapability = Field(
        default_factory=DataSourceCapability,
        description="Data source capability declaration",
    )

    # Reserved: cross-database settings
    cross_db_settings: Optional[Dict[str, Any]] = Field(
        None, description="Cross-database settings, currently null"
    )

    # ---------------------------------------------------------------------
    # Validators
    # ---------------------------------------------------------------------

    @field_validator("connection_type", mode="before")
    @classmethod
    def _parse_connection_type(cls, v: Any) -> ConnectionType:
        """Allow case-insensitive string values for ``connection_type``.

        The modern CLI and its test-suite frequently supply plain strings such
        as "CSV" instead of ``ConnectionType`` enum members.  This validator
        converts such strings to the corresponding enum in a tolerant manner
        using :py:meth:`ConnectionType.from_string`.
        """
        if isinstance(v, ConnectionType):
            return v
        if isinstance(v, str):
            return ConnectionType.from_string(v)
        raise OperationError("connection_type must be a str or ConnectionType instance")

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Validate host"""
        conn_type_value = info.data.get("connection_type")
        if isinstance(conn_type_value, ConnectionType):
            conn_type_value = conn_type_value.value

        if conn_type_value in [
            ConnectionType.MYSQL.value,
            ConnectionType.POSTGRESQL.value,
            ConnectionType.MSSQL.value,
            ConnectionType.ORACLE.value,
        ]:
            if not v:
                raise OperationError(
                    f"Host is required for {conn_type_value} connections"
                )
        return v

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: Optional[int], info: ValidationInfo) -> Optional[int]:
        """Validate port"""
        conn_type_value = info.data.get("connection_type")
        if isinstance(conn_type_value, ConnectionType):
            conn_type_value = conn_type_value.value

        if conn_type_value in [
            ConnectionType.MYSQL.value,
            ConnectionType.POSTGRESQL.value,
            ConnectionType.MSSQL.value,
            ConnectionType.ORACLE.value,
        ]:
            if not v:
                raise OperationError(
                    f"Port is required for {conn_type_value} connections"
                )

            # 👻 GHOST FIX: Add port range validation
            if v < 1 or v > 65535:
                raise OperationError(f"Port must be between 1 and 65535, got {v}")

        return v

    def supports_cross_db_comparison(self) -> bool:
        """Whether cross-database comparison is supported

        - always False in current version
        """
        return False
