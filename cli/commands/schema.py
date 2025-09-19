"""
Schema Command

Adds `vlite schema` command that parses parameters, performs minimal rules
file validation (supports both single-table and multi-table formats), and prints
output aligned with the existing CLI style.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import click

from cli.core.data_validator import DataValidator
from cli.core.source_parser import SourceParser
from shared.database.database_dialect import DatabaseDialectFactory
from shared.enums import RuleAction, RuleCategory, RuleType, SeverityLevel
from shared.enums.connection_types import ConnectionType
from shared.enums.data_types import DataType
from shared.schema.base import RuleTarget, TargetEntity
from shared.schema.connection_schema import ConnectionSchema
from shared.schema.rule_schema import RuleSchema
from shared.utils.console import safe_echo
from shared.utils.datetime_utils import now as _now
from shared.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CompatibilityResult:
    """Result of type compatibility analysis between native and desired types."""

    field_name: str
    table_name: str
    native_type: str
    desired_type: str
    compatibility: Literal["COMPATIBLE", "INCOMPATIBLE", "CONFLICTING"]
    reason: Optional[str] = None
    required_validation: Optional[str] = None  # "LENGTH", "REGEX", "DATE_FORMAT"
    validation_params: Optional[Dict[str, Any]] = None


class CompatibilityAnalyzer:
    """
    Analyzes type compatibility between native database types and desired types.

    Implements the compatibility matrix from the design document to determine:
    - COMPATIBLE: Skip desired_type validation (type conversions that always work)
    - INCOMPATIBLE: Require data validation (type conversions needing checks)
    - CONFLICTING: Report error immediately (impossible conversions)
    """

    def __init__(self, connection_type: ConnectionType):
        """Initialize with database connection type for dialect-specific patterns."""
        self.connection_type = connection_type
        # Map ConnectionType to DatabaseDialectFactory database type
        dialect_type_mapping = {
            ConnectionType.MYSQL: "mysql",
            ConnectionType.POSTGRESQL: "postgresql",
            ConnectionType.SQLITE: "sqlite",
            ConnectionType.MSSQL: "sqlserver",
        }
        dialect_type = dialect_type_mapping.get(connection_type)
        if dialect_type:
            self.dialect = DatabaseDialectFactory.get_dialect(dialect_type)
        else:
            # Fallback to MySQL for unsupported database types
            self.dialect = DatabaseDialectFactory.get_dialect("mysql")

    def analyze(
        self,
        native_type: str,
        desired_type: str,
        field_name: str,
        table_name: str,
        native_metadata: Optional[Dict[str, Any]] = None,
    ) -> CompatibilityResult:
        """
        Analyze compatibility between native and desired types.

        Args:
            native_type: Native database type (canonical, e.g. "STRING")
            desired_type: Desired type (canonical, e.g. "INTEGER")
            field_name: Name of the field being analyzed
            table_name: Name of the table containing the field
            native_metadata: Native type metadata (max_length, precision, etc.)

        Returns:
            CompatibilityResult with compatibility status and validation requirements
        """
        native_metadata = native_metadata or {}
        # Parse types using TypeParser to get canonical base types
        from shared.utils.type_parser import TypeParseError, TypeParser

        try:
            # For native type, it might already be canonical (e.g., "STRING")
            if str(native_type).upper() in [
                "STRING",
                "INTEGER",
                "FLOAT",
                "BOOLEAN",
                "DATE",
                "DATETIME",
            ]:
                native_canonical = str(native_type).upper()
            else:
                # Try to parse it as a type definition
                try:
                    native_parsed = TypeParser.parse_type_definition(str(native_type))
                    native_canonical = native_parsed.get(
                        "type", str(native_type)
                    ).upper()
                except Exception:
                    native_canonical = str(native_type).upper()
        except Exception:
            native_canonical = str(native_type).upper()

        try:
            # Parse desired_type to get base type
            desired_parsed = TypeParser.parse_type_definition(str(desired_type))
            desired_canonical = desired_parsed.get("type", str(desired_type)).upper()
        except TypeParseError:
            # Fallback to string comparison
            desired_canonical = str(desired_type).upper()

        # Same canonical type might still need validation if constraints are stricter
        if native_canonical == desired_canonical:
            # For STRING types, check if length constraints require validation
            if native_canonical == "STRING":
                try:
                    # Use native_metadata directly for native type constraints
                    native_max_length = native_metadata.get("max_length")

                    # Parse desired type to get constraints
                    desired_parsed = TypeParser.parse_type_definition(str(desired_type))
                    desired_max_length = desired_parsed.get("max_length")

                    # If desired type has stricter length constraint,
                    # validation is needed
                    if desired_max_length is not None:
                        if (
                            native_max_length is None
                            or native_max_length > desired_max_length
                        ):
                            return CompatibilityResult(
                                field_name=field_name,
                                table_name=table_name,
                                native_type=native_type,
                                desired_type=desired_type,
                                compatibility="INCOMPATIBLE",
                                reason=(
                                    f"Length constraint tightening: "
                                    f"{native_max_length or 'unlimited'} -> "
                                    f"{desired_max_length}"
                                ),
                                required_validation="LENGTH",
                                validation_params={
                                    "max_length": desired_max_length,
                                    "description": (
                                        f"Length validation for max "
                                        f"{desired_max_length} characters"
                                    ),
                                },
                            )
                except Exception:
                    # If parsing fails, fall back to compatible
                    pass

            # For INTEGER types, check if precision constraints require validation
            if native_canonical == "INTEGER":
                try:
                    # Parse desired type to get constraints
                    desired_parsed = TypeParser.parse_type_definition(str(desired_type))
                    desired_max_digits = desired_parsed.get(
                        "max_digits"
                    )  # For INTEGER constraints
                    desired_precision = desired_parsed.get(
                        "precision"
                    )  # For FLOAT constraints

                    if (
                        desired_canonical == "INTEGER"
                        and desired_max_digits is not None
                    ):
                        # INTEGER → INTEGER with digit constraint - use REGEX validation
                        pattern = self.dialect.generate_integer_regex_pattern(
                            desired_max_digits
                        )
                        return CompatibilityResult(
                            field_name=field_name,
                            table_name=table_name,
                            native_type=native_type,
                            desired_type=desired_type,
                            compatibility="INCOMPATIBLE",
                            reason=(
                                f"INTEGER precision constraint: unlimited -> "
                                f"{desired_max_digits} digits"
                            ),
                            required_validation="REGEX",
                            validation_params={
                                "pattern": pattern,
                                "description": (
                                    f"Integer precision validation for max "
                                    f"{desired_max_digits} digits"
                                ),
                            },
                        )
                except Exception:
                    # If parsing fails, fall back to compatible
                    pass

            # For FLOAT types, check if precision/scale constraints require validation
            if native_canonical == "FLOAT":
                try:
                    # Get native precision/scale from metadata
                    # These are extracted but not used in current logic
                    _ = native_metadata.get("precision")  # native_precision
                    _ = native_metadata.get("scale")  # native_scale

                    # Parse desired type to get constraints
                    desired_parsed = TypeParser.parse_type_definition(str(desired_type))
                    desired_precision = desired_parsed.get("precision")
                    desired_scale = desired_parsed.get("scale")

                    if desired_canonical == "FLOAT" and desired_precision is not None:
                        # FLOAT → FLOAT with precision/scale constraints
                        # For desired_type validation, always enforce constraints
                        # regardless of native metadata
                        # because actual data may not conform to
                        # database-reported constraints
                        scale = desired_scale or 0
                        integer_digits = desired_precision - scale
                        pattern = self.dialect.generate_float_regex_pattern(
                            desired_precision, scale
                        )

                        return CompatibilityResult(
                            field_name=field_name,
                            table_name=table_name,
                            native_type=native_type,
                            desired_type=desired_type,
                            compatibility="INCOMPATIBLE",
                            reason=(
                                f"FLOAT precision/scale constraint validation: "
                                f"desired ({desired_precision},{scale})"
                            ),
                            required_validation="REGEX",
                            validation_params={
                                "pattern": pattern,
                                "description": (
                                    f"Float precision/scale validation for "
                                    f"({desired_precision},{scale})"
                                ),
                            },
                        )
                except Exception:
                    # If parsing fails, fall back to compatible
                    pass

            # Same canonical type with no stricter constraints
            return CompatibilityResult(
                field_name=field_name,
                table_name=table_name,
                native_type=native_type,
                desired_type=desired_type,
                compatibility="COMPATIBLE",
                reason="Same canonical type with compatible constraints",
            )

        # Implement compatibility matrix from design document
        compatibility_matrix = {
            ("STRING", "STRING"): "COMPATIBLE",
            ("STRING", "INTEGER"): "INCOMPATIBLE",
            ("STRING", "FLOAT"): "INCOMPATIBLE",
            (
                "STRING",
                "DATE",
            ): "INCOMPATIBLE",  # String to Date requires date format validation
            ("STRING", "DATETIME"): "INCOMPATIBLE",
            ("INTEGER", "STRING"): "COMPATIBLE",
            ("INTEGER", "INTEGER"): "COMPATIBLE",
            ("INTEGER", "FLOAT"): "COMPATIBLE",
            (
                "INTEGER",
                "DATE",
            ): "INCOMPATIBLE",  # Integer to Date requires date format validation
            ("INTEGER", "DATETIME"): "INCOMPATIBLE",
            ("FLOAT", "STRING"): "COMPATIBLE",
            ("FLOAT", "INTEGER"): "INCOMPATIBLE",
            ("FLOAT", "FLOAT"): "COMPATIBLE",
            ("FLOAT", "DATE"): "CONFLICTING",  # Float to Date is not supported
            ("FLOAT", "DATETIME"): "CONFLICTING",
            ("DATE", "STRING"): "COMPATIBLE",
            ("DATE", "INTEGER"): "CONFLICTING",  # Date to Integer is not supported
            ("DATE", "FLOAT"): "CONFLICTING",  # Date to Float is not supported
            ("DATE", "DATE"): "COMPATIBLE",
            ("DATE", "DATETIME"): "COMPATIBLE",  # Date can be expanded to DateTime
            ("DATETIME", "STRING"): "COMPATIBLE",
            ("DATETIME", "INTEGER"): "CONFLICTING",
            ("DATETIME", "FLOAT"): "CONFLICTING",
            ("DATETIME", "DATE"): "COMPATIBLE",  # DateTime can be truncated to Date
            ("DATETIME", "DATETIME"): "COMPATIBLE",
        }

        compatibility_key = (native_canonical, desired_canonical)
        compatibility_status = cast(
            Literal["COMPATIBLE", "INCOMPATIBLE", "CONFLICTING"],
            compatibility_matrix.get(compatibility_key, "CONFLICTING"),
        )

        result = CompatibilityResult(
            field_name=field_name,
            table_name=table_name,
            native_type=native_type,
            desired_type=desired_type,
            compatibility=compatibility_status,
            reason=self._get_compatibility_reason(
                native_canonical, desired_canonical, compatibility_status
            ),
        )

        # For incompatible cases, determine required validation type
        if compatibility_status == "INCOMPATIBLE":
            validation_type, validation_params = (
                self._determine_validation_requirements(
                    native_canonical, desired_canonical, desired_type
                )
            )
            result.required_validation = validation_type
            result.validation_params = validation_params

        # Check for cross-type numeric constraints (even for COMPATIBLE cases)
        if (
            compatibility_status == "COMPATIBLE"
            and native_canonical == "INTEGER"
            and desired_canonical == "FLOAT"
        ):
            try:
                # Parse desired FLOAT type to get precision/scale constraints
                desired_parsed = TypeParser.parse_type_definition(str(desired_type))
                desired_precision = desired_parsed.get("precision")

                if desired_precision is not None:
                    desired_scale = desired_parsed.get("scale", 0)
                    integer_digits = desired_precision - desired_scale

                    if integer_digits > 0:
                        # Override compatibility status for cross-type precision
                        # constraints
                        pattern = self.dialect.generate_integer_regex_pattern(
                            integer_digits
                        )
                        result.compatibility = "INCOMPATIBLE"
                        result.reason = (
                            f"Cross-type precision constraint: INTEGER -> "
                            f"FLOAT({desired_precision},{desired_scale}) "
                            f"allows max {integer_digits} integer digits"
                        )
                        result.required_validation = "REGEX"
                        result.validation_params = {
                            "pattern": pattern,
                            "description": (
                                f"Cross-type integer-to-float precision validation "
                                f"for max {integer_digits} integer digits"
                            ),
                        }
            except Exception:
                # If parsing fails, keep original compatibility status
                pass

        # Check for cross-type length constraints (even for COMPATIBLE cases)
        if compatibility_status == "COMPATIBLE" and desired_canonical == "STRING":
            try:
                # Parse desired type to get constraints
                desired_parsed = TypeParser.parse_type_definition(str(desired_type))
                desired_max_length = desired_parsed.get("max_length")

                # If desired STRING type has length constraint, need validation for
                # cross-type conversions
                if desired_max_length is not None and native_canonical != "STRING":
                    # Override compatibility status for cross-type length constraints
                    result.compatibility = "INCOMPATIBLE"
                    result.reason = (
                        f"Cross-type length constraint: {native_canonical} -> "
                        f"STRING({desired_max_length})"
                    )
                    result.required_validation = "LENGTH"
                    result.validation_params = {
                        "max_length": desired_max_length,
                        "description": (
                            f"Cross-type length validation for max "
                            f"{desired_max_length} characters"
                        ),
                    }
            except Exception:
                # If parsing fails, keep original compatibility status
                pass

        return result

    @classmethod
    def _get_compatibility_reason(cls, native: str, desired: str, status: str) -> str:
        """Generate human-readable reason for compatibility status."""
        if status == "COMPATIBLE":
            if native == desired:
                return "Same canonical type"
            else:
                return f"{native} can be safely converted to {desired}"
        elif status == "INCOMPATIBLE":
            return f"{native} to {desired} conversion requires data validation"
        else:  # CONFLICTING
            return f"{native} to {desired} conversion is not supported"

    def _determine_validation_requirements(
        self, native: str, desired: str, desired_type_definition: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Determine what type of validation rules are needed for incompatible conversions.

        Returns:
            Tuple of (validation_type, validation_params) where:
            - validation_type: "LENGTH", "REGEX", "DATE_FORMAT", or "PRECISION"
            - validation_params: Parameters for the validation rule
        """
        if native == "STRING" and desired == "INTEGER":
            # String to integer needs regex validation
            pattern = self.dialect.generate_basic_integer_pattern()
            return "REGEX", {
                "pattern": pattern,
                "description": "Integer format validation",
            }

        elif native == "STRING" and desired == "FLOAT":
            # String to float needs regex validation
            pattern = self.dialect.generate_basic_float_pattern()
            return "REGEX", {
                "pattern": pattern,
                "description": "Float format validation",
            }

        elif native == "STRING" and desired == "DATE":
            # String to date needs date format validation
            format_pattern = "YYYY-MM-DD"  # default
            if desired_type_definition:
                try:
                    from shared.utils.type_parser import TypeParser

                    parsed = TypeParser.parse_type_definition(desired_type_definition)
                    format_pattern = parsed.get("format", format_pattern)
                except Exception:
                    pass  # use default if parsing fails
            return "DATE_FORMAT", {
                "format_pattern": format_pattern,
                "description": "String date format validation",
            }

        elif native == "STRING" and desired == "DATETIME":
            # String to datetime needs date format validation
            format_pattern = "YYYY-MM-DD"  # default
            if desired_type_definition:
                try:
                    from shared.utils.type_parser import TypeParser

                    parsed = TypeParser.parse_type_definition(desired_type_definition)
                    format_pattern = parsed.get("format", format_pattern)
                except Exception:
                    pass  # use default if parsing fails
            return "DATE_FORMAT", {
                "format_pattern": format_pattern,
                "description": "String datetime format validation",
            }

        elif native == "INTEGER" and desired == "DATE":
            # Integer to date needs date format validation
            format_pattern = "YYYYMMDD"  # default
            if desired_type_definition:
                try:
                    from shared.utils.type_parser import TypeParser

                    parsed = TypeParser.parse_type_definition(desired_type_definition)
                    format_pattern = parsed.get("format", format_pattern)
                except Exception:
                    pass  # use default if parsing fails
            return "DATE_FORMAT", {
                "format_pattern": format_pattern,
                "description": "Integer date format validation",
            }

        elif native == "INTEGER" and desired == "DATETIME":
            # Integer to datetime needs date format validation
            format_pattern = "YYYYMMDD"  # default
            if desired_type_definition:
                try:
                    from shared.utils.type_parser import TypeParser

                    parsed = TypeParser.parse_type_definition(desired_type_definition)
                    format_pattern = parsed.get("format", format_pattern)
                except Exception:
                    pass  # use default if parsing fails
            return "DATE_FORMAT", {
                "format_pattern": format_pattern,
                "description": "Integer datetime format validation",
            }

        elif native == "FLOAT" and desired == "INTEGER":
            # Float to integer needs validation that it's actually an integer value
            # Check if there are precision constraints (e.g., integer(2))
            if desired_type_definition:
                try:
                    from shared.utils.type_parser import TypeParser

                    parsed = TypeParser.parse_type_definition(desired_type_definition)
                    max_digits = parsed.get("max_digits")

                    if max_digits is not None:
                        # Generate pattern that checks both integer-like and digit limit
                        pattern = f"^-?[0-9]{{1,{max_digits}}}\\.0*$"
                        return "REGEX", {
                            "pattern": pattern,
                            "description": f"Integer-like float validation with max "
                            f"{max_digits} digits",
                        }
                except Exception:
                    pass  # Fall back to basic validation if parsing fails

            # Default: basic integer-like float validation
            pattern = self.dialect.generate_integer_like_float_pattern()
            return "REGEX", {
                "pattern": pattern,
                "description": "Integer-like float validation",
            }

        # Note: PRECISION validation types are handled by generating REGEX patterns
        # This is called from compatibility analysis when precision/scale
        # constraints are detected

        # Default: no specific validation requirements determined
        return None, None


class DesiredTypeRuleGenerator:
    """
    Generates validation rules for incompatible type conversions based on analysis.

    Transforms analysis results into concrete RuleSchema objects that can be
    executed by the core validation engine.
    """

    @classmethod
    def generate_rules(
        cls,
        compatibility_results: List[CompatibilityResult],
        table_name: str,
        source_db: str,
        desired_type_metadata: Dict[str, Dict[str, Any]],
        dialect: Any = None,  # Database dialect for pattern generation
    ) -> List[RuleSchema]:
        """
        Generate validation rules based on compatibility analysis results.

        Args:
            compatibility_results: Results from compatibility analysis
            table_name: Name of the table being validated
            source_db: Source database name
            desired_type_metadata: Metadata for desired types (precision, scale, etc.)

        Returns:
            List of RuleSchema objects for incompatible type conversions
        """
        generated_rules = []

        for result in compatibility_results:
            if result.compatibility != "INCOMPATIBLE":
                # Only generate rules for incompatible conversions
                continue

            if result.required_validation is None:
                # No validation requirements determined
                continue

            field_name = result.field_name
            validation_type = result.required_validation
            validation_params = result.validation_params or {}

            # Get desired type metadata for this field
            field_metadata = desired_type_metadata.get(field_name, {})

            if validation_type == "REGEX":
                safe_source_db = source_db if source_db is not None else "unknown"
                rule = cls._generate_regex_rule(
                    field_name,
                    table_name,
                    safe_source_db,
                    validation_params,
                    field_metadata,
                    dialect,
                )
                if rule:
                    generated_rules.append(rule)

            elif validation_type == "LENGTH":
                safe_source_db = source_db if source_db is not None else "unknown"
                rule = cls._generate_length_rule(
                    field_name,
                    table_name,
                    safe_source_db,
                    validation_params,
                    field_metadata,
                )
                if rule:
                    generated_rules.append(rule)

            elif validation_type == "DATE_FORMAT":
                safe_source_db = source_db if source_db is not None else "unknown"
                rule = cls._generate_date_format_rule(
                    field_name,
                    table_name,
                    safe_source_db,
                    validation_params,
                    field_metadata,
                )
                if rule:
                    generated_rules.append(rule)

        logger.debug(
            f"Generated {len(generated_rules)} desired_type validation rules "
            f"for table {table_name}"
        )
        return generated_rules

    @classmethod
    def _generate_regex_rule(
        cls,
        field_name: str,
        table_name: str,
        source_db: str,
        validation_params: Dict[str, Any],
        field_metadata: Dict[str, Any],
        dialect: Any = None,
    ) -> Optional[RuleSchema]:
        """Generate REGEX rule for string format validation."""
        pattern = validation_params.get("pattern")
        if not pattern:
            return None

        # Enhance pattern with desired type metadata if available
        if (
            dialect
            and "desired_precision" in field_metadata
            and "desired_scale" in field_metadata
        ):
            # For float patterns, use precision and scale from metadata
            precision = field_metadata["desired_precision"]
            scale = field_metadata["desired_scale"]
            if precision > 0 and scale >= 0:
                pattern = dialect.generate_float_regex_pattern(precision, scale)

        elif dialect and "desired_max_length" in field_metadata:
            # For string patterns, limit length
            max_length = field_metadata["desired_max_length"]
            if "integer" in validation_params.get("description", "").lower():
                pattern = dialect.generate_integer_regex_pattern(max_length)

        return _create_rule_schema(
            name=f"desired_type_regex_{field_name}",
            rule_type=RuleType.REGEX,
            column=field_name,
            parameters={
                "pattern": pattern,
                "description": validation_params.get(
                    "description", "format validation"
                ),
            },
            description=(
                f"Desired type validation: "
                f"{validation_params.get('description', 'format validation')}"
            ),
        )

    @classmethod
    def _generate_length_rule(
        cls,
        field_name: str,
        table_name: str,
        source_db: str,
        validation_params: Dict[str, Any],
        field_metadata: Dict[str, Any],
    ) -> Optional[RuleSchema]:
        """Generate LENGTH rule for length/precision validation."""
        max_length = field_metadata.get("desired_max_length")
        if not max_length:
            return None

        # Create rule with proper target information
        target = RuleTarget(
            entities=[
                TargetEntity(
                    database=source_db,
                    table=table_name,
                    column=field_name,
                    connection_id=None,
                    alias=None,
                )
            ],
            relationship_type="single_table",
        )

        # Use REGEX rule for length validation (more reliable than LENGTH)
        length_pattern = (
            rf"^.{{0,{max_length}}}$"  # Match strings with 0 to max_length characters
        )

        return RuleSchema(
            name=f"desired_type_length_{field_name}",
            description=f"Desired type length validation: max {max_length} characters",
            type=RuleType.REGEX,
            target=target,
            parameters={"pattern": length_pattern},
            cross_db_config=None,
            threshold=0.0,
            severity=SeverityLevel.MEDIUM,
            action=RuleAction.ALERT,
            category=RuleCategory.VALIDITY,
        )

    @classmethod
    def _generate_date_format_rule(
        cls,
        field_name: str,
        table_name: str,
        source_db: str,
        validation_params: Dict[str, Any],
        field_metadata: Dict[str, Any],
    ) -> Optional[RuleSchema]:
        """Generate DATE_FORMAT rule for date format validation."""
        # Use desired format from metadata if available, otherwise use default
        format_pattern = field_metadata.get(
            "desired_format", validation_params.get("format_pattern", "YYYY-MM-DD")
        )

        return _create_rule_schema(
            name=f"desired_type_date_{field_name}",
            rule_type=RuleType.DATE_FORMAT,
            column=field_name,
            parameters={"format_pattern": format_pattern},
            description=f"Desired type date format validation: {format_pattern}",
        )


_ALLOWED_TYPE_NAMES: set[str] = {
    "string",
    "integer",
    "float",
    "boolean",
    "date",
    "datetime",
}


def _validate_multi_table_rules_payload(payload: Any) -> Tuple[List[str], int]:
    """Validate the structure of multi-table schema rules file.

    Multi-table format:
    {
      "table1": {
        "rules": [...],
        "strict_mode": true
      },
      "table2": {
        "rules": [...]
      }
    }

    Returns:
        warnings, total_rules_count
    """
    warnings: List[str] = []
    total_rules = 0

    if not isinstance(payload, dict):
        raise click.UsageError("Rules file must be a JSON object")

    # Check if this is a multi-table format (has table names as keys)
    table_names = [key for key in payload.keys() if key != "rules"]

    if table_names:
        # Multi-table format
        for table_name in table_names:
            table_schema = payload[table_name]
            if not isinstance(table_schema, dict):
                raise click.UsageError(f"Table '{table_name}' schema must be an object")

            table_rules = table_schema.get("rules")
            if not isinstance(table_rules, list):
                raise click.UsageError(
                    f"Table '{table_name}' must have a 'rules' array"
                )

            # Validate each rule in this table
            for idx, item in enumerate(table_rules):
                if not isinstance(item, dict):
                    raise click.UsageError(
                        f"Table '{table_name}' rules[{idx}] must be an object"
                    )

                # Validate rule fields
                _validate_single_rule_item(item, f"Table '{table_name}' rules[{idx}]")

            total_rules += len(table_rules)

            # Validate optional table-level switches
            if "strict_mode" in table_schema and not isinstance(
                table_schema["strict_mode"], bool
            ):
                raise click.UsageError(
                    f"Table '{table_name}' strict_mode must be a boolean"
                )
            if "case_insensitive" in table_schema and not isinstance(
                table_schema["case_insensitive"], bool
            ):
                raise click.UsageError(
                    f"Table '{table_name}' case_insensitive must be a boolean"
                )
    else:
        # Single-table format (backward compatibility)
        warnings.append(
            "Single-table format detected; consider using multi-table format for "
            "better organization"
        )
        if "rules" not in payload:
            raise click.UsageError("Single-table format must have a 'rules' array")

        rules = payload["rules"]
        if not isinstance(rules, list):
            raise click.UsageError("'rules' must be an array")

        for idx, item in enumerate(rules):
            if not isinstance(item, dict):
                raise click.UsageError(f"rules[{idx}] must be an object")
            _validate_single_rule_item(item, f"rules[{idx}]")

        total_rules = len(rules)

    return warnings, total_rules


def _validate_single_rule_item(item: Dict[str, Any], context: str) -> None:
    """Validate a single rule item from the rules array."""
    # field
    field_name = item.get("field")
    if not isinstance(field_name, str) or not field_name:
        raise click.UsageError(f"{context}.field must be a non-empty string")

    # type - validate using TypeParser to support syntactic sugar
    if "type" in item:
        type_name = item["type"]
        if not isinstance(type_name, str):
            raise click.UsageError(f"{context}.type must be a string when provided")

        # Use TypeParser to validate the type definition
        from shared.utils.type_parser import TypeParseError, TypeParser

        try:
            TypeParser.parse_type_definition(type_name)
        except TypeParseError as e:
            allowed = ", ".join(sorted(_ALLOWED_TYPE_NAMES))
            raise click.UsageError(
                f"{context}.type '{type_name}' is not supported. Error: {str(e)}. "
                f"Supported formats: {allowed} or syntactic sugar like string(50), "
                "float(12,2), datetime('format')"
            )

    # required
    if "required" in item and not isinstance(item["required"], bool):
        raise click.UsageError(f"{context}.required must be a boolean when provided")

    # enum
    if "enum" in item and not isinstance(item["enum"], list):
        raise click.UsageError(f"{context}.enum must be an array when provided")

    # min/max
    for bound_key in ("min", "max"):
        if bound_key in item:
            value = item[bound_key]
            if not isinstance(value, (int, float)):
                raise click.UsageError(
                    f"{context}.{bound_key} must be numeric when provided"
                )

    # max_length - basic validation, TypeParser will handle type consistency
    if "max_length" in item:
        value = item["max_length"]
        if not isinstance(value, int) or value < 0:
            raise click.UsageError(
                f"{context}.max_length must be a non-negative integer when provided"
            )

    # precision - basic validation, TypeParser will handle type consistency
    if "precision" in item:
        value = item["precision"]
        if not isinstance(value, int) or value < 0:
            raise click.UsageError(
                f"{context}.precision must be a non-negative integer when provided"
            )

    # scale - basic validation, TypeParser will handle type consistency
    if "scale" in item:
        value = item["scale"]
        if not isinstance(value, int) or value < 0:
            raise click.UsageError(
                f"{context}.scale must be a non-negative integer when provided"
            )

    # desired_type - validate using TypeParser to support syntactic sugar
    if "desired_type" in item:
        desired_type = item["desired_type"]
        if not isinstance(desired_type, str):
            raise click.UsageError(
                f"{context}.desired_type must be a string when provided"
            )

        # Use TypeParser to validate the desired_type definition
        from shared.utils.type_parser import TypeParseError, TypeParser

        try:
            TypeParser.parse_type_definition(desired_type)
        except TypeParseError as e:
            allowed = ", ".join(sorted(_ALLOWED_TYPE_NAMES))
            raise click.UsageError(
                f"{context}.desired_type '{desired_type}' is not supported. "
                f"Error: {str(e)}. "
                f"Supported formats: {allowed} or syntactic sugar like string(50), "
                "float(12,2), datetime('format')"
            )


def _validate_rules_payload(payload: Any) -> Tuple[List[str], int]:
    """Validate the minimal structure of the schema rules file.

    This performs non-jsonschema checks for both single-table and multi-table formats.
    """
    return _validate_multi_table_rules_payload(payload)


def _map_type_name_to_datatype(type_name: str) -> DataType:
    """Map user-provided type string to DataType enum.

    Args:
        type_name: Input type name (case-insensitive), e.g. "string".

    Returns:
        DataType enum.

    Raises:
        click.UsageError: When the value is unsupported.
    """
    normalized = str(type_name).strip().lower()
    mapping: Dict[str, DataType] = {
        "string": DataType.STRING,
        "integer": DataType.INTEGER,
        "float": DataType.FLOAT,
        "boolean": DataType.BOOLEAN,
        "date": DataType.DATE,
        "datetime": DataType.DATETIME,
    }
    if normalized not in mapping:
        allowed = ", ".join(sorted(_ALLOWED_TYPE_NAMES))
        raise click.UsageError(f"Unsupported type '{type_name}'. Allowed: {allowed}")
    return mapping[normalized]


def _derive_category(rule_type: RuleType) -> RuleCategory:
    """Derive category from rule type per design mapping."""
    if rule_type == RuleType.SCHEMA:
        return RuleCategory.VALIDITY
    if rule_type == RuleType.NOT_NULL:
        return RuleCategory.COMPLETENESS
    if rule_type == RuleType.UNIQUE:
        return RuleCategory.UNIQUENESS
    # RANGE, LENGTH, ENUM, REGEX, DATE_FORMAT -> VALIDITY in v1
    return RuleCategory.VALIDITY


def _create_rule_schema(
    *,
    name: str,
    rule_type: RuleType,
    column: str | None,
    parameters: Dict[str, Any],
    description: str | None = None,
    severity: SeverityLevel = SeverityLevel.MEDIUM,
    action: RuleAction = RuleAction.ALERT,
) -> RuleSchema:
    """Create a `RuleSchema` with an empty target that will be completed later.

    The database and table will be filled by the validator based on the source.
    """
    target = RuleTarget(
        entities=[
            TargetEntity(
                database="unknown",
                table="unknown",
                column=column,
                connection_id=None,
                alias=None,
            )
        ],
        relationship_type="single_table",
    )
    return RuleSchema(
        name=name,
        description=description,
        type=rule_type,
        target=target,
        parameters=parameters,
        cross_db_config=None,
        threshold=0.0,
        category=_derive_category(rule_type),
        severity=severity,
        action=action,
        is_active=True,
        tags=[],
        template_id=None,
        validation_error=None,
    )


def _decompose_schema_payload(
    payload: Dict[str, Any], source_config: ConnectionSchema
) -> Tuple[List[RuleSchema], List[RuleSchema]]:
    """Decompose a schema payload into atomic RuleSchema objects, separated by phase.

    This function handles both single-table and multi-table formats in a
    source-agnostic way. Returns schema rules and non-schema rules separately
    to support two-phase execution.

    Returns:
        Tuple of (schema_rules, other_rules) for two-phase execution
    """
    all_atomic_rules = _decompose_schema_payload_atomic(payload, source_config)

    # Separate rules by type for two-phase execution
    schema_rules = [rule for rule in all_atomic_rules if rule.type == RuleType.SCHEMA]
    other_rules = [rule for rule in all_atomic_rules if rule.type != RuleType.SCHEMA]

    return schema_rules, other_rules


def _decompose_schema_payload_atomic(
    payload: Dict[str, Any], source_config: ConnectionSchema
) -> List[RuleSchema]:
    """Decompose a schema payload into atomic RuleSchema objects.

    This function handles both single-table and multi-table formats in a
    source-agnostic way.
    """
    all_atomic_rules: List[RuleSchema] = []
    source_db = source_config.db_name or "unknown"

    is_multi_table_format = "rules" not in payload

    if is_multi_table_format:
        tables_in_rules = list(payload.keys())
        available_tables_from_source = set(source_config.available_tables or [])

        for table_name in tables_in_rules:
            if (
                available_tables_from_source
                and table_name not in available_tables_from_source
            ):
                logger.warning(
                    f"Skipping rules for table '{table_name}' as it is not available "
                    "in the source."
                )
                continue

            table_schema = payload[table_name]
            if not isinstance(table_schema, dict):
                logger.warning(
                    f"Definition for table '{table_name}' is not a valid object, "
                    "skipping."
                )
                continue

            table_rules = _decompose_single_table_schema(
                table_schema, source_db, table_name
            )
            all_atomic_rules.extend(table_rules)
    else:
        table_name = "unknown"
        if source_config.available_tables:
            table_name = source_config.available_tables[0]
        else:
            logger.warning(
                "Could not determine table name for single-table schema. "
                "Consider using multi-table format for database sources."
            )

        table_rules = _decompose_single_table_schema(payload, source_db, table_name)
        all_atomic_rules.extend(table_rules)

    return all_atomic_rules


def _decompose_single_table_schema(
    table_schema: Dict[str, Any], source_db: str, table_name: str
) -> List[RuleSchema]:
    """Decompose a single table's schema definition into atomic RuleSchema objects.

    Args:
        table_schema: The schema definition for a single table
        source_db: Database name from source
        table_name: Name of the table being validated
    """
    rules_arr = table_schema.get("rules", [])

    # Build SCHEMA columns mapping first
    columns_map: Dict[str, Dict[str, Any]] = {}
    atomic_rules: List[RuleSchema] = []

    for item in rules_arr:
        field_name = item.get("field")
        if not isinstance(field_name, str) or not field_name:
            # Should have been validated earlier; keep defensive check
            raise click.UsageError("Each rule item must have a non-empty 'field'")

        # SCHEMA: collect column metadata using new TypeParser
        column_metadata = {}

        # Handle type definition using TypeParser (supports syntactic sugar)
        if "type" in item and item["type"] is not None:
            from shared.utils.type_parser import TypeParseError, TypeParser

            try:
                # Create a type definition dict for the parser
                type_def = {"type": item["type"]}

                # Add metadata fields if present in the item
                for metadata_field in ["max_length", "precision", "scale", "format"]:
                    if metadata_field in item:
                        type_def[metadata_field] = item[metadata_field]

                # Parse using TypeParser (handles both syntactic sugar
                # and detailed format)
                parsed_type = TypeParser.parse_type_definition(item["type"])

                # Add expected_type for schema validation
                column_metadata["expected_type"] = parsed_type["type"]

                # Add any parsed metadata
                for metadata_field in ["max_length", "precision", "scale", "format"]:
                    if metadata_field in parsed_type:
                        column_metadata[metadata_field] = parsed_type[metadata_field]

                # Also add any explicit metadata from the item (overrides parsed values)
                for metadata_field in ["max_length", "precision", "scale", "format"]:
                    if metadata_field in item:
                        column_metadata[metadata_field] = item[metadata_field]

                # Handle desired_type definition using TypeParser
                if "desired_type" in item and item["desired_type"] is not None:
                    try:
                        # Parse the desired_type using TypeParser for core layer
                        desired_type_fields = TypeParser.parse_desired_type_for_core(
                            item["desired_type"]
                        )

                        # Add all desired_type fields to column metadata
                        column_metadata.update(desired_type_fields)

                    except TypeParseError as dt_e:
                        raise click.UsageError(
                            f"Invalid desired_type definition for field '{field_name}'"
                            f": {str(dt_e)}"
                        )

            except TypeParseError as e:
                raise click.UsageError(
                    f"Invalid type definition for field '{field_name}': {str(e)}"
                )
            except Exception:
                # Fallback to original parsing for backward compatibility
                dt = _map_type_name_to_datatype(str(item["type"]))
                column_metadata["expected_type"] = dt.value

                # Add metadata fields if present
                if "max_length" in item:
                    column_metadata["max_length"] = item["max_length"]
                if "precision" in item:
                    column_metadata["precision"] = item["precision"]
                if "scale" in item:
                    column_metadata["scale"] = item["scale"]

        # Only add to columns_map if we have any metadata to store
        if column_metadata:
            columns_map[field_name] = column_metadata

        # NOT_NULL
        if bool(item.get("required", False)):
            atomic_rules.append(
                _create_rule_schema(
                    name=f"not_null_{field_name}",
                    rule_type=RuleType.NOT_NULL,
                    column=field_name,
                    parameters={},
                    description=f"CLI: required non-null for {field_name}",
                )
            )

        # RANGE
        has_min = "min" in item and isinstance(item.get("min"), (int, float))
        has_max = "max" in item and isinstance(item.get("max"), (int, float))
        if has_min or has_max:
            params: Dict[str, Any] = {}
            if has_min:
                params["min_value"] = item["min"]
            if has_max:
                params["max_value"] = item["max"]
            atomic_rules.append(
                _create_rule_schema(
                    name=f"range_{field_name}",
                    rule_type=RuleType.RANGE,
                    column=field_name,
                    parameters=params,
                    description=f"CLI: range for {field_name}",
                )
            )

        # ENUM
        if "enum" in item:
            values = item.get("enum")
            if not isinstance(values, list) or len(values) == 0:
                raise click.UsageError("'enum' must be a non-empty array when provided")
            atomic_rules.append(
                _create_rule_schema(
                    name=f"enum_{field_name}",
                    rule_type=RuleType.ENUM,
                    column=field_name,
                    parameters={"allowed_values": values},
                    description=f"CLI: enum for {field_name}",
                )
            )

    # Create one table-level SCHEMA rule if any columns were declared
    if columns_map:
        schema_params: Dict[str, Any] = {"columns": columns_map}
        # Optional switches at table level
        if isinstance(table_schema.get("strict_mode"), bool):
            schema_params["strict_mode"] = table_schema["strict_mode"]
        if isinstance(table_schema.get("case_insensitive"), bool):
            schema_params["case_insensitive"] = table_schema["case_insensitive"]

        atomic_rules.insert(
            0,
            _create_rule_schema(
                name=f"schema_{table_name}",
                rule_type=RuleType.SCHEMA,
                column=None,
                parameters=schema_params,
                description=f"CLI: table schema existence+type for {table_name}",
            ),
        )

    # Set the target table and database for all rules
    for rule in atomic_rules:
        if rule.target and rule.target.entities:
            rule.target.entities[0].database = source_db
            rule.target.entities[0].table = table_name

    return atomic_rules


def _build_prioritized_atomic_status(
    *,
    schema_results: List[Dict[str, Any]],
    atomic_rules: List[RuleSchema],
) -> Dict[str, Dict[str, str]]:
    """Return a mapping rule_id -> {status, skip_reason} applying prioritization."""
    mapping: Dict[str, Dict[str, str]] = {}
    schema_failures: Dict[str, str] = (
        {}
    )  # Key: f"{table}.{column}", Value: failure_code
    table_not_exists: set[str] = set()  # Set of table names that don't exist

    schema_rules_map = {
        str(rule.id): rule for rule in atomic_rules if rule.type == RuleType.SCHEMA
    }

    for res in schema_results:
        rule_id = str(res.get("rule_id", ""))
        rule = schema_rules_map.get(rule_id)
        if not rule:
            continue

        table = rule.get_target_info().get("table", "")

        # Check if table exists based on schema details
        schema_details = res.get("execution_plan", {}).get("schema_details", {})
        table_exists = schema_details.get("table_exists", True)

        if not table_exists and table:
            # Table doesn't exist - mark all rules for this table to be skipped
            table_not_exists.add(table)
            continue

        # Process field-level failures for existing tables
        field_results = schema_details.get("field_results", [])
        for item in field_results:
            code = item.get("failure_code")
            if code in ("FIELD_MISSING", "TYPE_MISMATCH"):
                col = item.get("column")
                if col:
                    schema_failures[f"{table}.{col}"] = code

    # Apply skip logic for all non-SCHEMA rules
    for rule in atomic_rules:
        if rule.type == RuleType.SCHEMA:
            continue

        table = rule.get_target_info().get("table", "")
        col = rule.get_target_column()

        # Skip all rules for tables that don't exist
        if table in table_not_exists:
            mapping[str(rule.id)] = {
                "status": "SKIPPED",
                "skip_reason": "TABLE_NOT_EXISTS",
            }
        # Skip specific column rules only when field is missing
        elif col and f"{table}.{col}" in schema_failures:
            reason = schema_failures[f"{table}.{col}"]
            # Only skip for missing fields, not for type mismatches
            if reason == "FIELD_MISSING":
                mapping[str(rule.id)] = {"status": "SKIPPED", "skip_reason": reason}

    return mapping


def _safe_echo(text: str, *, err: bool = False) -> None:
    """Compatibility shim; delegate to shared safe_echo."""
    safe_echo(text, err=err)


def _maybe_echo_analyzing(source: str, output: str) -> None:
    """Emit analyzing line unless JSON output."""
    if str(output).lower() != "json":
        _safe_echo(f"🔍 Analyzing source: {source}", err=True)


def _guard_empty_source_file(source: str) -> None:
    """Raise a ClickException if a provided file source is empty."""
    potential_path = Path(source)
    if potential_path.exists() and potential_path.is_file():
        if potential_path.stat().st_size == 0:
            raise click.ClickException(
                f"Error: Source file '{source}' is empty – nothing to validate."
            )


def _read_rules_payload(rules_file: str) -> Dict[str, Any]:
    """Read and parse JSON rules file, raising UsageError on invalid JSON."""
    try:
        with open(rules_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as e:
        raise click.UsageError(f"Invalid JSON in rules file: {rules_file}") from e
    return cast(Dict[str, Any], payload)


def _emit_warnings(warnings: List[str], output: str = "table") -> None:
    """Emit warnings only for non-JSON output to avoid polluting JSON output."""
    if output.lower() != "json":
        for msg in warnings:
            _safe_echo(f"⚠️ Warning: {msg}", err=True)


def _early_exit_when_no_rules(
    *, source: str, rules_file: str, output: str, fail_on_error: bool
) -> None:
    """Emit minimal output and exit when no rules are present."""
    if output.lower() == "json":
        payload = {
            "status": "ok",
            "source": source,
            "rules_file": rules_file,
            "rules_count": 0,
            "summary": {
                "total_rules": 0,
                "passed_rules": 0,
                "failed_rules": 0,
                "skipped_rules": 0,
                "total_failed_records": 0,
                "execution_time_s": 0.0,
            },
            "results": [],
            "fields": [],
        }
        _safe_echo(json.dumps(payload, default=str))
        raise click.exceptions.Exit(1 if fail_on_error else 0)
    else:
        _safe_echo(f"✓ Checking {source} (0 records)")
        raise click.exceptions.Exit(1 if fail_on_error else 0)


def _create_validator(
    *,
    source_config: Any,
    atomic_rules: List[RuleSchema] | List[Dict[str, Any]],
    core_config: Any,
    cli_config: Any,
) -> Any:
    try:
        return DataValidator(
            source_config=source_config,
            rules=cast(List[RuleSchema | Dict[str, Any]], atomic_rules),
            core_config=core_config,
            cli_config=cli_config,
        )
    except Exception as e:
        logger.error(f"Failed to create DataValidator: {str(e)}")
        raise click.UsageError(f"Failed to create validator: {str(e)}")


def _run_validation(validator: Any) -> Tuple[List[Any], float]:
    import asyncio

    start = _now()
    logger.debug("Starting validation")
    try:
        results = asyncio.run(validator.validate())
        logger.debug(f"Validation returned {len(results)} results")
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise
    exec_seconds = (_now() - start).total_seconds()
    return results, exec_seconds


def _extract_schema_results(
    *, atomic_rules: List[RuleSchema], results: List[Any]
) -> List[Dict[str, Any]]:
    """Extract all SCHEMA rule results from the list of validation results."""
    schema_results = []
    schema_rule_ids = {
        str(rule.id) for rule in atomic_rules if rule.type == RuleType.SCHEMA
    }
    if not schema_rule_ids:
        return []

    for r in results:
        if r is None:
            continue
        rid = ""
        if hasattr(r, "rule_id"):
            try:
                rid = str(getattr(r, "rule_id"))
            except Exception:
                rid = ""
        elif isinstance(r, dict):
            rid = str(r.get("rule_id", ""))

        if rid in schema_rule_ids:
            schema_results.append(
                r.model_dump() if hasattr(r, "model_dump") else cast(Dict[str, Any], r)
            )
    return schema_results


def _compute_skip_map(
    *, atomic_rules: List[RuleSchema], schema_results: List[Dict[str, Any]]
) -> Dict[str, Dict[str, str]]:
    try:
        return _build_prioritized_atomic_status(
            schema_results=schema_results, atomic_rules=atomic_rules
        )
    except Exception:
        return {}


def _emit_json_output(
    *,
    source: str,
    rules_file: str,
    atomic_rules: List[RuleSchema],
    results: List[Any],
    skip_map: Dict[str, Dict[str, str]],
    schema_results: List[Dict[str, Any]],
    exec_seconds: float,
) -> None:
    enriched_results: List[Dict[str, Any]] = []
    for r in results:
        rd: Dict[str, Any]
        if hasattr(r, "model_dump"):
            try:
                rd = cast(Dict[str, Any], r.model_dump())
            except Exception:
                rd = {}
        elif isinstance(r, dict):
            rd = r
        else:
            rd = {}
        rule_id = str(rd.get("rule_id", ""))
        if rule_id in skip_map:
            rd["status"] = skip_map[rule_id]["status"]
            rd["skip_reason"] = skip_map[rule_id]["skip_reason"]
        enriched_results.append(rd)

    rule_map: Dict[str, RuleSchema] = {str(rule.id): rule for rule in atomic_rules}

    def _failed_records_of(res: Dict[str, Any]) -> int:
        if "failed_records" in res and isinstance(res.get("failed_records"), int):
            return int(res.get("failed_records") or 0)
        dm = res.get("dataset_metrics") or []
        total = 0
        for m in dm:
            if hasattr(m, "failed_records"):
                total += int(getattr(m, "failed_records", 0) or 0)
            elif isinstance(m, dict):
                total += int(m.get("failed_records", 0) or 0)
        return total

    fields: List[Dict[str, Any]] = []
    schema_fields_index: Dict[str, Dict[str, Any]] = {}

    schema_rules_map = {
        str(rule.id): rule for rule in atomic_rules if rule.type == RuleType.SCHEMA
    }

    for schema_result in schema_results:
        schema_plan = (schema_result or {}).get("execution_plan", {}) or {}
        schema_details = schema_plan.get("schema_details", {}) or {}
        field_results = schema_details.get("field_results", []) or []

        rule_id = str(schema_result.get("rule_id", ""))
        rule = schema_rules_map.get(rule_id)
        table_name = rule.get_target_info().get("table") if rule else "unknown"

        for item in field_results:
            col_name = str(item.get("column"))
            entry: Dict[str, Any] = {
                "column": col_name,
                "table": table_name,
                "checks": {
                    "existence": {
                        "status": item.get("existence", "UNKNOWN"),
                        "failure_code": item.get("failure_code", "NONE"),
                    },
                    "type": {
                        "status": item.get("type", "UNKNOWN"),
                        "failure_code": item.get("failure_code", "NONE"),
                    },
                },
            }
            fields.append(entry)
            schema_fields_index[f"{table_name}.{col_name}"] = entry

    for rule in atomic_rules:
        if rule.type == RuleType.SCHEMA:
            params = rule.parameters or {}
            declared_cols = (params.get("columns") or {}).keys()
            table_name = rule.get_target_info().get("table")
            for col in declared_cols:
                if f"{table_name}.{str(col)}" not in schema_fields_index:
                    entry = {
                        "column": str(col),
                        "table": table_name,
                        "checks": {
                            "existence": {"status": "UNKNOWN", "failure_code": "NONE"},
                            "type": {"status": "UNKNOWN", "failure_code": "NONE"},
                        },
                    }
                    fields.append(entry)
                    schema_fields_index[f"{table_name}.{str(col)}"] = entry

    def _ensure_check(entry: Dict[str, Any], name: str) -> Dict[str, Any]:
        checks: Dict[str, Dict[str, Any]] = entry.setdefault("checks", {})
        if name not in checks:
            checks[name] = {
                "status": (
                    "SKIPPED"
                    if name
                    in {
                        "not_null",
                        "range",
                        "enum",
                        "regex",
                        "date_format",
                        "desired_type",
                    }
                    else "UNKNOWN"
                )
            }
        return checks[name]

    for rd in enriched_results:
        rule_id = str(rd.get("rule_id", ""))
        rule = rule_map.get(rule_id)
        if not rule or rule.type == RuleType.SCHEMA:
            continue

        column_name = rule.get_target_column() or ""
        if not column_name:
            continue

        table_name = "unknown"
        if rule.target and rule.target.entities:
            table_name = rule.target.entities[0].table

        l_entry = schema_fields_index.get(f"{table_name}.{column_name}")
        if not l_entry:
            l_entry = {"column": column_name, "table": table_name, "checks": {}}
            fields.append(l_entry)
            schema_fields_index[f"{table_name}.{column_name}"] = l_entry
        else:
            l_entry["table"] = table_name

        # Check if this is a desired_type validation rule
        rule_name = getattr(rule, "name", "")
        if rule_name and rule_name.startswith("desired_type_"):
            key = "desired_type"
        else:
            # Regular rule type mapping
            t = rule.type
            if t == RuleType.NOT_NULL:
                key = "not_null"
            elif t == RuleType.RANGE:
                key = "range"
            elif t == RuleType.ENUM:
                key = "enum"
            elif t == RuleType.REGEX:
                key = "regex"
            elif t == RuleType.DATE_FORMAT:
                key = "date_format"
            else:
                key = t.value.lower()

        check = _ensure_check(l_entry, key)
        check["status"] = str(rd.get("status", "UNKNOWN"))
        if rule_id in skip_map:
            check["status"] = skip_map[rule_id]["status"]
            check["skip_reason"] = skip_map[rule_id]["skip_reason"]

        fr = _failed_records_of(rd)
        if fr:
            check["failed_records"] = fr

    total_rules = len(enriched_results)
    passed_rules = sum(
        1 for r in enriched_results if str(r.get("status", "")).upper() == "PASSED"
    )
    failed_rules = sum(
        1 for r in enriched_results if str(r.get("status", "")).upper() == "FAILED"
    )
    skipped_rules = sum(
        1 for r in enriched_results if str(r.get("status", "")).upper() == "SKIPPED"
    )
    total_failed_records = sum(_failed_records_of(r) for r in enriched_results)

    schema_extras: List[str] = []
    for schema_result in schema_results:
        try:
            extras = (
                (schema_result or {})
                .get("execution_plan", {})
                .get("schema_details", {})
                .get("extras", [])
            )
            if isinstance(extras, list):
                schema_extras.extend([str(x) for x in extras])
        except Exception:
            pass

    payload: Dict[str, Any] = {
        "status": "ok",
        "source": source,
        "rules_file": rules_file,
        "rules_count": len(atomic_rules),
        "summary": {
            "total_rules": total_rules,
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
            "skipped_rules": skipped_rules,
            "total_failed_records": total_failed_records,
            "execution_time_s": round(exec_seconds, 3),
        },
        "results": enriched_results,
        "fields": fields,
    }
    if schema_extras:
        payload["schema_extras"] = sorted(list(set(schema_extras)))
    _safe_echo(json.dumps(payload, default=str))


class SchemaPhaseExecutor:
    """Executor for Phase 1: Schema rules only with native type collection."""

    def __init__(self, *, source_config: Any, core_config: Any, cli_config: Any):
        """Init SchemaPhaseExecutor object"""
        self.source_config = source_config
        self.core_config = core_config
        self.cli_config = cli_config

    async def execute_schema_phase(
        self, schema_rules: List[RuleSchema]
    ) -> Tuple[List[Any], float, List[Dict[str, Any]]]:
        """Execute schema rules and collect native type information.

        Returns:
            Tuple of (results, execution_seconds, schema_results)
        """
        logger.debug(f"Phase 1: Executing {len(schema_rules)} schema rules")

        if not schema_rules:
            return [], 0.0, []

        validator = _create_validator(
            source_config=self.source_config,
            atomic_rules=schema_rules,
            core_config=self.core_config,
            cli_config=self.cli_config,
        )

        results, exec_seconds = _run_validation(validator)
        schema_results = _extract_schema_results(
            atomic_rules=schema_rules, results=results
        )

        logger.debug(
            f"Phase 1: Completed in {exec_seconds:.3f}s with {len(schema_results)} "
            "schema results"
        )
        return results, exec_seconds, schema_results


class DesiredTypePhaseExecutor:
    """
    Executor for Phase 2: Desired type validation based on compatibility analysis.

    Analyzes schema results to extract native types, performs compatibility analysis
    with desired types, and generates validation rules for incompatible conversions.
    """

    def __init__(
        self, *, source_config: Any, core_config: Any, cli_config: Any
    ) -> None:
        """Init DesiredTypePhaseExecutor object"""
        self.source_config = source_config
        self.core_config = core_config
        self.cli_config = cli_config

    async def execute_desired_type_validation(
        self,
        schema_results: List[Dict[str, Any]],
        original_payload: Dict[str, Any],
        skip_map: Dict[str, Dict[str, str]],
    ) -> Tuple[List[Any], float, List[RuleSchema]]:
        """
        Execute desired_type validation with compatibility analysis and rule generation.

        Args:
            schema_results: Results from schema phase containing native type information
            original_payload: Original rules payload with desired_type definitions
            skip_map: Pre-computed skip decisions based on schema results

        Returns:
            Tuple of (results, execution_seconds, generated_rules)
        """
        logger.debug(
            "Phase 2: Starting desired_type validation with compatibility analysis"
        )
        logger.debug(f"Schema results count: {len(schema_results)}")
        logger.debug(f"Original payload keys: {list(original_payload.keys())}")

        # Create compatibility analyzer with database connection type
        connection_type = getattr(
            self.source_config, "connection_type", ConnectionType.MYSQL
        )
        analyzer = CompatibilityAnalyzer(connection_type)

        # Extract native types from schema results
        native_types = self._extract_native_types_from_schema_results(schema_results)

        # Extract desired_type definitions from payload
        desired_type_definitions = self._extract_desired_type_definitions(
            original_payload
        )

        logger.debug(f"Extracted native types: {native_types}")
        logger.debug(f"Extracted desired_type definitions: {desired_type_definitions}")

        if not desired_type_definitions:
            logger.debug("Phase 2: No desired_type definitions found, skipping")
            return [], 0.0, []

        # Perform compatibility analysis
        compatibility_results = []
        for field_name, table_info in desired_type_definitions.items():
            table_name = table_info["table"]
            desired_type = table_info["desired_type"]  # This is the canonical type
            original_desired_type = table_info.get(
                "original_desired_type", desired_type
            )  # Original string

            # Get native type for this field
            # First try exact match with table name
            field_key = f"{table_name}.{field_name}"
            native_type_info = native_types.get(field_key)

            # If not found, try to find by field name only (handles 'unknown' table
            # name issue)
            if not native_type_info:
                for key, info in native_types.items():
                    if key.endswith(f".{field_name}"):
                        native_type_info = info
                        logger.debug(
                            f"Found native type for {field_name} using fuzzy match: "
                            f"{key}"
                        )
                        break

            if not native_type_info:
                logger.debug(f"No native type info for {field_key}, skipping")
                continue

            native_type = native_type_info["canonical_type"]
            native_metadata = native_type_info.get("native_metadata", {})

            logger.debug(
                f"Analyzing compatibility for {field_name}: {native_type} -> "
                f"{original_desired_type}"
            )

            # Perform compatibility analysis using original desired_type for proper
            # parsing
            compatibility_result = analyzer.analyze(
                native_type=native_type,
                desired_type=original_desired_type,  # Use original string for parsing
                field_name=field_name,
                table_name=table_name,
                native_metadata=native_metadata,
            )
            logger.debug(
                f"Compatibility result: {compatibility_result.compatibility} - "
                f"{compatibility_result.reason}"
            )
            compatibility_results.append(compatibility_result)

            # Handle conflicting conversions immediately
            if compatibility_result.compatibility == "CONFLICTING":
                error_msg = (
                    f"Conflicting type conversion for {table_name}.{field_name}: "
                    f"{compatibility_result.reason}"
                )
                logger.error(error_msg)
                raise click.UsageError(error_msg)

        # Filter out fields that should be skipped
        valid_compatibility_results = []
        for result in compatibility_results:
            field_key = f"{result.table_name}.{result.field_name}"
            # Check if this field should be skipped based on schema failures
            should_skip = any(
                skip_info.get("skip_reason") in ["FIELD_MISSING", "TABLE_NOT_EXISTS"]
                for rule_id, skip_info in skip_map.items()
                if field_key in str(rule_id)  # Simple check, could be improved
            )
            if not should_skip:
                valid_compatibility_results.append(result)

        # Generate validation rules for incompatible conversions
        generated_rules: List[RuleSchema] = []
        if valid_compatibility_results:
            # Group by table for rule generation
            tables_with_incompatible_fields: dict = {}
            for result in valid_compatibility_results:
                if result.compatibility == "INCOMPATIBLE":
                    table_name = result.table_name
                    if table_name not in tables_with_incompatible_fields:
                        tables_with_incompatible_fields[table_name] = []
                    tables_with_incompatible_fields[table_name].append(result)

            # Generate rules for each table
            source_db = getattr(self.source_config, "db_name", None)
            source_db = source_db if source_db is not None else "unknown"
            for table_name, table_results in tables_with_incompatible_fields.items():
                # Extract desired type metadata for this table
                table_metadata = {
                    result.field_name: desired_type_definitions[result.field_name].get(
                        "metadata", {}
                    )
                    for result in table_results
                }

                table_rules = DesiredTypeRuleGenerator.generate_rules(
                    compatibility_results=table_results,
                    table_name=table_name,
                    source_db=source_db,
                    desired_type_metadata=table_metadata,
                    dialect=analyzer.dialect,
                )
                generated_rules.extend(table_rules)

        logger.debug(
            f"Phase 2: Generated {len(generated_rules)} desired_type validation rules"
        )
        for rule in generated_rules:
            logger.debug(
                f"Generated rule: {rule.name}, Type: {rule.type}, Target: "
                f"{rule.get_target_info()}"
            )

        # Execute generated rules if any
        if generated_rules:
            # Set target information for generated rules
            for rule in generated_rules:
                if rule.target and rule.target.entities:
                    entity = rule.target.entities[0]
                    # Ensure database name is never None
                    db_name = getattr(self.source_config, "db_name", None)
                    entity.database = db_name if db_name is not None else "unknown"

                    # Get table name from the field metadata using the column name
                    column_name: Optional[str] = entity.column
                    if column_name and column_name in desired_type_definitions:
                        entity.table = desired_type_definitions[column_name]["table"]
                    else:
                        # Fallback: try to extract from existing source config
                        if (
                            hasattr(self.source_config, "available_tables")
                            and self.source_config.available_tables
                        ):
                            entity.table = self.source_config.available_tables[0]
                        else:
                            entity.table = "unknown"

            validator = _create_validator(
                source_config=self.source_config,
                atomic_rules=generated_rules,
                core_config=self.core_config,
                cli_config=self.cli_config,
            )

            # Execute validation directly without _run_validation to avoid
            # asyncio.run() conflicts
            start = _now()
            logger.debug("Starting desired_type validation")
            try:
                results = await validator.validate()
                exec_seconds = (_now() - start).total_seconds()
                logger.debug(f"Desired_type validation returned {len(results)} results")
            except Exception as e:
                logger.error(f"Desired_type validation failed: {str(e)}")
                results, exec_seconds = [], 0.0
            logger.debug(
                f"Phase 2: Executed desired_type validation in {exec_seconds:.3f}s"
            )
            return results, exec_seconds, generated_rules
        else:
            logger.debug("Phase 2: No rules to execute")
            return [], 0.0, []

    def _extract_native_types_from_schema_results(
        self, schema_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract native type information from schema validation results.

        Args:
            schema_results: Results from schema phase execution

        Returns:
            Dict mapping "table.field" to native type information:
            {
                "table.field": {
                    "native_type": "VARCHAR(255)",
                    "canonical_type": "STRING",
                    "native_metadata": {"max_length": 255}
                }
            }
        """
        native_types = {}

        for result in schema_results:
            # Extract field results from schema execution plan
            execution_plan = result.get("execution_plan", {})
            schema_details = execution_plan.get("schema_details", {})
            field_results = schema_details.get("field_results", [])

            # Determine table name from the rule or result
            rule_id = result.get("rule_id")
            table_name = result.get(
                "table_name", "unknown"
            )  # Try to get table name from result

            # If still unknown, try to get it from target_info
            if table_name == "unknown":
                target_info = result.get("target_info", {})
                table_name = target_info.get("table", "unknown")

            logger.debug(f"Schema result for table '{table_name}', rule_id: {rule_id}")

            for field_result in field_results:
                column_name = field_result.get("column")
                native_type = field_result.get("native_type")
                canonical_type = field_result.get("canonical_type")
                native_metadata = field_result.get("native_metadata", {})

                if column_name and native_type and canonical_type:
                    field_key = f"{table_name}.{column_name}"
                    native_types[field_key] = {
                        "native_type": native_type,
                        "canonical_type": canonical_type,
                        "native_metadata": native_metadata,
                    }

        logger.debug(f"Extracted native types for {len(native_types)} fields")
        return native_types

    def _extract_desired_type_definitions(
        self, payload: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract desired_type definitions from the original rules payload.

        Args:
            payload: Original rules payload with desired_type definitions

        Returns:
            Dict mapping field names to desired type information:
            {
                "field_name": {
                    "table": "table_name",
                    "desired_type": "INTEGER",
                    "metadata": {"desired_max_length": 50}
                }
            }
        """
        desired_type_definitions = {}

        # Handle both single-table and multi-table formats
        is_multi_table = "rules" not in payload

        if is_multi_table:
            # Multi-table format
            for table_name, table_config in payload.items():
                if not isinstance(table_config, dict) or "rules" not in table_config:
                    continue

                rules = table_config.get("rules", [])
                for rule_item in rules:
                    if not isinstance(rule_item, dict):
                        continue

                    field_name = rule_item.get("field")
                    desired_type = rule_item.get("desired_type")

                    if field_name and desired_type:
                        # Parse desired type to get canonical type
                        from shared.utils.type_parser import TypeParseError, TypeParser

                        try:
                            parsed_desired = TypeParser.parse_type_definition(
                                desired_type
                            )
                            canonical_desired_type = parsed_desired.get("type")

                            # Extract metadata with desired_ prefix
                            desired_metadata = {}
                            for key, value in parsed_desired.items():
                                if key != "type":
                                    desired_metadata[f"desired_{key}"] = value

                            desired_type_definitions[field_name] = {
                                "table": table_name,
                                "desired_type": canonical_desired_type,
                                "original_desired_type": desired_type,
                                "metadata": desired_metadata,
                            }
                        except TypeParseError as e:
                            logger.warning(
                                f"Failed to parse desired_type '{desired_type}' for "
                                f"field '{field_name}': {e}"
                            )

        else:
            # Single-table format
            rules = payload.get("rules", [])
            table_name = "unknown"  # We don't have table name in single-table format

            for rule_item in rules:
                if not isinstance(rule_item, dict):
                    continue

                field_name = rule_item.get("field")
                desired_type = rule_item.get("desired_type")

                if field_name and desired_type:
                    # Parse desired type to get canonical type
                    from shared.utils.type_parser import TypeParseError, TypeParser

                    try:
                        parsed_desired = TypeParser.parse_type_definition(desired_type)
                        canonical_desired_type = parsed_desired.get("type")

                        # Extract metadata with desired_ prefix
                        desired_metadata = {}
                        for key, value in parsed_desired.items():
                            if key != "type":
                                desired_metadata[f"desired_{key}"] = value

                        desired_type_definitions[field_name] = {
                            "table": table_name,
                            "desired_type": canonical_desired_type,
                            "original_desired_type": desired_type,
                            "metadata": desired_metadata,
                        }
                    except TypeParseError as e:
                        logger.warning(
                            f"Failed to parse desired_type '{desired_type}' "
                            f"for field '{field_name}': {e}"
                        )

        logger.debug(
            "Extracted desired_type definitions for "
            f"{len(desired_type_definitions)} fields"
        )
        return desired_type_definitions

    async def execute_additional_rules_phase(
        self,
        other_rules: List[RuleSchema],
        schema_results: List[Dict[str, Any]],
        skip_map: Dict[str, Dict[str, str]],
    ) -> Tuple[List[Any], float]:
        """Execute additional rules with filtering based on schema results.

        Currently implements skip semantics for testing the two-phase framework.
        Future versions will implement desired_type compatibility analysis.

        Args:
            other_rules: Non-schema rules to execute
            schema_results: Results from schema phase for analysis
            skip_map: Pre-computed skip decisions based on schema results

        Returns:
            Tuple of (results, execution_seconds)
        """
        logger.debug(
            f"Phase 2: Executing {len(other_rules)} additional rules "
            "with skip semantics"
        )

        if not other_rules:
            return [], 0.0

        # Filter out rules that should be skipped based on schema results
        filtered_rules = []
        skipped_count = 0

        for rule in other_rules:
            rule_id = str(rule.id)
            if rule_id in skip_map:
                skipped_count += 1
                logger.debug(
                    f"Phase 2: Skipping rule {rule.name} - "
                    f"{skip_map[rule_id]['skip_reason']}"
                )
                continue
            filtered_rules.append(rule)

        logger.debug(
            f"Phase 2: Executing {len(filtered_rules)} rules, skipping {skipped_count}"
        )

        if not filtered_rules:
            return [], 0.0

        validator = _create_validator(
            source_config=self.source_config,
            atomic_rules=filtered_rules,
            core_config=self.core_config,
            cli_config=self.cli_config,
        )

        # Execute validation directly without _run_validation to avoid
        # asyncio.run() conflicts
        start = _now()
        logger.debug("Starting additional rules validation")
        try:
            results = await validator.validate()
            exec_seconds = (_now() - start).total_seconds()
            logger.debug(f"Additional rules validation returned {len(results)} results")
        except Exception as e:
            logger.error(f"Additional rules validation failed: {str(e)}")
            results, exec_seconds = [], 0.0

        logger.debug(f"Phase 2: Completed in {exec_seconds:.3f}s")

        return results, exec_seconds


class ResultMerger:
    """Merges results from two-phase execution to maintain existing output format."""

    @staticmethod
    def merge_results(
        schema_results_list: List[Any],
        additional_results_list: List[Any],
        schema_rules: List[RuleSchema],
        other_rules: List[RuleSchema],
        skip_map: Dict[str, Dict[str, str]],
        generated_desired_type_rules: Optional[List[RuleSchema]] = None,
    ) -> Tuple[List[Any], List[RuleSchema]]:
        """Merge results from both phases and reconstruct skipped results.

        Args:
            schema_results_list: Results from schema phase
            additional_results_list: Results from additional rules phase
            schema_rules: Schema rules that were executed
            other_rules: Other rules (some may have been skipped)
            skip_map: Information about skipped rules
            generated_desired_type_rules: Dynamically generated desired_type rules

        Returns:
            Tuple of (combined_results, all_atomic_rules)
        """
        logger.debug("Merging results from two-phase execution")

        # Combine all rules for consistent processing
        if generated_desired_type_rules is None:
            generated_desired_type_rules = []
        all_atomic_rules = schema_rules + other_rules + generated_desired_type_rules

        # Start with executed results
        combined_results = list(schema_results_list) + list(additional_results_list)

        # Create synthetic results for skipped rules to maintain output consistency
        executed_rule_ids = set()
        for result in combined_results:
            if hasattr(result, "rule_id"):
                executed_rule_ids.add(str(result.rule_id))
            elif isinstance(result, dict):
                executed_rule_ids.add(str(result.get("rule_id", "")))

        # Create placeholder results for skipped rules
        for rule in other_rules:
            rule_id = str(rule.id)
            if rule_id in skip_map and rule_id not in executed_rule_ids:
                # Create a synthetic result for skipped rule
                synthetic_result = {
                    "rule_id": rule.id,
                    "status": "SKIPPED",
                    "skip_reason": skip_map[rule_id]["skip_reason"],
                    "dataset_metrics": [],
                    "execution_time": 0.0,
                    "execution_message": "Skipped due to "
                    f"{skip_map[rule_id]['skip_reason']}",
                    "error_message": None,
                    "sample_data": None,
                    "cross_db_metrics": None,
                    "execution_plan": {},
                    "started_at": None,
                    "ended_at": None,
                }
                combined_results.append(synthetic_result)

        logger.debug(
            f"Merged {len(schema_results_list)} schema + "
            f"{len(additional_results_list)} additional + {len(skip_map)} "
            f"skipped = {len(combined_results)} total results"
        )

        return combined_results, all_atomic_rules


def _emit_table_output(
    *,
    source: str,
    atomic_rules: List[RuleSchema],
    results: List[Any],
    skip_map: Dict[str, Dict[str, str]],
    schema_results: List[Dict[str, Any]],
    exec_seconds: float,
) -> None:
    rule_map = {str(rule.id): rule for rule in atomic_rules}

    table_results: List[Dict[str, Any]] = []

    def _dataset_total(res: Dict[str, Any]) -> int:
        if isinstance(res.get("total_records"), int):
            return int(res.get("total_records") or 0)
        dm = res.get("dataset_metrics") or []
        total = 0
        for m in dm:
            if hasattr(m, "total_records"):
                total = max(total, int(getattr(m, "total_records", 0) or 0))
            elif isinstance(m, dict):
                total = max(total, int(m.get("total_records", 0) or 0))
        return total

    for r in results:
        rd: Dict[str, Any]
        if hasattr(r, "model_dump"):
            try:
                rd = cast(Dict[str, Any], r.model_dump())
            except Exception:
                rd = {}
        elif isinstance(r, dict):
            rd = r
        else:
            rd = {}
        rid = str(rd.get("rule_id", ""))
        rule = rule_map.get(rid)
        if rule is not None:
            rd["rule_type"] = rule.type.value
            rd["column_name"] = rule.get_target_column()
            rd.setdefault("rule_name", rule.name)
            if rule.target and rule.target.entities:
                rd["table_name"] = rule.target.entities[0].table
        if rid in skip_map:
            rd["status"] = skip_map[rid]["status"]
            rd["skip_reason"] = skip_map[rid]["skip_reason"]
        table_results.append(rd)

    table_records: Dict[str, int] = {}
    for rd in table_results:
        table_name = rd.get("table_name", "unknown")
        total = _dataset_total(rd)
        if total > 0:
            table_records[table_name] = max(table_records.get(table_name, 0), total)

    header_total_records = sum(table_records.values())

    def _calc_failed(res: Dict[str, Any]) -> int:
        if isinstance(res.get("failed_records"), int):
            return int(res.get("failed_records") or 0)
        dm = res.get("dataset_metrics") or []
        total = 0
        for m in dm:
            if hasattr(m, "failed_records"):
                total += int(getattr(m, "failed_records", 0) or 0)
            elif isinstance(m, dict):
                total += int(m.get("failed_records", 0) or 0)
        return total

    for rd in table_results:
        if "failed_records" not in rd:
            rd["failed_records"] = _calc_failed(rd)
        if "total_records" not in rd:
            rd["total_records"] = _dataset_total(rd)

    tables_grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for rd in table_results:
        if rd.get("rule_type") == RuleType.SCHEMA.value:
            continue
        table_name = rd.get("table_name", "unknown")
        if table_name not in tables_grouped:
            tables_grouped[table_name] = {}

        col = rd.get("column_name", "")
        if col:
            if col not in tables_grouped[table_name]:
                tables_grouped[table_name][col] = {"column": col, "issues": []}

            status: Any = str(rd.get("status", "UNKNOWN"))

            # Check if this is a desired_type validation rule by looking at rule name
            rule_name = rd.get("rule_name", "")
            if rule_name and rule_name.startswith("desired_type_"):
                key = "desired_type"
            elif rd.get("rule_type") == RuleType.NOT_NULL.value:
                key = "not_null"
            elif rd.get("rule_type") == RuleType.RANGE.value:
                key = "range"
            elif rd.get("rule_type") == RuleType.ENUM.value:
                key = "enum"
            else:
                key = rd.get("rule_type", "unknown").lower()

            if status in {"FAILED", "ERROR", "SKIPPED"}:
                tables_grouped[table_name][col]["issues"].append(
                    {
                        "check": key,
                        "status": status,
                        "failed_records": int(rd.get("failed_records", 0) or 0),
                        "skip_reason": rd.get("skip_reason"),
                    }
                )

    all_columns_by_table: Dict[str, List[str]] = {}
    for rule in atomic_rules:
        if rule.target and rule.target.entities:
            table_name = rule.target.entities[0].table
            if table_name not in all_columns_by_table:
                all_columns_by_table[table_name] = []

            if rule.type == RuleType.SCHEMA:
                if rule.parameters:
                    declared_cols = (rule.parameters.get("columns") or {}).keys()
                    for col in declared_cols:
                        if str(col) not in all_columns_by_table[table_name]:
                            all_columns_by_table[table_name].append(str(col))
            else:
                column_name = rule.get_target_column()
                if column_name and column_name not in all_columns_by_table[table_name]:
                    all_columns_by_table[table_name].append(column_name)

    for table_name, columns in all_columns_by_table.items():
        if table_name not in tables_grouped:
            tables_grouped[table_name] = {}
        for column_name in columns:
            if column_name not in tables_grouped[table_name]:
                tables_grouped[table_name][column_name] = {
                    "column": column_name,
                    "issues": [],
                }

    schema_rules_map = {
        str(rule.id): rule for rule in atomic_rules if rule.type == RuleType.SCHEMA
    }
    for schema_result in schema_results:
        rule_id = str(schema_result.get("rule_id", ""))
        rule = schema_rules_map.get(rule_id)
        if not rule:
            continue

        table_name = rule.get_target_info().get("table")
        if table_name is None or table_name not in tables_grouped:
            continue

        execution_plan = schema_result.get("execution_plan") or {}
        schema_details = execution_plan.get("schema_details", {}) or {}
        details = schema_details.get("field_results", []) or []
        for item in details:
            col = str(item.get("column"))
            if col not in tables_grouped[table_name]:
                continue
            if item.get("failure_code") == "FIELD_MISSING":
                tables_grouped[table_name][col]["issues"].append(
                    {"check": "missing", "status": "FAILED"}
                )
            elif item.get("failure_code") == "TYPE_MISMATCH":
                tables_grouped[table_name][col]["issues"].append(
                    {"check": "type", "status": "FAILED"}
                )
            elif item.get("failure_code") == "METADATA_MISMATCH":
                tables_grouped[table_name][col]["issues"].append(
                    {"check": "metadata", "status": "FAILED"}
                )

    lines: List[str] = []
    lines.append(f"✓ Checking {source}")

    total_failed_records = sum(
        int(r.get("failed_records", 0) or 0) for r in table_results
    )

    # Check which tables don't exist based on skip reasons
    tables_not_exist = set()
    for rule_id, skip_info in skip_map.items():
        if skip_info.get("skip_reason") == "TABLE_NOT_EXISTS":
            rule = rule_map.get(rule_id)
            if rule and rule.target and rule.target.entities:
                table_name = rule.target.entities[0].table
                tables_not_exist.add(table_name)

    # Include all tables (existing and non-existing) in sorted output
    all_table_names = set(tables_grouped.keys()) | tables_not_exist
    sorted_tables = sorted(all_table_names)

    for table_name in sorted_tables:
        records = table_records.get(table_name, 0)
        lines.append(f"\n📋 Table: {table_name} ({records:,} records)")

        # If table doesn't exist, show only that error
        if table_name in tables_not_exist:
            lines.append("✗ Table does not exist or cannot be accessed")
            continue

        table_grouped = tables_grouped[table_name]
        ordered_columns = all_columns_by_table.get(table_name, [])

        # Fallback for columns that might appear in results but not in rules
        # (e.g., from a different source)
        result_columns = sorted(table_grouped.keys())
        for col in result_columns:
            if col not in ordered_columns:
                ordered_columns.append(col)

        for col in ordered_columns:
            if col not in table_grouped:
                lines.append(f"✓ {col}: OK")
                continue

            issues = table_grouped[col]["issues"]

            if not issues:
                lines.append(f"✓ {col}: OK")
                continue

            is_missing = any(
                i.get("check") == "missing" or i.get("skip_reason") == "FIELD_MISSING"
                for i in issues
            )

            if is_missing:
                lines.append(f"✗ {col}: missing (skipped dependent checks)")
                continue

            unique_issues: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for issue in issues:
                key_ = (str(issue.get("status")), str(issue.get("check")))
                if key_ not in unique_issues:
                    unique_issues[key_] = issue

            final_issues = sorted(
                unique_issues.values(), key=lambda x: str(x.get("check"))
            )

            issue_descs: List[str] = []
            for i in final_issues:
                status = i.get("status")
                check = i.get("check", "unknown")

                if status in {"FAILED", "ERROR"}:
                    fr = i.get("failed_records", 0)
                    if status == "ERROR":
                        issue_descs.append(f"{check} error")
                    else:
                        # For structural validation issues (type, metadata),
                        # don't show record counts
                        if check in {"type", "metadata"}:
                            issue_descs.append(f"{check} failed")
                        else:
                            issue_descs.append(f"{check} failed ({fr} failures)")
                elif status == "SKIPPED":
                    skip_reason = i.get("skip_reason")
                    if skip_reason == "FIELD_MISSING":
                        issue_descs.append(f"{check} skipped (field missing)")
                    else:
                        reason_text = skip_reason or "unknown reason"
                        issue_descs.append(f"{check} skipped ({reason_text})")

            if not issue_descs:
                lines.append(f"✓ {col}: OK")
            else:
                lines.append(f"✗ {col}: { ', '.join(issue_descs)}")

    total_columns = sum(len(all_columns_by_table.get(t, [])) for t in sorted_tables)
    passed_columns = sum(
        sum(
            1
            for c in all_columns_by_table.get(t, [])
            if not tables_grouped.get(t, {}).get(c, {}).get("issues", [])
        )
        for t in sorted_tables
    )
    failed_columns = total_columns - passed_columns
    overall_error_rate = (
        0.0
        if header_total_records == 0
        else (total_failed_records / max(header_total_records, 1)) * 100
    )

    if len(tables_grouped) > 1:
        lines.append("\n📊 Multi-table Summary:")
        for table_name in sorted_tables:
            table_cols = all_columns_by_table.get(table_name, [])
            table_columns_count = len(table_cols)
            table_passed = sum(
                1
                for c in table_cols
                if not tables_grouped[table_name].get(c, {}).get("issues")
            )
            table_failed = table_columns_count - table_passed
            lines.append(
                f"  {table_name}: {table_passed} passed, {table_failed} failed"
            )

    lines.append(
        f"\nSummary: {passed_columns} passed, {failed_columns} failed"
        f" ({overall_error_rate:.2f}% overall error rate)"
    )
    lines.append(f"Time: {exec_seconds:.2f}s")

    _safe_echo("\n".join(lines))


@click.command("schema")
@click.option(
    "--conn",
    "connection_string",
    required=True,
    help="Database connection string or file path",
)
@click.option(
    "--rules",
    "rules_file",
    type=click.Path(exists=True, readable=True),
    required=True,
    help="Path to schema rules file (JSON) - supports both single-table "
    "and multi-table formats",
)
@click.option(
    "--output",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format",
)
@click.option(
    "--fail-on-error",
    is_flag=True,
    default=False,
    help="Return exit code 1 if any error occurs during execution",
)
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose output")
@click.option(
    "--table",
    "table_name",
    help=(
        "Table name (optional for single-table validation, takes precedence "
        "when JSON has no table names)"
    ),
)
def schema_command(
    connection_string: str,
    rules_file: str,
    output: str,
    fail_on_error: bool,
    verbose: bool,
    table_name: Optional[str],
) -> None:
    """
    Schema validation command with support for both single-table
    and multi-table validation.
    """

    from cli.core.config import get_cli_config
    from core.config import get_core_config

    try:
        _maybe_echo_analyzing(connection_string, output)
        _guard_empty_source_file(connection_string)

        # Load rules first to determine if we should use --table parameter
        rules_payload = _read_rules_payload(rules_file)
        is_multi_table_rules = "rules" not in rules_payload

        # Use --table parameter only for single-table format
        #  (when JSON has no table names)
        table_for_parser = None if is_multi_table_rules else table_name
        source_config = SourceParser().parse_source(connection_string, table_for_parser)
        if is_multi_table_rules:
            source_config.parameters["is_multi_table"] = True

        warnings, rules_count = _validate_rules_payload(rules_payload)
        _emit_warnings(warnings, output)

        # Two-phase execution: separate schema and other rules
        schema_rules, other_rules = _decompose_schema_payload(
            rules_payload, source_config
        )
        all_atomic_rules = schema_rules + other_rules

        if not all_atomic_rules:
            _early_exit_when_no_rules(
                source=connection_string,
                rules_file=rules_file,
                output=output,
                fail_on_error=fail_on_error,
            )
            return

        core_config = get_core_config()
        cli_config = get_cli_config()

        # Phase 1: Execute schema rules only
        # schema_executor = SchemaPhaseExecutor(
        #     source_config=source_config, core_config=core_config,
        #     cli_config=cli_config
        # )

        # Execute two-phase validation in a single event loop to avoid
        # connection issues
        async def execute_two_phase_validation() -> tuple:
            # start_time = _now()

            # Phase 1: Execute schema rules only
            if schema_rules:
                schema_validator = _create_validator(
                    source_config=source_config,
                    atomic_rules=schema_rules,
                    core_config=core_config,
                    cli_config=cli_config,
                )
                schema_start = _now()
                schema_results_list = await schema_validator.validate()
                schema_exec_seconds = (_now() - schema_start).total_seconds()
                schema_results = _extract_schema_results(
                    atomic_rules=schema_rules, results=schema_results_list
                )
            else:
                schema_results_list, schema_exec_seconds, schema_results = [], 0.0, []

            # Compute skip logic based on schema results
            skip_map = _compute_skip_map(
                atomic_rules=all_atomic_rules, schema_results=schema_results
            )

            # Phase 2: Execute desired_type validation and additional rules
            desired_type_executor = DesiredTypePhaseExecutor(
                source_config=source_config,
                core_config=core_config,
                cli_config=cli_config,
            )

            # Execute desired_type validation
            (
                desired_type_results,
                desired_type_exec_seconds,
                generated_desired_type_rules,
            ) = await desired_type_executor.execute_desired_type_validation(
                schema_results=schema_results,
                original_payload=rules_payload,
                skip_map=skip_map,
            )

            # Execute remaining additional rules (non-desired_type rules) with skip
            # semantics
            additional_results_list = []
            additional_exec_seconds = 0.0

            if other_rules:
                # Filter out rules that should be skipped based on schema results
                filtered_rules = [
                    rule for rule in other_rules if str(rule.id) not in skip_map
                ]

                if filtered_rules:
                    additional_results, additional_exec_seconds = (
                        await desired_type_executor.execute_additional_rules_phase(
                            other_rules=filtered_rules,
                            schema_results=schema_results,
                            skip_map=skip_map,
                        )
                    )
                    additional_results_list = additional_results

            # Combine desired_type and additional results
            combined_additional_results = list(desired_type_results) + list(
                additional_results_list
            )
            total_additional_exec_seconds = (
                desired_type_exec_seconds + additional_exec_seconds
            )

            return (
                schema_results_list,
                schema_exec_seconds,
                schema_results,
                combined_additional_results,
                total_additional_exec_seconds,
                skip_map,
                generated_desired_type_rules,
            )

        import asyncio

        (
            schema_results_list,
            schema_exec_seconds,
            schema_results,
            additional_results_list,
            additional_exec_seconds,
            skip_map,
            generated_desired_type_rules,
        ) = asyncio.run(execute_two_phase_validation())

        # Merge results to maintain existing output format
        results, atomic_rules = ResultMerger.merge_results(
            schema_results_list,
            additional_results_list,
            schema_rules,
            other_rules,
            skip_map,
            generated_desired_type_rules,
        )

        # Total execution time
        exec_seconds = schema_exec_seconds + additional_exec_seconds

        if output.lower() == "json":
            _emit_json_output(
                source=connection_string,
                rules_file=rules_file,
                atomic_rules=atomic_rules,
                results=results,
                skip_map=skip_map,
                schema_results=schema_results,
                exec_seconds=exec_seconds,
            )
        else:
            _emit_table_output(
                source=connection_string,
                atomic_rules=atomic_rules,
                results=results,
                skip_map=skip_map,
                schema_results=schema_results,
                exec_seconds=exec_seconds,
            )

        def _status_of(item: Any) -> str:
            if hasattr(item, "status"):
                try:
                    return str(getattr(item, "status") or "").upper()
                except Exception:
                    return ""
            if isinstance(item, dict):
                return str(item.get("status", "") or "").upper()
            return ""

        any_failed = any(_status_of(r) == "FAILED" for r in results)
        raise click.exceptions.Exit(1 if any_failed or fail_on_error else 0)

    except click.UsageError:
        raise
    except click.exceptions.Exit:
        raise
    except Exception as e:
        logger.error(f"Schema command error: {str(e)}")
        _safe_echo(f"❌ Error: {str(e)}", err=True)
        raise click.exceptions.Exit(1)
