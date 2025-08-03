"""
Rule Parser

Parse inline rules and rules files into RuleSchema objects.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from cli.exceptions import (
    RuleParsingError,
    ValidationError,
)
from shared.enums import RuleAction, RuleCategory, RuleType, SeverityLevel
from shared.schema import RuleSchema
from shared.schema.base import RuleTarget, TargetEntity
from shared.utils.logger import get_logger


class RuleParser:
    """
    Rule parser for inline rules and JSON rule files.

    Supported inline rule formats:
    - not_null(column)
    - unique(column)
    - length(column,min,max)
    - range(column,min,max)
    - enum(column,value1,value2,...)
    - regex(column,pattern)
    """

    def __init__(self) -> None:
        """Initialize RuleParser"""
        self.logger = get_logger(__name__)

        # Rule pattern regex - Fix: use smarter parenthesis matching
        # The original r'(\w+)\(([^)]+)\)' stops at the first )
        # The new pattern correctly handles nested parentheses and escape characters
        self.rule_pattern = re.compile(
            r'(\w+)\(((?:[^()"\'\\]|\\[^]|"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*'
            r"(?:\\.[^\'\\]*)*\'|\([^()]*\))*)\)",
            re.IGNORECASE,
        )

        # Simplified version: match up to the last ), then process parameters
        self.simple_rule_pattern = re.compile(r"(\w+)\((.*)\)", re.IGNORECASE)

        # Supported rule types mapping
        self.rule_type_mapping = {
            "not_null": RuleType.NOT_NULL,
            "unique": RuleType.UNIQUE,
            "length": RuleType.LENGTH,
            "range": RuleType.RANGE,
            "enum": RuleType.ENUM,
            "regex": RuleType.REGEX,
            "date_format": RuleType.DATE_FORMAT,
        }

    def parse_rules(
        self, inline_rules: Optional[List[str]] = None, rules_file: Optional[str] = None
    ) -> List[RuleSchema]:
        """
        Parse rules from inline expressions and/or rules file.

        Args:
            inline_rules: List of inline rule expressions
            rules_file: Path to JSON rules file

        Returns:
            List[RuleSchema]: Parsed rule configurations

        Raises:
            ValidationError: If rule syntax or parameters are invalid
            RuleParsingError: If rule parsing fails
            FileNotFoundError: If rules file is not found
        """
        rules = []

        # Parse inline rules
        if inline_rules:
            self.logger.info(f"Parsing {len(inline_rules)} inline rules")
            for rule_expr in inline_rules:
                try:
                    rule = self._parse_inline_rule(rule_expr)
                    rules.append(rule)
                except Exception as e:
                    self.logger.error(
                        f"Failed to parse inline rule '{rule_expr}': {str(e)}"
                    )
                    raise RuleParsingError(
                        message=f"Invalid rule syntax: {rule_expr}",
                        rule_expression=rule_expr,
                    ) from e

        # Parse rules file
        if rules_file:
            self.logger.info(f"Parsing rules file: {rules_file}")
            file_rules = self._parse_rules_file(rules_file)
            rules.extend(file_rules)

        if not rules:
            raise ValidationError(message="No valid rules parsed", field="rules")

        self.logger.info(f"Successfully parsed {len(rules)} rules")
        return rules

    def _parse_inline_rule(self, rule_expr: str) -> RuleSchema:
        """
        Parse a single inline rule expression.

        Examples:
        - not_null(id)
        - length(name,2,50)
        - enum(status,active,inactive)
        """
        rule_expr = rule_expr.strip()

        # First try smart parenthesis matching
        match = self.rule_pattern.match(rule_expr)

        # If smart matching fails, use simplified matching
        if not match:
            match = self.simple_rule_pattern.match(rule_expr)

        if not match:
            raise RuleParsingError(
                message=f"Invalid rule syntax: {rule_expr}", rule_expression=rule_expr
            )

        rule_type_str, params_str = match.groups()

        # Validate rule type
        rule_type_str = rule_type_str.lower()
        if rule_type_str not in self.rule_type_mapping:
            # valid_types = ", ".join(self.rule_type_mapping.keys())
            raise ValidationError(
                message=f"Unsupported rule type: {rule_type_str}",
                field="rule_type",
                value=rule_type_str,
            )

        rule_type = self.rule_type_mapping[rule_type_str]

        # For regex rules, special parameter parsing is required
        if rule_type == RuleType.REGEX:
            # Regex rule: regex(column,pattern)
            # Need to find the first comma, then treat everything after as the pattern
            comma_pos = params_str.find(",")
            if comma_pos == -1:
                raise ValidationError(
                    message="Regex rule requires column and pattern", field="parameters"
                )

            column_name = params_str[:comma_pos].strip().strip("\"'")
            pattern = params_str[comma_pos + 1 :].strip()

            # Create rule parameters
            rule_params = self._build_rule_parameters(rule_type, [pattern])
        else:
            # Other rule types: use standard parameter parsing
            # Parse parameters
            params = [p.strip() for p in params_str.split(",")]

            if not params or not params[0]:
                raise ValidationError(
                    message="Rule must specify at least a column name", field="column"
                )

            column_name = params[0].strip("\"'")  # Remove quotes if present
            rule_params = self._build_rule_parameters(rule_type, params[1:])

        # Create rule schema
        return self._create_rule_schema(
            rule_type=rule_type,
            column_name=column_name,
            parameters=rule_params,
            description=f"CLI rule: {rule_expr}",
        )

    def _parse_rules_file(self, rules_file: str) -> List[RuleSchema]:
        """Parse JSON rules file"""
        try:
            with open(rules_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValidationError(
                message="Invalid JSON in rules file",
                field="rules_file",
                value=rules_file,
            ) from e
        except FileNotFoundError:
            raise FileNotFoundError(f"Rules file not found: {rules_file}")
        except Exception as e:
            raise ValidationError(
                message="Cannot read rules file", field="rules_file", value=rules_file
            ) from e

        # Validate file structure
        if not isinstance(data, dict):
            raise ValidationError(
                message="Rules file must contain a JSON object", field="rules_file"
            )

        if "rules" not in data:
            raise ValidationError(
                message="Rules file must contain a 'rules' array",
                field="rules_file",
                value="missing 'rules' key",
            )

        if not isinstance(data["rules"], list):
            raise ValidationError(
                message="'rules' must be an array",
                field="rules",
                value=type(data["rules"]).__name__,
            )

        # Parse individual rules
        rules = []
        for i, rule_data in enumerate(data["rules"]):
            try:
                rule = self._parse_file_rule(rule_data)
                rules.append(rule)
            except Exception as e:
                raise ValidationError(
                    message=f"Error parsing rule {i+1} in file", field=f"rules[{i}]"
                ) from e

        return rules

    def _parse_file_rule(self, rule_data: Dict[str, Any]) -> RuleSchema:
        """Parse a single rule from JSON file"""
        if not isinstance(rule_data, dict):
            raise ValidationError(message="Rule must be a JSON object", field="rule")

        # Required fields
        if "type" not in rule_data:
            raise ValidationError(message="Rule must have 'type' field", field="type")

        if "column" not in rule_data:
            raise ValidationError(
                message="Rule must have 'column' field", field="column"
            )

        rule_type_str = rule_data["type"].lower()
        if rule_type_str not in self.rule_type_mapping:
            # valid_types = ", ".join(self.rule_type_mapping.keys())
            raise ValidationError(
                message=f"Unsupported rule type: {rule_type_str}",
                field="type",
                value=rule_type_str,
            )

        rule_type = self.rule_type_mapping[rule_type_str]
        column_name = rule_data["column"]

        # Build parameters from rule data
        rule_params: Dict[str, Any] = {}

        # Handle type-specific parameters
        if rule_type == RuleType.LENGTH:
            if "min" in rule_data:
                min_length = int(rule_data["min"])
                if min_length < 0:
                    raise ValidationError(
                        message="Minimum length cannot be negative",
                        field="min",
                        value=min_length,
                    )
                rule_params["min_length"] = min_length
            if "max" in rule_data:
                max_length = int(rule_data["max"])
                if (
                    "min_length" in rule_params
                    and max_length < rule_params["min_length"]
                ):
                    raise ValidationError(
                        message="Maximum length cannot be less than minimum length",
                        field="max",
                        value=max_length,
                    )
                rule_params["max_length"] = max_length

        elif rule_type == RuleType.RANGE:
            if "min" in rule_data:
                rule_params["min_value"] = float(rule_data["min"])
            if "max" in rule_data:
                max_value = float(rule_data["max"])
                if "min_value" in rule_params and max_value < rule_params["min_value"]:
                    raise ValidationError(
                        message="Maximum value cannot be less than minimum value",
                        field="max",
                        value=max_value,
                    )
                rule_params["max_value"] = max_value

        elif rule_type == RuleType.ENUM:
            if "values" in rule_data:
                values = rule_data["values"]
                if not values:
                    raise ValidationError(
                        message="Enum values cannot be empty", field="values"
                    )
                rule_params["allowed_values"] = values
            elif "allowed_values" in rule_data:
                values = rule_data["allowed_values"]
                if not values:
                    raise ValidationError(
                        message="Enum values cannot be empty", field="allowed_values"
                    )
                rule_params["allowed_values"] = values

        elif rule_type == RuleType.REGEX:
            if "pattern" in rule_data:
                pattern = rule_data["pattern"]
                if not pattern:
                    raise ValidationError(
                        message="Regex pattern cannot be empty", field="pattern"
                    )
                # 🔧 Fix: also handle escaping for regex in JSON files
                rule_params["pattern"] = self._process_regex_pattern_from_json(pattern)
            elif "regex_pattern" in rule_data:
                pattern = rule_data["regex_pattern"]
                if not pattern:
                    raise ValidationError(
                        message="Regex pattern cannot be empty", field="regex_pattern"
                    )
                rule_params["pattern"] = self._process_regex_pattern_from_json(pattern)

        elif rule_type == RuleType.DATE_FORMAT:
            # Support multiple date format parameter names
            date_format = None
            if "format" in rule_data:
                date_format = rule_data["format"]
            elif "format_pattern" in rule_data:
                date_format = rule_data["format_pattern"]

            if not date_format:
                raise ValidationError(
                    message="Date format cannot be empty", field="format"
                )
            # Set both parameter names for compatibility
            rule_params["format"] = date_format
            rule_params["format_pattern"] = date_format

        # Optional fields
        description = rule_data.get("description", f"Rule for {column_name}")
        severity = rule_data.get("severity", "medium").upper()

        return self._create_rule_schema(
            rule_type=rule_type,
            column_name=column_name,
            parameters=rule_params,
            description=description,
            severity=(
                SeverityLevel(severity)
                if hasattr(SeverityLevel, severity)
                else SeverityLevel.MEDIUM
            ),
        )

    def _build_rule_parameters(
        self, rule_type: RuleType, params: List[str]
    ) -> Dict[str, Any]:
        """Build rule parameters based on rule type"""
        rule_params: Dict[str, Any] = {}

        # Helper function to strip quotes from parameters
        def strip_quotes(value: str) -> str:
            """Remove surrounding quotes from a string value"""
            if value and len(value) >= 2:
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    return value[1:-1]
            return value

        # Helper function to handle regex pattern escaping
        def process_regex_pattern(pattern: str) -> str:
            r"""
            Handle command-line escaping issues for regex patterns

            Command-line escaping issues:
            1. User input \\ will be escaped by the shell to \
            2. User input \. will be escaped by the shell to .
            3. Need to restore to correct regex syntax
            """
            # Remove quotes
            pattern = strip_quotes(pattern)

            # Handle command-line escaping issues
            # In the command line, backslashes may be escaped and need to be restored
            # For example: \\ should become \
            pattern = pattern.replace("\\\\", "\\")

            # Handle common escape sequences
            pattern = pattern.replace("\\n", "\n")
            pattern = pattern.replace("\\t", "\t")
            pattern = pattern.replace("\\r", "\r")

            # Validate regex syntax
            try:
                import re

                re.compile(pattern)
            except re.error as e:
                raise ValidationError(
                    message=f"Invalid regex pattern: {e}",
                    field="pattern",
                    value=pattern,
                )

            return pattern

        if rule_type == RuleType.NOT_NULL:
            # No additional parameters needed
            pass

        elif rule_type == RuleType.UNIQUE:
            # No additional parameters needed
            pass

        elif rule_type == RuleType.LENGTH:
            if len(params) >= 1:
                min_length = int(strip_quotes(params[0]))
                if min_length < 0:
                    raise ValidationError(
                        message="Minimum length cannot be negative",
                        field="min_length",
                        value=min_length,
                    )
                rule_params["min_length"] = min_length
            if len(params) >= 2:
                max_length = int(strip_quotes(params[1]))
                if (
                    "min_length" in rule_params
                    and max_length < rule_params["min_length"]
                ):
                    raise ValidationError(
                        message="Maximum length cannot be less than minimum length",
                        field="max_length",
                        value=max_length,
                    )
                rule_params["max_length"] = max_length

        elif rule_type == RuleType.RANGE:
            if len(params) >= 1:
                rule_params["min_value"] = float(strip_quotes(params[0]))
            if len(params) >= 2:
                max_value = float(strip_quotes(params[1]))
                if "min_value" in rule_params and max_value < rule_params["min_value"]:
                    raise ValidationError(
                        message="Maximum value cannot be less than minimum value",
                        field="max_value",
                        value=max_value,
                    )
                rule_params["max_value"] = max_value

        elif rule_type == RuleType.ENUM:
            if not params:
                raise ValidationError(
                    message="Enum rule requires at least one allowed value",
                    field="allowed_values",
                )

            # Smart type conversion: try to convert value to appropriate type
            def convert_enum_value(value: str) -> Union[str, int, float]:
                """Convert enum value to appropriate type"""
                # First remove quotes
                stripped_value = strip_quotes(value)

                # Try to convert to integer
                try:
                    if stripped_value.isdigit() or (
                        stripped_value.startswith("-") and stripped_value[1:].isdigit()
                    ):
                        return int(stripped_value)
                except ValueError:
                    pass

                # Try to convert to float
                try:
                    if "." in stripped_value:
                        return float(stripped_value)
                except ValueError:
                    pass

                # If neither, keep as string
                return stripped_value

            # Convert all enum values
            rule_params["allowed_values"] = [convert_enum_value(p) for p in params]

        elif rule_type == RuleType.REGEX:
            if not params or not params[0]:
                raise ValidationError(
                    message="Regex rule requires a pattern", field="pattern"
                )
            # 🔧 Fix: use dedicated regex processing function
            rule_params["pattern"] = process_regex_pattern(params[0])

        elif rule_type == RuleType.DATE_FORMAT:
            if not params or not params[0]:
                raise ValidationError(
                    message="Date format rule requires a format string", field="format"
                )
            # Set both parameter names for compatibility
            format_value = strip_quotes(params[0])
            rule_params["format"] = format_value
            rule_params["format_pattern"] = format_value

        return rule_params

    def _process_regex_pattern_from_json(self, pattern: str) -> str:
        """
        Handle regex patterns in JSON files

        Regex patterns in JSON files usually do not require command-line escaping,
        but need to validate syntax correctness
        """
        if not pattern:
            raise ValidationError(
                message="Regex pattern cannot be empty", field="pattern"
            )

        # Validate regex syntax
        try:
            import re

            re.compile(pattern)
        except re.error as e:
            raise ValidationError(
                message=f"Invalid regex pattern: {e}", field="pattern", value=pattern
            )

        return pattern

    def _create_rule_schema(
        self,
        rule_type: RuleType,
        column_name: str,
        parameters: Dict[str, Any],
        description: str,
        severity: SeverityLevel = SeverityLevel.MEDIUM,
    ) -> RuleSchema:
        """Create a RuleSchema instance"""
        # Generate a unique name based on rule type and column
        rule_id = str(uuid4())
        rule_name = f"{rule_type.name.lower()}_{column_name}_{rule_id[-8:]}"

        # Determine rule category based on rule type
        category = self._get_rule_category(rule_type)

        try:
            return RuleSchema(
                id=rule_id,
                name=rule_name,
                description=description,
                connection_id=uuid4(),
                type=rule_type,
                category=category,
                severity=severity,
                action=RuleAction.ALERT,
                threshold=0.0,
                template_id=None,
                is_active=True,
                tags=[],
                target=RuleTarget(
                    entities=[TargetEntity(database="", table="", column=column_name)],
                    relationship_type="single_table",
                ),
                parameters=parameters,
            )
        except Exception as e:
            raise e
            # raise RuleConfigurationError(
            #     message=f"Failed to create rule schema: {str(e)}",
            #     rule_id=rule_id,
            #     config_key="parameters",
            #     config_value=parameters
            # ) from e

    def _get_rule_category(self, rule_type: RuleType) -> RuleCategory:
        """Determine rule category based on rule type"""
        if rule_type in [RuleType.NOT_NULL, RuleType.LENGTH]:
            return RuleCategory.COMPLETENESS
        elif rule_type == RuleType.UNIQUE:
            return RuleCategory.UNIQUENESS
        else:
            return RuleCategory.VALIDITY
