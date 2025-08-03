"""
Parameter conversion utility

Provides functions for rule parameter conversion and compatibility handling.
"""

import copy
import logging
from typing import Any, Dict, Optional, Tuple

# Configure logger
logger = logging.getLogger(__name__)


def convert_legacy_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert legacy parameter format to new format

    Converts legacy parameter format (fields like database, table_name, column_name)
    to the new parameter format (structures like target, rule_config)

    Args:
        params: Legacy parameters

    Returns:
        Dict[str, Any]: New parameters
    """
    # If already in new format, return directly (shallow copy)
    if "target" in params or "targets" in params:
        # Note: return a shallow copy of params to avoid modifying the original
        # parameters
        return {**params}

    # Create a deep copy of parameters to avoid modifying the original parameters
    params_copy = copy.deepcopy(params)

    # Create new format parameters
    new_params: Dict[str, Any] = {}

    # Extract target information
    target = {}

    # Handle database field
    if "database" in params_copy:
        target["database"] = params_copy.pop("database")

    # Handle table_name field
    if "table_name" in params_copy:
        target["table"] = params_copy.pop("table_name")
    elif "table" in params_copy:
        target["table"] = params_copy.pop("table")

    # Handle column_name field
    if "column_name" in params_copy:
        target["column"] = params_copy.pop("column_name")
    elif "column" in params_copy:
        target["column"] = params_copy.pop("column")

    # Add target information
    new_params["target"] = target

    # Create rule config
    rule_config = {}

    # Handle filter condition
    if "filter_condition" in params_copy:
        rule_config["filter_condition"] = params_copy.pop("filter_condition")

    # Handle enum rule specific parameters
    if "allowed_values" in params_copy:
        rule_config["allowed_values"] = params_copy.pop("allowed_values")
    elif "values" in params_copy:
        rule_config["allowed_values"] = params_copy.pop("values")

    if "extract_domain" in params_copy:
        rule_config["extract_domain"] = params_copy.pop("extract_domain")

    # Handle range rule specific parameters
    if "min" in params_copy:
        rule_config["min"] = params_copy.pop("min")
    if "max" in params_copy:
        rule_config["max"] = params_copy.pop("max")
    if "min_value" in params_copy:
        rule_config["min_value"] = params_copy.pop("min_value")
    if "max_value" in params_copy:
        rule_config["max_value"] = params_copy.pop("max_value")

    # Handle regex rule specific parameters
    if "pattern" in params_copy:
        rule_config["pattern"] = params_copy.pop("pattern")

    # Handle length rule specific parameters
    if "min_length" in params_copy:
        rule_config["min_length"] = params_copy.pop("min_length")
    if "max_length" in params_copy:
        rule_config["max_length"] = params_copy.pop("max_length")

    # Handle date format rule specific parameters
    if "format" in params_copy:
        rule_config["format"] = params_copy.pop("format")

    # Handle custom SQL rule specific parameters
    if "sql" in params_copy:
        rule_config["sql"] = params_copy.pop("sql")

    # Handle other config parameters
    for key, value in params_copy.items():
        # Skip already processed fields and internal fields
        if key in ["target", "targets", "rule_config"] or key.startswith("_"):
            continue

        rule_config[key] = value

    # Handle existing rule_config
    if "rule_config" in params_copy and isinstance(params_copy["rule_config"], dict):
        # Merge existing rule_config and extracted rule_config
        rule_config.update(params_copy["rule_config"])

    # Add rule config
    new_params["rule_config"] = rule_config if rule_config else {}

    return new_params


def extract_target_info(params: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
    """
    Extract target information from parameters

    Args:
        params: Rule parameters

    Returns:
        Tuple[str, str, Optional[str]]: (database, table, column)

    Raises:
        ValueError: Raised when parameter structure is invalid
    """
    # Save original parameters to check structure
    original_params = params.copy()

    # Convert legacy parameter format
    params = convert_legacy_params(params)

    # Check existence of target fields in original parameters, or if it's legacy format
    original_has_target = "target" in original_params
    original_has_targets = "targets" in original_params
    is_legacy_format = any(
        key in original_params for key in ["database", "table_name", "column_name"]
    )

    # If original parameters do not have any target-related fields and are not
    # legacy format, raise missing error
    if not original_has_target and not original_has_targets and not is_legacy_format:
        raise ValueError("Invalid parameter structure, missing target or targets")

    # Check targets field
    if "targets" in params:
        targets = params["targets"]
        if not isinstance(targets, list) or len(targets) == 0:
            raise ValueError("Invalid parameter structure, missing target or targets")
        else:
            # Use the first target
            target = targets[0]
            database = target.get("database", "")
            table = target.get("table", "")
            column = target.get("column")

            if not database or not table:
                raise ValueError(
                    "Target information incomplete, missing database or table"
                )

            return database, table, column

    # Check target field
    if "target" in params:
        target = params["target"]
        if not isinstance(target, dict):
            raise ValueError("Invalid parameter structure, missing target or targets")

        # target field exists and is a dict, check content
        database = target.get("database", "")
        table = target.get("table", "")
        column = target.get("column")

        # Check database and table
        if not database or not table:
            raise ValueError("Target information incomplete, missing database or table")

        return database, table, column

    # Should not reach here theoretically
    raise ValueError("Invalid parameter structure, missing target or targets")


def extract_rule_config(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract rule config from parameters

    Args:
        params: Rule parameters

    Returns:
        Dict[str, Any]: Rule config
    """
    # Convert legacy parameter format
    params = convert_legacy_params(params)

    # Extract rule config
    if "rule_config" in params and isinstance(params["rule_config"], dict):
        return params["rule_config"]
    else:
        # Try to extract rule config from parameters
        config = {}

        # Handle filter_condition field
        if "filter_condition" in params:
            config["filter_condition"] = params["filter_condition"]

        # Handle other possible config fields
        for key, value in params.items():
            # Skip already processed fields and internal fields
            if key in [
                "target",
                "targets",
                "rule_config",
                "filter_condition",
            ] or key.startswith("_"):
                continue

            config[key] = value

        return config


def extract_filter_condition(params: Dict[str, Any]) -> Optional[str]:
    """
    Extract filter condition from parameters

    Args:
        params: Rule parameters

    Returns:
        Optional[str]: Filter condition
    """
    # Convert legacy parameter format
    params = convert_legacy_params(params)

    # Extract from rule_config
    if "rule_config" in params and isinstance(params["rule_config"], dict):
        return params["rule_config"].get("filter_condition")

    # Extract directly from params
    return params.get("filter_condition")


def merge_params(
    base_params: Dict[str, Any], override_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge parameters

    Merge override parameters into base parameters. If there are parameters with
    the same name, the value in override_params will overwrite the value in
    base_params.

    Args:
        base_params: Base parameters
        override_params: Override parameters

    Returns:
        Dict[str, Any]: Merged parameters
    """
    # Convert to new format
    base_params = convert_legacy_params(base_params)
    override_params = convert_legacy_params(override_params)

    # Create a deep copy of result parameters (avoid modifying the original object)
    result_params = copy.deepcopy(base_params)

    # Merge target information
    if "target" in override_params:
        if "target" in result_params:
            # Deep merge target field
            for key, value in override_params["target"].items():
                result_params["target"][key] = value
        else:
            result_params["target"] = override_params["target"]

    if "targets" in override_params:
        result_params["targets"] = override_params["targets"]

    # Merge rule config
    if "rule_config" in override_params:
        if "rule_config" not in result_params:
            result_params["rule_config"] = {}

        # Deep merge rule config - override fields with the same name
        for key, value in override_params["rule_config"].items():
            result_params["rule_config"][key] = value

    return result_params
