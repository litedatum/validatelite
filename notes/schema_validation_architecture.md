# Validatelite Schema Validation Architecture Overview

This document outlines the end-to-end architecture of the schema validation process in `validatelite`, as of our last discussion. Its purpose is to serve as a reference for future development.

## End-to-End Workflow

The system is designed with a clear separation of concerns, divided into three main layers:

### 1. User-Facing Schema Definition (The "What")

- **File Format**: The user defines validation rules in a simple JSON file.
- **Structure**: The file contains a `rules` array, where each object specifies checks for a particular `field`.
- **Example (`rules.json`):**
  ```json
  {
    "rules": [
      { "field": "product_code", "type": "string", "required": true },
      { "field": "price", "type": "float", "min": 0 },
      { "field": "status", "type": "string", "enum": ["active", "inactive"] }
    ]
  }
  ```

### 2. CLI Command Layer (The "Translator")

- **Key File**: `cli/commands/schema.py`
- **Purpose**: This layer acts as a **translator** or **decomposer**. It parses the user-friendly `rules.json` and converts each check into one or more atomic, engine-readable `RuleSchema` objects.
- **Decomposition Logic**:
  - `"required": true` is decomposed into a `NOT_NULL` rule.
  - `"min": 0` is decomposed into a `RANGE` rule.
  - `"enum": [...]` is decomposed into an `ENUM` rule.
  - All fields with a `"type"` definition are collected and bundled into a **single, table-level `SCHEMA` rule**.

- **Example of Generated `SCHEMA` Rule**: The `rules.json` above would result in a `SCHEMA` rule with parameters like this, which is then sent to the core engine:
  ```python
  {
    "columns": {
      "product_code": { "expected_type": "STRING" },
      "price": { "expected_type": "FLOAT" },
      "status": { "expected_type": "STRING" }
    }
  }
  ```

### 3. Core Engine Layer (The "Executor")

- **Key Files**:
  - `core/registry/builtin_rule_types.py`: Defines the structure of the `SCHEMA` rule and its allowed parameters.
  - `core/executors/validity_executor.py`: Contains the `_execute_schema_rule` method that implements the validation logic.
- **Execution Logic**:
  1. The executor receives the `SCHEMA` rule from the CLI layer.
  2. It queries the database to get the actual table metadata (column names and types).
  3. **Crucially, it simplifies the database-specific type**. For example, `VARCHAR(100)` becomes `STRING`, and `DECIMAL(10, 2)` becomes `FLOAT`.
  4. It compares this simplified, canonical type with the `expected_type` from the rule's parameters.

- **Current Limitation**: By design, this process **only validates the general data type category** and deliberately **ignores physical storage attributes like length, precision, and scale**.

---

## Proposed Enhancement Plan

To add length and precision validation, we will extend the existing architecture at all three layers.

1.  **Enhance User-Facing Schema**: Officially support `length`, `precision`, and `scale` keys in the `rules.json` file.
    ```json
    { "field": "product_code", "type": "string", "length": 50 },
    { "field": "price", "type": "float", "precision": 10, "scale": 2 }
    ```

2.  **Modify CLI Translator (`cli/commands/schema.py`)**: Update the decomposition logic to read these new keys and include them in the parameters of the generated `SCHEMA` rule.
    ```python
    "price": {
      "expected_type": "FLOAT",
      "precision": 10,
      "scale": 2
    }
    ```

3.  **Modify Core Rule Definition (`core/registry/builtin_rule_types.py`)**: Update the `SCHEMA` rule's `parameters_schema` to officially allow these new keys.

4.  **Modify Core Executor (`core/executors/validity_executor.py`)**: Enhance the `_execute_schema_rule` method to:
    a. Fetch the **full, unmodified** data type from the database metadata (e.g., `VARCHAR(50)`).
    b. After checking the canonical type, perform additional checks by parsing the length/precision/scale from the database type string and comparing them against the values now present in the rule parameters.
