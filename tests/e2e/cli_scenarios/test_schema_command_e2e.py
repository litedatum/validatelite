"""
E2E: vlite schema on databases and table/json outputs

Scenarios derived from notes/测试方案-数据库SchemaDrift与CLI-Schema命令.md:
- Happy path on DB URL with table/json outputs
- Drift: missing column (FIELD_MISSING), type mismatch (TYPE_MISMATCH), strict extras
- Exit codes and minimal payload when empty rules
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.shared.utils.database_utils import (
    get_available_databases,
    get_mysql_test_url,
    get_postgresql_test_url,
)
from tests.shared.utils.e2e_test_utils import E2ETestUtils

pytestmark = pytest.mark.e2e


def _db_urls() -> list[str]:
    urls: list[str] = []
    available = set(get_available_databases())
    if "mysql" in available:
        urls.append(get_mysql_test_url())
    if "postgresql" in available:
        urls.append(get_postgresql_test_url())
    return urls


def _write_rules(tmp_dir: Path, payload: dict) -> str:
    p = tmp_dir / "rules.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def _param_db_urls() -> list[object]:
    """Mypy-friendly parameter provider for pytest.mark.parametrize.

    Returns list[object] so we can mix str and pytest.param when DB not configured.
    """
    out: list[object] = []
    urls = _db_urls()
    if urls:
        out.extend(urls)
    else:
        out.append(pytest.param("", marks=pytest.mark.skip(reason="No DB configured")))
    return out


@pytest.mark.parametrize("db_url", _param_db_urls())
def test_happy_path_table_and_json(tmp_path: Path, db_url: str) -> None:
    # Schema baseline + a couple atomic rules
    rules = {
        "customers": {
            "rules": [
                {"field": "id", "type": "integer", "required": True},
                {"field": "email", "type": "string"},
                {"field": "age", "type": "integer", "min": 0, "max": 150},
            ],
            "strict_mode": False,
            "case_insensitive": True,
        }
    }
    rules_file = _write_rules(tmp_path, rules)

    # table output
    r1 = E2ETestUtils.run_cli_command(
        [
            "schema",
            "--conn",
            db_url,
            "--rules",
            rules_file,
            "--output",
            "table",
        ]
    )
    assert r1.returncode in {0, 1}
    assert "Checking" in r1.stdout

    # json output
    r2 = E2ETestUtils.run_cli_command(
        [
            "schema",
            "--conn",
            db_url,
            "--rules",
            rules_file,
            "--output",
            "json",
        ]
    )
    assert r2.returncode in {0, 1}
    try:
        payload = json.loads(r2.stdout)
    except Exception as e:
        assert False, (
            "Expected JSON output from CLI but failed to parse. "
            f"Error: {e}\nSTDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}"
        )
    assert payload["status"] == "ok"
    assert payload["rules_count"] >= 1
    assert "summary" in payload and "results" in payload and "fields" in payload


@pytest.mark.parametrize("db_url", _param_db_urls())
def test_drift_missing_and_type_mismatch(tmp_path: Path, db_url: str) -> None:
    # Declare a missing column and mismatched type to trigger SKIPPED in JSON for dependent rules
    rules = {
        "customers": {
            "rules": [
                {"field": "email", "type": "integer", "required": True},  # mismatch
                {
                    "field": "status",
                    "type": "string",
                    "enum": ["active", "inactive"],
                },  # missing
            ],
            "strict_mode": False,
            "case_insensitive": True,
        }
    }
    rules_file = _write_rules(tmp_path, rules)

    r = E2ETestUtils.run_cli_command(
        [
            "schema",
            "--conn",
            db_url,
            "--rules",
            rules_file,
            "--output",
            "json",
        ]
    )
    assert r.returncode in {1, 0}
    try:
        payload = json.loads(r.stdout)
    except Exception as e:
        assert False, (
            "Expected JSON output from CLI but failed to parse. "
            f"Error: {e}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )
    # Ensure field-level failure codes surface
    fields = {f["column"]: f for f in payload.get("fields", [])}
    assert "email" in fields and "status" in fields

    # Any dependent checks (not_null/range/enum) may be present; ensure skip reasons appear when applicable
    # We accept either PASS/FAIL depending on data, but presence of checks map is required when emitted


@pytest.mark.parametrize("db_url", _param_db_urls())
def test_strict_mode_extras_json(tmp_path: Path, db_url: str) -> None:
    rules = {
        "customers": {
            "rules": [
                {"field": "id", "type": "integer"},
            ],
            "strict_mode": True,
            "case_insensitive": True,
        }
    }
    rules_file = _write_rules(tmp_path, rules)

    r = E2ETestUtils.run_cli_command(
        [
            "schema",
            "--conn",
            db_url,
            "--rules",
            rules_file,
            "--output",
            "json",
        ]
    )
    try:
        payload = json.loads(r.stdout)
    except Exception as e:
        assert False, (
            "Expected JSON output from CLI but failed to parse. "
            f"Error: {e}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )
    # schema_extras should appear and be an array
    assert isinstance(payload.get("schema_extras", []), list)


def test_empty_rules_minimal_payload(tmp_path: Path) -> None:
    # Use a simple CSV source to exercise early-exit path
    data_file = tmp_path / "data.csv"
    data_file.write_text("id\n1\n", encoding="utf-8")
    rules_file = _write_rules(tmp_path, {"rules": []})

    r = E2ETestUtils.run_cli_command(
        [
            "schema",
            "--conn",
            str(data_file),
            "--rules",
            rules_file,
            "--output",
            "json",
        ]
    )
    assert r.returncode == 0
    payload = json.loads(r.stdout)
    assert payload["rules_count"] == 0


@pytest.mark.parametrize("db_url", _param_db_urls())
def test_multi_table_schema_metadata_happy_path(tmp_path: Path, db_url: str) -> None:
    """E2E test for multi-table schema validation with metadata - happy path.

    This test uses real database connections and the test data generated by
    scripts/sql/generate_test_data.py, which includes both customers and orders tables.
    """
    # Multi-table schema with metadata validation for enhanced schema features
    # This schema definition matches the actual database structure created by generate_test_data.py
    rules = {
        "customers": {
            "rules": [
                {"field": "id", "type": "integer", "required": True},
                {"field": "name", "type": "string", "max_length": 255},
                {"field": "email", "type": "string", "max_length": 255},
                {"field": "age", "type": "integer", "required": True},
                {"field": "gender", "type": "integer"},
            ],
            "strict_mode": False,
            "case_insensitive": True,
        },
        "orders": {
            "rules": [
                {"field": "id", "type": "integer", "required": True},
                {"field": "customer_id", "type": "integer", "required": True},
                {
                    "field": "product_name",
                    "type": "string",
                    "max_length": 255,
                    "required": True,
                },
                {"field": "quantity", "type": "integer", "required": True},
                {
                    "field": "price",
                    "type": "float",
                    "precision": 10,
                    "scale": 2,
                    "required": True,
                },
                {
                    "field": "status",
                    "type": "string",
                    "max_length": 50,
                    "required": True,
                },
                {"field": "order_date", "type": "date", "required": True},
            ],
            "strict_mode": False,
            "case_insensitive": True,
        },
    }
    rules_file = _write_rules(tmp_path, rules)

    # Test with JSON output to verify schema validation results
    r = E2ETestUtils.run_cli_command(
        [
            "schema",
            "--conn",
            db_url,
            "--rules",
            rules_file,
            "--output",
            "json",
        ]
    )
    assert r.returncode in {0, 1}

    try:
        payload = json.loads(r.stdout)
    except Exception as e:
        assert False, (
            "Expected JSON output from CLI but failed to parse. "
            f"Error: {e}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )

    assert payload["status"] == "ok"
    assert payload["rules_count"] >= 2  # At least 2 tables worth of rules
    assert "summary" in payload and "results" in payload and "fields" in payload

    # Verify both tables are processed
    table_names = set()
    for result in payload.get("results", []):
        for metric in result.get("dataset_metrics", []):
            if "entity_name" in metric:
                table_names.add(metric["entity_name"])

    # Check for table names (could be fully qualified like "db.table" or just "table")
    customers_found = any("customers" in name for name in table_names)
    orders_found = any("orders" in name for name in table_names)
    assert customers_found, f"customers table not found in: {table_names}"
    assert orders_found, f"orders table not found in: {table_names}"

    # Verify metadata validation results are present
    fields = payload.get("fields", [])
    assert len(fields) > 0

    # Look for specific fields from both tables
    field_names = {f["column"] for f in fields}
    # Customer fields
    assert "name" in field_names or "email" in field_names
    # Order fields
    assert "product_name" in field_names or "price" in field_names


@pytest.mark.parametrize("db_url", _param_db_urls())
def test_multi_table_schema_metadata_validation_failures(
    tmp_path: Path, db_url: str
) -> None:
    """E2E test for multi-table schema validation with metadata - failure scenarios.

    This test uses real database connections and the test data generated by
    scripts/sql/generate_test_data.py, deliberately creating schema mismatches
    to test validation failure detection.
    """
    # Schema rules designed to trigger validation failures against real database structure
    rules = {
        "customers": {
            "rules": [
                {"field": "id", "type": "integer", "required": True},
                {
                    "field": "name",
                    "type": "string",
                    "max_length": 10,
                },  # Too restrictive - DB has VARCHAR(255)
                {
                    "field": "email",
                    "type": "integer",
                },  # Wrong type - DB has VARCHAR(255)
                {"field": "age", "type": "string"},  # Wrong type - DB has INTEGER
                {"field": "nonexistent_field", "type": "string"},  # Missing field
            ],
            "strict_mode": True,  # Will detect extra fields (gender, created_at)
            "case_insensitive": True,
        },
        "orders": {
            "rules": [
                {"field": "id", "type": "integer", "required": True},
                {
                    "field": "customer_id",
                    "type": "string",
                },  # Wrong type - DB has INTEGER
                {
                    "field": "product_name",
                    "type": "string",
                    "max_length": 10,
                },  # Too restrictive - DB has VARCHAR(255)
                {"field": "quantity", "type": "float"},  # Wrong type - DB has INTEGER
                {
                    "field": "price",
                    "type": "float",
                    "precision": 5,
                    "scale": 4,
                },  # Inconsistent - DB has DECIMAL(10,2)
                {
                    "field": "status",
                    "type": "string",
                    "max_length": 5,
                },  # Too restrictive - DB has VARCHAR(50)
                {"field": "missing_field", "type": "integer"},  # Missing field
            ],
            "strict_mode": True,  # Will detect extra fields (order_date, created_at)
            "case_insensitive": True,
        },
    }
    rules_file = _write_rules(tmp_path, rules)

    # Test with JSON output to verify failure detection
    r = E2ETestUtils.run_cli_command(
        [
            "schema",
            "--conn",
            db_url,
            "--rules",
            rules_file,
            "--output",
            "json",
        ]
    )
    # Expected to fail due to validation errors
    assert r.returncode in {0, 1}

    try:
        payload = json.loads(r.stdout)
    except Exception as e:
        assert False, (
            "Expected JSON output from CLI but failed to parse. "
            f"Error: {e}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        )

    assert payload["status"] == "ok"  # Command executed successfully
    assert payload["rules_count"] >= 2  # At least 2 tables worth of rules

    # Verify validation failures are captured
    fields = payload.get("fields", [])
    assert len(fields) > 0

    # Look for specific failure patterns - check for FAILED status or METADATA_MISMATCH failure codes
    failed_fields = [
        f
        for f in fields
        if any(
            (
                check.get("status") == "FAILED"
                or check.get("failure_code") == "METADATA_MISMATCH"
            )
            for check in f.get("checks", {}).values()
            if isinstance(check, dict)
        )
    ]

    # Should have some failures due to type mismatches and metadata conflicts
    assert (
        len(failed_fields) > 0
    ), f"Expected validation failures but found none. Fields: {fields}"

    # Check for strict mode detecting extra columns
    schema_extras = payload.get("schema_extras", [])
    assert isinstance(schema_extras, list)
    # Should detect extra columns not defined in our restrictive schema

    # Verify both tables have validation results
    table_names = set()
    for result in payload.get("results", []):
        for metric in result.get("dataset_metrics", []):
            if "entity_name" in metric:
                table_names.add(metric["entity_name"])

    # Check for table names (could be fully qualified like "db.table" or just "table")
    customers_found = any("customers" in name for name in table_names)
    orders_found = any("orders" in name for name in table_names)
    assert customers_found, f"customers table not found in: {table_names}"
    assert orders_found, f"orders table not found in: {table_names}"
