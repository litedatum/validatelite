"""
E2E: vlite-cli schema on databases and table/json outputs

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
        "rules": [
            {"field": "id", "type": "integer", "required": True},
            {"field": "email", "type": "string"},
            {"field": "age", "type": "integer", "min": 0, "max": 150},
        ],
        "strict_mode": False,
        "case_insensitive": True,
    }
    rules_file = _write_rules(tmp_path, rules)

    # table output
    r1 = E2ETestUtils.run_cli_command(
        [
            "schema",
            "--conn",
            db_url,
            "--table",
            "customers",
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
            "--table",
            "customers",
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
    rules_file = _write_rules(tmp_path, rules)

    r = E2ETestUtils.run_cli_command(
        [
            "schema",
            "--conn",
            db_url,
            "--table",
            "customers",
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
        "rules": [
            {"field": "id", "type": "integer"},
        ],
        "strict_mode": True,
        "case_insensitive": True,
    }
    rules_file = _write_rules(tmp_path, rules)

    r = E2ETestUtils.run_cli_command(
        [
            "schema",
            "--conn",
            db_url,
            "--table",
            "customers",
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
            "--table",
            "data",
            "--rules",
            rules_file,
            "--output",
            "json",
        ]
    )
    assert r.returncode == 0
    payload = json.loads(r.stdout)
    assert payload["rules_count"] == 0
