from __future__ import annotations

from shared.schema.result_schema import ExecutionResultSchema


class TestExecutionResultSchemaBasics:
    def test_success_and_failure_rates(self) -> None:
        # Success result
        ok = ExecutionResultSchema.create_success_result(
            rule_id="r1",
            entity_name="db.t",
            total_count=10,
            error_count=0,
            execution_time=0.1,
        )
        assert ok.is_success() is True
        assert ok.get_success_rate() == 1.0
        assert ok.get_failure_rate() == 0.0

        # Failure result
        fail = ExecutionResultSchema.create_success_result(
            rule_id="r2",
            entity_name="db.t",
            total_count=10,
            error_count=2,
            execution_time=0.1,
        )
        assert fail.is_failure() is True
        assert 0.0 < fail.get_failure_rate() <= 1.0

    def test_error_result_and_summary(self) -> None:
        err = ExecutionResultSchema.create_error_result(
            rule_id="r3", entity_name="db.t", error_message="table not exist"
        )
        assert err.is_error() is True
        hints = err.get_error_classification_hints()
        assert hints.get("resource_type") in {"table", "column", None}
