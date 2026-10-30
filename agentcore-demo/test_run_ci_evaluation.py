"""Offline fail-closed checks for the GitHub Actions quality gate."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import demo
import run_ci_evaluation as ci
from domain import load_dataset


def fake_scored_records(variant="baseline"):
    """Actual fixture tool execution with explicitly synthetic judge/trace evidence."""
    records = []
    for index, scenario in enumerate(load_dataset()["scenarios"]):
        with patch.object(demo, "persist", return_value=Path("/offline-unit-test.json")):
            record = demo.run_scenario(scenario, {"mode": "rehearsal", "variant": variant})
        record["mode"] = "bedrock"
        record["spans"] = []
        rows = []
        for turn_index, turn in enumerate(record["turns"]):
            trace_id = f"{index + 1:016x}{turn_index + 1:016x}"
            span_id = f"{turn_index + 1:016x}"
            turn["trace_id"] = trace_id
            attributes = {
                "session.id": record["session_id"],
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 5,
            }
            record["spans"].extend([
                {"name": "invoke_agent", "traceId": trace_id, "spanId": span_id,
                 "attributes": attributes},
                {"traceId": trace_id, "spanId": span_id,
                 "attributes": {"session.id": record["session_id"]},
                 "body": {"input": "Synthetic unit test input", "output": "Synthetic unit test output"}},
            ])
            rows.append({
                "evaluator_id": "Builtin.Correctness", "trace_id": trace_id,
                "value": 1.0, "explanation": "Synthetic unit test score.", "error": None,
            })
        rows.extend([
            {"evaluator_id": evaluator, "trace_id": None, "value": 1.0,
             "explanation": "Synthetic unit test score.", "error": None}
            for evaluator in ("Builtin.GoalSuccessRate", "Builtin.TrajectoryInOrderMatch")
        ])
        record["agentcore_evaluation"] = {
            "status": "PASS", "results": rows, "raw_responses": [],
        }
        records.append(record)
    return records


class QualityGateTests(unittest.TestCase):
    def test_complete_baseline_passes_and_tool_fault_fails_even_with_perfect_judges(self):
        self.assertEqual(ci.grade_records(fake_scored_records(), "baseline")["status"], "PASS")
        failed = ci.grade_records(fake_scored_records("regression"), "regression")
        self.assertEqual(failed["status"], "FAIL")
        self.assertTrue(all(item["failures"] for item in failed["scenarios"]))

    def test_one_low_turn_fails_even_if_overall_mean_would_pass(self):
        records = fake_scored_records()
        records[1]["agentcore_evaluation"]["results"][0]["value"] = 0.79
        result = ci.grade_records(records, "baseline")
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("0.79", " ".join(result["scenarios"][1]["failures"]))

    def test_missing_duplicate_foreign_and_wrong_scope_results_fail_closed(self):
        for mutation in ("missing", "duplicate", "foreign", "wrong_scope"):
            with self.subTest(mutation=mutation):
                records = fake_scored_records()
                rows = records[0]["agentcore_evaluation"]["results"]
                if mutation == "missing":
                    rows.pop()
                elif mutation == "duplicate":
                    rows.append(copy.deepcopy(rows[0]))
                elif mutation == "foreign":
                    rows[0]["trace_id"] = "f" * 32
                else:
                    rows[1]["trace_id"] = records[0]["turns"][0]["trace_id"]
                self.assertEqual(ci.grade_records(records, "baseline")["status"], "ERROR")

    def test_invalid_scores_and_eval_errors_never_pass(self):
        for value in (None, True, "1", -0.1, 1.01, float("nan"), float("inf")):
            with self.subTest(value=value):
                records = fake_scored_records()
                records[0]["agentcore_evaluation"]["results"][0]["value"] = value
                self.assertEqual(ci.grade_records(records, "baseline")["status"], "ERROR")
        records = fake_scored_records()
        records[0]["agentcore_evaluation"]["raw_responses"] = [{"error": "AccessDenied"}]
        self.assertEqual(ci.grade_records(records, "baseline")["status"], "ERROR")

    def test_missing_scenario_duplicate_session_changed_data_and_partial_turn_fail(self):
        for mutation in ("missing", "session", "dataset", "turn", "profile_variant", "rehearsal"):
            with self.subTest(mutation=mutation):
                records = fake_scored_records()
                if mutation == "missing":
                    records.pop()
                elif mutation == "session":
                    records[1]["session_id"] = records[0]["session_id"]
                elif mutation == "dataset":
                    records[0]["dataset_sha256"] = "changed"
                elif mutation == "turn":
                    records[1]["turns"].pop()
                elif mutation == "profile_variant":
                    records[1]["variant"] = "regression"
                else:
                    records[1]["mode"] = "rehearsal"
                self.assertEqual(ci.grade_records(records, "baseline")["status"], "ERROR")

    def test_missing_trace_content_and_missing_explanation_fail(self):
        records = fake_scored_records()
        records[0]["spans"] = records[0]["spans"][:1]
        self.assertEqual(ci.grade_records(records, "baseline")["status"], "ERROR")
        records = fake_scored_records()
        records[0]["agentcore_evaluation"]["results"][0]["explanation"] = ""
        self.assertEqual(ci.grade_records(records, "baseline")["status"], "ERROR")

    def test_saved_pass_and_lowered_threshold_do_not_override_checks(self):
        records = fake_scored_records("regression")
        for record in records:
            record["local_evaluation"]["status"] = "PASS"
            record["agentcore_evaluation"]["thresholds"] = {
                name: 0 for name in ci.DEFAULT_EVALUATORS
            }
        self.assertEqual(ci.grade_records(records, "regression")["status"], "FAIL")

    def test_saved_failure_and_tool_invocation_error_are_not_silently_ignored(self):
        for key in ("local_evaluation", "agentcore_evaluation"):
            with self.subTest(key=key):
                records = fake_scored_records()
                records[0][key]["status"] = "FAIL"
                self.assertEqual(ci.grade_records(records, "baseline")["status"], "FAIL")
        records = fake_scored_records()
        records[0]["turns"][0]["tool_calls"][0]["error"] = "Tool invocation failed"
        self.assertEqual(ci.grade_records(records, "baseline")["status"], "ERROR")

    def test_observed_usage_uses_only_roots_and_does_not_guess_cost(self):
        records = fake_scored_records()
        records[0]["spans"].append({
            "name": "tool.call", "attributes": {
                "gen_ai.operation.name": "execute_tool", "gen_ai.usage.input_tokens": 1000,
            },
        })
        usage = ci.observed_usage(records)
        self.assertEqual(usage["input_tokens"], 30)
        self.assertEqual(usage["output_tokens"], 15)
        self.assertIsNone(usage["cost"])

    def test_grade_cli_preserves_failure_exit_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index, record in enumerate(fake_scored_records("regression")):
                path = root / f"record-{index}.json"
                path.write_text(json.dumps(record))
                paths.append(str(path))
            output, summary = root / "report.json", root / "summary.md"
            status = ci.main([
                "grade", *paths, "--variant", "regression",
                "--output", str(output), "--summary", str(summary),
            ])
            self.assertEqual(status, 1)
            self.assertEqual(json.loads(output.read_text())["gate"]["status"], "FAIL")
            self.assertIn("Agent evaluation: FAIL", summary.read_text())

    def test_run_uses_ambient_credentials_and_guards_expected_account(self):
        sessions = []
        fake_boto = SimpleNamespace(Session=lambda **kwargs: (
            sessions.append(kwargs) or SimpleNamespace(client=lambda *args, **kw: SimpleNamespace(
                get_caller_identity=lambda: {"Account": "111111111111", "Arn": "test-arn"},
            ))
        ))
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "sys.modules", {
                "boto3": fake_boto,
                "botocore.config": SimpleNamespace(Config=lambda **kwargs: kwargs),
            },
        ), patch.object(demo, "run_scenario") as run:
            output = Path(temporary) / "report.json"
            status = ci.main([
                "run", "--expected-account-id", "222222222222", "--output", str(output),
            ])
            self.assertEqual(status, 2)
            self.assertIsNone(sessions[0]["profile_name"])
            run.assert_not_called()
            self.assertIn("account mismatch", output.read_text())

    def test_malformed_input_writes_error_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "bad.json", root / "report.json"
            source.write_text("{")
            self.assertEqual(ci.main(["grade", str(source), "--output", str(output)]), 2)
            self.assertEqual(json.loads(output.read_text())["gate"]["status"], "ERROR")

    def test_malformed_telemetry_still_produces_error_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            records = fake_scored_records()
            records[0]["spans"] = None
            records[1]["spans"][0]["attributes"] = None
            records[1]["agentcore_evaluation"]["results"][0]["evaluator_id"] = []
            for index, record in enumerate(records):
                path = root / f"{index}.json"
                path.write_text(json.dumps(record))
                paths.append(str(path))
            output = root / "report.json"
            self.assertEqual(ci.main(["grade", *paths, "--output", str(output)]), 2)
            self.assertEqual(json.loads(output.read_text())["gate"]["status"], "ERROR")


if __name__ == "__main__":
    unittest.main()
