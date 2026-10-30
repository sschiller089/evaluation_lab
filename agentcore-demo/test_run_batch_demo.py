"""Offline contracts for batch ground truth, telemetry completeness and resumability."""

import argparse
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from domain import load_dataset
from run_batch_demo import (
    build_request, checked_session, main, poll, result_events, save, submit,
    validate_completed_job, wait_for_telemetry,
)


def recorded_cases():
    scenarios = load_dataset()["scenarios"]
    records = []
    for index, scenario in enumerate(scenarios):
        session = f"fixture-session-{index}"
        turns, spans = [], []
        for turn_index, golden in enumerate(scenario["turns"]):
            trace = f"{index * 10 + turn_index + 1:032x}"
            span = f"{index * 10 + turn_index + 1:016x}"
            turns.append({"input": golden["input"], "trace_id": trace})
            spans.extend([
                {
                    "traceId": trace, "spanId": span, "name": "invoke_agent",
                    "attributes": {"session.id": session},
                    "resource": {"attributes": {"service.name": "test-demo"}},
                },
                {"traceId": trace, "spanId": span, "body": {"input": {}}},
            ])
        records.append({
            "session_id": session, "scenario_id": scenario["scenario_id"],
            "variant": "baseline", "mode": "bedrock", "turns": turns, "spans": spans,
            "dataset_sha256": "a" * 64, "fixture_sha256": "b" * 64,
        })
    return scenarios, records


def request(records, scenarios):
    return build_request(
        records, scenarios, variant="baseline", service_name="test-demo",
        log_groups=["/test/content", "aws/spans"], name="test_baseline",
        token="10000000-0000-0000-0000-000000000001",
    )


def completed_job():
    scenarios, records = recorded_cases()
    payload = request(records, scenarios)
    summaries, events = [], []
    for evaluator in (item["evaluatorId"] for item in payload["evaluators"]):
        evaluated = 0
        for record in records:
            scope = "Trace" if evaluator == "Builtin.Correctness" else "Session"
            targets = (
                [turn["trace_id"] for turn in record["turns"]]
                if scope == "Trace" else [record["session_id"]]
            )
            for target in targets:
                evaluated += 1
                event = {"attributes": {
                    "gen_ai.evaluation.name": evaluator,
                    "session.id": record["session_id"],
                    "gen_ai.response.id": target,
                    "gen_ai.evaluation.score.value": 0.0,
                    "aws.bedrock_agentcore.evaluation_level": scope,
                    "aws.bedrock_agentcore.evaluation_job.run_id": "test-job",
                }}
                if scope == "Trace":
                    event["traceId"] = target
                events.append({"message": json.dumps(event)})
        summaries.append({
            "evaluatorId": evaluator, "totalEvaluated": evaluated, "totalFailed": 0,
        })
    return {
        "request": payload, "response": {
            "batchEvaluationId": "test-job", "status": "COMPLETED",
            "evaluationResults": {
                "numberOfSessionsCompleted": len(records),
                "totalNumberOfSessions": len(records),
                "numberOfSessionsFailed": 0, "numberOfSessionsIgnored": 0,
                "numberOfSessionsInProgress": 0, "evaluatorSummaries": summaries,
            },
        },
        "result_events": events,
    }


class BatchGroundTruthTests(unittest.TestCase):
    def test_wire_contract_preserves_goldens_and_selects_only_actual_traces(self):
        from botocore.session import get_session
        from botocore.validate import validate_parameters

        scenarios, records = recorded_cases()
        original = copy.deepcopy(scenarios)
        payload = request(records, scenarios)
        shape = get_session().get_service_model("bedrock-agentcore").operation_model(
            "StartBatchEvaluation"
        ).input_shape
        validate_parameters(payload, shape)
        self.assertEqual(scenarios, original)
        sources = payload["dataSourceConfig"]["cloudWatchLogs"]
        self.assertEqual(sources["logGroupNames"], ["/test/content", "aws/spans"])
        selected = sources["filterConfig"]["sessionTraceIds"]
        for record, scenario, metadata, selection in zip(
            records, scenarios, payload["evaluationMetadata"]["sessionMetadata"], selected
        ):
            self.assertEqual(metadata["sessionId"], record["session_id"])
            self.assertEqual(selection["sessionId"], record["session_id"])
            self.assertEqual(selection["traceIds"], [t["trace_id"] for t in record["turns"]])
            golden = metadata["groundTruth"]["inline"]
            self.assertEqual(golden["assertions"], [{"text": a} for a in scenario["assertions"]])
            self.assertEqual(golden["expectedTrajectory"]["toolNames"], scenario["expected_trajectory"])
            self.assertEqual(
                [t["expectedResponse"]["text"] for t in golden["turns"]],
                [t["expected_response"] for t in scenario["turns"]],
            )

    def test_incomplete_or_mismatched_records_do_not_create_a_request(self):
        scenarios, records = recorded_cases()
        for mutate in (
            lambda r: r.pop(),
            lambda r: r[0]["turns"].clear(),
            lambda r: r[0]["turns"][0].update(trace_id="f" * 32),
            lambda r: r[0]["turns"][0].update(input="different prompt"),
            lambda r: r[0]["spans"][0]["attributes"].update({"session.id": "other"}),
            lambda r: r[0]["spans"][0]["resource"]["attributes"].update({"service.name": "other"}),
        ):
            changed = copy.deepcopy(records)
            mutate(changed)
            with self.assertRaises(ValueError):
                request(changed, scenarios)


class BatchResumeTests(unittest.TestCase):
    def test_cli_preserves_ambient_oidc_credentials_without_a_local_profile(self):
        with patch.dict(os.environ, {}, clear=True), patch("run_batch_demo.run") as run:
            self.assertEqual(main(["run", "--manifest", "/tmp/not-created.json"]), 0)
        self.assertIsNone(run.call_args.args[0].profile)

    def test_cli_preserves_an_explicit_local_profile(self):
        with patch.dict(os.environ, {"AWS_PROFILE": "demo-profile"}), \
             patch("run_batch_demo.run") as run:
            self.assertEqual(main(["run", "--manifest", "/tmp/not-created.json"]), 0)
        self.assertEqual(run.call_args.args[0].profile, "demo-profile")

    def test_wrong_aws_account_is_rejected_before_model_or_batch_usage(self):
        with patch("boto3.Session") as factory:
            factory.return_value.client.return_value.get_caller_identity.return_value = {
                "Account": "111111111111"
            }
            with self.assertRaisesRegex(ValueError, "Expected AWS account 123456789012"):
                checked_session({
                    "profile": "default", "region": "eu-central-1",
                    "account_id": "123456789012",
                })
            self.assertEqual(factory.return_value.client.call_args.args, ("sts",))

    def test_missing_content_blocks_submission_even_when_spans_are_present(self):
        _, records = recorded_cases()
        manifest = {
            "span_log_group": "aws/spans", "content_log_group": "/test/content",
            "started_epoch": 1,
        }
        found_spans = {
            (span["traceId"], span["spanId"])
            for record in records for span in record["spans"] if "name" in span
        }
        with patch("run_batch_demo.query_pairs", side_effect=[found_spans, set()]):
            with self.assertRaisesRegex(TimeoutError, "Telemetry is incomplete"):
                wait_for_telemetry(MagicMock(), manifest, records, timeout=0)

    def test_submit_retries_the_persisted_token_and_never_recreates_known_jobs(self):
        session = MagicMock()
        client = session.client.return_value
        from botocore.session import get_session
        client.meta.service_model = get_session().get_service_model("bedrock-agentcore")
        scenarios, records = recorded_cases()
        pending_request = request(records, scenarios)
        manifest = {
            "settings": {}, "records": [],
            "jobs": {
                "baseline": {"request": pending_request, "response": {"batchEvaluationId": "known"}},
                "regression": {"request": pending_request},
            },
        }
        client.start_batch_evaluation.return_value = {"batchEvaluationId": "retried"}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "batch.json"
            save(path, manifest)
            args = argparse.Namespace(manifest=path, ingestion_timeout=0)
            with patch("run_batch_demo.checked_session", return_value=session), \
                 patch("run_batch_demo.wait_for_telemetry"):
                submit(args)
                submit(args)
            client.start_batch_evaluation.assert_called_once_with(**pending_request)
            persisted = json.loads(path.read_text())
            self.assertEqual(
                persisted["jobs"]["regression"]["request"]["clientToken"],
                pending_request["clientToken"],
            )
            self.assertEqual(persisted["jobs"]["baseline"]["response"]["batchEvaluationId"], "known")

    def test_result_pages_are_all_saved_without_duplicating_terminal_page(self):
        logs = MagicMock()
        logs.get_log_events.side_effect = [
            {"events": [{"message": "one"}], "nextForwardToken": "first"},
            {"events": [{"message": "two"}], "nextForwardToken": "second"},
            {"events": [], "nextForwardToken": "second"},
        ]
        response = {"outputConfig": {"cloudWatchConfig": {
            "logGroupName": "/actual/results", "logStreamName": "actual-job",
        }}}
        self.assertEqual(result_events(logs, response), [{"message": "one"}, {"message": "two"}])
        self.assertEqual(logs.get_log_events.call_args.kwargs["logGroupName"], "/actual/results")


class BatchCompletionTests(unittest.TestCase):
    def test_complete_low_scores_are_valid_results_with_scope_specific_counts(self):
        validation = validate_completed_job(completed_job())
        self.assertTrue(validation["complete"])
        self.assertEqual(validation["expected_results"], 7)
        self.assertEqual(validation["observed_results"], 7)

    def test_nonempty_partial_logs_remain_pending(self):
        job = completed_job()
        job["result_events"] = job["result_events"][:1]
        validation = validate_completed_job(job)
        self.assertFalse(validation["complete"])
        self.assertEqual(validation["errors"], [])
        self.assertEqual(validation["observed_results"], 1)
        self.assertTrue(validation["pending"])

    def test_incomplete_summary_or_failed_ignored_sessions_are_errors(self):
        for mutate in (
            lambda r: r.update(numberOfSessionsCompleted=1),
            lambda r: r.update(totalNumberOfSessions=3),
            lambda r: r.update(numberOfSessionsFailed=1),
            lambda r: r.update(numberOfSessionsIgnored=1),
            lambda r: r["evaluatorSummaries"].pop(),
            lambda r: r["evaluatorSummaries"][0].update(totalFailed=1),
            lambda r: r["evaluatorSummaries"][0].update(totalEvaluated=1),
        ):
            job = completed_job()
            mutate(job["response"]["evaluationResults"])
            validation = validate_completed_job(job)
            self.assertFalse(validation["complete"])
            self.assertTrue(validation["errors"])

    def test_wrong_job_session_trace_or_error_result_cannot_fill_coverage(self):
        for field, value in (
            ("aws.bedrock_agentcore.evaluation_job.run_id", "another-job"),
            ("session.id", "another-session"),
            ("gen_ai.response.id", "f" * 32),
            ("error.type", "EvaluatorError"),
            ("gen_ai.evaluation.score.value", None),
        ):
            job = completed_job()
            event = json.loads(job["result_events"][0]["message"])
            event["attributes"][field] = value
            job["result_events"][0]["message"] = json.dumps(event)
            validation = validate_completed_job(job)
            self.assertFalse(validation["complete"])
            self.assertTrue(validation["errors"])
            self.assertEqual(validation["observed_results"], 6)

    def test_poll_keeps_partial_output_pending_and_marks_terminal_errors_distinctly(self):
        for failed in (False, True):
            job = completed_job()
            if failed:
                job["response"]["status"] = "FAILED"
            session = MagicMock()
            session.client.return_value.get_batch_evaluation.return_value = job["response"]
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "batch.json"
                save(path, {"settings": {}, "jobs": {"baseline": job, "regression": job}})
                args = argparse.Namespace(manifest=path, wait_seconds=0)
                with patch("run_batch_demo.checked_session", return_value=session), \
                     patch("run_batch_demo.result_events", return_value=job["result_events"][:1]):
                    if failed:
                        with self.assertRaisesRegex(RuntimeError, "ended with errors"):
                            poll(args)
                    else:
                        poll(args)
                saved = json.loads(path.read_text())
                self.assertEqual(saved["status"], "FINISHED_WITH_ERRORS" if failed else "SUBMITTED")


if __name__ == "__main__":
    unittest.main()
