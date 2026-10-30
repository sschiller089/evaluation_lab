#!/usr/bin/env python3
"""Run the existing fixture demo and publish new AgentCore batch evaluations.

Launch ``run`` with ADOT configured by the caller. ``submit`` waits for every
recorded span and content event in CloudWatch, then starts one job per variant.
``poll`` resumes existing jobs without invoking the agent or starting new jobs.

API contract: AWS batch-evaluations-start.html and the installed boto3 model.
No infrastructure, observability settings, or customer fixtures are changed.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aws_agent import DEFAULT_EVALUATORS
from demo import LOCAL_DEFAULTS, file_hash, run_scenario
from domain import DATASET_PATH, FIXTURE_PATH, load_dataset

TERMINAL = {"COMPLETED", "COMPLETED_WITH_ERRORS", "FAILED", "STOPPED"}
VARIANTS = ("baseline", "regression")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def checked_session(settings: dict) -> Any:
    import boto3
    from botocore.config import Config

    session = boto3.Session(
        profile_name=settings["profile"], region_name=settings["region"]
    )
    identity = session.client(
        "sts", config=Config(connect_timeout=10, read_timeout=20)
    ).get_caller_identity()
    if identity["Account"] != settings["account_id"]:
        raise ValueError(
            f"Expected AWS account {settings['account_id']}, "
            f"but the credentials resolve to {identity['Account']}."
        )
    return session


def validate_record(scenario: dict, record: dict, service_name: str) -> list[str]:
    if record.get("error") or record.get("mode") != "bedrock":
        raise ValueError("Batch evaluation requires a completed real Bedrock run.")
    if record.get("scenario_id") != scenario["scenario_id"]:
        raise ValueError("The recorded scenario does not match the golden scenario.")
    if len(record.get("turns", [])) != len(scenario["turns"]):
        raise ValueError("The recorded conversation is incomplete.")
    traces = []
    for expected, actual in zip(scenario["turns"], record["turns"]):
        trace = actual.get("trace_id", "")
        if actual.get("input") != expected["input"]:
            raise ValueError("The recorded prompt differs from the golden prompt.")
        if not re.fullmatch("[a-f0-9]{32}", trace):
            raise ValueError("Each golden turn needs its actual trace ID.")
        traces.append(trace)
    spans = [span for span in record.get("spans", []) if "name" in span]
    if not spans or set(traces) - {span.get("traceId") for span in spans}:
        raise ValueError("The golden turns are missing their recorded spans.")
    for span in spans:
        if span.get("attributes", {}).get("session.id") != record["session_id"]:
            raise ValueError("A recorded span belongs to another session.")
        resource = span.get("resource", {}).get("attributes", {})
        if resource.get("service.name") != service_name:
            raise ValueError("The recorded service.name differs from the batch source.")
    return traces


def build_request(
    records: list[dict], scenarios: list[dict], *, variant: str,
    service_name: str, log_groups: list[str], name: str, token: str,
) -> dict:
    """Keep golden content unchanged; bind ordered turns to actual sessions/traces."""
    selected = [record for record in records if record["variant"] == variant]
    by_id = {scenario["scenario_id"]: scenario for scenario in scenarios}
    if len(selected) != len(by_id) or {r["scenario_id"] for r in selected} != set(by_id):
        raise ValueError("Each variant must contain every golden scenario exactly once.")
    metadata, filters = [], []
    for record in selected:
        scenario = by_id[record["scenario_id"]]
        traces = validate_record(scenario, record, service_name)
        ground_truth: dict[str, Any] = {"turns": []}
        for turn in scenario["turns"]:
            item = {"input": {"prompt": turn["input"]}}
            if "expected_response" in turn:
                item["expectedResponse"] = {"text": turn["expected_response"]}
            ground_truth["turns"].append(item)
        if scenario.get("assertions"):
            ground_truth["assertions"] = [
                {"text": assertion} for assertion in scenario["assertions"]
            ]
        if scenario.get("expected_trajectory"):
            ground_truth["expectedTrajectory"] = {
                "toolNames": copy.deepcopy(scenario["expected_trajectory"])
            }
        metadata.append({
            "sessionId": record["session_id"],
            "testScenarioId": scenario["scenario_id"],
            "groundTruth": {"inline": ground_truth},
            "metadata": {
                "variant": variant,
                "dataset_sha256": record["dataset_sha256"],
                "fixture_sha256": record["fixture_sha256"],
            },
        })
        filters.append({"sessionId": record["session_id"], "traceIds": traces})
    return {
        "batchEvaluationName": name,
        "clientToken": token,
        "evaluators": [{"evaluatorId": item} for item in DEFAULT_EVALUATORS],
        "dataSourceConfig": {"cloudWatchLogs": {
            "serviceNames": [service_name],
            "logGroupNames": list(log_groups),
            "filterConfig": {"sessionTraceIds": filters},
        }},
        "evaluationMetadata": {"sessionMetadata": metadata},
        "description": f"Connectivity fixture demo: {variant}; original no_sim_guard golden cases.",
    }


def expected_pairs(records: list[dict], *, events: bool) -> set[tuple[str, str]]:
    return {
        (span["traceId"], span["spanId"])
        for record in records for span in record["spans"]
        if ("name" not in span) == events
    }


def query_pairs(
    logs: Any, group: str, trace_ids: list[str], start_time: int, *, events: bool,
) -> set[tuple[str, str]]:
    # Read identifiers only. Do not print customer conversations or tool payloads.
    # Logs Insights discovers scalar JSON fields, but a nested body object may
    # have only its leaf fields discovered. Match ADOT's Strands content scope
    # and event timestamp instead of testing the undiscovered parent "body".
    predicate = (
        'scope.name = "strands.telemetry.tracer" and ispresent(timeUnixNano)'
        if events else "ispresent(name)"
    )
    query = (
        f"filter traceId in {json.dumps(trace_ids)} and {predicate} "
        "| stats count(*) as records by traceId, spanId"
    )
    response = logs.start_query(
        logGroupName=group, startTime=start_time - 300,
        endTime=int(time.time()) + 60, queryString=query,
    )
    deadline = time.monotonic() + 60
    while True:
        result = logs.get_query_results(queryId=response["queryId"])
        if result["status"] == "Complete":
            rows = [
                {item["field"]: item["value"] for item in row}
                for row in result.get("results", [])
            ]
            return {(row["traceId"], row["spanId"]) for row in rows}
        if result["status"] in {"Failed", "Cancelled", "Timeout", "Unknown"}:
            raise RuntimeError(f"CloudWatch ingestion query: {result['status']}")
        if time.monotonic() >= deadline:
            raise TimeoutError("CloudWatch ingestion query did not complete in 60 seconds.")
        time.sleep(2)


def wait_for_telemetry(
    logs: Any, manifest: dict, records: list[dict], *, timeout: int,
) -> None:
    span_pairs = expected_pairs(records, events=False)
    event_pairs = expected_pairs(records, events=True)
    traces = sorted({trace for trace, _ in span_pairs})
    if not span_pairs or not event_pairs:
        raise ValueError("The split-telemetry run needs both real spans and content events.")
    deadline = time.monotonic() + timeout
    while True:
        found_spans = query_pairs(
            logs, manifest["span_log_group"], traces, manifest["started_epoch"],
            events=False,
        )
        found_events = query_pairs(
            logs, manifest["content_log_group"], traces, manifest["started_epoch"],
            events=True,
        )
        missing_spans = span_pairs - found_spans
        missing_events = event_pairs - found_events
        if not missing_spans and not missing_events:
            print("CloudWatch contains every recorded span and content event.", flush=True)
            return
        print(
            f"Waiting for telemetry: {len(missing_spans)} spans and "
            f"{len(missing_events)} content events still missing.", flush=True,
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("Telemetry is incomplete; no new batch job was submitted.")
        time.sleep(min(15, max(0, deadline - time.monotonic())))


def flush_telemetry() -> None:
    from opentelemetry import trace
    from opentelemetry._logs import get_logger_provider

    provider = trace.get_tracer_provider()
    if not hasattr(provider, "force_flush") or not provider.force_flush(30000):
        raise RuntimeError("The configured trace exporter did not flush successfully.")
    logger = get_logger_provider()
    if hasattr(logger, "force_flush") and not logger.force_flush(30000):
        raise RuntimeError("The configured content-event exporter did not flush successfully.")


def run(args: argparse.Namespace) -> None:
    path = args.manifest.resolve()
    if path.exists():
        raise ValueError("Manifest already exists; use submit/poll to resume it.")
    settings = {
        "profile": args.profile, "region": args.region, "account_id": args.account_id,
        "mode": "bedrock", "model_id": args.model_id,
    }
    required_env = {
        "AGENT_OBSERVABILITY_ENABLED": "true",
        "OTEL_PYTHON_DISTRO": "aws_distro",
        "OTEL_PYTHON_CONFIGURATOR": "aws_configurator",
        "AWS_REGION": args.region,
        "AWS_DEFAULT_REGION": args.region,
        "AWS_PROFILE": args.profile,
    }
    for key, value in required_env.items():
        if os.environ.get(key) != value:
            raise ValueError(f"Launch through batch_demo.sh with {key}={value}.")
    checked_session(settings)  # Guard before incurring any model or evaluation usage.
    if os.environ.get("OTEL_SERVICE_NAME") != args.service_name:
        raise ValueError("Launch with ADOT and the matching OTEL_SERVICE_NAME.")
    if not os.environ.get("OTEL_EXPORTER_OTLP_LOGS_HEADERS"):
        raise ValueError("The split-telemetry ADOT log exporter must be configured.")
    manifest = {
        "schema_version": 1, "created_at": now(), "started_epoch": int(time.time()),
        "settings": settings, "service_name": args.service_name,
        "content_log_group": args.content_log_group,
        "span_log_group": args.span_log_group,
        "dataset_sha256": file_hash(DATASET_PATH),
        "fixture_sha256": file_hash(FIXTURE_PATH),
        "source_revision": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "status": "RUNNING_AGENT", "records": [], "jobs": {},
    }
    save(path, manifest)
    scenarios = load_dataset()["scenarios"]
    records = []
    try:
        for variant in VARIANTS:
            for scenario in scenarios:
                record = run_scenario(
                    scenario, {**settings, "variant": variant}, aws_score=False
                )
                records.append(record)
                manifest["records"].append({
                    "path": record["result_file"], "variant": variant,
                    "session_id": record["session_id"],
                    "scenario_id": record["scenario_id"],
                    "local_status": record["local_evaluation"]["status"],
                })
                save(path, manifest)
                validate_record(scenario, record, args.service_name)
                print(
                    f"{variant}: {scenario['scenario_id']} "
                    f"local={record['local_evaluation']['status']} "
                    f"session={record['session_id']}", flush=True,
                )
        flush_telemetry()
        suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        for variant in VARIANTS:
            manifest["jobs"][variant] = {"request": build_request(
                records, scenarios, variant=variant, service_name=args.service_name,
                log_groups=list(dict.fromkeys(
                    [args.content_log_group, args.span_log_group]
                )),
                name=f"evaluation_lab_{variant}_{suffix}", token=str(uuid.uuid4()),
            )}
        manifest["status"] = "READY_TO_SUBMIT"
        manifest["traces_flushed_at"] = now()
        save(path, manifest)
    except Exception as exc:
        manifest["status"] = "AGENT_RUN_ERROR"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        save(path, manifest)
        raise
    print(f"Saved {path}. Next: submit --manifest {path}", flush=True)


def read_records(manifest: dict) -> list[dict]:
    records = []
    for entry in manifest["records"]:
        record = json.loads(Path(entry["path"]).read_text())
        if record["session_id"] != entry["session_id"]:
            raise ValueError("A saved record no longer matches this manifest.")
        for key in ("dataset_sha256", "fixture_sha256"):
            if record[key] != manifest[key]:
                raise ValueError("The run contains inconsistent golden source hashes.")
        records.append(record)
    return records


def submit(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text())
    if set(manifest.get("jobs", {})) != set(VARIANTS):
        raise ValueError("Run did not finish preparing both batch requests.")
    session = checked_session(manifest["settings"])
    client = session.client("bedrock-agentcore")
    from botocore.validate import validate_parameters

    shape = client.meta.service_model.operation_model("StartBatchEvaluation").input_shape
    for job in manifest["jobs"].values():
        validate_parameters(job["request"], shape)
    pending = [job for job in manifest["jobs"].values() if not job.get("response")]
    if pending:
        records = read_records(manifest)
        wait_for_telemetry(
            session.client("logs"), manifest, records, timeout=args.ingestion_timeout
        )
    for variant, job in manifest["jobs"].items():
        if not job.get("response"):
            # The request token was persisted before this call. A retry is idempotent
            # even if the process stops between AWS acceptance and saving the response.
            job["response"] = client.start_batch_evaluation(**job["request"])
            save(args.manifest, manifest)
        print(f"{variant}: {job['response']['batchEvaluationId']}", flush=True)
    manifest["status"] = "SUBMITTED"
    save(args.manifest, manifest)


def result_events(logs: Any, response: dict) -> list[dict]:
    output = response.get("outputConfig", {}).get("cloudWatchConfig", {})
    if not output.get("logGroupName") or not output.get("logStreamName"):
        return []
    args = {
        "logGroupName": output["logGroupName"],
        "logStreamName": output["logStreamName"], "startFromHead": True,
    }
    events = []
    while True:
        try:
            page = logs.get_log_events(**args)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code == "ResourceNotFoundException":
                return []  # A terminal job's output can still be propagating.
            raise
        events.extend(page.get("events", []))
        token = page.get("nextForwardToken")
        if not token or token == args.get("nextToken"):
            return events
        args["nextToken"] = token


def validate_completed_job(job: dict) -> dict:
    """Require every requested session, evaluator and result identity.

    A completed job can become visible before all of its CloudWatch result
    events. Missing events remain pending; invalid summaries or error results
    fail validation. Low quality scores are valid results, not processing errors.
    """
    request, response = job["request"], job["response"]
    errors, pending = [], []
    selections = request["dataSourceConfig"]["cloudWatchLogs"]["filterConfig"]["sessionTraceIds"]
    session_ids = {item["sessionId"] for item in selections}
    evaluators = {item["evaluatorId"] for item in request["evaluators"]}
    expected = {
        (evaluator, selection["sessionId"], target)
        for evaluator in evaluators for selection in selections
        for target in (
            selection["traceIds"] if evaluator == "Builtin.Correctness"
            else [selection["sessionId"]]
        )
    }
    counts = {
        "numberOfSessionsCompleted": len(session_ids),
        "totalNumberOfSessions": len(session_ids),
        "numberOfSessionsFailed": 0,
        "numberOfSessionsIgnored": 0,
        "numberOfSessionsInProgress": 0,
    }
    aggregate = response.get("evaluationResults", {})
    if response.get("status") != "COMPLETED":
        errors.append(f"Job status is {response.get('status')}, not COMPLETED.")
    if response.get("errorDetails"):
        errors.append("The job returned errorDetails.")
    for field, wanted in counts.items():
        if aggregate.get(field) != wanted:
            errors.append(f"{field}: expected {wanted}, received {aggregate.get(field)}.")
    summaries = aggregate.get("evaluatorSummaries", [])
    actual_evaluators = [item.get("evaluatorId") for item in summaries]
    if len(actual_evaluators) != len(evaluators) or set(actual_evaluators) != evaluators:
        errors.append("Evaluator summaries do not cover exactly the requested evaluators.")
    for summary in summaries:
        evaluator = summary.get("evaluatorId")
        wanted = sum(identity[0] == evaluator for identity in expected)
        if summary.get("totalEvaluated") != wanted or summary.get("totalFailed") != 0:
            errors.append(f"{evaluator}: expected {wanted} evaluated results and zero failures.")

    observed = set()
    job_id = response["batchEvaluationId"]
    for entry in job.get("result_events", []):
        try:
            event = json.loads(entry["message"])
        except (KeyError, TypeError, ValueError):
            errors.append("A result log event is not valid JSON.")
            continue
        attributes = event.get("attributes", {})
        evaluator = attributes.get("gen_ai.evaluation.name")
        if attributes.get("error.type") or attributes.get("error.message"):
            errors.append(f"{evaluator or 'Unknown evaluator'}: result contains an error.")
            continue
        if not evaluator:
            continue  # Ignore non-evaluation records such as log metadata.
        session_id = attributes.get("session.id")
        target = attributes.get("gen_ai.response.id")
        identity = (evaluator, session_id, target)
        scope = "Trace" if evaluator == "Builtin.Correctness" else "Session"
        if attributes.get("aws.bedrock_agentcore.evaluation_job.run_id") != job_id:
            errors.append("A result belongs to a different batch job.")
            continue
        if attributes.get("aws.bedrock_agentcore.evaluation_level") != scope:
            errors.append(f"{evaluator}: unexpected evaluation scope.")
            continue
        if scope == "Trace" and event.get("traceId") != target:
            errors.append(f"{evaluator}: result trace IDs disagree.")
            continue
        if identity not in expected:
            errors.append(f"{evaluator}: result does not match a requested session/trace.")
            continue
        score = attributes.get("gen_ai.evaluation.score.value")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            errors.append(f"{evaluator}: result has no finite numeric score.")
            continue
        observed.add(identity)
    if missing := expected - observed:
        pending.append(f"{len(missing)} expected evaluation result(s) are not available yet.")
    return {
        "complete": not errors and not pending,
        "errors": errors, "pending": pending,
        "expected_results": len(expected), "observed_results": len(observed),
    }


def poll(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text())
    if not manifest.get("jobs") or any(
        not job.get("response") for job in manifest["jobs"].values()
    ):
        raise ValueError("Submit both jobs before polling.")
    session = checked_session(manifest["settings"])
    client, logs = session.client("bedrock-agentcore"), session.client("logs")
    deadline = time.monotonic() + args.wait_seconds
    while True:
        for variant, job in manifest["jobs"].items():
            response = client.get_batch_evaluation(
                batchEvaluationId=job["response"]["batchEvaluationId"]
            )
            job["response"] = response
            save(args.manifest, manifest)
            print(
                f"{variant}: {response['status']} "
                f"{json.dumps(response.get('evaluationResults', {}))}", flush=True,
            )
            if response["status"] in TERMINAL:
                job["result_events"] = result_events(logs, response)
                job["result_validation"] = validate_completed_job(job)
                output = response.get("outputConfig", {}).get("cloudWatchConfig", {})
                print(f"CloudWatch result location: {json.dumps(output)}", flush=True)
                print(f"Result validation: {json.dumps(job['result_validation'])}", flush=True)
            save(args.manifest, manifest)
        terminal = all(
            job["response"]["status"] in TERMINAL for job in manifest["jobs"].values()
        )
        validation_errors = any(
            job.get("result_validation", {}).get("errors")
            for job in manifest["jobs"].values()
        )
        output_pending = any(
            job.get("result_validation", {}).get("pending")
            for job in manifest["jobs"].values()
        )
        complete = terminal and (validation_errors or not output_pending)
        manifest["status"] = (
            ("FINISHED_WITH_ERRORS" if validation_errors else "FINISHED")
            if complete else "SUBMITTED"
        )
        manifest["last_checked_at"] = now()
        save(args.manifest, manifest)
        if complete:
            if validation_errors:
                raise RuntimeError("At least one batch job ended with errors; inspect manifest.")
            return
        if time.monotonic() >= deadline:
            print(
                "Jobs or result logs are still pending. Repeat poll with this manifest.",
                flush=True,
            )
            return
        time.sleep(min(15, max(0, deadline - time.monotonic())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Invoke fresh fixture sessions with ADOT.")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE") or None)
    run_parser.add_argument("--region", default=os.environ.get("AWS_REGION", LOCAL_DEFAULTS["region"]))
    run_parser.add_argument("--account-id", default=os.environ.get("AWS_ACCOUNT_ID", ""))
    run_parser.add_argument("--model-id", default=LOCAL_DEFAULTS["model_id"])
    run_parser.add_argument("--service-name", default="agentcore-evaluation-lab")
    run_parser.add_argument(
        "--content-log-group", default="/aws/bedrock-agentcore/runtimes/agentcore-evaluation-lab"
    )
    run_parser.add_argument("--span-log-group", default="aws/spans")
    submit_parser = commands.add_parser("submit", help="Wait for telemetry and start batch jobs.")
    submit_parser.add_argument("--manifest", type=Path, required=True)
    submit_parser.add_argument("--ingestion-timeout", type=int, default=600)
    poll_parser = commands.add_parser("poll", help="Read and save existing batch results.")
    poll_parser.add_argument("--manifest", type=Path, required=True)
    poll_parser.add_argument("--wait-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    try:
        {"run": run, "submit": submit, "poll": poll}[args.command](args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
