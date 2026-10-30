#!/usr/bin/env python3
"""Run the two golden demo scenarios as an on-demand CI quality gate.

Exit codes: 0 = PASS, 1 = quality FAIL, 2 = ERROR/incomplete evidence.
The regression variant is deliberately faulty and uses exactly the same gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from aws_agent import DEFAULT_EVALUATORS
from checks import evaluate_local
from domain import DATASET_PATH, FIXTURE_PATH, load_dataset

ROOT = Path(__file__).resolve().parent
DEMO_THRESHOLD = 0.8
DEFAULT_MODEL = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
SUPPORTED_SCENARIOS = {
    "no-sim-blocks-enable-and-reset",
    "no-sim-attach-and-enable-in-one-call",
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_score(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _attributes(span: dict) -> dict:
    value = span.get("attributes")
    return value if isinstance(value, dict) else {}


def _telemetry_errors(record: dict, trace_ids: list[str]) -> list[str]:
    spans = record.get("spans")
    if not isinstance(spans, list) or not spans or any(
        not isinstance(span, dict) for span in spans
    ):
        return ["Actual session trace records are missing or malformed."]
    errors = []
    if any(
        _attributes(span).get("session.id") != record.get("session_id")
        for span in spans
    ):
        errors.append("A trace record belongs to a different session.")
    roots = [
        span for span in spans
        if span.get("name")
        and _attributes(span).get("gen_ai.operation.name") == "invoke_agent"
    ]
    if Counter(root.get("traceId") for root in roots) != Counter(trace_ids):
        errors.append("Every golden turn needs exactly one captured agent root span.")
    for root in roots:
        if not any(
            span.get("traceId") == root.get("traceId")
            and span.get("spanId") == root.get("spanId")
            and span.get("body")
            for span in spans
        ):
            errors.append(f"Trace {root.get('traceId')} is missing conversation content.")
    return errors


def grade_record(scenario: dict, record: dict, variant: str) -> dict:
    """Recompute hard checks and validate every score, never trust a stored PASS."""
    errors: list[str] = []
    failures: list[str] = []
    if record.get("error"):
        errors.append(f"Agent execution failed: {record['error']}")
    if record.get("mode") != "bedrock":
        errors.append("CI requires a real Bedrock run, not a scripted rehearsal.")
    if record.get("variant") != variant:
        errors.append("The record's agent variant does not match this gate run.")
    if not isinstance(record.get("session_id"), str) or not record["session_id"]:
        errors.append("The record has no session ID.")
    for label, path in (("dataset", DATASET_PATH), ("fixture", FIXTURE_PATH)):
        if record.get(f"{label}_sha256") != file_hash(path):
            errors.append(f"The {label} hash does not match the version under test.")
    turns = record.get("turns")
    turns = turns if isinstance(turns, list) else []
    if any(not isinstance(turn, dict) for turn in turns):
        errors.append("A recorded turn is malformed.")
        turns = [turn for turn in turns if isinstance(turn, dict)]
    if [turn.get("input") for turn in turns] != [
        turn["input"] for turn in scenario["turns"]
    ]:
        errors.append("The recorded prompts do not match all golden turns in order.")
    for turn in turns:
        calls = turn.get("tool_calls")
        if isinstance(calls, list):
            errors.extend(
                f"Tool invocation failed: {call['error']}"
                for call in calls if isinstance(call, dict) and call.get("error")
            )
    trace_ids = [turn.get("trace_id") for turn in turns]
    if any(
        not isinstance(trace_id, str) or not re.fullmatch(r"[0-9a-f]{32}", trace_id)
        for trace_id in trace_ids
    ) or len(set(str(trace_id) for trace_id in trace_ids)) != len(trace_ids):
        errors.append("Every turn requires a distinct valid trace ID.")
    else:
        errors.extend(_telemetry_errors(record, trace_ids))
    try:
        local = evaluate_local(scenario, record)
        failures.extend(
            f"Local check {check['name']}: {check['detail']}"
            for check in local["checks"] if check.get("passed") is not True
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"Could not validate recorded tool behavior: {exc}")
        local = {"status": "ERROR", "checks": []}
    saved_local = record.get("local_evaluation")
    if not isinstance(saved_local, dict) or saved_local.get("status") not in {"PASS", "FAIL"}:
        errors.append("The original local evaluation did not complete.")
    elif saved_local["status"] == "FAIL":
        failures.append("The original local evaluation reported a failure.")

    cloud = record.get("agentcore_evaluation")
    if not isinstance(cloud, dict):
        cloud = {}
    if cloud.get("error") or cloud.get("status") == "ERROR":
        errors.append(f"AgentCore evaluation failed: {cloud.get('error') or 'see results'}")
    elif cloud.get("status") not in {"PASS", "FAIL"}:
        errors.append("AgentCore evaluation did not complete.")
    elif cloud["status"] == "FAIL":
        failures.append("AgentCore evaluation reported a failure.")
    raw_responses = cloud.get("raw_responses", [])
    if isinstance(raw_responses, list) and any(
        isinstance(response, dict) and response.get("error")
        for response in raw_responses
    ):
        errors.append("An AgentCore evaluator invocation returned an error.")
    rows = cloud.get("results")
    rows = rows if isinstance(rows, list) else []
    expected = Counter({
        ("Builtin.Correctness", turn.get("trace_id")): 1
        for turn, golden in zip(turns, scenario["turns"])
        if "expected_response" in golden and isinstance(turn.get("trace_id"), str)
    })
    expected.update({
        ("Builtin.GoalSuccessRate", None): 1,
        ("Builtin.TrajectoryInOrderMatch", None): 1,
    })
    observed: Counter = Counter()
    scores = {name: [] for name in DEFAULT_EVALUATORS}
    for row in rows:
        if not isinstance(row, dict):
            errors.append("AgentCore returned a malformed result.")
            continue
        evaluator = row.get("evaluator_id")
        trace_id = row.get("trace_id")
        if (
            not isinstance(evaluator, str) or evaluator not in scores
            or trace_id is not None and not isinstance(trace_id, str)
        ):
            errors.append("AgentCore returned an unexpected evaluator or invalid trace ID.")
            continue
        observed[(evaluator, trace_id)] += 1
        if row.get("error"):
            errors.append(f"{evaluator}: {row['error']}")
            continue
        value = row.get("value")
        if not valid_score(value):
            errors.append(f"{evaluator}: invalid or missing score.")
            continue
        if not isinstance(row.get("explanation"), str) or not row["explanation"].strip():
            errors.append(f"{evaluator}: missing evaluation explanation.")
        scores[evaluator].append(value)
        if value < DEMO_THRESHOLD:
            failures.append(
                f"{evaluator} ({trace_id or 'session'}): {value:g} < {DEMO_THRESHOLD:g}."
            )
    if observed != expected:
        errors.append(
            "Evaluator coverage is incomplete, duplicated, or mismatched: "
            f"expected {sum(expected.values())} exact result identities, "
            f"received {sum(observed.values())}."
        )
    return {
        "scenario_id": scenario["scenario_id"],
        "session_id": record.get("session_id"),
        "status": "ERROR" if errors else "FAIL" if failures else "PASS",
        "errors": errors,
        "failures": failures,
        "local_evaluation": local,
        "scores": scores,
    }


def grade_records(records: list[dict], variant: str) -> dict:
    scenarios = load_dataset()["scenarios"]
    expected_ids = {scenario["scenario_id"] for scenario in scenarios}
    errors = []
    if expected_ids != SUPPORTED_SCENARIOS:
        errors.append("This gate must be extended before evaluating a different golden set.")
    observed = Counter(
        record.get("scenario_id") for record in records
        if isinstance(record, dict) and isinstance(record.get("scenario_id"), str)
    )
    if observed != Counter({scenario_id: 1 for scenario_id in expected_ids}):
        errors.append("Exactly one completed run of each of the two golden scenarios is required.")
    session_ids = [
        record.get("session_id") for record in records if isinstance(record, dict)
    ]
    if any(not isinstance(item, str) for item in session_ids) or len(
        {str(item) for item in session_ids}
    ) != len(session_ids):
        errors.append("Each scenario must have its own fresh session ID.")
    results = []
    for scenario in scenarios:
        matching = [
            record for record in records
            if isinstance(record, dict) and record.get("scenario_id") == scenario["scenario_id"]
        ]
        if len(matching) == 1:
            results.append(grade_record(scenario, matching[0], variant))
    statuses = {result["status"] for result in results}
    return {
        "status": "ERROR" if errors or "ERROR" in statuses else "FAIL" if "FAIL" in statuses else "PASS",
        "errors": errors,
        "variant": variant,
        "thresholds": {name: DEMO_THRESHOLD for name in DEFAULT_EVALUATORS},
        "threshold_note": "Demo thresholds; not calibrated release criteria or AWS recommendations.",
        "scenario_count": len(scenarios),
        "scenarios": results,
    }


def source_metadata() -> dict:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {
        "git_revision": revision,
        "github_sha": os.getenv("GITHUB_SHA"),
        "github_repository": os.getenv("GITHUB_REPOSITORY"),
        "github_run_id": os.getenv("GITHUB_RUN_ID"),
        "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
        "hashes": {
            "dataset": file_hash(DATASET_PATH),
            "fixture": file_hash(FIXTURE_PATH),
            "agent": file_hash(ROOT / "aws_agent.py"),
            "tool_adapter": file_hash(ROOT / "domain.py"),
            "hard_checks": file_hash(ROOT / "checks.py"),
            "quality_gate": file_hash(Path(__file__)),
        },
    }


def observed_usage(records: list[dict]) -> dict:
    roots = [
        span for record in records if isinstance(record, dict)
        for span in (
            record["spans"] if isinstance(record.get("spans"), list) else []
        ) if isinstance(span, dict)
        if span.get("name")
        and _attributes(span).get("gen_ai.operation.name") == "invoke_agent"
    ]
    usage = {}
    for name in ("input_tokens", "output_tokens"):
        values = [
            _attributes(span).get(f"gen_ai.usage.{name}") for span in roots
        ]
        usage[name] = sum(values) if values and all(
            type(value) is int and value >= 0 for value in values
        ) else None
    usage["source"] = "Actual invoke_agent root span usage; null means unavailable."
    usage["cost"] = None
    usage["cost_note"] = "No monetary cost estimate; evaluator usage is not included."
    return usage


def markdown_summary(report: dict) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ").replace("<", "&lt;")

    gate = report["gate"]
    lines = [
        f"## Agent evaluation: {gate['status']}",
        "",
        f"Variant: **{cell(gate['variant'])}**. Each score must be ≥ {DEMO_THRESHOLD}; "
        "every deterministic check must pass.",
        "",
        "| Scenario | Status | Correctness | Goal success | Tool order |",
        "|---|---|---|---|---|",
    ]
    for result in gate.get("scenarios", []):
        values = [
            ", ".join(f"{value:g}" for value in result["scores"][name]) or "missing"
            for name in DEFAULT_EVALUATORS
        ]
        # DEFAULT_EVALUATORS is Correctness, GoalSuccessRate, TrajectoryInOrderMatch.
        lines.append(
            f"| {cell(result['scenario_id'])} | {result['status']} | "
            + " | ".join(values) + " |"
        )
    problems = list(gate.get("errors", []))
    for result in gate.get("scenarios", []):
        problems.extend(
            f"{result['scenario_id']}: {item}"
            for item in result["errors"] + result["failures"]
        )
    if problems:
        lines.extend(["", "Failures and incomplete evidence:", ""])
        lines.extend(f"- {cell(problem)}" for problem in problems)
    lines.extend([
        "",
        "The artifact contains recorded conversations, tool calls, trace IDs, "
        "individual evaluator explanations, source hashes, and observed token usage.",
        "These are the two implemented fixture scenarios. Passing this demo does "
        "not certify a production deployment.",
        "",
    ])
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_report(report: dict, output: Path, summary: str | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(report), indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    if summary:
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write(markdown_summary(report))


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    commands = command.add_subparsers(dest="command", required=True)
    for name in ("run", "grade"):
        sub = commands.add_parser(name)
        sub.add_argument("--variant", choices=("baseline", "regression"), default="baseline")
        sub.add_argument("--output", type=Path, default=ROOT / "results" / "ci-evaluation.json")
        sub.add_argument("--summary", default=os.getenv("GITHUB_STEP_SUMMARY"))
        if name == "run":
            sub.add_argument("--profile", default=None)
            sub.add_argument("--region", default=os.getenv("AWS_REGION", "eu-central-1"))
            sub.add_argument("--model-id", default=os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL))
            sub.add_argument("--expected-account-id", default=os.getenv("AWS_ACCOUNT_ID"))
        else:
            sub.add_argument("records", type=Path, nargs="+")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    started = time.monotonic()
    records = []
    fatal_error = None
    identity = None
    try:
        if args.command == "run":
            import boto3
            from botocore.config import Config
            from demo import run_scenario

            identity = boto3.Session(
                profile_name=args.profile, region_name=args.region
            ).client(
                "sts",
                config=Config(connect_timeout=10, read_timeout=30, retries={"total_max_attempts": 2}),
            ).get_caller_identity()
            identity = {key: identity[key] for key in ("Account", "Arn")}
            if args.expected_account_id and identity["Account"] != args.expected_account_id:
                raise ValueError(
                    f"AWS account mismatch: expected {args.expected_account_id}, "
                    f"received {identity['Account']}."
                )
            settings = {
                "mode": "bedrock", "variant": args.variant,
                "profile": args.profile, "region": args.region, "model_id": args.model_id,
            }
            for scenario in load_dataset()["scenarios"]:
                print(f"Evaluating {scenario['scenario_id']} ({args.variant})", flush=True)
                records.append(run_scenario(scenario, settings, aws_score=True))
        else:
            for path in args.records:
                record = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(record, dict):
                    raise ValueError(f"{path.name}: expected a single recorded scenario object.")
                records.append(record)
        gate = grade_records(records, args.variant)
    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        gate = {
            "status": "ERROR", "variant": args.variant,
            "errors": [fatal_error], "scenarios": [],
            "thresholds": {name: DEMO_THRESHOLD for name in DEFAULT_EVALUATORS},
        }
    report = {
        "schema_version": 1,
        "command": args.command,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source_metadata(),
        "aws_identity": identity,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "usage": observed_usage(records),
        "gate": gate,
        "records": records,
    }
    save_report(report, args.output, args.summary)
    print(f"Quality gate: {gate['status']}. Report: {args.output}", flush=True)
    if fatal_error:
        print(fatal_error, file=sys.stderr)
    return {"PASS": 0, "FAIL": 1, "ERROR": 2}[gate["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
