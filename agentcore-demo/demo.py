#!/usr/bin/env python3
"""A small learning lab using the synthetic no_sim_guard golden dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

from checks import evaluate_local
from domain import DATASET_PATH, FIXTURE_PATH, ScriptedAgent, load_dataset

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WORK_LOCK = threading.RLock()
SESSIONS: dict = {}
JOBS: dict = {}
# Portable defaults; account and credentials come from the caller.
# Selecting AWS mode and opting into AgentCore scoring remain explicit actions.
LOCAL_DEFAULTS = {
    "profile": None,
    "region": "eu-central-1",
    "model_id": "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
    "account_id": "",
}


def connection_defaults() -> dict:
    profile = os.environ.get("AWS_PROFILE", LOCAL_DEFAULTS["profile"])
    return {
        "profile": profile,
        "region": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
                  or LOCAL_DEFAULTS["region"],
        "model_id": os.environ.get("BEDROCK_MODEL_ID", LOCAL_DEFAULTS["model_id"]),
        "account_id": os.environ.get("AWS_ACCOUNT_ID", ""),
    }


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_settings(settings: dict) -> dict:
    defaults = connection_defaults()
    mode = settings.get("mode", "rehearsal")
    variant = settings.get("variant", "baseline")
    if mode not in ("rehearsal", "bedrock"):
        raise ValueError("Choose rehearsal or bedrock.")
    if variant not in ("baseline", "regression"):
        raise ValueError("Choose baseline or regression.")
    model_id = str(settings.get("model_id", defaults["model_id"]) or "").strip()
    region = str(settings.get("region") or defaults["region"]).strip()
    profile = str(settings.get("profile", defaults["profile"]) or "").strip() or None
    if mode == "bedrock" and not model_id:
        raise ValueError("Enter a Bedrock model or inference profile ID for AWS mode.")
    return dict(mode=mode, variant=variant, profile=profile, region=region,
                model_id=model_id)


def new_session(settings: dict, scenario_id: str = "interactive") -> dict:
    settings = validate_settings(settings)
    if settings["mode"] == "bedrock":
        from aws_agent import BedrockSupportAgent
        agent = BedrockSupportAgent(
            profile=settings["profile"], region=settings["region"],
            model_id=settings["model_id"], variant=settings["variant"],
        )
    else:
        agent = ScriptedAgent(variant=settings["variant"])
    session_id = getattr(agent, "session_id", None) or str(uuid.uuid4())
    record = {
        "session_id": session_id,
        "scenario_id": scenario_id,
        **settings,
        "dataset_path": str(DATASET_PATH),
        "fixture_path": str(FIXTURE_PATH),
        "dataset_sha256": file_hash(DATASET_PATH),
        "fixture_sha256": file_hash(FIXTURE_PATH),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "turns": [],
        "spans": [],
        "pending_actions": [],
    }
    session = {"agent": agent, "record": record}
    SESSIONS[session_id] = session
    return session


def run_turn(session: dict, prompt: str) -> dict:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a question.")
    if len(prompt) > 12000:
        raise ValueError("Keep demo questions below 12,000 characters.")
    output = session["agent"].run(prompt)
    turn = {"input": prompt, **output}
    turn.setdefault("trace_id", uuid.uuid4().hex)
    session["record"]["turns"].append(turn)
    session["record"]["pending_actions"] = copy.deepcopy(
        session["agent"].tools.pending_actions
    )
    session["record"]["spans"] = copy.deepcopy(
        getattr(session["agent"], "spans", [])
    )
    return turn


def persist(record: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{record['variant']}-{record['session_id']}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, default=str) + "\n")
    temporary.replace(path)
    return path


def saved_results() -> dict:
    """Read recent demo AWS results from the fixed results folder; never invoke AWS."""
    scenarios = {s["scenario_id"]: s for s in load_dataset()["scenarios"]}
    current_hash = file_hash(DATASET_PATH)
    current_fixture_hash = file_hash(FIXTURE_PATH)
    results = []
    warnings = []
    total_bytes = 0
    allowed_keys = {
        "session_id", "scenario_id", "mode", "variant", "profile", "region",
        "model_id", "account_id", "aws_account_id", "dataset_sha256",
        "fixture_sha256", "created_at", "turns", "spans", "pending_actions",
        "error", "local_evaluation", "agentcore_evaluation",
    }
    if not RESULTS.is_dir():
        return {"results": [], "warnings": []}
    candidates = [
        path for path in RESULTS.iterdir()
        if not path.is_symlink() and path.is_file()
        and re.fullmatch(r"(baseline|regression)-[A-Za-z0-9_-]+\.json", path.name)
    ]
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
        if len(results) >= 20:
            break
        try:
            size = path.stat().st_size
            if size > 4_000_000 or total_bytes + size > 16_000_000:
                warnings.append(f"Skipped oversized saved record: {path.name}")
                continue
            data = json.loads(path.read_text())
            if not isinstance(data, dict) or data.get("mode") != "bedrock":
                continue
            if data.get("scenario_id") not in scenarios:
                continue
            record = {key: data[key] for key in allowed_keys if key in data}
            turns = record.get("turns")
            expected_turns = len(scenarios[record["scenario_id"]]["turns"])
            visible_turns = [turn for turn in turns if isinstance(turn, dict)] \
                if isinstance(turns, list) else []
            completed_turns = sum(
                isinstance(turn.get("input"), str)
                and isinstance(turn.get("response"), str)
                and bool(turn["response"].strip())
                for turn in visible_turns
            )
            local = record.get("local_evaluation")
            cloud = record.get("agentcore_evaluation")
            local_status = local.get("status") if isinstance(local, dict) else None
            cloud_status = cloud.get("status") if isinstance(cloud, dict) else None
            if completed_turns != expected_turns or len(visible_turns) != expected_turns:
                status = "INCOMPLETE"
            elif record.get("error") or "ERROR" in (local_status, cloud_status):
                status = "ERROR"
            elif "FAIL" in (local_status, cloud_status):
                status = "FAIL"
            elif local_status == "PASS" and cloud_status == "PASS":
                status = "PASS"
            else:
                status = "NOT SCORED"
            record.update({
                "turns": visible_turns,
                "saved_record": True,
                "read_only": True,
                "result_file": str(path),
                "saved_status": status,
                "completed_turns": completed_turns,
                "expected_turns": expected_turns,
                "matches_current_dataset": record.get("dataset_sha256") == current_hash,
                "matches_current_fixture": record.get("fixture_sha256") == current_fixture_hash,
            })
            results.append(record)
            total_bytes += size
        except (OSError, ValueError, TypeError) as exc:
            warnings.append(f"Could not read {path.name}: {type(exc).__name__}")
    return {"results": results, "warnings": warnings}


def run_scenario(scenario: dict, settings: dict, aws_score: bool = False) -> dict:
    session = new_session(settings, scenario["scenario_id"])
    record = session["record"]
    try:
        for turn in scenario["turns"]:
            if not isinstance(turn["input"], str):
                raise ValueError("This small demo supports the no_sim_guard string turns.")
            run_turn(session, turn["input"])
        record["local_evaluation"] = evaluate_local(scenario, record)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["local_evaluation"] = {
            "status": "ERROR", "checks": [],
            "semantic_evaluation": "NOT_RUN",
        }
    if aws_score:
        try:
            if record.get("error"):
                raise ValueError("Agent execution was incomplete; evaluation was not submitted.")
            if record["mode"] != "bedrock":
                raise ValueError("AgentCore scoring requires an actual Bedrock agent run.")
            from aws_agent import score_agentcore
            record["agentcore_evaluation"] = score_agentcore(
                scenario, record, profile=record["profile"], region=record["region"]
            )
        except Exception as exc:
            record["agentcore_evaluation"] = {
                "status": "ERROR", "results": [], "raw_responses": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
    record["result_file"] = str(persist(record))
    return copy.deepcopy(record)


def start_job(settings: dict, aws_score: bool = False) -> str:
    settings = validate_settings(settings)
    if aws_score and settings["mode"] != "bedrock":
        raise ValueError("Use Bedrock mode to request real AgentCore scores.")
    scenarios = load_dataset()["scenarios"]
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "id": job_id, "status": "running", "current": "Starting",
        "total": len(scenarios), "completed": 0, "results": [],
    }

    def worker():
        try:
            with WORK_LOCK:
                for scenario in scenarios:
                    JOBS[job_id]["current"] = scenario["scenario_id"]
                    result = run_scenario(scenario, settings, aws_score)
                    JOBS[job_id]["results"].append(result)
                    JOBS[job_id]["completed"] += 1
            JOBS[job_id]["status"] = "complete"
        except Exception as exc:
            JOBS[job_id].update(status="error", error=str(exc))

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def config() -> dict:
    dataset = load_dataset()
    return {
        "dataset": dataset,
        "fixture": json.loads(FIXTURE_PATH.read_text()),
        "dataset_path": str(DATASET_PATH),
        "fixture_path": str(FIXTURE_PATH),
        "dataset_sha256": file_hash(DATASET_PATH),
        "defaults": connection_defaults(),
    }


def main():
    defaults = connection_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "chat"), nargs="?",
                        default="run")
    parser.add_argument("--mode", choices=("rehearsal", "bedrock"), default="rehearsal")
    parser.add_argument("--variant", choices=("baseline", "regression"), default="baseline")
    parser.add_argument("--region", default=defaults["region"])
    parser.add_argument("--profile", default=defaults["profile"])
    parser.add_argument("--model-id", default=defaults["model_id"])
    parser.add_argument("--aws-score", action="store_true",
                        help="Request actual AgentCore scores (Bedrock mode only)")
    args = parser.parse_args()
    settings = dict(mode=args.mode, variant=args.variant, profile=args.profile,
                    region=args.region, model_id=args.model_id)
    try:
        validate_settings(settings)
        if args.aws_score and args.mode != "bedrock":
            parser.error("--aws-score requires --mode bedrock")
        if args.command == "chat":
            session = new_session(settings)
            print(f"{args.mode} / {args.variant}. Type /quit to finish.")
            while True:
                try:
                    prompt = input("You: ")
                except EOFError:
                    break
                if prompt.strip() == "/quit":
                    break
                with WORK_LOCK:
                    turn = run_turn(session, prompt)
                print(f"Agent: {turn['response']}")
                for call in turn["tool_calls"]:
                    print(f"  tool: {call['name']} {json.dumps(call['arguments'])}")
            return
        results = []
        for scenario in load_dataset()["scenarios"]:
            with WORK_LOCK:
                result = run_scenario(scenario, settings, args.aws_score)
            results.append(result)
            print(f"{scenario['scenario_id']}: LOCAL {result['local_evaluation']['status']}")
            if result.get("error"):
                print(f"  ERROR: {result['error']}")
            for check in result["local_evaluation"].get("checks", []):
                if not check["passed"]:
                    print(f"  FAIL: {check['name']} — {check['detail']}")
            if args.aws_score:
                print(json.dumps(result.get("agentcore_evaluation", {}), indent=2))
            print(f"  {result['result_file']}")
        print("Local checks cover explicit tool constraints; semantic assertions need AWS scoring.")
        failed = any(r["local_evaluation"]["status"] != "PASS" for r in results)
        if args.aws_score:
            failed |= any(r.get("agentcore_evaluation", {}).get("status") != "PASS"
                          for r in results)
        raise SystemExit(1 if failed else 0)
    except (ValueError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
