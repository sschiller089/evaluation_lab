#!/usr/bin/env python3
"""Publish a batch comparison summary only after every expected result exists."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

from run_batch_demo import VARIANTS, validate_completed_job


def console_url(region: str, group: str, stream: str) -> str:
    def encode(value: str) -> str:
        return quote(quote(value, safe=""), safe="").replace("%", "$")

    return (
        f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}"
        f"#logsV2:log-groups/log-group/{encode(group)}/log-events/{encode(stream)}"
    )


def summarize(manifest: dict) -> str:
    if manifest.get("status") != "FINISHED":
        raise ValueError("Batch jobs or result events are incomplete; inspect the manifest.")
    jobs = manifest.get("jobs", {})
    if set(jobs) != set(VARIANTS):
        raise ValueError("Both baseline and regression batch jobs are required.")
    lines = [
        "## AgentCore batch comparison",
        "",
        "Batch completion means grading finished. The regression is intentionally faulty;",
        "use the separate Agent quality gate workflow for a release pass/fail decision.",
        "",
        "| Variant | Evaluator | Average score | Evaluated |",
        "|---|---|---:|---:|",
    ]
    links = []
    for variant in VARIANTS:
        job = jobs[variant]
        validation = validate_completed_job(job)
        if not validation["complete"]:
            raise ValueError(f"{variant}: {json.dumps(validation)}")
        response = job["response"]
        for result in response["evaluationResults"]["evaluatorSummaries"]:
            score = result.get("averageScore")
            if score is None:
                # AWS exposes aggregate values in scoreDistribution on some versions.
                # Do not invent an aggregate; individual validated records are canonical.
                values = []
                for event in job["result_events"]:
                    item = json.loads(event["message"])
                    attributes = item.get("attributes", {})
                    if attributes.get("gen_ai.evaluation.name") == result["evaluatorId"]:
                        values.append(attributes["gen_ai.evaluation.score.value"])
                score = sum(values) / len(values)
            lines.append(
                f"| {variant} | {result['evaluatorId']} | {score:.2f} | "
                f"{result['totalEvaluated']} |"
            )
        destination = response["outputConfig"]["cloudWatchConfig"]
        url = console_url(
            manifest["settings"]["region"],
            destination["logGroupName"],
            destination["logStreamName"],
        )
        links.append(f"- [{variant}: scores and explanations]({url})")
    lines.extend(["", *links, "", "Raw responses and the resumable manifest are in this run's artifact."])
    if manifest.get("source_revision"):
        lines.extend(["", f"Evaluated revision: `{manifest['source_revision']}`"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = summarize(json.loads(args.manifest.read_text()))
        args.manifest.with_suffix(".md").write_text(summary)
        if path := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(path).open("a") as output:
                output.write(summary)
        print(summary)
        return 0
    except Exception as exc:
        print(f"Batch report ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
