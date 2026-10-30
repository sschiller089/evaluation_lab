"""Run a real Bedrock agent and capture its traces for AgentCore evaluation.

The default path captures locally; the batch launcher also enables ADOT export.
No Runtime deployment or fixture API calls are used.
The wire adapter follows the AWS local-span sample and the Strands event
grouping in aws-opentelemetry-distro 0.21.0's llo_handler.py:
https://github.com/aws-samples/sample-agentcore-optimization-local-langgraph-agent
https://github.com/aws-observability/aws-otel-python-instrumentation

Third-party imports are deferred so the local rehearsal needs only Python.
"""

from __future__ import annotations

import copy
import json
import math
import os
import threading
import time
import uuid
from typing import Any

SCOPE = "strands.telemetry.tracer"
DEFAULT_EVALUATORS = (
    "Builtin.Correctness",
    "Builtin.GoalSuccessRate",
    "Builtin.TrajectoryInOrderMatch",
)
# Teaching thresholds, not AWS recommendations or calibrated release criteria.
THRESHOLDS = {name: 0.8 for name in DEFAULT_EVALUATORS}
MAX_MODEL_CALLS = 8
MAX_SESSION_TURNS = 10
MAX_SESSION_BYTES = 2_000_000
_CAPTURE_LOCK = threading.RLock()
_TELEMETRY: tuple[Any, Any] | None = None

SYSTEM_PROMPT = """You are a concise connectivity support assistant.
Use the supplied tools for device and SIM facts. Their data is a fixed demo
world. Never invent identifiers or report a proposed action as completed.
External identifiers are synthetic strings generated for this lab, including
all-zero ICCIDs. Resolve exact fixture matches without live-format validation;
the returned internal SIM id is the only id allowed in an update proposal.

First resolve the user's device with get_device_info_and_id. Check whether
a SIM is attached BEFORE following the Disabled-status branch. If sim is
null, that is the blocking cause: enabling or resetting cannot solve missing
hardware. Explain the missing SIM, mention Disabled when relevant, ask which
SIM to attach (ICCID or SIM id), and stop. Do not call or propose update_device
or reset_device_connectivity while the SIM is unknown.

When the user supplies a SIM identifier, resolve it with get_sim_info_and_id.
Use only the returned internal SIM id. If the previously resolved device is
disabled and the SIM is unattached, call update_device ONCE with endpoint_id,
new_sim_id, and new_status_id=0 together. These are one proposal, not two
changes. Omit fields that were not requested; never send placeholder ids,
null values, or tags. The update tool only records a pending approval.
Then describe the attach-and-enable proposal and explicitly ask for approval.
Do not claim the change ran. Do not reset connectivity before attachment and
retesting. A pending proposal ends this turn; do not retry or split it.
For unrelated requests, explain that this demo covers missing-SIM support.
"""

REGRESSION_PROMPT = """You are a connectivity support assistant.
This is a synthetic lab: external identifiers, including all-zero ICCIDs, are
fixture strings. Resolve exact matches without live-format validation.
Use get_device_info_and_id to look up a device. In this intentionally flawed
version, prioritize Disabled status: when a device is Disabled, immediately
call update_device with its endpoint_id and new_status_id=0 and ask the user
to approve enabling it. Do this even if no SIM is attached; do not stop to
ask which SIM to attach. Changes are proposals only, so never claim execution.
If a SIM identifier is later supplied, resolve it through get_sim_info_and_id
and propose attaching it. Do not invent ids or call tools repeatedly.
"""


def _json_safe(value: Any) -> Any:
    """Normalize OTel tuples/mappings to boto3's document-compatible values."""
    return json.loads(json.dumps(value))


def span_records(span: Any) -> list[dict[str, Any]]:
    """Convert an actual OTel ReadableSpan and its content events to wire records.

Strands' default (legacy) content events hold dictionaries such as
{"content": "[...]"} and {"message": "...", "finish_reason": "end_turn"}.
Keep these dictionaries intact: AgentCore's Strands parser expects them.
As in ADOT's content collector, the last event of each kind is retained.
"""
    scope = getattr(span, "instrumentation_scope", None)
    if scope is None or scope.name != SCOPE:
        return []
    attributes = _json_safe(dict(span.attributes or {}))
    if not attributes.get("session.id"):
        raise ValueError("A Strands span is missing session.id.")
    context = span.get_span_context()
    record = {
        "traceId": format(context.trace_id, "032x"),
        "spanId": format(context.span_id, "016x"),
        "name": span.name,
        "kind": span.kind.name,
        "startTimeUnixNano": span.start_time,
        "endTimeUnixNano": span.end_time,
        "durationNano": (span.end_time or 0) - (span.start_time or 0),
        "scope": {"name": scope.name, "version": scope.version or ""},
        "attributes": attributes,
        "status": {"code": span.status.status_code.name},
        "resource": {"attributes": _json_safe(dict(span.resource.attributes or {}))},
    }
    if span.parent is not None:
        record["parentSpanId"] = format(span.parent.span_id, "016x")

    roles = {
        "gen_ai.system.message": ("input", "system"),
        "gen_ai.user.message": ("input", "user"),
        "gen_ai.tool.message": ("input", "tool"),
        "gen_ai.assistant.message": ("output", "assistant"),
        "gen_ai.choice": ("output", "assistant"),
    }
    content_by_name: dict[str, Any] = {}
    for event in span.events:
        if event.name in roles:
            content_by_name[event.name] = _json_safe(dict(event.attributes or {}))
        if event.name == "gen_ai.client.inference.operation.details":
            raise ValueError(
                "This demo uses Strands' default telemetry format. Unset "
                "OTEL_SEMCONV_STABILITY_OPT_IN and restart before running AWS mode."
            )
    body: dict[str, Any] = {}
    for name, content in content_by_name.items():
        direction, role = roles[name]
        body.setdefault(direction, {"messages": []})["messages"].append(
            {"role": role, "content": content}
        )
    if not body:
        return [record]
    event_record = {
        "traceId": record["traceId"],
        "spanId": record["spanId"],
        "scope": record["scope"],
        "timeUnixNano": span.end_time,
        "observedTimeUnixNano": span.end_time,
        "severityNumber": 9,
        "severityText": "",
        "body": body,
        "attributes": {
            "event.name": SCOPE,
            "session.id": attributes["session.id"],
        },
        "resource": record["resource"],
    }
    return [record, event_record]


class _TraceRecordCapture:
    """Snapshot content before ADOT moves it from spans into CloudWatch logs.

    OTel's _on_ending hook runs before any on_end exporter, including exporters
    registered by auto-instrumentation before this processor. ADOT mutates the
    exported ReadableSpan in place, so retaining span references loses content.
    This hook is supported by the pinned OTel SDK bundled with ADOT 0.21.
    """

    def __init__(self) -> None:
        self._finished: list[list[dict[str, Any]]] = []
        self._errors: list[str] = []
        self._lock = threading.Lock()

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def _on_ending(self, span: Any) -> None:
        try:
            records = span_records(span)
        except Exception as exc:
            # Processor hooks must not throw into the agent framework. Surface
            # incomplete capture when the caller reads the finished records.
            with self._lock:
                self._errors.append(f"{type(exc).__name__}: {exc}")
            return
        if records:
            with self._lock:
                self._finished.append(records)

    def on_end(self, span: Any) -> None:
        pass

    def get_finished_records(self) -> list[dict[str, Any]]:
        with self._lock:
            if self._errors:
                raise RuntimeError("Trace capture failed: " + "; ".join(self._errors))
            groups = sorted(
                self._finished, key=lambda records: records[0]["startTimeUnixNano"] or 0
            )
            return copy.deepcopy([record for records in groups for record in records])

    def clear(self) -> None:
        with self._lock:
            self._finished.clear()
            self._errors.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        self.clear()


def _telemetry() -> tuple[Any, Any]:
    global _TELEMETRY
    with _CAPTURE_LOCK:
        if _TELEMETRY is None:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import SpanProcessor, TracerProvider

            if not hasattr(SpanProcessor, "_on_ending"):
                raise RuntimeError(
                    "The installed OpenTelemetry SDK cannot safely capture traces "
                    "before export. Install the demo's pinned requirements."
                )

            provider = trace.get_tracer_provider()
            if not hasattr(provider, "add_span_processor"):
                provider = TracerProvider(
                    resource=Resource.create({"service.name": "agentcore-evaluation-lab"})
                )
                trace.set_tracer_provider(provider)
            capture = _TraceRecordCapture()
            provider.add_span_processor(capture)
            _TELEMETRY = (provider, capture)
        return _TELEMETRY


class _ToolRecorder:
    """Keep model-requested arguments before SDK validation/coercion.

    FixtureTools.calls describes handler invocations. These hook records also
    include invalid or unknown tool attempts, which must remain visible to the
    deterministic checks even when the SDK never invokes a fixture handler.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def register_hooks(self, registry: Any, **kwargs: Any) -> None:
        from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent

        registry.add_callback(BeforeToolCallEvent, self.before)
        registry.add_callback(AfterToolCallEvent, self.after)

    def before(self, event: Any) -> None:
        tool_use = event.tool_use
        with self._lock:
            self.calls.append({
                "name": tool_use.get("name"),
                "arguments": copy.deepcopy(tool_use.get("input")),
                "tool_call_id": tool_use.get("toolUseId"),
                "result": {"status": "requested", "executed": False},
                "tool_status": "requested",
            })

    def after(self, event: Any) -> None:
        raw_result = copy.deepcopy(event.result)
        result: Any = raw_result
        content = raw_result.get("content", [])
        if raw_result.get("status") == "success" and len(content) == 1:
            item = content[0]
            if "json" in item:
                result = item["json"]
            elif isinstance(item.get("text"), str):
                try:
                    result = json.loads(item["text"])
                except ValueError:
                    pass
        with self._lock:
            call = next(
                call for call in reversed(self.calls)
                if call["tool_call_id"] == event.tool_use.get("toolUseId")
                and call["tool_status"] == "requested"
            )
            call["result"] = result
            call["tool_status"] = raw_result.get("status", "unknown")
            call["tool_result"] = raw_result
            if event.exception is not None:
                call["error"] = str(event.exception)[:1600]


class BedrockSupportAgent:
    """One real Strands conversation with fixture-backed tools."""

    def __init__(
        self,
        profile: str | None,
        region: str,
        model_id: str,
        variant: str = "baseline",
    ):
        if variant not in ("baseline", "regression"):
            raise ValueError("variant must be baseline or regression")
        if "gen_ai_latest_experimental" in os.getenv("OTEL_SEMCONV_STABILITY_OPT_IN", ""):
            raise ValueError("Unset OTEL_SEMCONV_STABILITY_OPT_IN for this demo.")
        from domain import FixtureTools

        self.session_id = str(uuid.uuid4())
        self.tools = FixtureTools()
        self.spans: list[dict[str, Any]] = []
        self._turn_count = 0
        self._failed = False
        self._provider, self._exporter = _telemetry()
        self._tool_recorder = _ToolRecorder()

        import boto3
        from botocore.config import Config
        from strands import Agent, tool
        from strands.models import BedrockModel

        fixture_tools = self.tools

        @tool
        def get_device_info_and_id(identifier: str) -> dict:
            """Look up a device by its exact name or endpoint id.

            Args:
                identifier: Device name or endpoint id supplied by the user.
            """
            return fixture_tools.get_device_info_and_id(identifier)

        @tool
        def get_sim_info_and_id(identifier: str) -> dict:
            """Resolve a user-supplied ICCID or SIM id to a SIM record.

            Args:
                identifier: ICCID or SIM id supplied by the user.
            """
            return fixture_tools.get_sim_info_and_id(identifier)

        @tool(context=True)
        def update_device(
            endpoint_id: int,
            new_sim_id: int | None = None,
            new_status_id: int | None = None,
            tool_context: Any = None,
        ) -> dict:
            """Record one proposed device update and await user approval.

            Never executes an update. Supply only fields the user needs changed.

            Args:
                endpoint_id: Internal device id returned by the device lookup.
                new_sim_id: Internal SIM id from SIM lookup, if attaching a SIM.
                new_status_id: 0 to propose enabling, 1 to propose disabling.
            """
            # Pydantic supplies optional defaults and may coerce numbers. Keep
            # the actual model input: explicit null is different from omission.
            supplied = tool_context.tool_use["input"]
            if set(supplied) - {"endpoint_id", "new_sim_id", "new_status_id"}:
                raise ValueError("Unsupported update fields; only SIM and status changes are supported.")
            fields = {
                name: supplied[name]
                for name in ("new_sim_id", "new_status_id") if name in supplied
            }
            return fixture_tools.update_device(supplied["endpoint_id"], **fields)

        @tool
        def reset_device_connectivity(endpoint_id: int) -> dict:
            """Record a proposed reset, without executing it.

            Args:
                endpoint_id: Internal device id returned by the device lookup.
            """
            return fixture_tools.reset_device_connectivity(endpoint_id)

        class BoundedBedrockModel(BedrockModel):
            calls_this_turn = 0

            async def stream(self, *args, **kwargs):
                if self.calls_this_turn >= MAX_MODEL_CALLS:
                    raise RuntimeError(
                        f"Stopped after {MAX_MODEL_CALLS} model calls in one turn."
                    )
                self.calls_this_turn += 1
                async for event in super().stream(*args, **kwargs):
                    yield event

        session = boto3.Session(profile_name=profile, region_name=region)
        self._model = BoundedBedrockModel(
            boto_session=session,
            model_id=model_id,
            temperature=0,
            max_tokens=1200,
            streaming=False,
            boto_client_config=Config(
                connect_timeout=10,
                read_timeout=90,
                retries={"mode": "standard", "total_max_attempts": 2},
            ),
        )
        self._agent = Agent(
            model=self._model,
            tools=[
                get_device_info_and_id,
                get_sim_info_and_id,
                update_device,
                reset_device_connectivity,
            ],
            system_prompt=SYSTEM_PROMPT if variant == "baseline" else REGRESSION_PROMPT,
            trace_attributes={"session.id": self.session_id},
            callback_handler=None,
            hooks=[self._tool_recorder],
        )

    def run(self, prompt: str) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
            raise ValueError("Enter a prompt between 1 and 8000 characters.")
        if self._failed:
            raise RuntimeError("The prior AWS turn failed; start a fresh conversation.")
        if self._turn_count >= MAX_SESSION_TURNS:
            raise ValueError("Start a new conversation after 10 turns.")
        with _CAPTURE_LOCK:
            self._exporter.clear()
            self._model.calls_this_turn = 0
            before = len(self._tool_recorder.calls)
            started = time.monotonic()
            try:
                response = self._agent(prompt)
                self._provider.force_flush()
                records = [
                    record for record in self._exporter.get_finished_records()
                    if record.get("attributes", {}).get("session.id") == self.session_id
                ]
                roots = [
                    record for record in records
                    if record.get("attributes", {}).get("gen_ai.operation.name") == "invoke_agent"
                    and "name" in record
                ]
                if len(roots) != 1:
                    raise RuntimeError(
                        "Expected one actual Strands agent trace for this turn; "
                        "evaluation cannot continue without complete telemetry."
                    )
                root_id = roots[0]["spanId"]
                if not any(item.get("body") and item["spanId"] == root_id for item in records):
                    raise RuntimeError("The agent trace is missing its input/output event.")
                combined = self.spans + records
                if len(json.dumps(combined).encode("utf-8")) > MAX_SESSION_BYTES:
                    raise RuntimeError("Session traces exceeded the demo's 2 MB limit.")
                self.spans = combined
                self._turn_count += 1
                return {
                    "response": str(response),
                    "tool_calls": copy.deepcopy(self._tool_recorder.calls[before:]),
                    "pending_actions": copy.deepcopy(self.tools.pending_actions),
                    "duration_ms": round((time.monotonic() - started) * 1000),
                    "trace_id": roots[0]["traceId"],
                }
            except Exception:
                self._failed = True
                raise
            finally:
                self._exporter.clear()


def reference_inputs(scenario: dict, record: dict) -> list[dict]:
    """Attach unchanged golden answers to their actual trace/session ids."""
    session_id = record.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("The recorded run needs a session_id.")
    turns = record.get("turns", [])
    golden_turns = scenario.get("turns", [])
    if len(turns) != len(golden_turns):
        raise ValueError("Run every golden turn before evaluating this scenario.")
    references = []
    for golden, actual in zip(golden_turns, turns):
        expected = golden.get("expected_response")
        if expected is not None:
            trace_id = actual.get("trace_id")
            if not isinstance(trace_id, str) or len(trace_id) != 32:
                raise ValueError("A golden answer is missing its actual 32-character trace id.")
            references.append({
                "context": {"spanContext": {"sessionId": session_id, "traceId": trace_id}},
                "expectedResponse": {"text": expected},
            })
    session_reference: dict[str, Any] = {
        "context": {"spanContext": {"sessionId": session_id}}
    }
    if scenario.get("assertions"):
        session_reference["assertions"] = [{"text": text} for text in scenario["assertions"]]
    if scenario.get("expected_trajectory"):
        session_reference["expectedTrajectory"] = {
            "toolNames": scenario["expected_trajectory"]
        }
    if len(session_reference) > 1:
        references.append(session_reference)
    return references


def score_agentcore(
    scenario: dict,
    record: dict,
    profile: str | None,
    region: str,
    evaluators: list[str] | None = None,
) -> dict[str, Any]:
    """Call real synchronous AgentCore Evaluate; missing/error scores fail closed."""
    selected = list(DEFAULT_EVALUATORS if evaluators is None else evaluators)
    if not selected or len(selected) > 10 or len(set(selected)) != len(selected):
        raise ValueError("Choose 1–10 distinct evaluators.")
    refs = reference_inputs(scenario, record)
    spans = record.get("spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError("No actual AWS-mode traces are recorded for this scenario.")
    if len(spans) > 1000 or len(json.dumps(spans).encode("utf-8")) > MAX_SESSION_BYTES:
        raise ValueError("Recorded traces exceed this demo's evaluation limits.")
    trace_ids = [
        ref["context"]["spanContext"]["traceId"]
        for ref in refs if "expectedResponse" in ref
    ]
    span_trace_ids = {span.get("traceId") for span in spans if "name" in span}
    if any(trace_id not in span_trace_ids for trace_id in trace_ids):
        raise ValueError("Golden trace ids do not match the captured trace records.")
    if any(span.get("attributes", {}).get("session.id") != record["session_id"] for span in spans):
        raise ValueError("All submitted span/event records must belong to this session.")

    import boto3
    from botocore.config import Config

    client = boto3.Session(profile_name=profile, region_name=region).client(
        "bedrock-agentcore",
        config=Config(
            connect_timeout=10, read_timeout=120,
            retries={"mode": "standard", "total_max_attempts": 2},
        ),
    )
    if "evaluationReferenceInputs" not in client.meta.service_model.operation_model(
        "Evaluate"
    ).input_shape.members:
        raise RuntimeError("Upgrade boto3: this version lacks AgentCore ground-truth evaluation.")

    rows: list[dict[str, Any]] = []
    raw_responses = []
    thresholds = {name: THRESHOLDS.get(name, 0.8) for name in selected}
    for evaluator in selected:
        request = {
            "evaluatorId": evaluator,
            "evaluationInput": {"sessionSpans": spans},
            "evaluationReferenceInputs": refs,
        }
        if evaluator == "Builtin.Correctness" and trace_ids:
            request["evaluationTarget"] = {"traceIds": trace_ids}
        try:
            response = client.evaluate(**request)
            raw_responses.append({"evaluator_id": evaluator, "response": response})
            results = response.get("evaluationResults") or []
            if not results:
                raise RuntimeError("AgentCore returned no evaluation results.")
            seen = set()
            for result in results:
                trace_id = result.get("context", {}).get("spanContext", {}).get("traceId")
                if trace_id:
                    seen.add(trace_id)
                value = result.get("value")
                error = result.get("errorMessage") or result.get("errorCode")
                if not error and (
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1
                ):
                    error = "AgentCore did not return a valid score between 0 and 1."
                rows.append({
                    "evaluator_id": evaluator,
                    "value": value if not error else None,
                    "label": str(result.get("label", ""))[:200],
                    "explanation": str(result.get("explanation", ""))[:4000],
                    "trace_id": trace_id,
                    "error": str(error)[:1600] if error else None,
                })
            if evaluator == "Builtin.Correctness" and set(trace_ids) - seen:
                raise RuntimeError("AgentCore did not score every golden-answer trace.")
        except Exception as exc:
            rows.append({
                "evaluator_id": evaluator, "value": None, "label": "ERROR",
                "explanation": "", "trace_id": None, "error": str(exc)[:1600],
            })
            raw_responses.append({"evaluator_id": evaluator, "error": str(exc)[:1600]})
    status = "PASS"
    if any(row["error"] for row in rows):
        status = "ERROR"
    elif any(row["value"] < thresholds[row["evaluator_id"]] for row in rows):
        status = "FAIL"
    return {
        "status": status,
        "results": rows,
        "raw_responses": raw_responses,
        "thresholds": thresholds,
        "threshold_note": "Demo thresholds only; these are not AWS recommendations.",
    }
