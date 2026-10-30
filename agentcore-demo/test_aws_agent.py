"""Offline contract checks for real-trace serialization and AgentCore scoring."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from enum import Enum
from types import SimpleNamespace
from unittest.mock import patch

import aws_agent


class Kind(Enum):
    INTERNAL = 0


class Status(Enum):
    OK = 1


SESSION = "test-session"
TRACE = "a" * 32
SPAN = "b" * 16


def readable_span(*, events=None, session=SESSION, trace_id=TRACE):
    context = SimpleNamespace(trace_id=int(trace_id, 16), span_id=int(SPAN, 16))
    return SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name=aws_agent.SCOPE, version="test"),
        attributes={
            "session.id": session,
            "gen_ai.operation.name": "invoke_agent",
            "demo.tuple": ("one", "two"),
        },
        get_span_context=lambda: context,
        parent=None,
        kind=Kind.INTERNAL,
        name="invoke_agent test",
        start_time=1000,
        end_time=2000,
        status=SimpleNamespace(status_code=Status.OK),
        resource=SimpleNamespace(attributes={"service.name": "offline-test"}),
        events=events if events is not None else [
            SimpleNamespace(
                name="gen_ai.user.message",
                attributes={"content": json.dumps([{"text": "Why is my device offline?"}])},
            ),
            SimpleNamespace(
                name="gen_ai.choice",
                attributes={"message": "No SIM is attached. Which SIM?", "finish_reason": "end_turn"},
            ),
        ],
    )


def golden_and_record():
    scenario = {
        "turns": [{"input": "Why offline?", "expected_response": "No SIM is attached."}],
        "assertions": ["Ask which SIM to attach."],
        "expected_trajectory": ["get_device_info_and_id"],
    }
    record = {
        "session_id": SESSION,
        "turns": [{"input": "Why offline?", "trace_id": TRACE}],
        "spans": aws_agent.span_records(readable_span()),
    }
    return scenario, record


class FakeClient:
    def __init__(self, result_for=None):
        self.requests = []
        self.result_for = result_for
        self.meta = SimpleNamespace(service_model=SimpleNamespace(
            operation_model=lambda name: SimpleNamespace(
                input_shape=SimpleNamespace(members={"evaluationReferenceInputs": {}})
            )
        ))

    def evaluate(self, **request):
        self.requests.append(copy.deepcopy(request))
        if self.result_for:
            return self.result_for(request)
        return {
            "evaluationResults": [{
                "value": 1.0, "label": "Correct", "explanation": "Offline fake SDK response.",
                "context": {"spanContext": {"sessionId": SESSION, "traceId": TRACE}},
            }]
        }


def sdk_modules(client):
    return {
        "boto3": SimpleNamespace(
            Session=lambda **kwargs: SimpleNamespace(client=lambda *args, **kw: client)
        ),
        "botocore.config": SimpleNamespace(Config=lambda **kwargs: kwargs),
    }


class TraceContractTests(unittest.TestCase):
    def test_keeps_actual_ids_and_nested_strands_content(self):
        span, event = aws_agent.span_records(readable_span())
        self.assertEqual(span["traceId"], TRACE)
        self.assertEqual(span["spanId"], SPAN)
        self.assertEqual(span["attributes"]["demo.tuple"], ["one", "two"])
        self.assertNotIn("spanEvents", span)
        self.assertEqual(event["traceId"], span["traceId"])
        self.assertEqual(event["spanId"], span["spanId"])
        self.assertEqual(event["attributes"]["event.name"], aws_agent.SCOPE)
        user = event["body"]["input"]["messages"][0]
        self.assertEqual(user["role"], "user")
        self.assertIsInstance(user["content"], dict)
        self.assertEqual(json.loads(user["content"]["content"])[0]["text"],
                         "Why is my device offline?")
        self.assertEqual(event["body"]["output"]["messages"][0]["content"]["finish_reason"],
                         "end_turn")

    def test_last_user_event_matches_adot_content_collection(self):
        span = readable_span()
        span.events.insert(0, SimpleNamespace(
            name="gen_ai.user.message", attributes={"content": "an earlier turn"}
        ))
        event = aws_agent.span_records(span)[1]
        self.assertEqual(len(event["body"]["input"]["messages"]), 1)
        self.assertNotIn("earlier", event["body"]["input"]["messages"][0]["content"]["content"])

    def test_tool_proposal_is_preserved_as_tool_output(self):
        span = readable_span(events=[
            SimpleNamespace(name="gen_ai.tool.message", attributes={
                "content": '{"endpoint_id":2,"new_sim_id":9,"new_status_id":0}',
                "id": "tool-1", "role": "tool",
            }),
            SimpleNamespace(name="gen_ai.choice", attributes={
                "message": '[{"json":{"status":"pending_approval","executed":false}}]',
                "id": "tool-1",
            }),
        ])
        span.name = "tool.update_device"
        span.attributes["gen_ai.operation.name"] = "execute_tool"
        span.attributes["gen_ai.tool.name"] = "update_device"
        records = aws_agent.span_records(span)
        self.assertEqual(records[0]["attributes"]["gen_ai.tool.name"], "update_device")
        body = records[1]["body"]
        self.assertEqual(body["input"]["messages"][0]["role"], "tool")
        self.assertIn("pending_approval", body["output"]["messages"][0]["content"]["message"])

    def test_rejects_unsupported_semconv_instead_of_silently_losing_content(self):
        span = readable_span(events=[SimpleNamespace(
            name="gen_ai.client.inference.operation.details",
            attributes={"gen_ai.input.messages": "[]"},
        )])
        with self.assertRaisesRegex(ValueError, "default telemetry"):
            aws_agent.span_records(span)

    def test_rejects_missing_session_and_ignores_transport_spans(self):
        span = readable_span()
        span.attributes.pop("session.id")
        with self.assertRaisesRegex(ValueError, "session.id"):
            aws_agent.span_records(span)
        span.instrumentation_scope.name = "requests"
        self.assertEqual(aws_agent.span_records(span), [])

    def test_actual_otel_readable_spans_when_dependency_installed(self):
        try:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        except ImportError:
            self.skipTest("Install demo requirements for the actual OTel SDK contract test.")
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer(aws_agent.SCOPE)
        with tracer.start_as_current_span(
            "invoke_agent offline", attributes={
                "session.id": SESSION, "gen_ai.operation.name": "invoke_agent"
            },
        ) as span:
            span.add_event("gen_ai.user.message", {"content": '[{"text":"Check my device"}]'})
            span.add_event("gen_ai.choice", {
                "message": "No SIM attached. Which SIM?", "finish_reason": "end_turn"
            })
        actual = exporter.get_finished_spans()[0]
        records = aws_agent.span_records(actual)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["spanId"], format(actual.context.span_id, "016x"))
        self.assertEqual(records[1]["body"]["output"]["messages"][0]["content"]["message"],
                         "No SIM attached. Which SIM?")
        provider.shutdown()

    def test_capture_keeps_content_when_adot_exporter_mutates_original_span(self):
        try:
            from amazon.opentelemetry.distro.llo_handler import LLOHandler
            from opentelemetry.sdk._logs import LoggerProvider
            from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
        except ImportError:
            self.skipTest("Install demo requirements for the ADOT capture regression test.")

        log_provider = LoggerProvider()
        handler = LLOHandler(log_provider)
        exported_spans = []

        class ExtractingExporterProcessor(SpanProcessor):
            def on_end(self, span):
                # ADOT's real exporter performs this destructive extraction.
                handler.process_spans([span])
                exported_spans.append(span)

        provider = TracerProvider()
        # Auto-instrumentation registers its exporter before our local capture.
        provider.add_span_processor(ExtractingExporterProcessor())
        capture = aws_agent._TraceRecordCapture()
        provider.add_span_processor(capture)
        try:
            tracer = provider.get_tracer(aws_agent.SCOPE)
            with tracer.start_as_current_span(
                "invoke_agent offline",
                attributes={"session.id": SESSION, "gen_ai.operation.name": "invoke_agent"},
            ) as span:
                span.add_event("gen_ai.user.message", {"content": '[{"text":"Check my device"}]'})
                span.add_event("gen_ai.choice", {
                    "message": "No SIM attached. Which SIM?", "finish_reason": "end_turn",
                })
            provider.force_flush()
            self.assertEqual(len(exported_spans), 1)
            self.assertEqual(len(exported_spans[0].events), 0)
            records = capture.get_finished_records()
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["traceId"], format(span.context.trace_id, "032x"))
            self.assertEqual(records[1]["body"]["output"]["messages"][0]["content"]["message"],
                             "No SIM attached. Which SIM?")
            records[1]["body"].clear()
            self.assertIn("output", capture.get_finished_records()[1]["body"])
            capture.clear()
            self.assertEqual(capture.get_finished_records(), [])
        finally:
            provider.shutdown()
            log_provider.shutdown()


class EvaluationContractTests(unittest.TestCase):
    def test_ground_truth_uses_actual_trace_and_session_ids(self):
        scenario, record = golden_and_record()
        refs = aws_agent.reference_inputs(scenario, record)
        self.assertEqual(refs[0]["context"]["spanContext"],
                         {"sessionId": SESSION, "traceId": TRACE})
        self.assertEqual(refs[0]["expectedResponse"], {"text": "No SIM is attached."})
        self.assertEqual(refs[1]["context"]["spanContext"], {"sessionId": SESSION})
        self.assertEqual(refs[1]["expectedTrajectory"]["toolNames"],
                         ["get_device_info_and_id"])

    def test_all_evaluators_receive_real_recorded_spans_and_expected_targets(self):
        scenario, record = golden_and_record()
        client = FakeClient()
        with patch.dict(sys.modules, sdk_modules(client)):
            result = aws_agent.score_agentcore(scenario, record, None, "eu-west-1")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["results"]), 3)
        self.assertEqual(len(result["raw_responses"]), 3)
        self.assertEqual(client.requests[0]["evaluationTarget"], {"traceIds": [TRACE]})
        for request in client.requests:
            self.assertEqual(request["evaluationInput"]["sessionSpans"], record["spans"])
        self.assertNotIn("evaluationTarget", client.requests[1])

    def test_low_score_fails_demo_threshold(self):
        scenario, record = golden_and_record()
        client = FakeClient(lambda request: {"evaluationResults": [{
            "value": 0.5, "label": "Partially correct",
            "context": {"spanContext": {"traceId": TRACE}},
        }]})
        with patch.dict(sys.modules, sdk_modules(client)):
            result = aws_agent.score_agentcore(scenario, record, None, "eu-west-1")
        self.assertEqual(result["status"], "FAIL")

    def test_empty_error_nonnumeric_and_missing_trace_results_never_pass(self):
        scenario, record = golden_and_record()
        responses = [
            {"evaluationResults": []},
            {"evaluationResults": [{"errorCode": "MappingFailed", "errorMessage": "missing input"}]},
            {"evaluationResults": [{"label": "PASS", "value": None}]},
            {"evaluationResults": [{"label": "PASS", "value": float("nan")}]},
            {"evaluationResults": [{"label": "PASS", "value": 1.0}]},
        ]
        for response in responses:
            with self.subTest(response=response):
                client = FakeClient(lambda request: response)
                with patch.dict(sys.modules, sdk_modules(client)):
                    result = aws_agent.score_agentcore(
                        scenario, record, None, "eu-west-1",
                        evaluators=["Builtin.Correctness"],
                    )
                self.assertEqual(result["status"], "ERROR")
                self.assertTrue(any(row["error"] for row in result["results"]))

    def test_api_exception_is_reported_as_error(self):
        scenario, record = golden_and_record()

        def fail(request):
            raise RuntimeError("AccessDenied: no evaluation permission")

        with patch.dict(sys.modules, sdk_modules(FakeClient(fail))):
            result = aws_agent.score_agentcore(scenario, record, None, "eu-west-1")
        self.assertEqual(result["status"], "ERROR")
        self.assertTrue(all(row["error"] for row in result["results"]))

    def test_incomplete_mismatched_or_absent_traces_are_rejected_before_aws(self):
        scenario, record = golden_and_record()
        for kind in ("incomplete", "wrong_trace", "wrong_session", "no_spans"):
            bad = copy.deepcopy(record)
            if kind == "incomplete":
                bad["turns"] = []
            elif kind == "wrong_trace":
                bad["turns"][0]["trace_id"] = "c" * 32
            elif kind == "wrong_session":
                bad["session_id"] = "another-session"
            else:
                bad["spans"] = []
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                aws_agent.score_agentcore(scenario, bad, None, "eu-west-1")


class ActualSdkOfflineTests(unittest.TestCase):
    """Exercise installed Strands, OTel and boto3; every network call is stubbed."""

    def setUp(self):
        try:
            import boto3
            import strands
            from botocore.stub import Stubber
        except ImportError:
            self.skipTest("Install demo requirements for actual SDK offline tests.")
        self.stubber = Stubber
        # Explicit dummy credentials prevent credentials/metadata network lookup.
        self.session = boto3.Session(
            aws_access_key_id="offline-test",
            aws_secret_access_key="offline-test",
            region_name="eu-west-1",
        )

    def make_agent(self):
        with patch("boto3.Session", return_value=self.session):
            return aws_agent.BedrockSupportAgent(None, "eu-west-1", "offline-model")

    def converse(self, *, tool_name=None, arguments=None, call_id="call-1"):
        content = (
            [{"toolUse": {"toolUseId": call_id, "name": tool_name, "input": arguments}}]
            if tool_name else [{"text": "Offline stub response; not a model quality test."}]
        )
        return {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": "tool_use" if tool_name else "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 10, "totalTokens": 20},
            "metrics": {"latencyMs": 1},
        }

    def test_real_sdk_two_turn_tool_flow_and_otel_capture(self):
        from checks import evaluate_local
        from domain import load_dataset

        agent = self.make_agent()
        scenario = load_dataset()["scenarios"][1]
        with self.stubber(agent._model.client) as stub:
            for response in [
                self.converse(tool_name="get_device_info_and_id",
                              arguments={"identifier": "Demo-Device-01"}, call_id="lookup"),
                self.converse(),
                self.converse(tool_name="get_sim_info_and_id",
                              arguments={"identifier": "0000000000000000000"}, call_id="sim"),
                self.converse(tool_name="update_device", arguments={
                    "endpoint_id": 2, "new_sim_id": 9, "new_status_id": 0,
                }, call_id="proposal"),
                self.converse(),
            ]:
                stub.add_response("converse", response)
            turns = [agent.run(turn["input"]) for turn in scenario["turns"]]
            stub.assert_no_pending_responses()
        record = {"session_id": agent.session_id, "turns": turns,
                  "spans": agent.spans, "pending_actions": agent.tools.pending_actions}
        self.assertEqual(evaluate_local(scenario, record)["status"], "PASS")
        self.assertNotEqual(turns[0]["trace_id"], turns[1]["trace_id"])
        proposal = turns[1]["tool_calls"][-1]
        self.assertEqual(proposal["result"]["arguments"], proposal["arguments"])
        self.assertIs(proposal["result"]["executed"], False)
        self.assertEqual(proposal["tool_status"], "success")
        tool_spans = [span for span in agent.spans if span.get("attributes", {}).get(
            "gen_ai.tool.name") == "update_device"]
        self.assertTrue(tool_spans)
        self.assertTrue(all(span["attributes"]["session.id"] == agent.session_id
                            for span in agent.spans))
        update_events = [span for span in agent.spans
                         if span.get("body") and span["spanId"] == tool_spans[0]["spanId"]]
        self.assertIn("pending_approval", json.dumps(update_events))

    def test_explicit_null_is_not_silently_omitted(self):
        agent = self.make_agent()
        arguments = {"endpoint_id": 2, "new_sim_id": None, "new_status_id": 0}
        with self.stubber(agent._model.client) as stub:
            stub.add_response("converse", self.converse(
                tool_name="update_device", arguments=arguments))
            stub.add_response("converse", self.converse())
            turn = agent.run("Offline explicit-null regression test")
            stub.assert_no_pending_responses()
        call = turn["tool_calls"][0]
        self.assertEqual(call["arguments"], arguments)
        self.assertEqual(call["result"]["arguments"], arguments)
        self.assertEqual(agent.tools.pending_actions[0]["arguments"], arguments)

    def test_extra_fields_schema_errors_and_unknown_tools_remain_visible(self):
        cases = [
            ("update_device", {"endpoint_id": 2, "new_status_id": 0, "tags": None}),
            ("update_device", {"endpoint_id": "not-an-integer", "new_status_id": 0}),
            ("unknown_tool", {"endpoint_id": 2}),
        ]
        for name, arguments in cases:
            with self.subTest(name=name, arguments=arguments):
                agent = self.make_agent()
                with self.stubber(agent._model.client) as stub:
                    stub.add_response("converse", self.converse(
                        tool_name=name, arguments=arguments))
                    stub.add_response("converse", self.converse())
                    turn = agent.run("Offline invalid-tool regression test")
                    stub.assert_no_pending_responses()
                self.assertEqual(len(turn["tool_calls"]), 1)
                call = turn["tool_calls"][0]
                self.assertEqual(call["name"], name)
                self.assertEqual(call["arguments"], arguments)
                self.assertEqual(call["tool_status"], "error")
                self.assertEqual(call["result"]["status"], "error")
                self.assertEqual(agent.tools.pending_actions, [])
                self.assertEqual(agent.tools.calls, [])

    def test_model_call_limit_stops_repeated_tools_without_extra_request(self):
        from strands.types.exceptions import EventLoopException

        agent = self.make_agent()
        with self.stubber(agent._model.client) as stub:
            for index in range(aws_agent.MAX_MODEL_CALLS):
                stub.add_response("converse", self.converse(
                    tool_name="get_device_info_and_id",
                    arguments={"identifier": "Demo-Device-01"}, call_id=f"lookup-{index}"))
            with self.assertRaisesRegex(EventLoopException, "Stopped after 8 model calls"):
                agent.run("Offline bounded-loop test")
            stub.assert_no_pending_responses()
        with self.assertRaisesRegex(RuntimeError, "fresh conversation"):
            agent.run("Do not reuse a failed conversation")

    def test_agentcore_request_validates_against_installed_boto3_schema(self):
        scenario, record = golden_and_record()
        client = self.session.client("bedrock-agentcore")
        response = {"evaluationResults": [{
            "evaluatorArn": "arn:aws:bedrock-agentcore:eu-west-1::evaluator/Builtin.Correctness",
            "evaluatorId": "Builtin.Correctness",
            "evaluatorName": "Correctness",
            "context": {"spanContext": {"sessionId": SESSION, "traceId": TRACE}},
            "value": 1.0, "label": "Correct",
        }]}
        expected = {
            "evaluatorId": "Builtin.Correctness",
            "evaluationInput": {"sessionSpans": record["spans"]},
            "evaluationTarget": {"traceIds": [TRACE]},
            "evaluationReferenceInputs": aws_agent.reference_inputs(scenario, record),
        }
        session_stub = SimpleNamespace(client=lambda *args, **kwargs: client)
        with self.stubber(client) as stub:
            stub.add_response("evaluate", response, expected)
            with patch("boto3.Session", return_value=session_stub):
                result = aws_agent.score_agentcore(
                    scenario, record, None, "eu-west-1", ["Builtin.Correctness"])
            stub.assert_no_pending_responses()
        self.assertEqual(result["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
