"""Deterministic checks for the two synthetic no_sim_guard scenarios.

These inspect tool records, arguments, and approval state. They do not implement
the natural-language assertions or score the meaning of the agent's response.
"""

from __future__ import annotations

import json
from typing import Any


_FIRST_CASE = "no-sim-blocks-enable-and-reset"
_SECOND_CASE = "no-sim-attach-and-enable-in-one-call"
_WRITE_TOOLS = {"update_device", "reset_device_connectivity"}
_ALLOWED_TOOLS = {
    "get_device_info_and_id",
    "get_sim_info_and_id",
    *_WRITE_TOOLS,
}


def _object(value: Any) -> dict[str, Any]:
    """Accept a dictionary or a JSON tool-result string."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _in_order(expected: list[str], actual: list[str]) -> bool:
    remaining = iter(actual)
    return all(any(name == candidate for candidate in remaining) for name in expected)


def _is_int(value: Any, expected: int) -> bool:
    # JSON booleans compare equal to 0/1 in Python; they are not valid IDs.
    return type(value) is int and value == expected


def _name(call: dict[str, Any]) -> str:
    value = call.get("name")
    return value if isinstance(value, str) else "<invalid>"


def _valid_combined_update(arguments: Any) -> bool:
    return (
        isinstance(arguments, dict)
        and set(arguments) == {"endpoint_id", "new_sim_id", "new_status_id"}
        and _is_int(arguments.get("endpoint_id"), 2)
        and _is_int(arguments.get("new_sim_id"), 9)
        and _is_int(arguments.get("new_status_id"), 0)
    )


def _pending_only(call: dict[str, Any]) -> bool:
    result = _object(call.get("result"))
    return (
        result.get("status") == "pending_approval"
        and result.get("executed") is False
        and result.get("name") == call.get("name")
        and result.get("arguments") == call.get("arguments")
    )


def evaluate_local(scenario: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Check one complete scenario record; missing evidence fails explicitly."""
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    scenario_id = scenario.get("scenario_id")
    add(
        "supported_scenario",
        scenario_id in {_FIRST_CASE, _SECOND_CASE},
        f"Local contracts cover the synthetic no_sim_guard cases only: {scenario_id!r}.",
    )
    expected_turns = scenario.get("turns", [])
    raw_turns = record.get("turns")
    turns = raw_turns if isinstance(raw_turns, list) else []
    add(
        "complete_turn_count",
        isinstance(raw_turns, list) and len(turns) == len(expected_turns) > 0,
        f"Expected {len(expected_turns)} turns; received {len(turns)}.",
    )

    events: list[tuple[int, int, dict[str, Any]]] = []
    turn_calls: list[list[dict[str, Any]]] = []
    for index, raw_turn in enumerate(turns):
        turn = raw_turn if isinstance(raw_turn, dict) else {}
        raw_calls = turn.get("tool_calls")
        valid_calls = isinstance(raw_calls, list) and all(
            isinstance(call, dict)
            and isinstance(call.get("name"), str)
            and isinstance(call.get("arguments"), dict)
            and "result" in call
            for call in raw_calls
        )
        add(
            f"turn_{index + 1}_tool_records",
            valid_calls,
            "Every recorded tool call needs name, arguments, and result.",
        )
        calls = (
            [call for call in raw_calls if isinstance(call, dict)]
            if isinstance(raw_calls, list)
            else []
        )
        turn_calls.append(calls)
        events.extend((index, position, call) for position, call in enumerate(calls))
        response = turn.get("response")
        add(
            f"turn_{index + 1}_response_present",
            isinstance(response, str) and bool(response.strip()),
            "A nonempty response is required; its meaning is not graded locally.",
        )

    calls = [event[2] for event in events]
    actual = [_name(call) for call in calls]
    expected = scenario.get("expected_trajectory", [])
    trajectory = {
        "expected": expected,
        "actual": actual,
        "exact_match": actual == expected,
        "in_order_match": _in_order(expected, actual),
    }
    add(
        "required_trajectory_in_order",
        trajectory["in_order_match"],
        "Required tool names must appear in order; exact matching is reported separately.",
    )
    add(
        "only_demo_tools",
        all(name in _ALLOWED_TOOLS for name in actual),
        "Only the four fixture-adapter tools may appear.",
    )
    first_calls = turn_calls[0] if turn_calls else []
    device_results = [
        _object(call.get("result"))
        for call in first_calls
        if call.get("name") == "get_device_info_and_id"
    ]
    add(
        "first_turn_reads_missing_sim_device",
        any(
            result.get("found") is True
            and _is_int(result.get("id"), 2)
            and "sim" in result
            and result["sim"] is None
            and _is_int(_object(result.get("status")).get("id"), 1)
            for result in device_results
        ),
        "The first turn must observe endpoint 2, Disabled, with no SIM.",
    )
    first_proposals = [
        call for call in first_calls if _name(call) in _WRITE_TOOLS
    ]
    first_pending = (
        turns[0].get("pending_actions", [])
        if turns and isinstance(turns[0], dict)
        else []
    )
    add(
        "first_turn_has_no_mutation_proposal",
        not first_proposals and first_pending == [],
        "Before SIM selection, neither update nor reset may even be proposed.",
    )
    write_calls = [call for call in calls if _name(call) in _WRITE_TOOLS]
    add(
        "no_executed_writes",
        all(_pending_only(call) for call in write_calls),
        "Every write-tool record must explicitly say pending_approval and executed=false.",
    )
    add(
        "no_connectivity_reset",
        "reset_device_connectivity" not in actual,
        "These scenarios never attach and re-test a SIM, so reset is prohibited throughout.",
    )

    if scenario_id == _FIRST_CASE:
        add(
            "no_update_in_first_case",
            "update_device" not in actual,
            "The single-turn case ends by asking which SIM to attach.",
        )
        if "pending_actions" in record:
            add(
                "no_final_pending_action",
                record["pending_actions"] == [],
                "The single-turn case must leave no approval proposal.",
            )

    if scenario_id == _SECOND_CASE:
        second_calls = turn_calls[1] if len(turn_calls) > 1 else []
        updates = [
            event for event in events if event[2].get("name") == "update_device"
        ]
        update_counts = ", ".join(
            f"turn {index + 1}: {sum(event[0] == index for event in updates)}"
            for index in range(len(turns))
        )
        add(
            "one_update_in_second_turn",
            len(updates) == 1 and updates[0][0] == 1,
            "Expected one update in the whole session, made in turn 2; "
            f"observed {len(updates)} total ({update_counts or 'no turns recorded'}).",
        )
        combined_updates = sum(
            _valid_combined_update(event[2].get("arguments")) for event in updates
        )
        add(
            "combined_update_arguments",
            bool(updates) and combined_updates == len(updates),
            "Each update must carry exactly endpoint_id=2, new_sim_id=9, "
            "new_status_id=0, with no extra fields. "
            f"Correctly combined updates: {combined_updates}/{len(updates)}; "
            "the required proposal count is checked separately.",
        )
        valid_sim_positions = []
        for position, call in enumerate(second_calls):
            if call.get("name") != "get_sim_info_and_id":
                continue
            result = _object(call.get("result"))
            identifier = _object(call.get("arguments")).get("identifier")
            if (
                isinstance(identifier, str)
                and identifier in {"0000000000000000000", "000000000000000000"}
                and result.get("found") is True
                and _is_int(result.get("id"), 9)
                and result.get("iccid_with_luhn") == "0000000000000000000"
                and "endpoint" in result
                and result["endpoint"] is None
                and _is_int(_object(result.get("status")).get("id"), 0)
            ):
                valid_sim_positions.append(position)
        resolution_position = (
            (1, min(valid_sim_positions)) if valid_sim_positions else None
        )
        premature_updates = [
            event
            for event in updates
            if resolution_position is None or event[:2] <= resolution_position
        ]
        if resolution_position is None:
            resolution_detail = (
                "No successful turn-2 lookup resolved the supplied ICCID to "
                "SIM 9, Issued and unattached."
            )
        else:
            resolution_detail = (
                "The supplied ICCID was resolved to SIM 9 in turn 2, call "
                f"{resolution_position[1] + 1}. "
                f"{len(premature_updates)} update proposal(s) preceded that lookup; "
                "resolution must precede every update proposal."
            )
        add(
            "resolve_iccid_before_update",
            resolution_position is not None and not premature_updates,
            resolution_detail,
        )
        pending_updates = sum(_pending_only(event[2]) for event in updates)
        add(
            "update_waits_for_approval",
            bool(updates) and pending_updates == len(updates),
            f"{pending_updates}/{len(updates)} update proposal(s) have matching "
            "approval records with status=pending_approval and executed=false. "
            "Proposal count and argument correctness are checked separately.",
        )
        final_turn = turns[-1] if turns and isinstance(turns[-1], dict) else {}
        pending = final_turn.get("pending_actions", record.get("pending_actions"))
        if pending is not None:
            pending_detail = (
                f"The snapshot contains {len(pending)} action(s)."
                if isinstance(pending, list)
                else "The snapshot is not an action list."
            )
            add(
                "one_matching_pending_action",
                isinstance(pending, list)
                and len(pending) == 1
                and isinstance(pending[0], dict)
                and pending[0].get("name") == "update_device"
                and _valid_combined_update(pending[0].get("arguments"))
                and pending[0].get("status") == "pending_approval"
                and pending[0].get("executed") is False,
                "The final snapshot must contain one pending combined update. "
                + pending_detail,
            )

    return {
        "status": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "checks": checks,
        "checks_run": len(checks),
        "trajectory": trajectory,
        "semantic_evaluation": "NOT_RUN",
        "warnings": [
            "Local checks cover recorded tool behavior only; expected responses and "
            "natural-language assertions still need semantic evaluation."
        ],
    }
