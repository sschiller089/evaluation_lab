"""Small, offline fixture adapter for the synthetic no-SIM golden dataset.

The scripted agent is a rehearsal of the harness, not an AI-quality measurement.
Neither the adapter nor the rehearsal makes network calls or changes fixture data.
"""

from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from typing import Any


EVALUATIONS_PATH = Path(__file__).resolve().parent.parent / "evaluations"
DATASET_PATH = EVALUATIONS_PATH / "datasets" / "no_sim_guard.json"
FIXTURE_PATH = EVALUATIONS_PATH / "fixtures" / "no_sim_guard.json"

# A sentinel preserves the difference between omission and an explicitly supplied
# null. The public model tool schema can use ordinary optional integer arguments.
_UNSET = object()


def load_dataset() -> dict[str, Any]:
    """Read the versioned synthetic lab scenarios."""
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


class FixtureTools:
    """Per-session tool data, call records, and approval proposals."""

    def __init__(self, fixture_path: str | Path = FIXTURE_PATH):
        self.fixture_path = Path(fixture_path)
        self.fixture = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        self.calls: list[dict[str, Any]] = []
        self.pending_actions: list[dict[str, Any]] = []

    def _record(
        self, name: str, arguments: dict[str, Any], result: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "name": name,
                "arguments": copy.deepcopy(arguments),
                "result": copy.deepcopy(result),
            }
        )
        return copy.deepcopy(result)

    def get_device_info_and_id(self, identifier: str) -> dict[str, Any]:
        """Find a fixture device by its name or endpoint ID."""
        key = str(identifier).strip().casefold()
        items = self.fixture["graphql"]["endpointsQuery"]["items"]
        matches = [
            item
            for item in items
            if key in (str(item["id"]), item["name"].strip().casefold())
        ]
        if len(matches) == 1:
            result = {"found": True, **copy.deepcopy(matches[0])}
        else:
            result = {
                "found": False,
                "identifier": identifier,
                "message": "No unique device matches this identifier in the fixture.",
            }
        return self._record(
            "get_device_info_and_id", {"identifier": identifier}, result
        )

    def get_sim_info_and_id(self, identifier: str) -> dict[str, Any]:
        """Find a fixture SIM by ID, base ICCID, or ICCID including its check digit."""
        key = str(identifier).strip()
        items = self.fixture["graphql"]["sims"]["items"]
        matches = [
            item
            for item in items
            if key
            in (
                str(item["id"]),
                item["iccid"],
                item["iccid_with_luhn"],
            )
        ]
        if len(matches) == 1:
            result = {"found": True, **copy.deepcopy(matches[0])}
        else:
            result = {
                "found": False,
                "identifier": identifier,
                "message": "No unique SIM matches this identifier in the fixture.",
            }
        return self._record(
            "get_sim_info_and_id", {"identifier": identifier}, result
        )

    def _propose(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        action = {
            "name": name,
            "arguments": copy.deepcopy(arguments),
            "status": "pending_approval",
            "executed": False,
        }
        self.pending_actions.append(copy.deepcopy(action))
        return self._record(name, arguments, action)

    def update_device(
        self,
        endpoint_id: int,
        new_sim_id: int | None = _UNSET,
        new_status_id: int | None = _UNSET,
    ) -> dict[str, Any]:
        """Propose a partial update; never execute a change or consume PATCH data.

        Unrequested fields are absent. An explicitly supplied null remains in the
        call record so the checks can identify it instead of silently correcting it.
        """
        arguments: dict[str, Any] = {"endpoint_id": endpoint_id}
        if new_sim_id is not _UNSET:
            arguments["new_sim_id"] = new_sim_id
        if new_status_id is not _UNSET:
            arguments["new_status_id"] = new_status_id
        return self._propose("update_device", arguments)

    def reset_device_connectivity(self, endpoint_id: int) -> dict[str, Any]:
        """Trap tool: record an approval proposal, without resetting anything."""
        return self._propose(
            "reset_device_connectivity", {"endpoint_id": endpoint_id}
        )


class ScriptedAgent:
    """An explicitly non-LLM rehearsal, using tools and conversation state.

    ``baseline`` checks for a missing SIM before the disabled-device branch.
    ``regression`` deliberately reverses that priority and proposes enabling a
    device with no SIM. Responses are built from tool results, not golden answers.
    """

    def __init__(
        self, variant: str = "baseline", tools: FixtureTools | None = None
    ):
        if variant not in {"baseline", "regression"}:
            raise ValueError("variant must be 'baseline' or 'regression'")
        self.variant = variant
        self.tools = tools if tools is not None else FixtureTools()
        self.last_endpoint: dict[str, Any] | None = None
        self.history: list[dict[str, str]] = []
        self.answered_turns = 0

    def run(self, prompt: str) -> dict[str, Any]:
        if not isinstance(prompt, str):
            raise TypeError("The rehearsal accepts a string prompt.")
        started = time.perf_counter()
        call_start = len(self.tools.calls)
        response = self._answer(prompt)
        self.history.append({"prompt": prompt, "response": response})
        self.answered_turns += 1
        return {
            "response": response,
            "tool_calls": copy.deepcopy(self.tools.calls[call_start:]),
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "pending_actions": copy.deepcopy(self.tools.pending_actions),
        }

    def _answer(self, prompt: str) -> str:
        sim_selection = re.search(
            r"\b(?:ICCID|SIM(?:\s+id)?)\s+(\d+)\b", prompt, re.IGNORECASE
        )
        if sim_selection:
            return self._select_sim(sim_selection.group(1))

        identifier = next(
            (
                item["name"]
                for item in self.tools.fixture["graphql"]["endpointsQuery"]["items"]
                if item["name"].casefold() in prompt.casefold()
            ),
            None,
        )
        if identifier is None:
            return "Please name the device you want me to inspect."

        device = self.tools.get_device_info_and_id(identifier)
        if not device["found"]:
            return "I could not identify that device. Please check its name or ID."
        self.last_endpoint = device
        label = f"{device['name']} (endpoint id {device['id']})"
        status = device["status"]["description"]

        if self.variant == "regression" and device["status"]["id"] == 1:
            self.tools.update_device(endpoint_id=device["id"], new_status_id=0)
            return (
                f"{label} is {status}. I propose enabling it. "
                "Please approve the action; no change has been made."
            )

        if device["sim"] is None:
            return (
                f"{label} is {status} and has no SIM attached. "
                "The missing SIM blocks connectivity; enabling or resetting the "
                "device cannot help until a SIM is attached. "
                "Which SIM should I attach? Please provide its ICCID or SIM ID."
            )
        return f"{label} is {status} and has a SIM attached."

    def _select_sim(self, identifier: str) -> str:
        if self.last_endpoint is None:
            return "Please identify the device before selecting its SIM."
        sim = self.tools.get_sim_info_and_id(identifier)
        if not sim["found"]:
            return "I could not find that SIM. Please check its ICCID or SIM ID."
        if sim["endpoint"] is not None:
            return "That SIM is already attached to a device. Please select another SIM."

        device = self.last_endpoint
        self.tools.update_device(
            endpoint_id=device["id"], new_sim_id=sim["id"], new_status_id=0
        )
        return (
            f"I found SIM {sim['id']} (ICCID {sim['iccid_with_luhn']}, "
            f"{sim['status']['description']}, unattached). "
            f"I propose attaching it to {device['name']} and enabling the device "
            "in one action. Please approve the action; no change has been made."
        )
