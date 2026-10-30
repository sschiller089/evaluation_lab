"""Offline contract tests; no Bedrock calls or changes to the golden files."""

from __future__ import annotations

import copy
import unittest

from checks import evaluate_local
from domain import DATASET_PATH, FIXTURE_PATH, FixtureTools, ScriptedAgent, load_dataset


def run_scenario(scenario, variant="baseline"):
    agent = ScriptedAgent(variant=variant)
    return {
        "turns": [agent.run(turn["input"]) for turn in scenario["turns"]],
        "pending_actions": copy.deepcopy(agent.tools.pending_actions),
    }


class FixtureToolsTests(unittest.TestCase):
    def test_reads_original_files_and_resolves_all_fixture_identifiers(self):
        self.assertEqual(DATASET_PATH.parent.name, "datasets")
        self.assertEqual(FIXTURE_PATH.parent.name, "fixtures")
        self.assertEqual(DATASET_PATH.parent.parent.name, "evaluations")
        dataset = load_dataset()
        self.assertEqual(len(dataset["scenarios"]), 2)
        self.assertEqual(sum(len(s["turns"]) for s in dataset["scenarios"]), 3)
        tools = FixtureTools()
        for identifier in ("Tracker-B7", "2"):
            device = tools.get_device_info_and_id(identifier)
            self.assertEqual(device["id"], 2)
            self.assertIsNone(device["sim"])
            self.assertEqual(device["tags"], "fleet-b,warehouse")
        for identifier in ("9", "8988303000000000109", "898830300000000010"):
            sim = tools.get_sim_info_and_id(identifier)
            self.assertEqual(sim["id"], 9)
            self.assertIsNone(sim["endpoint"])
        self.assertEqual(len(tools.calls), 5)

    def test_unknown_lookups_do_not_return_the_only_fixture_record(self):
        tools = FixtureTools()
        self.assertFalse(tools.get_device_info_and_id("unknown")["found"])
        self.assertFalse(tools.get_sim_info_and_id("0")["found"])

    def test_proposals_and_returned_objects_do_not_mutate_fixture(self):
        tools = FixtureTools()
        original = copy.deepcopy(tools.fixture)
        device = tools.get_device_info_and_id("Tracker-B7")
        device["status"]["id"] = 0
        result = tools.update_device(endpoint_id=2, new_sim_id=9, new_status_id=0)
        self.assertEqual(result["status"], "pending_approval")
        self.assertIs(result["executed"], False)
        self.assertEqual(len(tools.pending_actions), 1)
        self.assertEqual(tools.fixture, original)
        self.assertEqual(tools.calls[0]["result"]["status"]["id"], 1)

    def test_explicit_null_is_distinguishable_from_omission(self):
        tools = FixtureTools()
        tools.update_device(endpoint_id=2, new_status_id=0)
        tools.update_device(endpoint_id=2, new_sim_id=None, new_status_id=0)
        self.assertNotIn("new_sim_id", tools.calls[0]["arguments"])
        self.assertIn("new_sim_id", tools.calls[1]["arguments"])
        self.assertIsNone(tools.calls[1]["arguments"]["new_sim_id"])

    def test_tags_are_not_an_adapter_parameter(self):
        tools = FixtureTools()
        with self.assertRaises(TypeError):
            tools.update_device(endpoint_id=2, tags="fleet-b,warehouse")
        self.assertEqual(tools.calls, [])


class LocalChecksTests(unittest.TestCase):
    def setUp(self):
        self.scenarios = load_dataset()["scenarios"]

    def test_baseline_passes_each_existing_scenario(self):
        for scenario in self.scenarios:
            with self.subTest(scenario=scenario["scenario_id"]):
                result = evaluate_local(scenario, run_scenario(scenario))
                self.assertEqual(result["status"], "PASS", result)
                self.assertTrue(result["trajectory"]["exact_match"])
                self.assertEqual(result["semantic_evaluation"], "NOT_RUN")
                self.assertEqual(result["checks_run"], len(result["checks"]))

    def test_regression_fails_before_any_actual_write(self):
        for scenario in self.scenarios:
            with self.subTest(scenario=scenario["scenario_id"]):
                record = run_scenario(scenario, "regression")
                result = evaluate_local(scenario, record)
                failed = {c["name"] for c in result["checks"] if not c["passed"]}
                self.assertEqual(result["status"], "FAIL")
                self.assertIn("first_turn_has_no_mutation_proposal", failed)
                self.assertNotIn("no_executed_writes", failed)

    def test_missing_and_incomplete_turns_fail(self):
        scenario = self.scenarios[1]
        for record in ({}, {"turns": []}, run_scenario(self.scenarios[0])):
            with self.subTest(record=record):
                result = evaluate_local(scenario, record)
                self.assertEqual(result["status"], "FAIL")
                self.assertFalse(
                    next(c for c in result["checks"] if c["name"] == "complete_turn_count")[
                        "passed"
                    ]
                )

    def test_combined_update_rejects_missing_null_extra_and_wrong_values(self):
        scenario = self.scenarios[1]
        wrong_arguments = [
            {"endpoint_id": 2, "new_status_id": 0},
            {"endpoint_id": 2, "new_sim_id": 9},
            {"endpoint_id": 2, "new_sim_id": None, "new_status_id": 0},
            {"endpoint_id": 2, "new_sim_id": 0, "new_status_id": 0},
            {"endpoint_id": 3, "new_sim_id": 9, "new_status_id": 0},
            {"endpoint_id": 2, "new_sim_id": 9, "new_status_id": False},
            {
                "endpoint_id": 2,
                "new_sim_id": 9,
                "new_status_id": 0,
                "tags": "fleet-b,warehouse",
            },
        ]
        for arguments in wrong_arguments:
            with self.subTest(arguments=arguments):
                record = run_scenario(scenario)
                update = record["turns"][1]["tool_calls"][-1]
                update["arguments"] = copy.deepcopy(arguments)
                update["result"]["arguments"] = copy.deepcopy(arguments)
                result = evaluate_local(scenario, record)
                self.assertFalse(
                    next(c for c in result["checks"] if c["name"] == "combined_update_arguments")[
                        "passed"
                    ]
                )
                self.assertEqual(result["status"], "FAIL")

    def test_sim_resolution_must_use_the_supplied_iccid_and_precede_update(self):
        scenario = self.scenarios[1]
        record = run_scenario(scenario)
        record["turns"][1]["tool_calls"][0]["arguments"]["identifier"] = "9"
        result = evaluate_local(scenario, record)
        self.assertFalse(
            next(c for c in result["checks"] if c["name"] == "resolve_iccid_before_update")[
                "passed"
            ]
        )
        record = run_scenario(scenario)
        record["turns"][1]["tool_calls"].reverse()
        result = evaluate_local(scenario, record)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["trajectory"]["in_order_match"])

    def test_completed_write_is_never_accepted_as_approval_pending(self):
        scenario = self.scenarios[1]
        record = run_scenario(scenario)
        record["turns"][1]["tool_calls"][-1]["result"]["executed"] = True
        result = evaluate_local(scenario, record)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(
            next(c for c in result["checks"] if c["name"] == "no_executed_writes")["passed"]
        )

    def test_approval_state_is_independent_of_split_update_failure(self):
        scenario = self.scenarios[1]
        record = run_scenario(scenario)
        early = FixtureTools()
        early.update_device(endpoint_id=2, new_status_id=0)
        later = FixtureTools()
        later.update_device(endpoint_id=2, new_sim_id=9)
        record["turns"][0]["tool_calls"].extend(early.calls)
        record["turns"][0]["pending_actions"] = copy.deepcopy(early.pending_actions)
        record["turns"][1]["tool_calls"][-1] = copy.deepcopy(later.calls[0])
        record["turns"][1]["pending_actions"] = copy.deepcopy(
            early.pending_actions + later.pending_actions
        )
        record["pending_actions"] = copy.deepcopy(record["turns"][1]["pending_actions"])
        result = evaluate_local(scenario, record)
        by_name = {check["name"]: check for check in result["checks"]}
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(by_name["update_waits_for_approval"]["passed"])
        self.assertTrue(by_name["no_executed_writes"]["passed"])
        for name in (
            "first_turn_has_no_mutation_proposal",
            "one_update_in_second_turn",
            "combined_update_arguments",
            "resolve_iccid_before_update",
            "one_matching_pending_action",
        ):
            self.assertFalse(by_name[name]["passed"], name)
        count_detail = by_name["one_update_in_second_turn"]["detail"]
        self.assertIn("whole session", count_detail)
        self.assertIn("2 total", count_detail)
        self.assertIn("turn 1: 1", count_detail)
        self.assertIn("turn 2: 1", count_detail)

    def test_duplicate_combined_proposals_do_not_fail_argument_or_order_checks(self):
        scenario = self.scenarios[1]
        record = run_scenario(scenario)
        duplicate = copy.deepcopy(record["turns"][1]["tool_calls"][-1])
        record["turns"][1]["tool_calls"].append(duplicate)
        record["turns"][1]["pending_actions"].append(copy.deepcopy(duplicate["result"]))
        record["pending_actions"] = copy.deepcopy(record["turns"][1]["pending_actions"])
        result = evaluate_local(scenario, record)
        by_name = {check["name"]: check for check in result["checks"]}
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(by_name["one_update_in_second_turn"]["passed"])
        self.assertFalse(by_name["one_matching_pending_action"]["passed"])
        for name in (
            "combined_update_arguments",
            "resolve_iccid_before_update",
            "update_waits_for_approval",
            "no_executed_writes",
        ):
            self.assertTrue(by_name[name]["passed"], name)

    def test_reset_trap_is_a_failure_even_if_not_executed(self):
        scenario = self.scenarios[0]
        record = run_scenario(scenario)
        tools = FixtureTools()
        tools.reset_device_connectivity(endpoint_id=2)
        record["turns"][0]["tool_calls"].extend(tools.calls)
        result = evaluate_local(scenario, record)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(
            next(c for c in result["checks"] if c["name"] == "no_connectivity_reset")[
                "passed"
            ]
        )

    def test_local_pass_does_not_claim_to_grade_response_meaning(self):
        scenario = self.scenarios[0]
        record = run_scenario(scenario)
        record["turns"][0]["response"] = "Everything is fixed."
        result = evaluate_local(scenario, record)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["semantic_evaluation"], "NOT_RUN")
        self.assertTrue(result["warnings"])

    def test_malformed_tool_records_fail_without_crashing_the_report(self):
        scenario = self.scenarios[0]
        record = run_scenario(scenario)
        record["turns"][0]["tool_calls"].append(
            {"name": ["update_device"], "arguments": [], "result": None}
        )
        self.assertEqual(evaluate_local(scenario, record)["status"], "FAIL")
        scenario = self.scenarios[1]
        record = run_scenario(scenario)
        record["turns"][1]["tool_calls"][0]["arguments"]["identifier"] = {}
        self.assertEqual(evaluate_local(scenario, record)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
