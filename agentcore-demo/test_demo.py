"""Integration checks for orchestration, without network calls."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import demo
from domain import ScriptedAgent, load_dataset


class OrchestrationTests(unittest.TestCase):
    def test_cloud_scoring_error_preserves_completed_local_results(self):
        scenario = load_dataset()["scenarios"][0]
        agent = ScriptedAgent()
        session = {
            "agent": agent,
            "record": {
                "session_id": "test-session",
                "scenario_id": scenario["scenario_id"],
                "variant": "baseline",
                "mode": "bedrock",
                "profile": None,
                "region": "eu-west-1",
                "turns": [],
                "spans": [],
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(demo, "RESULTS", Path(folder)), \
                 patch.object(demo, "new_session", return_value=session), \
                 patch("aws_agent.score_agentcore", side_effect=RuntimeError("Scorer unavailable")):
                result = demo.run_scenario(scenario, {}, aws_score=True)
            self.assertEqual(result["local_evaluation"]["status"], "PASS")
            self.assertGreater(len(result["local_evaluation"]["checks"]), 0)
            self.assertEqual(result["agentcore_evaluation"]["status"], "ERROR")
            self.assertIn("Scorer unavailable", result["agentcore_evaluation"]["error"])

    def test_browser_defaults_cannot_silently_enable_aws(self):
        self.assertEqual(demo.validate_settings({})["mode"], "rehearsal")
        with self.assertRaisesRegex(ValueError, "model"):
            demo.validate_settings({"mode": "bedrock", "model_id": ""})
        with self.assertRaisesRegex(ValueError, "Bedrock mode"):
            demo.start_job({}, aws_score=True)

    def test_shared_nonsecret_defaults_and_environment_overrides(self):
        with patch.dict(demo.os.environ, {}, clear=True):
            defaults = demo.connection_defaults()
            self.assertEqual(defaults, demo.LOCAL_DEFAULTS)
            self.assertEqual(demo.config()["defaults"], defaults)
            configured = demo.validate_settings({"mode": "bedrock"})
            self.assertEqual(configured["model_id"], defaults["model_id"])
            self.assertEqual(configured["region"], defaults["region"])
            self.assertEqual(configured["profile"], defaults["profile"])
        with patch.dict(demo.os.environ, {
            "AWS_PROFILE": "other", "AWS_REGION": "us-east-1",
            "BEDROCK_MODEL_ID": "test-model",
        }, clear=True):
            defaults = demo.connection_defaults()
            self.assertEqual(defaults["profile"], "other")
            self.assertEqual(defaults["region"], "us-east-1")
            self.assertEqual(defaults["model_id"], "test-model")
            self.assertEqual(defaults["account_id"], "")

    def test_scenarios_create_fresh_fixture_worlds(self):
        scenario = load_dataset()["scenarios"][1]
        golden_before = copy.deepcopy(scenario)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(demo, "RESULTS", Path(folder)):
                one = demo.run_scenario(scenario, {})
                two = demo.run_scenario(scenario, {})
        self.assertNotEqual(one["session_id"], two["session_id"])
        self.assertEqual(one["local_evaluation"]["status"], "PASS")
        self.assertEqual(two["local_evaluation"]["status"], "PASS")
        self.assertEqual(len(two["pending_actions"]), 1)
        self.assertEqual(scenario, golden_before)


class SavedResultsTests(unittest.TestCase):
    def record(self):
        return {
            "session_id": "saved-one",
            "scenario_id": load_dataset()["scenarios"][0]["scenario_id"],
            "mode": "bedrock",
            "variant": "baseline",
            "model_id": "recorded-model",
            "region": "recorded-region",
            "profile": "recorded-profile",
            "dataset_sha256": demo.file_hash(demo.DATASET_PATH),
            "fixture_sha256": demo.file_hash(demo.FIXTURE_PATH),
            "turns": [{"input": "original input", "response": "original response",
                       "tool_calls": []}],
            "local_evaluation": {"status": "PASS"},
            "agentcore_evaluation": {"status": "PASS", "results": []},
        }

    def test_saved_aws_records_are_read_only_and_never_invoke_an_agent(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baseline-saved-one.json"
            record = self.record()
            record["result_file"] = "/untrusted/outside/path"
            record["credentials"] = {"should_not_be_returned": True}
            path.write_text(json.dumps(record))
            with patch.object(demo, "RESULTS", Path(folder)), \
                 patch.object(demo, "new_session", side_effect=AssertionError("no invocation")):
                found = demo.saved_results()["results"]
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0]["read_only"])
        self.assertEqual(found[0]["result_file"], str(path))
        self.assertNotIn("credentials", found[0])
        self.assertEqual(found[0]["model_id"], "recorded-model")
        self.assertEqual(found[0]["saved_status"], "PASS")
        self.assertTrue(found[0]["matches_current_dataset"])
        self.assertTrue(found[0]["matches_current_fixture"])

    def test_saved_errors_incomplete_turns_and_unscored_runs_stay_distinct(self):
        states = []
        for change in (
            {"agentcore_evaluation": {"status": "ERROR", "error": "service error"}},
            {"turns": []},
            {"turns": [{"input": "unfinished"}]},
            {"agentcore_evaluation": None},
            {"local_evaluation": {"status": "FAIL"}},
        ):
            with tempfile.TemporaryDirectory() as folder:
                record = self.record()
                record.update(change)
                (Path(folder) / "baseline-state.json").write_text(json.dumps(record))
                with patch.object(demo, "RESULTS", Path(folder)):
                    states.append(demo.saved_results()["results"][0]["saved_status"])
        self.assertEqual(states, ["ERROR", "INCOMPLETE", "INCOMPLETE", "NOT SCORED", "FAIL"])

    def test_reader_ignores_rehearsal_bad_filenames_and_symlinks(self):
        with tempfile.TemporaryDirectory() as folder, \
             tempfile.TemporaryDirectory() as outside:
            root = Path(folder)
            external = Path(outside) / "external.json"
            external.write_text(json.dumps(self.record()))
            (root / "baseline-link.json").symlink_to(external)
            (root / "credentials.json").write_text(json.dumps(self.record()))
            rehearsal = self.record()
            rehearsal["mode"] = "rehearsal"
            (root / "baseline-local.json").write_text(json.dumps(rehearsal))
            (root / "baseline-broken.json").write_text("{")
            with patch.object(demo, "RESULTS", root):
                found = demo.saved_results()
        self.assertEqual(found["results"], [])
        self.assertEqual(len(found["warnings"]), 1)
        self.assertIn("baseline-broken.json", found["warnings"][0])

    def test_saved_source_changes_are_visible_without_regrading_history(self):
        with tempfile.TemporaryDirectory() as folder:
            record = self.record()
            record["dataset_sha256"] = "old-dataset"
            record["fixture_sha256"] = "old-fixture"
            (Path(folder) / "baseline-history.json").write_text(json.dumps(record))
            with patch.object(demo, "RESULTS", Path(folder)):
                found = demo.saved_results()["results"][0]
        self.assertEqual(found["saved_status"], "PASS")
        self.assertFalse(found["matches_current_dataset"])
        self.assertFalse(found["matches_current_fixture"])


if __name__ == "__main__":
    unittest.main()
