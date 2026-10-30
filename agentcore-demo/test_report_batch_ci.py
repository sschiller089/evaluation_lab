"""A green comparison workflow must have complete AWS evaluation evidence."""

import copy
import unittest

from report_batch_ci import summarize
from test_run_batch_demo import completed_job


def manifest():
    job = completed_job()
    job["response"]["outputConfig"] = {"cloudWatchConfig": {
        "logGroupName": "/aws/bedrock-agentcore/evaluations/batch-evaluations/results/default",
        "logStreamName": "run-test-job",
    }}
    return {
        "status": "FINISHED", "settings": {"region": "eu-central-1"},
        "jobs": {"baseline": job, "regression": copy.deepcopy(job)},
    }


class BatchReportTests(unittest.TestCase):
    def test_completed_grading_can_report_low_scores_without_claiming_gate_pass(self):
        text = summarize(manifest())
        self.assertIn("| regression | Builtin.GoalSuccessRate | 0.00 | 2 |", text)
        self.assertIn("intentionally faulty", text)
        self.assertIn("$252Faws$252Fbedrock-agentcore", text)

    def test_pending_jobs_cannot_create_a_successful_report(self):
        value = manifest()
        value["status"] = "SUBMITTED"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            summarize(value)

    def test_missing_score_events_cannot_create_a_successful_report(self):
        value = manifest()
        value["jobs"]["baseline"]["result_events"].pop()
        with self.assertRaisesRegex(ValueError, "baseline"):
            summarize(value)
