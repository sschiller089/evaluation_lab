# AgentCore evaluation lab

A small connectivity-support agent demonstrates how to detect an agent regression
with Amazon Bedrock AgentCore Evaluations and GitHub Actions.

The repository is `sschiller089/evaluation_lab`. The agent uses Claude Haiku 4.5
through its EU inference profile. All device tools return fixed fixture data and
record approval proposals; they never change a device.

The fixture values were generated for this lab. Device `Demo-Device-01` and the
all-zero ICCID, IMSI, IMEI, and MSISDN strings are deliberately fictional and
invalid for live use. Small numeric IDs such as endpoint `2` and SIM `9` are
internal mock keys with no external meaning. The fixture tools never make
network calls; only the model and evaluator integrations call AWS.

## What the experiment checks

Demo-Device-01 is disabled and has no SIM attached. Two golden scenarios require the
agent to:

1. Identify the missing SIM, ask which SIM to attach, and stop without proposing
   an enable or connectivity reset.
2. Once the user supplies the SIM identifier, resolve it and propose attaching
   the SIM and enabling the device in one update.

The **baseline** follows those instructions. The intentionally faulty
**regression** prioritizes Disabled status and proposes enabling the device
before checking for a SIM. Both variants face the same questions and assertions.

The gate combines exact tool checks with three AgentCore evaluators:

| Evaluator | Evidence |
|---|---|
| `Builtin.Correctness` | Each turn's response against its expected answer |
| `Builtin.GoalSuccessRate` | The session against its business assertions |
| `Builtin.TrajectoryInOrderMatch` | Expected tool names in sequence |

Every deterministic check and every individual score must pass. The score
threshold is **0.8**, chosen for this demonstration. Missing scores, mismatched
traces, incomplete coverage, and invocation errors reject the run. Tool order
alone can pass even when a premature update violates the business rule.

## GitHub Actions

- **Agent quality gate:** runs offline contract tests on pull requests and pushes
  to `main`; runs real on-demand evaluations after those tests on `main`.
  A manual run can select `regression` to demonstrate a failed gate.
- **Agent batch comparison:** starts fresh baseline and regression sessions,
  exports telemetry, submits two batch jobs, and adds AWS console links to the
  GitHub run summary. This workflow runs only when manually dispatched.

Both workflows retain recorded evidence as GitHub artifacts for 14 days.
See [GITHUB-ACTIONS.md](GITHUB-ACTIONS.md) for AWS access and usage.

## Run locally

```bash
python3 -m venv agentcore-demo/.venv
agentcore-demo/.venv/bin/python -m pip install -r agentcore-demo/requirements.lock.txt
cd agentcore-demo
.venv/bin/python -m unittest discover -v
.venv/bin/python demo.py run --variant baseline
.venv/bin/python demo.py run --variant regression
```

The last two commands use a scripted rehearsal and make no AWS calls. Baseline
should exit `0`; regression should exit `1`.

To run real on-demand evaluation with your configured AWS credentials:

```bash
export AWS_REGION=eu-central-1
export AWS_ACCOUNT_ID=123456789012  # Replace with your expected account.
.venv/bin/python run_ci_evaluation.py run \
  --variant baseline \
  --expected-account-id "$AWS_ACCOUNT_ID" \
  --output results/ci-evaluation.json
```

This invokes Bedrock and AgentCore and incurs AWS usage. Credentials come from
the normal AWS provider chain; `--profile` is optional for a local named profile.
Results include conversations, tool calls, scores, explanations, trace IDs,
source hashes, duration, and available token counts.

## Scope

This is a two-scenario adaptation of the CI quality-gate and comparison patterns
in the AWS post
[Building trust in AI agents: A production-ready evaluation framework with Amazon Bedrock AgentCore](https://aws.amazon.com/blogs/machine-learning/building-trust-in-ai-agents-a-production-ready-evaluation-framework-with-amazon-bedrock-agentcore/).
It does not implement the post's entire nine-stage production system.

The agent runs in the GitHub runner or your local Python process. This repository
does not deploy an AgentCore Runtime or a production agent. The two cases are the
complete implemented set, not evidence of broad agent reliability. There is no
production-baseline comparison, latency or monetary-cost gate, canary schedule,
or live-traffic monitor.
