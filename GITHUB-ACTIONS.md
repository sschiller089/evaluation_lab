# GitHub Actions setup

The repository is `sschiller089/evaluation_lab`. AWS jobs run from
`main` in `eu-central-1`. Pull requests run offline tests without AWS access.

## Configure AWS access

Deploy [deploy/github-actions-iam.yaml](deploy/github-actions-iam.yaml) into the
intended AWS account. It creates an evaluation role authenticated through
GitHub OpenID Connect; no AWS access keys belong in repository files or secrets.

The required `GitHubOidcSubject` parameter is the exact `sub` claim for this
repository's `main` branch. GitHub repositories created after July 15, 2026 use
immutable owner and repository IDs:

```text
repo:sschiller089@246036503/evaluation_lab@1439530816:ref:refs/heads/main
```

These are this repository's owner and repository IDs. Older repositories can use
`repo:OWNER/REPOSITORY:ref:refs/heads/main`. The template accepts an exact subject
for `main`; it rejects wildcard, pull-request, and environment subjects. See
[GitHub's AWS OIDC guidance](https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-aws).

Set `ExistingGitHubOidcProviderArn` to the existing GitHub provider ARN if the
account already has one. Leave it empty to create the provider. Set
`EnableBatchEvaluation=true` to enable the manual batch comparison and create
its dedicated content log group and stream.

```bash
export GITHUB_OIDC_SUBJECT='repo:sschiller089@246036503/evaluation_lab@1439530816:ref:refs/heads/main'
aws cloudformation deploy \
  --region eu-central-1 \
  --stack-name agentcore-evaluation-lab-github \
  --template-file deploy/github-actions-iam.yaml \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    GitHubOidcSubject="$GITHUB_OIDC_SUBJECT" \
    EnableBatchEvaluation=true
```

Verify the CLI is using the intended account before deploying. When reusing an
existing provider, add its parameter to the command.

In **Settings → Secrets and variables → Actions → Variables**, set:

| Repository variable | Value |
|---|---|
| `AWS_ROLE_ARN` | The stack's `GitHubActionsRoleArn` output |
| `AWS_ACCOUNT_ID` | The intended 12-digit account ID |

The role permits this Haiku inference profile and its current six EU destination
models, the three selected evaluators, and optional batch telemetry/results
access. The policies grant no permission to deploy or modify a real device or
AgentCore Runtime.

Batch execution requires CloudWatch Transaction Search to be enabled already.
The stack creates
`/aws/bedrock-agentcore/runtimes/agentcore-evaluation-lab` with `runtime-logs` and
a 14-day retention period. It preserves these resources on stack deletion so
recent evidence is not deleted during teardown. It also preserves a newly
created GitHub OIDC provider, which can become shared with other roles.

## Run the quality gate

Open **Actions → Agent quality gate → Run workflow** on `main`.

- Select **baseline** to grade the current baseline prompt and code.
- Select **regression** to demonstrate an intentionally failed quality gate.

The workflow also runs automatically after a push to `main`. It evaluates the
checked-out revision against both versioned golden scenarios. Every local hard
check must pass, and every individual AgentCore score must be at least `0.8`.
The workflow fails on quality failures or incomplete evidence; the regression
variant does not invert failure into success.

Read the scenario score table in the run summary. Download the run artifact for
complete conversations, tool arguments, scores, explanations, source hashes,
trace IDs, and recorded token usage. Artifacts expire after 14 days.

The runner returns:

| Exit code | Meaning |
|---:|---|
| `0` | All required evidence is complete and every check passes |
| `1` | At least one quality requirement fails |
| `2` | Invocation error, missing evidence, or invalid evaluation result |

Pull-request checks are offline tests. This repository has no production
deployment job or automatic pre-merge AWS evaluation. To gate a later deployment,
call the workflow as a reusable job and make deployment depend on its success:

```yaml
permissions:
  contents: read
  id-token: write

jobs:
  quality:
    uses: ./.github/workflows/agent-evaluation.yml
    with:
      variant: baseline
  deploy:
    needs: quality
    runs-on: ubuntu-24.04
    steps:
      - run: echo "Replace this with the actual deployment command."
```

Call it from `main`; the AWS trust policy only permits that branch.

## View batch results in AWS

Open **Actions → Agent batch comparison → Run workflow** on `main`.
The workflow generates four fresh sessions, waits for complete telemetry, and
starts two jobs named:

```text
evaluation_lab_baseline_<timestamp>
evaluation_lab_regression_<timestamp>
```

In AWS, choose Frankfurt and open **Amazon Bedrock AgentCore → Batch evaluations**.
The GitHub summary also links directly to each job's detailed CloudWatch results.
The default result group is
`/aws/bedrock-agentcore/evaluations/batch-evaluations/results/default`.

**COMPLETED means grading finished.** The batch comparison is successful when
both jobs finish and every expected result is available, even when the
intentionally faulty variant receives low scores. Use **Agent quality gate**
for a pass/fail release decision.

The artifact contains a resumable manifest and each session's records. The
batch runner's `submit` and `poll` commands can resume from that evidence without
invoking the model again. Batch execution is manual; no recurring schedule is
enabled.
