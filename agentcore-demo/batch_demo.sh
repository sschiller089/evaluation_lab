#!/usr/bin/env bash
# Run the local fixture agent with AWS tracing, then submit/resume batch jobs.
# Usage:
#   bash batch_demo.sh run --manifest results/batch-20261030.json
#   bash batch_demo.sh submit --manifest results/batch-20261030.json
#   bash batch_demo.sh poll --manifest results/batch-20261030.json --wait-seconds 600
set -euo pipefail

DEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"
DEMO_PYTHON="${DEMO_ROOT}/.venv/bin/python"
if [[ ! -x "${DEMO_PYTHON}" ]]; then
    echo "Install the demo dependencies into agentcore-demo/.venv first." >&2
    exit 1
fi

# Preserve an explicitly selected profile; otherwise use the AWS provider chain.
# GitHub Actions supplies short-lived credentials without a local named profile.
export AWS_REGION="${AWS_REGION:-eu-central-1}"
export AWS_DEFAULT_REGION="${AWS_REGION}"
export AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID to the expected AWS account.}"
export PYTHONUNBUFFERED=1

# Only agent execution needs instrumentation. Submission and polling do not
# emit extra agent traces or overwrite the recorded source configuration.
if [[ "${1:-}" == "run" ]]; then
    export AGENT_OBSERVABILITY_ENABLED=true
    export OTEL_PYTHON_DISTRO=aws_distro
    export OTEL_PYTHON_CONFIGURATOR=aws_configurator
    export OTEL_SERVICE_NAME=agentcore-evaluation-lab
    export OTEL_RESOURCE_ATTRIBUTES="service.name=agentcore-evaluation-lab,aws.log.group.names=/aws/bedrock-agentcore/runtimes/agentcore-evaluation-lab"
    export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
    export OTEL_TRACES_EXPORTER=otlp
    export OTEL_EXPORTER_OTLP_LOGS_HEADERS="x-aws-log-group=/aws/bedrock-agentcore/runtimes/agentcore-evaluation-lab,x-aws-log-stream=runtime-logs,x-aws-metric-namespace=bedrock-agentcore"
    # Reuse the account's existing Transaction Search destination and policy.
    # Spans go to aws/spans; conversation content goes to the demo log group.
    export AWS_GENAI_CONTENT_EXTRACTION_OPT_OUT=false
    unset OTEL_EXPORTER_OTLP_TRACES_HEADERS
    unset OTEL_SEMCONV_STABILITY_OPT_IN
    export OTEL_PYTHON_DISABLED_INSTRUMENTATIONS="botocore,urllib3,requests,httpx"
    exec "${DEMO_ROOT}/.venv/bin/opentelemetry-instrument" \
        "${DEMO_PYTHON}" "${DEMO_ROOT}/run_batch_demo.py" "$@"
fi

exec "${DEMO_PYTHON}" "${DEMO_ROOT}/run_batch_demo.py" "$@"
