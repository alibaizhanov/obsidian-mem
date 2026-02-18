#!/usr/bin/env bash
# SECURITY MANIFEST:
# Environment variables accessed: MENGRAM_API_KEY (only)
# External endpoints called: https://mengram.io/v1/add (only)
# Local files read: none
# Local files written: none
set -euo pipefail

MENGRAM_BASE_URL="${MENGRAM_BASE_URL:-https://mengram.io}"
API_KEY="${MENGRAM_API_KEY:-}"

if [ -z "$API_KEY" ]; then
  echo "ERROR: MENGRAM_API_KEY not set. Get your free key at https://mengram.io"
  exit 1
fi

WORKFLOW_NAME="${1:-}"
STEPS="${2:-}"
OUTCOME="${3:-success}"

if [ -z "$WORKFLOW_NAME" ] || [ -z "$STEPS" ]; then
  echo "Usage: mengram-workflow.sh \"workflow name\" \"step1; step2; step3\" [success|failure]"
  exit 1
fi

# Build workflow description as assistant message (triggers procedural extraction)
SAFE_DESC=$(python3 -c "
import json, sys
name = sys.argv[1]
steps = sys.argv[2]
outcome = sys.argv[3]
desc = f'Completed workflow: {name}. Steps taken: {steps}. Result: {outcome}.'
print(json.dumps(desc))
" "$WORKFLOW_NAME" "$STEPS" "$OUTCOME")

RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X POST "${MENGRAM_BASE_URL}/v1/add" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"messages\": [{\"role\": \"assistant\", \"content\": ${SAFE_DESC}}]}")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" -ne 200 ] && [ "$HTTP_CODE" -ne 202 ]; then
  echo "ERROR: Mengram API returned HTTP ${HTTP_CODE}"
  echo "$BODY"
  exit 1
fi

echo "Workflow '${WORKFLOW_NAME}' saved. Mengram will learn this procedure for future use."
