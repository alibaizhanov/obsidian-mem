#!/usr/bin/env bash
# SECURITY MANIFEST:
# Environment variables accessed: MENGRAM_API_KEY (only)
# External endpoints called: https://mengram.io/v1/search/all (only)
# Local files read: none
# Local files written: none
set -euo pipefail

MENGRAM_BASE_URL="${MENGRAM_BASE_URL:-https://mengram.io}"
API_KEY="${MENGRAM_API_KEY:-}"

if [ -z "$API_KEY" ]; then
  echo "ERROR: MENGRAM_API_KEY not set. Get your free key at https://mengram.io"
  exit 1
fi

QUERY="${1:-}"
if [ -z "$QUERY" ]; then
  echo "Usage: mengram-search.sh \"your search query\""
  exit 1
fi

# Sanitize query for JSON
SAFE_QUERY=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$QUERY")

RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X POST "${MENGRAM_BASE_URL}/v1/search/all" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"query\": ${SAFE_QUERY}, \"limit\": 5}")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" -ne 200 ]; then
  echo "ERROR: Mengram API returned HTTP ${HTTP_CODE}"
  echo "$BODY"
  exit 1
fi

# Format output for the agent
python3 -c "
import json, sys

try:
    data = json.loads(sys.argv[1])
except json.JSONDecodeError:
    print('No results found.')
    sys.exit(0)

parts = []

# Semantic facts
semantic = data.get('semantic', [])
if semantic:
    facts = []
    for r in semantic[:5]:
        entity = r.get('entity', '')
        for f in r.get('facts', [])[:5]:
            facts.append(f'{entity}: {f}')
    if facts:
        parts.append('KNOWN FACTS:\n' + '\n'.join(f'- {f}' for f in facts))

# Episodic events
episodic = data.get('episodic', [])
if episodic:
    events = []
    for ep in episodic[:5]:
        line = ep.get('summary', '')
        if ep.get('outcome'):
            line += f' -> Outcome: {ep[\"outcome\"]}'
        if ep.get('when'):
            line += f' ({ep[\"when\"]})'
        events.append(line)
    if events:
        parts.append('PAST EVENTS:\n' + '\n'.join(f'- {e}' for e in events))

# Procedural workflows
procedural = data.get('procedural', [])
if procedural:
    procs = []
    for pr in procedural[:5]:
        name = pr.get('name', '')
        steps = pr.get('steps', [])
        steps_str = ' -> '.join(s.get('action', '') for s in steps[:10])
        success = pr.get('success_count', 0)
        fail = pr.get('fail_count', 0)
        procs.append(f'{name}: {steps_str} (success: {success}, fail: {fail})')
    if procs:
        parts.append('KNOWN WORKFLOWS:\n' + '\n'.join(f'- {p}' for p in procs))

if parts:
    print('\n\n'.join(parts))
else:
    print('No relevant memories found for this query.')
" "$BODY"
