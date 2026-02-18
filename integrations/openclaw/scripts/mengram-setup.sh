#!/usr/bin/env bash
# SECURITY MANIFEST:
# Environment variables accessed: MENGRAM_API_KEY (only)
# External endpoints called: https://mengram.io/v1/health, https://mengram.io/v1/search/all (only)
# Local files read: none
# Local files written: none
set -euo pipefail

MENGRAM_BASE_URL="${MENGRAM_BASE_URL:-https://mengram.io}"
API_KEY="${MENGRAM_API_KEY:-}"

echo "=== Mengram Setup Check ==="
echo ""

# Check 1: API key set
if [ -z "$API_KEY" ]; then
  echo "FAIL: MENGRAM_API_KEY is not set"
  echo "Get your free API key at https://mengram.io"
  exit 1
fi
echo "OK: MENGRAM_API_KEY is set (${API_KEY:0:10}...)"
echo "OK: MENGRAM_BASE_URL = ${MENGRAM_BASE_URL}"

# Check 2: curl available
if ! command -v curl &>/dev/null; then
  echo "FAIL: curl is not installed"
  exit 1
fi
echo "OK: curl available"

# Check 3: python3 available
if ! command -v python3 &>/dev/null; then
  echo "FAIL: python3 is not installed"
  exit 1
fi
echo "OK: python3 available"

# Check 4: API reachable
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X GET "${MENGRAM_BASE_URL}/v1/health" \
  --connect-timeout 5 2>/dev/null || true)

HEALTH_CODE=$(echo "$HEALTH_RESPONSE" | tail -1)

if [ "$HEALTH_CODE" = "200" ]; then
  echo "OK: Mengram API is reachable"
else
  echo "FAIL: Cannot reach ${MENGRAM_BASE_URL} (network error)"
  exit 1
fi

# Check 5: API key valid
AUTH_RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X POST "${MENGRAM_BASE_URL}/v1/search/all" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "limit": 1}' \
  --connect-timeout 5 2>/dev/null || true)

AUTH_CODE=$(echo "$AUTH_RESPONSE" | tail -1)

if [ "$AUTH_CODE" = "200" ]; then
  echo "OK: API key is valid"
else
  echo "FAIL: API key is invalid or expired. Get a new key at https://mengram.io"
  exit 1
fi

echo ""
echo "=== All checks passed! Mengram is ready. ==="
