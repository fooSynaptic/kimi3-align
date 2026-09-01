#!/usr/bin/env bash
# Tier-1: verify report claims C0–C7 against checked-in artifacts.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$SCRIPT_DIR/verify_reported_results.py" "$@"
