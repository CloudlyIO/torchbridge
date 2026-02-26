#!/usr/bin/env bash
# Run TorchBridge claim benchmarks and save JSON results.
#
# Usage:
#   ./scripts/benchmarks/run_claim_benchmarks.sh                    # stdout only
#   ./scripts/benchmarks/run_claim_benchmarks.sh --output out.json  # save results
#   ./scripts/benchmarks/run_claim_benchmarks.sh --ci               # JSON-only CI output
#
# Device is auto-detected: CUDA if available, MPS on Apple Silicon, CPU otherwise.
# On CUDA hardware, all 5 claims run. On CPU, tensor_core_alignment,
# channels_last_layout, and amd_tunableop are skipped automatically.
#
# Intended for cloud GPU instances where GPU-dependent claims run live.

set -euo pipefail

OUTPUT=""
CI_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --output) OUTPUT="$2"; shift 2 ;;
        --ci) CI_FLAG="--ci"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=== TorchBridge Claim Benchmarks ==="
echo ""

ARGS="--type claims"
if [[ -n "$OUTPUT" ]]; then
    ARGS="$ARGS --output $OUTPUT"
fi
if [[ -n "$CI_FLAG" ]]; then
    ARGS="$ARGS $CI_FLAG"
fi

# shellcheck disable=SC2086
tb-benchmark $ARGS
