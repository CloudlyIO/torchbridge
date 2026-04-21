#!/usr/bin/env bash
# pytest_capture.sh — Run pytest and save full output to a file.
#
# Background Bash tasks in Claude Code truncate stdout at ~30KB, so long
# test runs lose the summary line. This script:
#   1. Runs pytest (forwarding all args), OR runs in batched mode (--batched)
#   2. Tees full output to $TB_TEST_OUTPUT (default: /tmp/tb_pytest.txt)
#   3. Prints the summary line after completion
#   4. Exits with pytest's real exit code
#
# Usage:
#   scripts/ci/pytest_capture.sh [pytest-args...]          # passthrough mode
#   scripts/ci/pytest_capture.sh --batched                 # 4-batch full suite
#   scripts/ci/pytest_capture.sh --batched --ignore=tests/stress
#
# After the task completes, read the summary with:
#   tail -6 /tmp/tb_pytest.txt

OUTPUT_FILE="${TB_TEST_OUTPUT:-/tmp/tb_pytest.txt}"

# Use the correct Python interpreter (python@3.14 on this Mac lacks pytest/torch)
PYTHON="${TORCHBRIDGE_PYTHON:-/Library/Frameworks/Python.framework/Versions/3.11/bin/python3}"

# ─── Batched mode ─────────────────────────────────────────────────────────────
if [[ "$1" == "--batched" ]]; then
    shift  # consume --batched; remaining args passed to every pytest invocation

    # 4 batches sized to stay under macOS memory pressure threshold (~1k tests):
    #   Batch 1: backends + cli                (~250 tests)
    #   Batch 2: e2e + integration             (~300 tests)
    #   Batch 3: models + robustness + security (~150 tests)
    #   Batch 4: unit                          (~1,400 tests)
    # NOTE: tests/benchmark, tests/features, tests/regression, tests/distributed
    #       were deleted in v0.5.94 structural cleanup; their tests live in tests/unit/
    BATCHES=(
        "tests/backends tests/cli"
        "tests/e2e tests/integration"
        "tests/models tests/robustness tests/security"
        "tests/unit"
    )

    > "$OUTPUT_FILE"  # truncate/create output file

    OVERALL_EXIT=0
    TOTAL_PASSED=0
    TOTAL_FAILED=0
    TOTAL_SKIPPED=0
    TOTAL_ERRORS=0

    for i in "${!BATCHES[@]}"; do
        BATCH_NUM=$((i + 1))
        DIRS="${BATCHES[$i]}"
        echo "" | tee -a "$OUTPUT_FILE"
        echo "════════════════════════════════════════════════════════════" | tee -a "$OUTPUT_FILE"
        echo "  Batch $BATCH_NUM / ${#BATCHES[@]}: $DIRS" | tee -a "$OUTPUT_FILE"
        echo "════════════════════════════════════════════════════════════" | tee -a "$OUTPUT_FILE"

        # Run this batch, appending to the output file
        "$PYTHON" -m pytest $DIRS "$@" -q --tb=short 2>&1 | tee -a "$OUTPUT_FILE"
        BATCH_EXIT="${PIPESTATUS[0]}"

        if [[ $BATCH_EXIT -ne 0 ]]; then
            OVERALL_EXIT=$BATCH_EXIT
        fi

        # Parse this batch's summary line for totals
        SUMMARY=$(grep -E 'passed|failed|error' "$OUTPUT_FILE" | tail -1)
        PASSED=$(echo "$SUMMARY" | grep -oE '[0-9]+ passed'  | grep -oE '[0-9]+' || echo 0)
        FAILED=$(echo "$SUMMARY" | grep -oE '[0-9]+ failed'  | grep -oE '[0-9]+' || echo 0)
        SKIPPED=$(echo "$SUMMARY" | grep -oE '[0-9]+ skipped' | grep -oE '[0-9]+' || echo 0)
        ERRORS=$(echo "$SUMMARY"  | grep -oE '[0-9]+ error'   | grep -oE '[0-9]+' || echo 0)
        TOTAL_PASSED=$((TOTAL_PASSED + PASSED))
        TOTAL_FAILED=$((TOTAL_FAILED + FAILED))
        TOTAL_SKIPPED=$((TOTAL_SKIPPED + SKIPPED))
        TOTAL_ERRORS=$((TOTAL_ERRORS + ERRORS))
    done

    echo "" | tee -a "$OUTPUT_FILE"
    echo "════════════════════════════════════════════════════════════" | tee -a "$OUTPUT_FILE"
    echo "  TOTAL: $TOTAL_PASSED passed, $TOTAL_FAILED failed, $TOTAL_SKIPPED skipped, $TOTAL_ERRORS errors" | tee -a "$OUTPUT_FILE"
    echo "  Output: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
    echo "════════════════════════════════════════════════════════════" | tee -a "$OUTPUT_FILE"

    exit $OVERALL_EXIT
fi

# ─── Passthrough mode ─────────────────────────────────────────────────────────
"$PYTHON" -m pytest "$@" 2>&1 | tee "$OUTPUT_FILE"
EXIT_CODE="${PIPESTATUS[0]}"

echo ""
echo "--- Full output: $OUTPUT_FILE ---"
echo "--- Summary: $(grep -E 'passed|failed|error' "$OUTPUT_FILE" | tail -1) ---"

exit "$EXIT_CODE"
