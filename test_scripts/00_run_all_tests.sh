#!/bin/bash
# Master Test Script - Runs all tests in sequence

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   GraDeT-HTR Context-Aware Extension - Full Test Suite    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Test results tracking
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

# Function to run a test
run_test() {
    local test_name=$1
    local test_script=$2

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: $test_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    TESTS_RUN=$((TESTS_RUN + 1))

    if [ -f "$test_script" ]; then
        chmod +x "$test_script"

        if bash "$test_script"; then
            echo "✅ $test_name PASSED"
            TESTS_PASSED=$((TESTS_PASSED + 1))
            return 0
        else
            echo "❌ $test_name FAILED"
            TESTS_FAILED=$((TESTS_FAILED + 1))
            FAILED_TESTS+=("$test_name")
            return 1
        fi
    else
        echo "⚠️  Test script not found: $test_script"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        FAILED_TESTS+=("$test_name (not found)")
        return 1
    fi
}

run_python_test() {
    local test_name=$1
    local test_script=$2

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: $test_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    TESTS_RUN=$((TESTS_RUN + 1))

    if [ -f "$test_script" ]; then
        chmod +x "$test_script"

        if python3 "$test_script"; then
            echo "✅ $test_name PASSED"
            TESTS_PASSED=$((TESTS_PASSED + 1))
            return 0
        else
            echo "❌ $test_name FAILED"
            TESTS_FAILED=$((TESTS_FAILED + 1))
            FAILED_TESTS+=("$test_name")
            return 1
        fi
    else
        echo "⚠️  Test script not found: $test_script"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        FAILED_TESTS+=("$test_name (not found)")
        return 1
    fi
}

# Parse arguments
RUN_QUICK_ONLY=false
RUN_FULL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            RUN_QUICK_ONLY=true
            shift
            ;;
        --full)
            RUN_FULL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick] [--full]"
            echo "  --quick : Run only quick tests (no training)"
            echo "  --full  : Run all tests including full training"
            exit 1
            ;;
    esac
done

# ===== UNIT TESTS =====
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                     UNIT TESTS                             ║"
echo "╚════════════════════════════════════════════════════════════╝"

run_python_test "Test 1: Dataset Loading" "01_test_dataset_loading.py"
run_python_test "Test 2: Model Forward Pass" "02_test_model_forward.py"

# ===== INTEGRATION TESTS (Quick) =====
if [ "$RUN_QUICK_ONLY" = false ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                QUICK INTEGRATION TESTS                     ║"
    echo "╚════════════════════════════════════════════════════════════╝"

    run_test "Test 3: Quick Standard Training" "03_quick_train_standard.sh"
    run_test "Test 4: Quick Context Training" "04_quick_train_context.sh"
    run_python_test "Test 9: Context Benefit Validation" "09_validate_context_benefit.py"
fi

# ===== FULL TESTS (Optional) =====
if [ "$RUN_FULL" = true ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                    FULL TEST SUITE                         ║"
    echo "╚════════════════════════════════════════════════════════════╝"

    run_test "Test 5: Full Context Training" "05_train_context_full.sh"
    run_test "Test 6: Context vs Standard Comparison" "06_compare_context_vs_standard.sh"
    run_test "Test 7: Batch Size Testing" "07_test_all_batch_sizes.sh"
    run_test "Test 8: Learning Rate Testing" "08_test_learning_rates.sh"
fi

# ===== SUMMARY =====
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                     TEST SUMMARY                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Total tests run: $TESTS_RUN"
echo "Passed: $TESTS_PASSED ✅"
echo "Failed: $TESTS_FAILED ❌"
echo ""

if [ $TESTS_FAILED -gt 0 ]; then
    echo "Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  ❌ $test"
    done
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                  ⚠️  SOME TESTS FAILED                     ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    exit 1
else
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║              🎉 ALL TESTS PASSED! 🎉                       ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    exit 0
fi
