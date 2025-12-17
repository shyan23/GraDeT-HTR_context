#!/bin/bash
# Test 7: Test Different Batch Sizes
# Tests context training with various batch sizes to find optimal memory usage

echo "============================================================"
echo "TEST 7: Context Training with Different Batch Sizes"
echo "============================================================"

cd "$(dirname "$0")/../GraDeT_HTR" || exit 1

if [ ! -d "../out/single_words/images" ]; then
    echo "❌ ERROR: Dataset not found"
    exit 1
fi

echo ""
echo "Testing batch sizes: 2, 4, 8, 16, 32"
echo "Each will run for 1 epoch to test memory requirements"
echo ""

# Create log directory
mkdir -p ../test_logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="../test_logs/${TIMESTAMP}_batch_sizes"
mkdir -p "$LOG_DIR"

# Array of batch sizes to test
BATCH_SIZES=(2 4 8 16 32)
RESULTS_FILE="$LOG_DIR/batch_size_results.txt"

echo "Batch Size Test Results" > "$RESULTS_FILE"
echo "=======================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Testing batch size: $BATCH_SIZE"
    echo "------------------------------------------------------------"

    START_TIME=$(date +%s)

    python train.py \
        --context \
        --images_dir ../out/single_words/images \
        --json_dir ../out/single_words/json \
        --batch_size $BATCH_SIZE \
        --epochs 1 \
        --lr 1e-4 \
        2>&1 | tee "$LOG_DIR/batch_${BATCH_SIZE}.log"

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo "" >> "$RESULTS_FILE"
    echo "Batch Size: $BATCH_SIZE" >> "$RESULTS_FILE"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "  Status: ✅ SUCCESS" >> "$RESULTS_FILE"
        echo "  Duration: ${DURATION}s" >> "$RESULTS_FILE"

        # Extract final metrics
        LAST_LOSS=$(grep "Train - Loss:" "$LOG_DIR/batch_${BATCH_SIZE}.log" | tail -1 | awk '{print $4}')
        echo "  Final Loss: $LAST_LOSS" >> "$RESULTS_FILE"

        echo "✅ Batch size $BATCH_SIZE: SUCCESS (${DURATION}s)"
    else
        echo "  Status: ❌ FAILED (likely OOM)" >> "$RESULTS_FILE"
        echo "❌ Batch size $BATCH_SIZE: FAILED (Out of Memory?)"
    fi

    # Clean up checkpoints
    rm -f checkpoint_context_epoch_*.pt final_context_model.pth 2>/dev/null
done

echo ""
echo "============================================================"
echo "BATCH SIZE TEST SUMMARY"
echo "============================================================"
cat "$RESULTS_FILE"

echo ""
echo "Detailed logs saved to: $LOG_DIR"
echo ""
echo "Recommendation:"
echo "  Use the largest batch size that completed successfully"
echo "  for optimal training speed."
