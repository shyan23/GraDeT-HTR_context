#!/bin/bash
# Test 8: Test Different Learning Rates
# Tests context training with various learning rates

echo "============================================================"
echo "TEST 8: Context Training with Different Learning Rates"
echo "============================================================"

cd "$(dirname "$0")/../GraDeT_HTR" || exit 1

if [ ! -d "../out/single_words/images" ]; then
    echo "❌ ERROR: Dataset not found"
    exit 1
fi

echo ""
echo "Testing learning rates: 5e-6, 1e-5, 5e-5, 1e-4, 5e-4"
echo "Each will run for 2 epochs"
echo ""

# Create log directory
mkdir -p ../test_logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="../test_logs/${TIMESTAMP}_learning_rates"
mkdir -p "$LOG_DIR"

# Array of learning rates
LEARNING_RATES=("5e-6" "1e-5" "5e-5" "1e-4" "5e-4")
BATCH_SIZE=8
EPOCHS=2
RESULTS_FILE="$LOG_DIR/lr_results.txt"

echo "Learning Rate Test Results" > "$RESULTS_FILE"
echo "===========================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

for LR in "${LEARNING_RATES[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Testing learning rate: $LR"
    echo "------------------------------------------------------------"

    python train.py \
        --context \
        --images_dir ../out/single_words/images \
        --json_dir ../out/single_words/json \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        2>&1 | tee "$LOG_DIR/lr_${LR}.log"

    EXIT_CODE=$?

    echo "" >> "$RESULTS_FILE"
    echo "Learning Rate: $LR" >> "$RESULTS_FILE"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "  Status: ✅ SUCCESS" >> "$RESULTS_FILE"

        # Extract metrics for both epochs
        echo "  Epoch 1:" >> "$RESULTS_FILE"
        grep "Epoch 1/$EPOCHS:" "$LOG_DIR/lr_${LR}.log" -A 4 | grep "Train - Loss:" | awk '{print "    Train Loss: "$4"  Acc: "$7}' >> "$RESULTS_FILE"
        grep "Epoch 1/$EPOCHS:" "$LOG_DIR/lr_${LR}.log" -A 4 | grep "Val   - Loss:" | awk '{print "    Val Loss:   "$4"  Acc: "$7}' >> "$RESULTS_FILE"

        echo "  Epoch 2:" >> "$RESULTS_FILE"
        grep "Epoch 2/$EPOCHS:" "$LOG_DIR/lr_${LR}.log" -A 4 | grep "Train - Loss:" | awk '{print "    Train Loss: "$4"  Acc: "$7}' >> "$RESULTS_FILE"
        grep "Epoch 2/$EPOCHS:" "$LOG_DIR/lr_${LR}.log" -A 4 | grep "Val   - Loss:" | awk '{print "    Val Loss:   "$4"  Acc: "$7}' >> "$RESULTS_FILE"

        echo "✅ Learning rate $LR: SUCCESS"
    else
        echo "  Status: ❌ FAILED" >> "$RESULTS_FILE"
        echo "❌ Learning rate $LR: FAILED"
    fi

    # Clean up checkpoints
    rm -f checkpoint_context_epoch_*.pt final_context_model.pth 2>/dev/null
done

echo ""
echo "============================================================"
echo "LEARNING RATE TEST SUMMARY"
echo "============================================================"
cat "$RESULTS_FILE"

echo ""
echo "Detailed logs saved to: $LOG_DIR"
echo ""
echo "Recommendation:"
echo "  Choose the learning rate with:"
echo "  - Lowest final validation loss"
echo "  - Stable training (no loss spikes)"
echo "  - Consistent improvement across epochs"
