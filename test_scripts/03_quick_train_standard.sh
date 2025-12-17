#!/bin/bash
# Test 3: Quick Standard Training (No Context)
# Runs 1 epoch with small batch size to verify standard training works

echo "============================================================"
echo "TEST 3: Quick Standard Training (No Context)"
echo "============================================================"

cd "$(dirname "$0")/../GraDeT_HTR" || exit 1

echo ""
echo "Configuration:"
echo "  Mode: Standard (no context)"
echo "  Batch size: 4"
echo "  Epochs: 1"
echo "  Learning rate: 1e-4"
echo ""

python train.py \
    --images_dir ../sample_train/images \
    --labels_file ../sample_train/labels/label.csv \
    --batch_size 4 \
    --epochs 1 \
    --lr 1e-4

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Standard training test PASSED"

    # Check if checkpoint was created
    if [ -f "checkpoint_epoch_1.pt" ]; then
        echo "✅ Checkpoint file created successfully"
    else
        echo "⚠️  Checkpoint file not found"
    fi
else
    echo "❌ Standard training test FAILED with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
