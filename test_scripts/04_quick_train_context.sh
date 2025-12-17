#!/bin/bash
# Test 4: Quick Context-Aware Training
# Runs 1 epoch with small batch size to verify context-aware training works

echo "============================================================"
echo "TEST 4: Quick Context-Aware Training"
echo "============================================================"

cd "$(dirname "$0")/../GraDeT_HTR" || exit 1

# Check if dataset exists
if [ ! -d "../out/single_words/images" ]; then
    echo "❌ ERROR: Dataset not found at ../out/single_words/images"
    echo "   Please ensure the dataset is available before running this test."
    exit 1
fi

if [ ! -d "../out/single_words/json" ]; then
    echo "❌ ERROR: JSON directory not found at ../out/single_words/json"
    echo "   Please ensure the JSON files are available before running this test."
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Mode: Context-aware (dual-path)"
echo "  Batch size: 4"
echo "  Epochs: 1"
echo "  Learning rate: 1e-4"
echo ""

python train.py \
    --context \
    --images_dir ../out/single_words/images \
    --json_dir ../out/single_words/json \
    --batch_size 4 \
    --epochs 1 \
    --lr 1e-4

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Context-aware training test PASSED"

    # Check if checkpoint was created
    if [ -f "checkpoint_context_epoch_1.pt" ]; then
        echo "✅ Context checkpoint file created successfully"
    else
        echo "⚠️  Context checkpoint file not found"
    fi

    # Check for final model
    if [ -f "final_context_model.pth" ]; then
        echo "✅ Final context model saved successfully"
    fi
else
    echo "❌ Context-aware training test FAILED with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
