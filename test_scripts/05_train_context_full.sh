#!/bin/bash
# Test 5: Full Context-Aware Training
# Runs full training with recommended settings

echo "============================================================"
echo "TEST 5: Full Context-Aware Training"
echo "============================================================"

cd "$(dirname "$0")/../GraDeT_HTR" || exit 1

# Check if dataset exists
if [ ! -d "../out/single_words/images" ]; then
    echo "❌ ERROR: Dataset not found at ../out/single_words/images"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Mode: Context-aware (dual-path)"
echo "  Batch size: 16"
echo "  Epochs: 10"
echo "  Learning rate: 1e-4"
echo ""
echo "⚠️  This will take significant time (hours depending on dataset size)"
echo ""

read -p "Continue with full training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

python train.py \
    --context \
    --images_dir ../out/single_words/images \
    --json_dir ../out/single_words/json \
    --batch_size 16 \
    --epochs 10 \
    --lr 1e-4

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 Full training completed successfully!"
    echo ""
    echo "Generated files:"
    ls -lh checkpoint_context_epoch_*.pt 2>/dev/null
    ls -lh final_context_model.pth 2>/dev/null
else
    echo "❌ Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
