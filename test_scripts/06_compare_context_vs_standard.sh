#!/bin/bash
# Test 6: Compare Context vs Standard Performance
# Trains both modes for 2 epochs and compares results

echo "============================================================"
echo "TEST 6: Context vs Standard Comparison"
echo "============================================================"

cd "$(dirname "$0")/../GraDeT_HTR" || exit 1

EPOCHS=2
BATCH_SIZE=8
LR=1e-4

echo ""
echo "This test will:"
echo "  1. Train standard mode for $EPOCHS epochs"
echo "  2. Train context-aware mode for $EPOCHS epochs"
echo "  3. Compare final validation metrics"
echo ""

# Create log directory
mkdir -p ../test_logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="../test_logs/${TIMESTAMP}_comparison"
mkdir -p "$LOG_DIR"

echo "Logs will be saved to: $LOG_DIR"
echo ""

# ===== STANDARD TRAINING =====
echo "============================================================"
echo "Phase 1: Standard Training"
echo "============================================================"
echo ""

python train.py \
    --images_dir ../sample_train/images \
    --labels_file ../sample_train/labels/label.csv \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    2>&1 | tee "$LOG_DIR/standard_training.log"

STANDARD_EXIT=$?

if [ $STANDARD_EXIT -eq 0 ]; then
    echo "✅ Standard training completed"
    mv checkpoint_epoch_*.pt "$LOG_DIR/" 2>/dev/null
    mv final_model.pth "$LOG_DIR/final_standard_model.pth" 2>/dev/null
else
    echo "❌ Standard training failed"
fi

echo ""
echo "============================================================"
echo "Phase 2: Context-Aware Training"
echo "============================================================"
echo ""

# Check dataset
if [ ! -d "../out/single_words/images" ]; then
    echo "⚠️  Context dataset not found, skipping context training"
    echo "   To run full comparison, ensure ../out/single_words/ exists"
    exit 0
fi

python train.py \
    --context \
    --images_dir ../out/single_words/images \
    --json_dir ../out/single_words/json \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    2>&1 | tee "$LOG_DIR/context_training.log"

CONTEXT_EXIT=$?

if [ $CONTEXT_EXIT -eq 0 ]; then
    echo "✅ Context-aware training completed"
    mv checkpoint_context_epoch_*.pt "$LOG_DIR/" 2>/dev/null
    mv final_context_model.pth "$LOG_DIR/" 2>/dev/null
else
    echo "❌ Context-aware training failed"
fi

# ===== COMPARISON =====
echo ""
echo "============================================================"
echo "COMPARISON RESULTS"
echo "============================================================"
echo ""

# Extract final metrics from logs
echo "Standard Training:"
grep "Epoch: $EPOCHS" "$LOG_DIR/standard_training.log" | tail -1
echo ""

echo "Context-Aware Training:"
grep "Epoch $EPOCHS/$EPOCHS:" "$LOG_DIR/context_training.log" | tail -4
echo ""

if [ $STANDARD_EXIT -eq 0 ] && [ $CONTEXT_EXIT -eq 0 ]; then
    echo "🎉 Both training modes completed successfully!"
    echo ""
    echo "Files saved to: $LOG_DIR"
    exit 0
else
    echo "⚠️  Some training modes failed"
    exit 1
fi
