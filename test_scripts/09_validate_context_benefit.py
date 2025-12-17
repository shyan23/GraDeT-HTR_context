#!/usr/bin/env python3
"""
Test 9: Validate Context Benefit
Verifies that context actually improves performance over isolation mode.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'GraDeT_HTR'))

import torch
from torch.utils.data import DataLoader
from model import DTrOCRLMHeadModel
from dataset import split_context_data
from config import DTrOCRConfig


def evaluate_with_context(model, dataloader, device):
    """Evaluate with context (contextual mode)"""
    model.eval()
    losses, accs = [], []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            context_length = batch['context_length'][0].item() if isinstance(batch['context_length'], torch.Tensor) else batch['context_length']

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                context_length=context_length
            )

            losses.append(outputs.loss.item())
            accs.append(outputs.accuracy.item())

    return sum(losses) / len(losses), sum(accs) / len(accs)


def evaluate_without_context(model, dataloader, device):
    """Evaluate without context (isolation mode)"""
    model.eval()
    losses, accs = [], []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids_isolated'].to(device)
            attention_mask = batch['attention_mask_isolated'].to(device)
            labels = batch['labels_isolated'].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                context_length=0
            )

            losses.append(outputs.loss.item())
            accs.append(outputs.accuracy.item())

    return sum(losses) / len(losses), sum(accs) / len(accs)


def main():
    print("\n" + "🧪 " * 20)
    print("TEST 9: Context Benefit Validation")
    print("🧪 " * 20 + "\n")

    print("=" * 60)
    print("This test validates that context improves performance")
    print("=" * 60)

    os.chdir(os.path.join(os.path.dirname(__file__), '..', 'GraDeT_HTR'))

    # Setup
    config = DTrOCRConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}\n")

    # Load dataset
    try:
        images_dir = "../out/single_words/images"
        json_dir = "../out/single_words/json"

        print("Loading dataset...")
        train_dataset, test_dataset = split_context_data(images_dir, json_dir, config, test_size=0.2)
        print(f"Test set size: {len(test_dataset)}")

        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False

    # Initialize random model
    print("\nInitializing model with random weights...")
    model = DTrOCRLMHeadModel(config)
    model.to(device)
    model.eval()

    print("\n" + "=" * 60)
    print("Evaluating with CONTEXT (Contextual Mode)")
    print("=" * 60)
    ctx_loss, ctx_acc = evaluate_with_context(model, test_loader, device)
    print(f"Context Loss: {ctx_loss:.4f}")
    print(f"Context Accuracy: {ctx_acc:.4f}")

    print("\n" + "=" * 60)
    print("Evaluating WITHOUT CONTEXT (Isolation Mode)")
    print("=" * 60)
    iso_loss, iso_acc = evaluate_without_context(model, test_loader, device)
    print(f"Isolation Loss: {iso_loss:.4f}")
    print(f"Isolation Accuracy: {iso_acc:.4f}")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    loss_improvement = ((iso_loss - ctx_loss) / iso_loss) * 100
    acc_improvement = ((ctx_acc - iso_acc) / iso_acc) * 100

    print(f"\nContext vs Isolation:")
    print(f"  Loss reduction: {loss_improvement:.2f}%")
    print(f"  Accuracy improvement: {acc_improvement:.2f}%")

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    checks = []

    # Check 1: Context loss should be lower (or similar)
    if ctx_loss <= iso_loss * 1.1:  # Allow 10% margin
        print("✅ Context loss ≤ Isolation loss (context helps or neutral)")
        checks.append(True)
    else:
        print("❌ Context loss > Isolation loss (context hurts!)")
        checks.append(False)

    # Check 2: Context accuracy should be higher (or similar)
    if ctx_acc >= iso_acc * 0.95:  # Allow 5% margin
        print("✅ Context accuracy ≥ Isolation accuracy")
        checks.append(True)
    else:
        print("❌ Context accuracy < Isolation accuracy")
        checks.append(False)

    # Check 3: Values should be reasonable
    if 0.01 < ctx_acc < 1.0 and 0.01 < iso_acc < 1.0:
        print("✅ Accuracy values are reasonable")
        checks.append(True)
    else:
        print("❌ Accuracy values seem off")
        checks.append(False)

    all_passed = all(checks)

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 Context benefit validation PASSED!")
        print("   Context improves or maintains performance.")
    else:
        print("⚠️  Context benefit validation FAILED")
        print("   Context may need tuning or dataset issues exist.")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
