#!/usr/bin/env python3
"""
Test 2: Model Forward Pass Validation
Tests model forward pass with and without context.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'GraDeT_HTR'))

import torch
from model import DTrOCRLMHeadModel
from config import DTrOCRConfig


def test_model_standard_forward():
    """Test standard forward pass (no context)"""
    print("=" * 60)
    print("TEST 2A: Model Forward Pass (Standard Mode)")
    print("=" * 60)

    try:
        config = DTrOCRConfig()
        model = DTrOCRLMHeadModel(config)
        model.eval()

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create dummy inputs
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 32, 128)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, 10))
        attention_mask = torch.ones(batch_size, 10)
        labels = torch.randint(0, config.vocab_size, (batch_size, 10))

        print(f"\nInput shapes:")
        print(f"  pixel_values: {pixel_values.shape}")
        print(f"  input_ids: {input_ids.shape}")
        print(f"  attention_mask: {attention_mask.shape}")
        print(f"  labels: {labels.shape}")

        # Forward pass
        with torch.no_grad():
            output = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                context_length=0  # No context
            )

        print(f"\nOutput:")
        print(f"  loss: {output.loss.item():.4f}")
        print(f"  accuracy: {output.accuracy.item():.4f}")
        print(f"  logits shape: {output.logits.shape}")

        # Validate
        assert not torch.isnan(output.loss), "Loss is NaN!"
        assert output.logits.shape[0] == batch_size, "Batch size mismatch"
        assert output.logits.shape[-1] == config.vocab_size, "Vocab size mismatch"

        print("\n✅ Standard forward pass PASSED\n")
        return True

    except Exception as e:
        print(f"❌ Standard forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_context_forward():
    """Test forward pass with context"""
    print("=" * 60)
    print("TEST 2B: Model Forward Pass (Context Mode)")
    print("=" * 60)

    try:
        config = DTrOCRConfig()
        model = DTrOCRLMHeadModel(config)
        model.eval()

        # Create dummy inputs with context
        batch_size = 2
        context_length = 3
        pixel_values = torch.randn(batch_size, 3, 32, 128)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, context_length + 10))  # context + target
        attention_mask = torch.ones(batch_size, context_length + 10)
        labels = torch.randint(0, config.vocab_size, (batch_size, context_length + 10))

        print(f"\nInput shapes (with context):")
        print(f"  pixel_values: {pixel_values.shape}")
        print(f"  input_ids: {input_ids.shape}")
        print(f"  context_length: {context_length}")

        # Forward pass
        with torch.no_grad():
            output = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                context_length=context_length,
                return_per_sample_loss=True
            )

        print(f"\nOutput:")
        print(f"  loss: {output.loss.item():.4f}")
        print(f"  accuracy: {output.accuracy.item():.4f}")
        print(f"  per_sample_loss shape: {output.per_sample_loss.shape}")
        print(f"  per_sample_loss: {output.per_sample_loss}")

        # Validate
        assert not torch.isnan(output.loss), "Loss is NaN!"
        assert output.per_sample_loss is not None, "Per-sample loss not returned"
        assert output.per_sample_loss.shape[0] == batch_size, "Per-sample loss batch size mismatch"

        print("\n✅ Context forward pass PASSED\n")
        return True

    except Exception as e:
        print(f"❌ Context forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dual_path_forward():
    """Test dual-path forward pass (contextual + isolation)"""
    print("=" * 60)
    print("TEST 2C: Dual-Path Forward Pass")
    print("=" * 60)

    try:
        config = DTrOCRConfig()
        model = DTrOCRLMHeadModel(config)
        model.eval()

        batch_size = 2
        context_length = 3
        pixel_values = torch.randn(batch_size, 3, 32, 128)

        # Contextual inputs
        input_ids_ctx = torch.randint(0, config.vocab_size, (batch_size, context_length + 10))
        attention_mask_ctx = torch.ones(batch_size, context_length + 10)
        labels_ctx = torch.randint(0, config.vocab_size, (batch_size, context_length + 10))

        # Isolation inputs (no context)
        input_ids_iso = torch.randint(0, config.vocab_size, (batch_size, 10))
        attention_mask_iso = torch.ones(batch_size, 10)
        labels_iso = torch.randint(0, config.vocab_size, (batch_size, 10))

        print("\nContextual forward pass...")
        with torch.no_grad():
            output_ctx = model(
                pixel_values=pixel_values,
                input_ids=input_ids_ctx,
                attention_mask=attention_mask_ctx,
                labels=labels_ctx,
                context_length=context_length,
                return_per_sample_loss=True
            )

        print("Isolation forward pass...")
        with torch.no_grad():
            output_iso = model(
                pixel_values=pixel_values,
                input_ids=input_ids_iso,
                attention_mask=attention_mask_iso,
                labels=labels_iso,
                context_length=0,
                return_per_sample_loss=True
            )

        print(f"\nResults:")
        print(f"  Contextual loss: {output_ctx.loss.item():.4f}")
        print(f"  Isolation loss: {output_iso.loss.item():.4f}")
        print(f"  Contextual acc: {output_ctx.accuracy.item():.4f}")
        print(f"  Isolation acc: {output_iso.accuracy.item():.4f}")

        # Compute difficulty weights
        weights = (output_iso.per_sample_loss / output_iso.per_sample_loss.sum()) * batch_size
        weights = torch.clamp(weights, min=0.1, max=10.0)

        print(f"\nDifficulty weights: {weights}")

        # Combined loss
        combined_loss = output_ctx.per_sample_loss + output_iso.per_sample_loss
        weighted_loss = (weights * combined_loss).mean()

        print(f"Combined per-sample loss: {combined_loss}")
        print(f"Weighted loss: {weighted_loss.item():.4f}")

        # Validate
        assert not torch.isnan(weighted_loss), "Weighted loss is NaN!"
        assert weights.min() >= 0.1 and weights.max() <= 10.0, "Weights out of bounds!"

        print("\n✅ Dual-path forward pass PASSED\n")
        return True

    except Exception as e:
        print(f"❌ Dual-path forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "🧪 " * 20)
    print("MODEL FORWARD PASS TESTS")
    print("🧪 " * 20 + "\n")

    os.chdir(os.path.join(os.path.dirname(__file__), '..', 'GraDeT_HTR'))

    results = []
    results.append(("Standard Forward", test_model_standard_forward()))
    results.append(("Context Forward", test_model_context_forward()))
    results.append(("Dual-Path Forward", test_dual_path_forward()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("🎉 ALL TESTS PASSED!" if all_passed else "⚠️  SOME TESTS FAILED"))

    sys.exit(0 if all_passed else 1)
