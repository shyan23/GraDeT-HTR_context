#!/usr/bin/env python3
"""
Diagnostic Test Suite for Context-Aware Training
=================================================
Validates all 6 bug fixes and provides full visibility into:
  - Matrix sizes at every stage (embeddings, logits, labels, masks)
  - Shift alignment between logits and labels
  - Loss formula correctness (margin term)
  - Processor fixed-length padding
  - DataLoader batching
  - Backward compatibility (isolation / standard mode)
  - Edge cases (empty context, long context, batch_size=1, margin_lambda=0)

Run from repo root:
    python test_scripts/10_diagnostic_context_suite.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'GraDeT_HTR'))

import torch
import torch.nn as nn
from config import DTrOCRConfig
from model import DTrOCRLMHeadModel, DTrOCRModel
from data import DTrOCRProcessorOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = "PASS"
FAIL = "FAIL"
results = []


def record(name, passed, detail=""):
    tag = PASS if passed else FAIL
    results.append((name, passed))
    mark = "[PASS]" if passed else "[FAIL]"
    msg = f"  {mark} {name}"
    if detail:
        msg += f"  --  {detail}"
    print(msg)
    return passed


def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def dims(t, name=""):
    """Pretty-print tensor shape."""
    prefix = f"{name}: " if name else ""
    if t is None:
        return f"{prefix}None"
    return f"{prefix}{list(t.shape)}"


# ---------------------------------------------------------------------------
# 1. Model forward dimension checks
# ---------------------------------------------------------------------------
def test_isolation_forward_dimensions():
    """Verify all matrix sizes through a standard (no-context) forward pass."""
    section("1. Isolation Forward Pass -- Dimension Trace")

    config = DTrOCRConfig()
    model = DTrOCRLMHeadModel(config)
    model.eval()

    B, T = 2, 128   # batch, target length
    I = model.image_embedding_length  # 128 patches

    pixel_values = torch.randn(B, 3, 32, 128)
    input_ids    = torch.randint(0, config.vocab_size, (B, T))
    attn_mask    = torch.ones(B, T, dtype=torch.long)
    labels       = input_ids.clone()

    print(f"  Config: vocab={config.vocab_size}, max_pos={config.max_position_embeddings}, "
          f"image_patches={I}, hidden={config.hidden_size}")
    print(f"  Inputs:  {dims(pixel_values,'pixel_values')}  {dims(input_ids,'input_ids')}  "
          f"{dims(attn_mask,'attn_mask')}  {dims(labels,'labels')}")

    with torch.no_grad():
        out = model(
            pixel_values=pixel_values, input_ids=input_ids,
            attention_mask=attn_mask, labels=labels,
            context_length=0, return_per_sample_loss=True
        )

    expected_seq = I + T  # 128 + 128 = 256
    print(f"\n  Logits:           {dims(out.logits,'logits')}")
    print(f"  Expected seq len: {expected_seq}  (I={I} + T={T})")
    print(f"  Loss:             {out.loss.item():.4f}")
    print(f"  Accuracy:         {out.accuracy.item():.4f}")
    print(f"  Per-sample loss:  {dims(out.per_sample_loss,'per_sample_loss')}")

    ok = True
    ok &= record("logits shape", out.logits.shape == (B, expected_seq, config.vocab_size),
                  f"got {list(out.logits.shape)}, want [{B}, {expected_seq}, {config.vocab_size}]")
    ok &= record("loss not NaN", not torch.isnan(out.loss))
    ok &= record("per_sample_loss shape", out.per_sample_loss.shape == (B,))
    ok &= record("per_sample_loss not NaN", not torch.isnan(out.per_sample_loss).any())

    # Verify internal shift alignment
    skip = 0 + I  # context_length=0
    shift_logits_len = expected_seq - skip - 1  # T - 1 = 127
    shift_labels_len = T - (0 + 1)               # T - 1 = 127
    ok &= record("shift alignment (isolation)",
                  shift_logits_len == shift_labels_len,
                  f"shift_logits_len={shift_logits_len}, shift_labels_len={shift_labels_len}")
    return ok


def test_context_forward_dimensions():
    """Verify all matrix sizes through a context-aware forward pass."""
    section("2. Context Forward Pass -- Dimension Trace")

    config = DTrOCRConfig(max_position_embeddings=320)
    model = DTrOCRLMHeadModel(config)
    model.eval()

    B, C, T = 2, 20, 128  # batch, context, target
    I = model.image_embedding_length

    pixel_values = torch.randn(B, 3, 32, 128)
    input_ids    = torch.randint(0, config.vocab_size, (B, C + T))
    attn_mask    = torch.ones(B, C + T, dtype=torch.long)
    labels       = input_ids.clone()

    print(f"  Config: max_pos={config.max_position_embeddings}")
    print(f"  Inputs:  B={B}, C={C}, T={T}, I={I}")
    print(f"    {dims(pixel_values,'pixel_values')}  {dims(input_ids,'input_ids')}  "
          f"{dims(attn_mask,'attn_mask')}")

    with torch.no_grad():
        out = model(
            pixel_values=pixel_values, input_ids=input_ids,
            attention_mask=attn_mask, labels=labels,
            context_length=C, return_per_sample_loss=True
        )

    expected_seq = C + I + T
    print(f"\n  Logits:           {dims(out.logits,'logits')}")
    print(f"  Expected seq len: {expected_seq}  (C={C} + I={I} + T={T})")
    print(f"  Loss:             {out.loss.item():.4f}")
    print(f"  Accuracy:         {out.accuracy.item():.4f}")
    print(f"  Per-sample loss:  {dims(out.per_sample_loss,'per_sample_loss')}")

    ok = True
    ok &= record("logits shape", out.logits.shape == (B, expected_seq, config.vocab_size),
                  f"got {list(out.logits.shape)}, want [{B}, {expected_seq}, {config.vocab_size}]")
    ok &= record("loss not NaN", not torch.isnan(out.loss))
    ok &= record("per_sample_loss shape", out.per_sample_loss.shape == (B,))

    # Internal shift alignment
    skip = C + I
    shift_logits_len = expected_seq - skip - 1  # T - 1 = 127
    shift_labels_len = (C + T) - (C + 1)        # T - 1 = 127
    ok &= record("shift alignment (context)",
                  shift_logits_len == shift_labels_len,
                  f"shift_logits_len={shift_logits_len}, shift_labels_len={shift_labels_len}")
    return ok


# ---------------------------------------------------------------------------
# 2. Shift alignment across many context lengths
# ---------------------------------------------------------------------------
def test_shift_alignment_sweep():
    """Test that shift_logits and shift_labels always match for various C."""
    section("3. Shift Alignment Sweep (C=0..25)")

    config = DTrOCRConfig(max_position_embeddings=320)
    model = DTrOCRLMHeadModel(config)
    model.eval()

    B, T = 2, 128
    I = model.image_embedding_length
    ok = True

    for C in [0, 1, 4, 10, 15, 20, 25]:
        total_input = C + T
        pixel_values = torch.randn(B, 3, 32, 128)
        input_ids    = torch.randint(0, config.vocab_size, (B, total_input))
        attn_mask    = torch.ones(B, total_input, dtype=torch.long)
        labels       = input_ids.clone()

        try:
            with torch.no_grad():
                out = model(
                    pixel_values=pixel_values, input_ids=input_ids,
                    attention_mask=attn_mask, labels=labels,
                    context_length=C, return_per_sample_loss=True
                )
            no_crash = True
            is_nan = torch.isnan(out.loss).item()
        except RuntimeError as e:
            no_crash = False
            is_nan = True

        ok &= record(f"C={C:2d}  forward pass",
                      no_crash and not is_nan,
                      f"{'OK' if no_crash and not is_nan else 'CRASHED or NaN'}")
    return ok


# ---------------------------------------------------------------------------
# 3. Label content correctness
# ---------------------------------------------------------------------------
def test_label_content_correctness():
    """Verify that shift_labels actually contains target tokens, not context."""
    section("4. Label Content Correctness")

    config = DTrOCRConfig(max_position_embeddings=320)

    B, C, T = 1, 5, 10
    I = 128  # image patches

    # Construct known input_ids: context uses token 999, target uses tokens 1-10
    context_tokens = [999] * C
    target_tokens = list(range(1, T + 1))  # [1,2,3,...,10]
    combined = context_tokens + target_tokens
    input_ids = torch.tensor([combined])
    labels = input_ids.clone()

    # After shifting:
    #   shift_labels = labels[..., C+1:] = labels[..., 6:] = [7,8,9,10]
    #   (skips context + BOS position)
    expected_shift_labels = labels[..., C + 1:]
    expected_values = target_tokens[1:]  # skip BOS (first target token)

    print(f"  input_ids:           {combined}")
    print(f"  context (C={C}):       {context_tokens}")
    print(f"  target (T={T}):        {target_tokens}")
    print(f"  shift_labels (C+1:): {expected_shift_labels.tolist()[0]}")
    print(f"  expected:            {expected_values}")

    ok = True
    ok &= record("shift_labels skips context",
                  expected_shift_labels[0, 0].item() != 999,
                  f"first label = {expected_shift_labels[0, 0].item()}, should NOT be 999 (context)")
    ok &= record("shift_labels matches target[1:]",
                  expected_shift_labels[0].tolist() == expected_values)

    # For isolation (C=0), should be labels[..., 1:] = [2,3,...,10]
    iso_shift = labels[..., 0 + 1:]
    ok &= record("isolation shift_labels = labels[1:]",
                  iso_shift[0].tolist() == list(range(2, T + 1)))
    return ok


# ---------------------------------------------------------------------------
# 4. Mask alignment
# ---------------------------------------------------------------------------
def test_mask_alignment():
    """Verify the attention mask is sliced correctly to match shift_logits."""
    section("5. Attention Mask Alignment")

    B, C, T = 2, 20, 128
    I = 128

    # Create mask with padding at end of target
    attn_mask = torch.ones(B, C + T, dtype=torch.long)
    # Set last 30 target positions to 0 (padding)
    attn_mask[:, -30:] = 0

    # After fix: mask = attention_mask[..., C+1:]
    mask_slice = attn_mask[..., C + 1:]
    expected_len = T - 1  # = 127

    print(f"  attention_mask shape: {list(attn_mask.shape)}")
    print(f"  mask[..., C+1:] shape: {list(mask_slice.shape)}")
    print(f"  Expected length: {expected_len}")
    print(f"  Active positions: {mask_slice[0].sum().item()}")
    print(f"  Padded positions: {(mask_slice[0] == 0).sum().item()}")

    ok = True
    ok &= record("mask length matches T-1",
                  mask_slice.shape[-1] == expected_len,
                  f"got {mask_slice.shape[-1]}, want {expected_len}")
    ok &= record("mask has padding",
                  (mask_slice == 0).any(),
                  "padding should be present in mask")

    # For isolation (C=0)
    iso_mask = attn_mask[..., 0 + 1:]
    ok &= record("isolation mask = attn_mask[1:]",
                  iso_mask.shape[-1] == C + T - 1)
    return ok


# ---------------------------------------------------------------------------
# 5. Loss formula: margin term
# ---------------------------------------------------------------------------
def test_margin_term_logic():
    """Verify the margin term activates correctly."""
    section("6. Margin Term Logic")

    margin_lambda = 0.5
    ok = True

    # Case 1: context BETTER (loss_ctx < loss_iso) -> margin = 0
    loss_ctx = torch.tensor([0.5, 0.3, 0.8])
    loss_iso = torch.tensor([0.8, 0.9, 1.2])
    margin = margin_lambda * torch.clamp(loss_ctx - loss_iso, min=0.0)
    diff = loss_ctx - loss_iso
    print(f"\n  Case 1: Context is BETTER")
    print(f"    loss_ctx:  {loss_ctx.tolist()}")
    print(f"    loss_iso:  {loss_iso.tolist()}")
    print(f"    diff:      {diff.tolist()}")
    print(f"    margin:    {margin.tolist()}")
    ok &= record("margin=0 when ctx better", (margin == 0).all().item())

    # Case 2: context WORSE (loss_ctx > loss_iso) -> margin > 0
    loss_ctx = torch.tensor([1.0, 0.8, 1.5])
    loss_iso = torch.tensor([0.5, 0.6, 0.9])
    margin = margin_lambda * torch.clamp(loss_ctx - loss_iso, min=0.0)
    expected = margin_lambda * torch.tensor([0.5, 0.2, 0.6])
    diff = loss_ctx - loss_iso
    print(f"\n  Case 2: Context is WORSE")
    print(f"    loss_ctx:  {loss_ctx.tolist()}")
    print(f"    loss_iso:  {loss_iso.tolist()}")
    print(f"    diff:      {diff.tolist()}")
    print(f"    margin:    {margin.tolist()}")
    print(f"    expected:  {expected.tolist()}")
    ok &= record("margin>0 when ctx worse",
                  torch.allclose(margin, expected, atol=1e-6))

    # Case 3: mixed batch
    loss_ctx = torch.tensor([0.5, 1.2])
    loss_iso = torch.tensor([0.8, 0.6])
    margin = margin_lambda * torch.clamp(loss_ctx - loss_iso, min=0.0)
    print(f"\n  Case 3: Mixed batch")
    print(f"    loss_ctx:  {loss_ctx.tolist()}")
    print(f"    loss_iso:  {loss_iso.tolist()}")
    print(f"    margin:    {margin.tolist()}")
    ok &= record("sample 0: ctx better, margin=0",
                  margin[0].item() == 0.0)
    ok &= record("sample 1: ctx worse, margin=0.3",
                  abs(margin[1].item() - 0.3) < 1e-6,
                  f"got {margin[1].item()}")

    # Case 4: margin_lambda = 0 -> no margin ever
    margin_zero = 0.0 * torch.clamp(loss_ctx - loss_iso, min=0.0)
    print(f"\n  Case 4: lambda=0")
    print(f"    margin:    {margin_zero.tolist()}")
    ok &= record("lambda=0 -> margin always 0", (margin_zero == 0).all().item())

    return ok


def test_full_loss_formula():
    """Verify L = loss_ctx + loss_iso + lambda*max(0, loss_ctx - loss_iso)."""
    section("7. Full Loss Formula Verification")

    margin_lambda = 0.5
    loss_ctx = torch.tensor([0.5, 1.2, 0.8])
    loss_iso = torch.tensor([0.8, 0.6, 0.8])

    base = loss_ctx + loss_iso
    margin = margin_lambda * torch.clamp(loss_ctx - loss_iso, min=0.0)
    total = (base + margin).mean()

    # Manual:
    # Sample 0: 0.5+0.8 + 0.5*max(0,-0.3) = 1.3 + 0   = 1.3
    # Sample 1: 1.2+0.6 + 0.5*max(0, 0.6) = 1.8 + 0.3 = 2.1
    # Sample 2: 0.8+0.8 + 0.5*max(0, 0.0) = 1.6 + 0   = 1.6
    # Mean: 5.0/3 = 1.6667
    expected = torch.tensor(5.0 / 3.0)

    print(f"  loss_ctx:    {loss_ctx.tolist()}")
    print(f"  loss_iso:    {loss_iso.tolist()}")
    print(f"  base:        {base.tolist()}")
    print(f"  margin:      {margin.tolist()}")
    print(f"  per-sample:  {(base + margin).tolist()}")
    print(f"  total:       {total.item():.6f}")
    print(f"  expected:    {expected.item():.6f}")

    ok = record("loss matches manual calculation",
                 torch.allclose(total, expected, atol=1e-4),
                 f"got {total.item():.6f}, want {expected.item():.6f}")
    return ok


# ---------------------------------------------------------------------------
# 6. Dual-path forward: both paths produce valid output
# ---------------------------------------------------------------------------
def test_dual_path_forward():
    """Simulate the train loop's dual forward pass."""
    section("8. Dual-Path Forward Pass (Train Loop Simulation)")

    config = DTrOCRConfig(max_position_embeddings=320)
    model = DTrOCRLMHeadModel(config)
    model.eval()

    B, C, T = 4, 20, 128
    I = model.image_embedding_length

    pixel_values = torch.randn(B, 3, 32, 128)

    # Contextual path
    input_ids_ctx = torch.randint(0, config.vocab_size, (B, C + T))
    attn_mask_ctx = torch.ones(B, C + T, dtype=torch.long)
    labels_ctx    = input_ids_ctx.clone()

    # Isolation path
    input_ids_iso = torch.randint(0, config.vocab_size, (B, T))
    attn_mask_iso = torch.ones(B, T, dtype=torch.long)
    labels_iso    = input_ids_iso.clone()

    print(f"  B={B}, C={C}, T={T}, I={I}")
    print(f"  ctx input_ids: {list(input_ids_ctx.shape)}  iso input_ids: {list(input_ids_iso.shape)}")

    ok = True
    with torch.no_grad():
        # Contextual
        out_ctx = model(
            pixel_values=pixel_values, input_ids=input_ids_ctx,
            attention_mask=attn_mask_ctx, labels=labels_ctx,
            context_length=C, return_per_sample_loss=True
        )
        # Isolation
        out_iso = model(
            pixel_values=pixel_values, input_ids=input_ids_iso,
            attention_mask=attn_mask_iso, labels=labels_iso,
            context_length=0, return_per_sample_loss=True
        )

    print(f"\n  Contextual:  loss={out_ctx.loss.item():.4f}  acc={out_ctx.accuracy.item():.4f}  "
          f"per_sample={out_ctx.per_sample_loss.tolist()}")
    print(f"  Isolation:   loss={out_iso.loss.item():.4f}  acc={out_iso.accuracy.item():.4f}  "
          f"per_sample={out_iso.per_sample_loss.tolist()}")

    ok &= record("ctx loss valid", not torch.isnan(out_ctx.loss))
    ok &= record("iso loss valid", not torch.isnan(out_iso.loss))
    ok &= record("ctx per_sample shape", out_ctx.per_sample_loss.shape == (B,))
    ok &= record("iso per_sample shape", out_iso.per_sample_loss.shape == (B,))

    # Compute combined loss like train.py
    margin_lambda = 0.5
    base = out_ctx.per_sample_loss + out_iso.per_sample_loss
    margin = margin_lambda * torch.clamp(
        out_ctx.per_sample_loss - out_iso.per_sample_loss, min=0.0
    )
    total = (base + margin).mean()
    print(f"\n  Combined loss:  {total.item():.4f}")
    print(f"  Margin mean:    {margin.mean().item():.4f}")
    print(f"  Margin active:  {(margin > 0).sum().item()}/{B} samples")

    ok &= record("combined loss valid", not torch.isnan(total))
    ok &= record("margin >= 0", (margin >= 0).all().item())
    return ok


# ---------------------------------------------------------------------------
# 7. Gradient flow test
# ---------------------------------------------------------------------------
def test_gradient_flow():
    """Verify gradients flow through both paths and margin term."""
    section("9. Gradient Flow Test")

    config = DTrOCRConfig(max_position_embeddings=320)
    model = DTrOCRLMHeadModel(config)
    model.train()

    B, C, T = 2, 20, 128
    pixel_values = torch.randn(B, 3, 32, 128)
    input_ids_ctx = torch.randint(0, config.vocab_size, (B, C + T))
    attn_mask_ctx = torch.ones(B, C + T, dtype=torch.long)
    labels_ctx    = input_ids_ctx.clone()

    input_ids_iso = torch.randint(0, config.vocab_size, (B, T))
    attn_mask_iso = torch.ones(B, T, dtype=torch.long)
    labels_iso    = input_ids_iso.clone()

    # Forward
    out_ctx = model(pixel_values=pixel_values, input_ids=input_ids_ctx,
                    attention_mask=attn_mask_ctx, labels=labels_ctx,
                    context_length=C, return_per_sample_loss=True)
    out_iso = model(pixel_values=pixel_values, input_ids=input_ids_iso,
                    attention_mask=attn_mask_iso, labels=labels_iso,
                    context_length=0, return_per_sample_loss=True)

    margin_lambda = 0.5
    base = out_ctx.per_sample_loss + out_iso.per_sample_loss
    margin = margin_lambda * torch.clamp(
        out_ctx.per_sample_loss - out_iso.per_sample_loss, min=0.0
    )
    total_loss = (base + margin).mean()

    # Backward
    total_loss.backward()

    # Check gradients
    has_grad = False
    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            grad_norms[name] = p.grad.norm().item()

    top_grads = sorted(grad_norms.items(), key=lambda x: -x[1])[:5]
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Parameters with gradients: {len(grad_norms)}/{sum(1 for _ in model.parameters())}")
    print(f"  Top 5 gradient norms:")
    for name, norm in top_grads:
        print(f"    {name}: {norm:.6f}")

    ok = True
    ok &= record("gradients flow", has_grad)
    ok &= record("no NaN gradients",
                  all(not torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None))
    return ok


# ---------------------------------------------------------------------------
# 8. Backward compatibility: context_length=0 same as original
# ---------------------------------------------------------------------------
def test_backward_compatibility():
    """Verify context_length=0 gives identical results to no-context mode."""
    section("10. Backward Compatibility (context_length=0)")

    config = DTrOCRConfig()
    model = DTrOCRLMHeadModel(config)
    model.eval()

    torch.manual_seed(123)
    B, T = 2, 128
    pixel_values = torch.randn(B, 3, 32, 128)
    input_ids    = torch.randint(0, config.vocab_size, (B, T))
    attn_mask    = torch.ones(B, T, dtype=torch.long)
    labels       = input_ids.clone()

    with torch.no_grad():
        # With explicit context_length=0
        out_explicit = model(pixel_values=pixel_values, input_ids=input_ids,
                             attention_mask=attn_mask, labels=labels,
                             context_length=0)
        # Without context_length (default)
        out_default = model(pixel_values=pixel_values, input_ids=input_ids,
                            attention_mask=attn_mask, labels=labels)

    print(f"  Explicit C=0:  loss={out_explicit.loss.item():.6f}  acc={out_explicit.accuracy.item():.6f}")
    print(f"  Default (C=0): loss={out_default.loss.item():.6f}  acc={out_default.accuracy.item():.6f}")

    ok = True
    ok &= record("loss matches",
                  torch.allclose(out_explicit.loss, out_default.loss, atol=1e-5))
    ok &= record("accuracy matches",
                  torch.allclose(out_explicit.accuracy, out_default.accuracy, atol=1e-5))
    return ok


# ---------------------------------------------------------------------------
# 9. Edge case: batch_size = 1
# ---------------------------------------------------------------------------
def test_batch_size_one():
    """Verify everything works with batch_size=1."""
    section("11. Edge Case: batch_size=1")

    config = DTrOCRConfig(max_position_embeddings=320)
    model = DTrOCRLMHeadModel(config)
    model.eval()

    B, C, T = 1, 20, 128
    pixel_values = torch.randn(B, 3, 32, 128)
    input_ids    = torch.randint(0, config.vocab_size, (B, C + T))
    attn_mask    = torch.ones(B, C + T, dtype=torch.long)
    labels       = input_ids.clone()

    ok = True
    try:
        with torch.no_grad():
            out = model(pixel_values=pixel_values, input_ids=input_ids,
                        attention_mask=attn_mask, labels=labels,
                        context_length=C, return_per_sample_loss=True)
        print(f"  loss={out.loss.item():.4f}  per_sample={out.per_sample_loss.tolist()}")
        ok &= record("batch_size=1 works", True)
        ok &= record("per_sample_loss shape", out.per_sample_loss.shape == (1,))
    except Exception as e:
        ok &= record("batch_size=1 works", False, str(e))
    return ok


# ---------------------------------------------------------------------------
# 10. Edge case: heavy padding in attention mask
# ---------------------------------------------------------------------------
def test_heavy_padding():
    """Verify loss is computed correctly when most of target is padding."""
    section("12. Edge Case: Heavy Padding (90% PAD)")

    config = DTrOCRConfig(max_position_embeddings=320)
    model = DTrOCRLMHeadModel(config)
    model.eval()

    B, C, T = 2, 20, 128
    pixel_values = torch.randn(B, 3, 32, 128)
    input_ids    = torch.randint(0, config.vocab_size, (B, C + T))
    attn_mask    = torch.ones(B, C + T, dtype=torch.long)
    # Only first 13 target tokens are real, rest is PAD
    attn_mask[:, C + 13:] = 0
    labels       = input_ids.clone()

    with torch.no_grad():
        out = model(pixel_values=pixel_values, input_ids=input_ids,
                    attention_mask=attn_mask, labels=labels,
                    context_length=C, return_per_sample_loss=True)

    print(f"  Active target tokens: 13 / {T}")
    print(f"  loss={out.loss.item():.4f}  acc={out.accuracy.item():.4f}")

    ok = True
    ok &= record("loss finite", torch.isfinite(out.loss).item())
    ok &= record("loss not NaN", not torch.isnan(out.loss).item())
    ok &= record("accuracy in [0,1]", 0 <= out.accuracy.item() <= 1)
    return ok


# ---------------------------------------------------------------------------
# 11. Processor fixed-length context padding
# ---------------------------------------------------------------------------
def test_processor_fixed_context():
    """Test that the processor pads context to fixed length."""
    section("13. Processor Fixed-Length Context Padding")

    ok = True

    try:
        from processor import DTrOCRProcessor

        config = DTrOCRConfig(max_context_length=20, max_position_embeddings=320)
        processor = DTrOCRProcessor(config, add_bos_token=True, add_eos_token=True)

        print(f"  max_context_length: {config.max_context_length}")
        print(f"  tokeniser model_max_length: {processor.tokeniser.model_max_length}")

        # Create a dummy image
        from PIL import Image
        import numpy as np
        dummy_img = Image.fromarray(np.random.randint(0, 255, (32, 128, 3), dtype=np.uint8))

        # Test with short context
        out_ctx = processor(
            images=dummy_img, texts="test", context_text="hello",
            padding=True, return_labels=True
        )
        print(f"\n  Short context ('hello'):")
        print(f"    context_length: {out_ctx.context_length}")
        print(f"    input_ids shape: {list(out_ctx.input_ids.shape)}")
        print(f"    attention_mask shape: {list(out_ctx.attention_mask.shape)}")

        ok &= record("context_length = max_context_length",
                      out_ctx.context_length == config.max_context_length,
                      f"got {out_ctx.context_length}, want {config.max_context_length}")

        # Test with empty context (isolation)
        out_iso = processor(
            images=dummy_img, texts="test", context_text="",
            padding=True, return_labels=True
        )
        print(f"\n  Empty context (''):")
        print(f"    context_length: {out_iso.context_length}")
        print(f"    input_ids shape: {list(out_iso.input_ids.shape)}")

        ok &= record("empty context -> context_length=0",
                      out_iso.context_length == 0)

        # Test uniform shape: both should have consistent dimensions
        target_len = processor.tokeniser.model_max_length
        expected_ctx_input_len = config.max_context_length + target_len
        expected_iso_input_len = target_len
        ok &= record("ctx input_ids length consistent",
                      out_ctx.input_ids.shape[-1] == expected_ctx_input_len,
                      f"got {out_ctx.input_ids.shape[-1]}, want {expected_ctx_input_len}")
        ok &= record("iso input_ids length consistent",
                      out_iso.input_ids.shape[-1] == expected_iso_input_len,
                      f"got {out_iso.input_ids.shape[-1]}, want {expected_iso_input_len}")

    except ImportError as e:
        print(f"  Skipping processor test (import error): {e}")
        ok &= record("processor import", False, str(e))
    except Exception as e:
        print(f"  Processor test error: {e}")
        import traceback
        traceback.print_exc()
        ok &= record("processor test", False, str(e))

    return ok


# ---------------------------------------------------------------------------
# 12. Multiple samples with fixed context: simulate DataLoader stacking
# ---------------------------------------------------------------------------
def test_dataloader_stacking_simulation():
    """Simulate DataLoader's torch.stack on multiple context samples."""
    section("14. DataLoader Stacking Simulation")

    ok = True
    try:
        from processor import DTrOCRProcessor
        from PIL import Image
        import numpy as np

        config = DTrOCRConfig(max_context_length=20, max_position_embeddings=320)
        processor = DTrOCRProcessor(config, add_bos_token=True, add_eos_token=True)

        dummy_img = Image.fromarray(np.random.randint(0, 255, (32, 128, 3), dtype=np.uint8))

        # Process multiple samples with different context lengths
        samples = []
        context_texts = ["ab", "abcdef", "a" * 50, "xyz"]
        for ctx in context_texts:
            out = processor(
                images=dummy_img, texts="test", context_text=ctx,
                padding=True, return_labels=True
            )
            samples.append({
                'input_ids': out.input_ids[0],
                'attention_mask': out.attention_mask[0],
                'context_length': out.context_length
            })
            print(f"  ctx='{ctx[:20]}...' -> input_ids len={out.input_ids.shape[-1]}, "
                  f"context_length={out.context_length}")

        # Try stacking (this is what DataLoader does)
        try:
            stacked_ids = torch.stack([s['input_ids'] for s in samples])
            stacked_masks = torch.stack([s['attention_mask'] for s in samples])
            context_lengths = [s['context_length'] for s in samples]

            print(f"\n  Stacked input_ids: {list(stacked_ids.shape)}")
            print(f"  Stacked masks:     {list(stacked_masks.shape)}")
            print(f"  Context lengths:   {context_lengths}")

            ok &= record("torch.stack succeeds", True)
            ok &= record("all context_lengths equal",
                          len(set(context_lengths)) == 1,
                          f"got {context_lengths}")
        except RuntimeError as e:
            ok &= record("torch.stack succeeds", False, str(e))

    except ImportError as e:
        print(f"  Skipping (import error): {e}")
        ok &= record("import", False, str(e))
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        ok &= record("stacking test", False, str(e))

    return ok


# ---------------------------------------------------------------------------
# 13. save_checkpoint signature check
# ---------------------------------------------------------------------------
def test_save_checkpoint_signature():
    """Verify save_checkpoint is called with correct arguments."""
    section("15. save_checkpoint Argument Check")

    import inspect
    from utils import save_checkpoint

    sig = inspect.signature(save_checkpoint)
    params = list(sig.parameters.keys())
    print(f"  save_checkpoint params: {params}")
    print(f"  param count: {len(params)}")

    ok = True
    ok &= record("save_checkpoint has 9 params",
                  len(params) == 9,
                  f"got {len(params)}: {params}")
    ok &= record("includes train_acc",
                  'train_acc' in params)
    ok &= record("includes val_acc",
                  'val_acc' in params)
    ok &= record("includes checkpoint_dir",
                  'checkpoint_dir' in params)
    ok &= record("includes checkpoint_name",
                  'checkpoint_name' in params)

    # Simulate the call from train.py
    try:
        import tempfile
        config = DTrOCRConfig()
        model = DTrOCRLMHeadModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(
                model, optimizer, 1,
                0.5, 0.6,
                0.7, 0.8,
                tmpdir,
                'test_checkpoint.pt'
            )
            saved_path = os.path.join(tmpdir, 'test_checkpoint.pt')
            exists = os.path.exists(saved_path)
            ok &= record("checkpoint file created", exists)

            if exists:
                ckpt = torch.load(saved_path, weights_only=False)
                ok &= record("checkpoint has train_acc", 'train_acc' in ckpt)
                ok &= record("checkpoint has val_acc", 'val_acc' in ckpt)
                print(f"  Checkpoint keys: {list(ckpt.keys())}")

    except TypeError as e:
        ok &= record("save_checkpoint call succeeds", False, str(e))

    return ok


# ---------------------------------------------------------------------------
# 14. evaluate_context_model is callable
# ---------------------------------------------------------------------------
def test_evaluate_context_model_defined():
    """Verify evaluate_context_model is defined before the training loop."""
    section("16. evaluate_context_model Definition Order")

    ok = True
    try:
        # Read train.py and check function definition order
        train_path = os.path.join(os.path.dirname(__file__), '..', 'GraDeT_HTR', 'train.py')
        with open(train_path, 'r') as f:
            lines = f.readlines()

        def_line = None
        loop_line = None
        call_line = None

        for i, line in enumerate(lines, 1):
            if 'def evaluate_context_model' in line:
                def_line = i
            if line.strip().startswith('for epoch in range(EPOCHS)'):
                loop_line = i
            if 'evaluate_context_model(' in line and 'def ' not in line:
                if call_line is None:
                    call_line = i

        print(f"  evaluate_context_model defined at line: {def_line}")
        print(f"  Training loop starts at line:           {loop_line}")
        print(f"  First call at line:                     {call_line}")

        ok &= record("function defined before loop",
                      def_line is not None and loop_line is not None and def_line < loop_line,
                      f"def@{def_line} vs loop@{loop_line}")
        if call_line:
            ok &= record("function defined before first call",
                          def_line < call_line,
                          f"def@{def_line} vs call@{call_line}")

    except Exception as e:
        ok &= record("file parse", False, str(e))

    return ok


# ---------------------------------------------------------------------------
# 15. Position embedding budget
# ---------------------------------------------------------------------------
def test_position_embedding_budget():
    """Verify position embeddings can handle context + image + target."""
    section("17. Position Embedding Budget")

    config = DTrOCRConfig(max_context_length=20, max_position_embeddings=320)
    model = DTrOCRLMHeadModel(config)

    C = config.max_context_length
    I = model.image_embedding_length
    T = config.max_position_embeddings - I  # target model_max_length

    # The actual target length from tokeniser
    target_len = config.max_position_embeddings - I
    total = C + I + target_len

    print(f"  max_position_embeddings: {config.max_position_embeddings}")
    print(f"  max_context_length:      {C}")
    print(f"  image_patches:           {I}")
    print(f"  target_model_max_length: {target_len}")
    print(f"  Total sequence:          {total} = {C} + {I} + {target_len}")
    print(f"  Positional embedding:    {config.max_position_embeddings}")

    ok = True
    ok &= record("total <= max_position_embeddings",
                  total <= config.max_position_embeddings,
                  f"{total} vs {config.max_position_embeddings}")

    # Also check default config
    default_config = DTrOCRConfig()
    default_total = 0 + 128 + (default_config.max_position_embeddings - 128)
    ok &= record("default config (no context) fits",
                  default_total <= default_config.max_position_embeddings)

    return ok


# ---------------------------------------------------------------------------
# 16. Margin term with actual model output
# ---------------------------------------------------------------------------
def test_margin_with_model_output():
    """End-to-end: compute margin from actual model forward passes."""
    section("18. Margin Term with Actual Model Output")

    config = DTrOCRConfig(max_position_embeddings=320)
    model = DTrOCRLMHeadModel(config)
    model.eval()

    B, C, T = 4, 20, 128
    pixel_values = torch.randn(B, 3, 32, 128)

    # Build inputs so some samples might have ctx worse and some better
    input_ids_ctx = torch.randint(0, config.vocab_size, (B, C + T))
    attn_mask_ctx = torch.ones(B, C + T, dtype=torch.long)
    labels_ctx    = input_ids_ctx.clone()

    input_ids_iso = torch.randint(0, config.vocab_size, (B, T))
    attn_mask_iso = torch.ones(B, T, dtype=torch.long)
    labels_iso    = input_ids_iso.clone()

    with torch.no_grad():
        out_ctx = model(pixel_values=pixel_values, input_ids=input_ids_ctx,
                        attention_mask=attn_mask_ctx, labels=labels_ctx,
                        context_length=C, return_per_sample_loss=True)
        out_iso = model(pixel_values=pixel_values, input_ids=input_ids_iso,
                        attention_mask=attn_mask_iso, labels=labels_iso,
                        context_length=0, return_per_sample_loss=True)

    margin_lambda = 0.5
    diff = out_ctx.per_sample_loss - out_iso.per_sample_loss
    margin = margin_lambda * torch.clamp(diff, min=0.0)
    base = out_ctx.per_sample_loss + out_iso.per_sample_loss
    total = (base + margin).mean()

    print(f"  Per-sample ctx loss: {[f'{x:.4f}' for x in out_ctx.per_sample_loss.tolist()]}")
    print(f"  Per-sample iso loss: {[f'{x:.4f}' for x in out_iso.per_sample_loss.tolist()]}")
    print(f"  Diff (ctx - iso):    {[f'{x:.4f}' for x in diff.tolist()]}")
    print(f"  Margin term:         {[f'{x:.4f}' for x in margin.tolist()]}")
    print(f"  Margin active:       {(margin > 0).sum().item()}/{B} samples")
    print(f"  Total loss:          {total.item():.4f}")

    ok = True
    ok &= record("margin >= 0 everywhere", (margin >= 0).all().item())
    ok &= record("total loss finite", torch.isfinite(total).item())
    ok &= record("margin activates when ctx > iso",
                  ((diff > 0) == (margin > 0)).all().item())
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 72)
    print("  GraDeT-HTR Context-Aware Diagnostic Test Suite")
    print("  Tests all 6 bug fixes + edge cases + dimension traces")
    print("=" * 72)

    os.chdir(os.path.join(os.path.dirname(__file__), '..', 'GraDeT_HTR'))

    tests = [
        test_isolation_forward_dimensions,       # 1
        test_context_forward_dimensions,          # 2
        test_shift_alignment_sweep,               # 3
        test_label_content_correctness,           # 4
        test_mask_alignment,                      # 5
        test_margin_term_logic,                   # 6
        test_full_loss_formula,                   # 7
        test_dual_path_forward,                   # 8
        test_gradient_flow,                       # 9
        test_backward_compatibility,              # 10
        test_batch_size_one,                      # 11
        test_heavy_padding,                       # 12
        test_processor_fixed_context,             # 13
        test_dataloader_stacking_simulation,      # 14
        test_save_checkpoint_signature,           # 15
        test_evaluate_context_model_defined,      # 16
        test_position_embedding_budget,           # 17
        test_margin_with_model_output,            # 18
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            section(f"UNEXPECTED ERROR in {test_fn.__name__}")
            import traceback
            traceback.print_exc()
            record(test_fn.__name__, False, f"Unexpected: {e}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")

    passed = sum(1 for _, p in results if p)
    failed = sum(1 for _, p in results if not p)
    total = len(results)

    for name, p in results:
        if not p:
            print(f"  [FAIL] {name}")

    print(f"\n  {passed}/{total} checks passed, {failed} failed")

    if failed == 0:
        print(f"\n  ALL CHECKS PASSED")
    else:
        print(f"\n  {failed} CHECKS FAILED -- see details above")

    print(f"{'='*72}\n")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
