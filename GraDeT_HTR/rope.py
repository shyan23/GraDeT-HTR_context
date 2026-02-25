"""
Rotary Position Embedding (RoPE) implementation.

Based on: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
(Su et al., 2023) - arXiv:2104.09864v5

Core idea: encode position by rotating Q and K vectors so that their dot product
depends on relative position (m-n) through rotation matrix R^d_{Θ,n-m}.

Efficient realization (Eq. 34 from paper):
    R^d_{Θ,m} x = x ⊗ cos(mΘ) + rotate_half(x) ⊗ sin(mΘ)

where Θ = {θ_i = base^{-2(i-1)/d}, i ∈ [1, ..., d/2]} and d = head_dim.
"""

import torch
from torch import nn, Tensor
from typing import Tuple


class RotaryEmbedding(nn.Module):
    """
    Precomputes inverse frequencies and generates cos/sin embeddings for given positions.

    inv_freq = 1 / (base^(2i/dim)) for i in [0, dim/2)
    Given position_ids, computes:
        freqs = outer_product(position_ids, inv_freq)
        cos = cos(freqs), sin = sin(freqs)
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 320,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # θ_i = base^(-2i/dim) for i in [0, dim/2)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: Tensor,
        position_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: any tensor, used only for device/dtype inference
            position_ids: [batch, seq_len] or [1, seq_len]

        Returns:
            cos: [batch, seq_len, head_dim]
            sin: [batch, seq_len, head_dim]
        """
        # inv_freq_expanded: [batch, dim/2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, -1
        )
        # position_ids_expanded: [batch, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()

        # Disable autocast for numerical precision
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # freqs: [batch, dim/2, seq_len] -> [batch, seq_len, dim/2]
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            # Duplicate for rotate_half pattern: [batch, seq_len, dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[Tensor, Tensor]:
    """
    Applies RoPE to query and key tensors (Eq. 34 from paper).

    Args:
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_heads, seq_len, head_dim]
        cos: [batch, seq_len, head_dim]
        sin: [batch, seq_len, head_dim]
        unsqueeze_dim: dimension to unsqueeze for broadcasting over num_heads

    Returns:
        q_embed, k_embed: rotated Q and K with same shapes as input
    """
    cos = cos.unsqueeze(unsqueeze_dim)  # [batch, 1, seq_len, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)  # [batch, 1, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
