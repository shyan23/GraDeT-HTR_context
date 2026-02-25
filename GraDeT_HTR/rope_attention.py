"""
Custom transformer block with RoPE attention, replacing HuggingFace GPT2Block.

Architecture matches GPT-2 (pre-norm):
    x -> LayerNorm -> RoPEAttention -> + residual -> LayerNorm -> MLP -> + residual

Key difference from GPT2Block: RoPE is applied to Q and K inside attention,
instead of adding absolute positional embeddings before the transformer stack.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from rope import apply_rotary_pos_emb


class RoPEAttention(nn.Module):
    """
    Multi-head self-attention with Rotary Position Embeddings.

    Uses separate Q, K, V projections (nn.Linear) instead of GPT2's combined Conv1D.
    Applies RoPE to Q and K after projection, before attention computation.
    Compatible with KV caching for autoregressive generation.
    """

    def __init__(self, config, layer_idx: int = None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.layer_idx = layer_idx

        assert self.head_dim * self.num_heads == self.embed_dim, (
            f"hidden_size ({self.embed_dim}) must be divisible by "
            f"num_attention_heads ({self.num_heads})"
        )

        self.scale_attn_weights = config.scale_attn_weights

        # Separate Q, K, V projections
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self._attn_implementation = config._attn_implementation

    def _split_heads(self, tensor: Tensor) -> Tensor:
        """[batch, seq, hidden] -> [batch, num_heads, seq, head_dim]"""
        batch, seq_len, _ = tensor.shape
        tensor = tensor.view(batch, seq_len, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor: Tensor) -> Tensor:
        """[batch, num_heads, seq, head_dim] -> [batch, seq, hidden]"""
        tensor = tensor.transpose(1, 2).contiguous()
        batch, seq_len, _, _ = tensor.shape
        return tensor.view(batch, seq_len, self.embed_dim)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        layer_past: Optional[Tuple[Tensor, Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, ...]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            cos, sin: [batch, seq_len, head_dim] - RoPE embeddings for current positions
            layer_past: (past_key, past_value) each [batch, num_heads, past_len, head_dim]
            attention_mask: [batch, 1, seq_len, total_len] (4D causal mask) or None
            use_cache: whether to return (key, value) for caching
        """
        # Project Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Split into heads: [batch, num_heads, seq_len, head_dim]
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # Apply RoPE to Q and K BEFORE concatenating with cache
        # Past keys already have RoPE applied (rotated when first computed)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Concatenate with past KV cache
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None

        # Compute attention via SDPA
        if self._attn_implementation == "sdpa":
            is_causal = attention_mask is None and query.shape[-2] > 1
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            # Manual attention fallback
            attn_weights = torch.matmul(query, key.transpose(-1, -2))
            if self.scale_attn_weights:
                attn_weights = attn_weights / (self.head_dim ** 0.5)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, value)

        # Merge heads and project output
        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return (attn_output, present)


class RoPEMLP(nn.Module):
    """Feed-forward network, equivalent to GPT2MLP."""

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.c_fc = nn.Linear(hidden_size, inner_dim)
        self.c_proj = nn.Linear(inner_dim, hidden_size)
        self.act = nn.GELU(approximate="tanh")  # matches GPT-2's gelu_new
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class RoPEBlock(nn.Module):
    """
    Transformer block with RoPE attention. Replaces GPT2Block.

    Pre-norm structure (same as GPT-2):
        x -> LayerNorm -> RoPEAttention(cos, sin) -> + residual
          -> LayerNorm -> MLP -> + residual
    """

    def __init__(self, config, layer_idx: int = None):
        super().__init__()
        hidden_size = config.hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = RoPEAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = RoPEMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        layer_past: Optional[Tuple[Tensor, Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, ...]:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            cos=cos,
            sin=sin,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )
        attn_output = attn_outputs[0]
        present = attn_outputs[1]
        hidden_states = attn_output + residual

        # Feed-forward with residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_output = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_output

        if use_cache:
            return (hidden_states, present)
        else:
            return (hidden_states,)
