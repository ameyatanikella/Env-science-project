"""
RNA 3D Structure Prediction Model.

A transformer-based architecture for predicting 3D C1' atom coordinates
from RNA nucleotide sequences. Inspired by AlphaFold's structure module
and RhoFold's approach.

Architecture:
1. Sequence Embedding: nucleotide tokens + positional encoding
2. Pairwise Representation: outer product of single representations
3. Transformer Encoder: self-attention over sequence positions
4. Structure Module: predicts 3D coordinates via IPA-inspired attention
5. Multi-prediction Head: outputs 5 coordinate sets for ensemble scoring
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .dataset import VOCAB_SIZE


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PairwiseModule(nn.Module):
    """
    Computes pairwise representations from single representations
    using outer product mean, then refines with triangle attention.
    """

    def __init__(self, d_model: int, d_pair: int = 64):
        super().__init__()
        self.proj_left = nn.Linear(d_model, d_pair)
        self.proj_right = nn.Linear(d_model, d_pair)
        self.pair_norm = nn.LayerNorm(d_pair)
        self.pair_mlp = nn.Sequential(
            nn.Linear(d_pair, d_pair * 2),
            nn.GELU(),
            nn.Linear(d_pair * 2, d_pair),
        )
        self.pair_to_bias = nn.Linear(d_pair, 1)

    def forward(self, single_repr: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            single_repr: (B, L, d_model)
            mask: (B, L)
        Returns:
            pair_repr: (B, L, L, d_pair)
            attn_bias: (B, L, L) - bias for self-attention
        """
        left = self.proj_left(single_repr)   # (B, L, d_pair)
        right = self.proj_right(single_repr)  # (B, L, d_pair)

        # Outer product: (B, L, d_pair) x (B, L, d_pair) -> (B, L, L, d_pair)
        pair_repr = torch.einsum("bid,bjd->bijd", left, right)
        pair_repr = self.pair_norm(pair_repr)
        pair_repr = pair_repr + self.pair_mlp(pair_repr)

        # Mask pair representation
        pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (B, L, L)
        pair_repr = pair_repr * pair_mask.unsqueeze(-1)

        attn_bias = self.pair_to_bias(pair_repr).squeeze(-1)  # (B, L, L)
        return pair_repr, attn_bias


class StructureAwareAttention(nn.Module):
    """
    Multi-head self-attention with pairwise bias.
    Incorporates structural information via pair representations.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        B, L, D = x.shape

        q = rearrange(self.q_proj(x), "b l (h d) -> b h l d", h=self.n_heads)
        k = rearrange(self.k_proj(x), "b l (h d) -> b h l d", h=self.n_heads)
        v = rearrange(self.v_proj(x), "b l (h d) -> b h l d", h=self.n_heads)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)

        if attn_bias is not None:
            attn = attn + attn_bias.unsqueeze(1)  # broadcast over heads

        # Mask: set padding positions to -inf
        mask_2d = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        attn = attn.masked_fill(~mask_2d, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with structure-aware attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = StructureAwareAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask, attn_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class StructureModule(nn.Module):
    """
    Converts sequence representations to 3D coordinates.

    Uses an MLP to predict per-residue local frames and coordinates,
    inspired by AlphaFold2's structure module but simplified.
    """

    def __init__(self, d_model: int, num_predictions: int = 5):
        super().__init__()
        self.num_predictions = num_predictions

        # Per-prediction coordinate heads
        self.coord_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 3),  # x, y, z
            )
            for _ in range(num_predictions)
        ])

        # Confidence head (pLDDT-like per-residue confidence)
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_predictions),
            nn.Sigmoid(),
        )

    def forward(self, single_repr: torch.Tensor) -> dict:
        """
        Args:
            single_repr: (B, L, d_model)
        Returns:
            coords: (B, L, num_predictions, 3)
            confidence: (B, L, num_predictions)
        """
        coords_list = [head(single_repr) for head in self.coord_heads]
        coords = torch.stack(coords_list, dim=2)  # (B, L, num_predictions, 3)

        confidence = self.confidence_head(single_repr)  # (B, L, num_predictions)

        return {"coords": coords, "confidence": confidence}


class RNAFoldModel(nn.Module):
    """
    End-to-end RNA 3D structure prediction model.

    Pipeline:
        Sequence -> Embedding -> Pairwise -> Transformer Encoder -> Structure Module -> 3D Coords
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        num_predictions: int = 5,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_predictions = num_predictions

        # Embedding
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len + 100)

        # Pairwise module
        self.pairwise = PairwiseModule(d_model, d_pair=64)

        # Transformer encoder
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Structure module
        self.structure_module = StructureModule(d_model, num_predictions)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'token_emb' not in name:
                nn.init.xavier_uniform_(p)
        # Re-zero the padding embedding after init
        with torch.no_grad():
            self.token_emb.weight[0].zero_()

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict:
        """
        Args:
            tokens: (B, L) integer token IDs
            mask: (B, L) boolean mask (True = valid, False = padding)
        Returns:
            dict with:
                coords: (B, L, num_predictions, 3) predicted C1' coordinates
                confidence: (B, L, num_predictions) per-residue confidence
        """
        # Embed
        x = self.token_emb(tokens)  # (B, L, d_model)
        x = self.pos_enc(x)

        # Pairwise
        _, attn_bias = self.pairwise(x, mask)

        # Transformer
        for layer in self.layers:
            x = x * mask.unsqueeze(-1).float()  # Zero out padding
            x = layer(x, mask, attn_bias)

        # Structure prediction
        output = self.structure_module(x)
        return output

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
