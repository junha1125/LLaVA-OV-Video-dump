import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import Qwen2_5_VLConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from llava.utils import rank0_print, rank_print

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states

def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed

class VisualRotaryEmbedding(nn.Module):
    """
    Additive 3D sinusoidal PE for visual features (temporal + height + width).

    Added to visual features BEFORE they enter the LLM, complementary to
    the LLM's own 1D RoPE (which remains untouched).

    Hidden dimension is split into three sections:
        [ temporal section | height section | width section ]
    Each section receives sinusoidal PE from its corresponding axis.

    For video: temporal positions = raw frame indices (e.g. [0, 25, 50, 75])
               → frames far apart in time get far-apart PE (like QwenVL's M-RoPE)
    For image: temporal positions = 0, only spatial PE is active

    Args:
        hidden_size: LLM hidden dim (e.g. 1536, 3584)
        base:        frequency base, same role as RoPE theta (default: 10000.0)
        use_gate:    if True, gate = tanh(learnable_scalar) controls PE strength.
                     Initialized to 0 → tanh(0) = 0 → PE starts inactive.
                     Preserves pretrained weight compatibility during early training.
    """

    def __init__(self, hidden_size, base=10000.0, use_gate=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.base = base
        self.use_gate = use_gate

        # Gate: sigmoid(gate_param) scales PE contribution, always in (0, 1)
        # init -4 → sigmoid(-4)≈0.018 → PE nearly off at start → learns to open toward 1
        if use_gate:
            self.gate_param = nn.Parameter(torch.tensor([-4.0]))

        # Split hidden_size into 3 sections: temporal, height, width
        s = hidden_size // 3
        self.section_sizes = [s, s, hidden_size - 2 * s]

        # Inverse frequencies for each section (like RoPE's inv_freq)
        for i, sec_size in enumerate(self.section_sizes):
            half = sec_size // 2
            inv_freq = 1.0 / (base ** (torch.arange(0, half).float() / half))
            self.register_buffer(f"inv_freq_{i}", inv_freq, persistent=False)

    def _sinusoidal_pe(self, positions, inv_freq, target_dim):
        """
        Build sinusoidal PE vectors from positions and frequencies.

        Args:
            positions:  [N] position values (float)
            inv_freq:   [half_dim] inverse frequencies
            target_dim: desired output dimension for this section

        Returns:
            [N, target_dim] sinusoidal PE vectors
        """
        # [N, 1] × [1, half_dim] → [N, half_dim]
        angles = positions.unsqueeze(-1).float() * inv_freq.unsqueeze(0).float()
        pe = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [N, half_dim*2]

        # Adjust to target_dim (handles odd section sizes)
        if pe.shape[-1] < target_dim:
            pad = torch.zeros(
                *pe.shape[:-1], target_dim - pe.shape[-1],
                device=pe.device, dtype=pe.dtype,
            )
            pe = torch.cat([pe, pad], dim=-1)
        elif pe.shape[-1] > target_dim:
            pe = pe[..., :target_dim]

        return pe

    def forward(self, features, frame_indices=None, grid_h=None, grid_w=None):
        """
        Add 3D positional embedding to visual features.

        Args:
            features:      [num_frames, num_patches, hidden_dim]
                           video  → num_frames = number of sampled frames
                           image  → num_frames = 1 (or num_crops)
            frame_indices: list[int] of raw frame indices from the video decoder.
                           e.g. [0, 25, 50, 75, 100, 125]
                           None for images (all temporal positions become 0).
            grid_h:        spatial grid height  (default: sqrt(num_patches))
            grid_w:        spatial grid width   (default: sqrt(num_patches))

        Returns:
            [num_frames, num_patches, hidden_dim]  features + PE
        """
        num_frames, num_patches, hidden_dim = features.shape
        device = features.device

        # --- Infer spatial grid ---
        if grid_h is None or grid_w is None:
            grid_h = grid_w = int(math.sqrt(num_patches))
            assert grid_h * grid_w == num_patches, (
                f"num_patches={num_patches} is not a perfect square"
            )

        # --- Temporal positions ---
        if frame_indices is not None:
            t_pos = torch.tensor(frame_indices, dtype=torch.float32, device=device)
        else:
            # Image: no temporal variation
            t_pos = torch.zeros(num_frames, dtype=torch.float32, device=device)

        # --- Spatial positions ---
        h_pos = torch.arange(grid_h, dtype=torch.float32, device=device)
        w_pos = torch.arange(grid_w, dtype=torch.float32, device=device)

        # --- Build PE per axis ---
        t_pe = self._sinusoidal_pe(t_pos, self.inv_freq_0, self.section_sizes[0])  # [F, d_t]
        h_pe = self._sinusoidal_pe(h_pos, self.inv_freq_1, self.section_sizes[1])  # [H, d_h]
        w_pe = self._sinusoidal_pe(w_pos, self.inv_freq_2, self.section_sizes[2])  # [W, d_w]

        # --- Broadcast to [F, H, W, section_dim] then concatenate ---
        t_pe = t_pe[:, None, None, :].expand(num_frames, grid_h, grid_w, -1)
        h_pe = h_pe[None, :, None, :].expand(num_frames, grid_h, grid_w, -1)
        w_pe = w_pe[None, None, :, :].expand(num_frames, grid_h, grid_w, -1)

        pe_3d = torch.cat([t_pe, h_pe, w_pe], dim=-1)   # [F, H, W, hidden_dim]
        pe_3d = pe_3d.reshape(num_frames, num_patches, hidden_dim)

        if self.use_gate:
            gate = torch.sigmoid(self.gate_param)  # scalar in (0, 1), starts at ~0.018
            rank0_print(f"[VisualRoPE] gate_param={self.gate_param.item():.4f}, gate(sigmoid)={gate.item():.4f}, requires_grad={self.gate_param.requires_grad}")
            return features + gate * pe_3d.to(features.dtype)
        else:
            return features + pe_3d.to(features.dtype)


class Qwen2_5_VLRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen2_5_VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
