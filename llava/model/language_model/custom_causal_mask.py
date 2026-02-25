from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn.functional as F

from llava.constants import MODALITY_IMAGE, MODALITY_COMPRESSED_CONTEXT


# ---------------------------------------------------------------------------
# Compressed-Context (CC) restricted attention mask
# ---------------------------------------------------------------------------
"""example:
=== Visual Mask (. = attend, X = masked) ===
Q\K   T0   I1   I2   C3   I4   I5   C6   T7
 T0    O    X    X    X    X    X    X    X 
 I1    O    O    X    X    X    X    X    X 
 I2    O    O    O    X    X    X    X    X 
 C3    X    O    O    O    X    X    X    X 
 I4    O    O    O    O    O    X    X    X 
 I5    O    O    O    O    O    O    X    X 
 C6    X    X    X    X    O    O    O    X 
 T7    O    O    O    O    O    O    O    O 
"""


# ---------------------------------------------------------------------------
# Compressed-Context (CC) restricted attention mask
# ---------------------------------------------------------------------------
def build_cc_restricted_causal_mask(
    attention_mask_2d: Optional[torch.Tensor],
    modality_types: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a 4D causal attention mask where compressed-context tokens can
    only attend to their own frame's IMAGE patches and same-frame earlier CC
    tokens.  All other token types keep the standard causal mask.

    Args:
        attention_mask_2d: ``[B, S]`` with 1=valid / 0=pad, or *None*.
        modality_types:    ``[B, S]`` with values from ``llava.constants``
                           (MODALITY_TEXT / IMAGE / COMPRESSED_CONTEXT / PAD).
        dtype:  dtype of the returned mask (should match ``inputs_embeds``).
        device: device of the returned mask.

    Returns:
        ``[B, 1, S, S]`` attention mask (0 = attend, ``min_dtype`` = masked).
    """
    batch_size, seq_len = modality_types.shape
    min_dtype = torch.finfo(dtype).min

    # --- 1. Standard lower-triangular causal mask ---------------------------
    row_idx = torch.arange(seq_len, device=device)
    causal_mask = torch.where(
        row_idx.unsqueeze(1) >= row_idx.unsqueeze(0),          # [S, S]
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(min_dtype, dtype=dtype, device=device),
    )
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(
        batch_size, 1, -1, -1
    ).clone()                                                   # [B, 1, S, S]

    # --- 2. Padding mask (key positions that are pad → masked) --------------
    if attention_mask_2d is not None:
        padding_positions = (attention_mask_2d == 0)             # [B, S]
        causal_mask.masked_fill_(
            padding_positions[:, None, None, :], min_dtype,
        )

    # --- 3. Compute per-token frame IDs (vectorised) -----------------------
    #   Token order example:
    #     [TEXT … IMG IMG … CC CC IMG IMG … CC CC … TEXT …]
    #   A new frame starts whenever an IMAGE token follows a non-IMAGE token.
    is_image = (modality_types == MODALITY_IMAGE)                # [B, S]
    is_cc    = (modality_types == MODALITY_COMPRESSED_CONTEXT)   # [B, S]
    is_visual = is_image | is_cc                                 # [B, S]

    prev_is_image = F.pad(is_image[:, :-1], (1, 0), value=False)
    frame_starts  = is_image & ~prev_is_image                   # [B, S]

    # cumsum gives 1-based IDs; non-visual tokens stay 0
    frame_ids = frame_starts.long().cumsum(dim=1) * is_visual.long()  # [B, S]

    # --- 4. For CC query rows, mask out keys NOT in the same frame ----------
    #   same_frame[b,i,j] is True when frame_ids match.
    #   Non-visual tokens have frame_id=0, CC tokens have ≥1 → never equal.
    same_frame    = (frame_ids.unsqueeze(2) == frame_ids.unsqueeze(1))  # [B,S,S]
    restrict_mask = is_cc.unsqueeze(2) & ~same_frame                   # [B,S,S]

    causal_mask.masked_fill_(restrict_mask.unsqueeze(1), min_dtype)

    return causal_mask
