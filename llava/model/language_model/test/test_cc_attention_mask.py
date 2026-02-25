"""Unit tests for build_cc_restricted_causal_mask().

Run:
    python -m llava.model.language_model.test.test_cc_attention_mask
"""

import torch
import sys
import os

# Ensure the project root is on sys.path so `llava.*` resolves correctly.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from llava.constants import (
    MODALITY_TEXT,
    MODALITY_IMAGE,
    MODALITY_COMPRESSED_CONTEXT,
    MODALITY_PAD,
)
from llava.model.language_model.custom_causal_mask import build_cc_restricted_causal_mask

# Short aliases
T, I, C, P = MODALITY_TEXT, MODALITY_IMAGE, MODALITY_COMPRESSED_CONTEXT, MODALITY_PAD


# =====================================================================
# Helpers
# =====================================================================

def _can_attend(mask_4d, batch, query, key):
    """True if query position can attend to key position (not masked)."""
    val = mask_4d[batch, 0, query, key].item()
    return val > -1e30  # 0.0 = attend, min_dtype ≈ -3.4e38 = masked


# =====================================================================
# Tests
# =====================================================================

def test_single_frame_basic():
    """Single image frame: [TEXT, IMG, IMG, IMG, CC, CC, TEXT, TEXT]
    CC at pos 4,5 should only attend to IMG(1,2,3) + CC within same frame.
    """
    modality = torch.tensor([[T, I, I, I, C, C, T, T]])
    attn_2d = torch.ones(1, 8, dtype=torch.long)
    mask = build_cc_restricted_causal_mask(attn_2d, modality, torch.float32, torch.device("cpu"))

    assert mask.shape == (1, 1, 8, 8), f"Expected (1,1,8,8), got {mask.shape}"

    # -- CC at pos 4 --
    assert not _can_attend(mask, 0, 4, 0), "CC@4 should NOT attend to TEXT@0"
    assert     _can_attend(mask, 0, 4, 1), "CC@4 should attend to IMG@1 (same frame)"
    assert     _can_attend(mask, 0, 4, 2), "CC@4 should attend to IMG@2 (same frame)"
    assert     _can_attend(mask, 0, 4, 3), "CC@4 should attend to IMG@3 (same frame)"
    assert     _can_attend(mask, 0, 4, 4), "CC@4 should attend to self"
    assert not _can_attend(mask, 0, 4, 5), "CC@4 should NOT attend to CC@5 (causal)"

    # -- CC at pos 5 --
    assert not _can_attend(mask, 0, 5, 0), "CC@5 should NOT attend to TEXT@0"
    assert     _can_attend(mask, 0, 5, 1), "CC@5 should attend to IMG@1"
    assert     _can_attend(mask, 0, 5, 4), "CC@5 should attend to CC@4 (same frame)"
    assert     _can_attend(mask, 0, 5, 5), "CC@5 should attend to self"

    # -- TEXT tokens keep normal causal mask --
    assert     _can_attend(mask, 0, 6, 0), "TEXT@6 should attend to TEXT@0 (causal)"
    assert     _can_attend(mask, 0, 6, 4), "TEXT@6 should attend to CC@4 (causal)"
    assert     _can_attend(mask, 0, 7, 6), "TEXT@7 should attend to TEXT@6 (causal)"
    assert not _can_attend(mask, 0, 6, 7), "TEXT@6 should NOT attend to TEXT@7 (future)"

    # -- IMG tokens keep normal causal mask --
    assert     _can_attend(mask, 0, 2, 0), "IMG@2 should attend to TEXT@0 (causal)"
    assert     _can_attend(mask, 0, 2, 1), "IMG@2 should attend to IMG@1 (causal)"
    assert not _can_attend(mask, 0, 2, 3), "IMG@2 should NOT attend to IMG@3 (future)"

    print("PASSED: test_single_frame_basic")


def test_multi_frame():
    """Two video frames:
    [TEXT, IMG, IMG, CC, IMG, IMG, CC, TEXT]
      0     1    2   3    4    5   6    7
    Frame 1: pos 1,2 (IMG) + 3 (CC)
    Frame 2: pos 4,5 (IMG) + 6 (CC)
    """
    modality = torch.tensor([[T, I, I, C, I, I, C, T]])
    attn_2d = torch.ones(1, 8, dtype=torch.long)
    mask = build_cc_restricted_causal_mask(attn_2d, modality, torch.float32, torch.device("cpu"))

    # -- CC@3 (frame 1): attend to IMG@1,2 and self --
    assert not _can_attend(mask, 0, 3, 0), "CC@3 should NOT attend to TEXT@0"
    assert     _can_attend(mask, 0, 3, 1), "CC@3 should attend to IMG@1 (frame 1)"
    assert     _can_attend(mask, 0, 3, 2), "CC@3 should attend to IMG@2 (frame 1)"
    assert     _can_attend(mask, 0, 3, 3), "CC@3 should attend to self"

    # -- CC@6 (frame 2): attend to IMG@4,5 and self, NOT to frame 1 --
    assert not _can_attend(mask, 0, 6, 0), "CC@6 should NOT attend to TEXT@0"
    assert not _can_attend(mask, 0, 6, 1), "CC@6 should NOT attend to IMG@1 (frame 1)"
    assert not _can_attend(mask, 0, 6, 2), "CC@6 should NOT attend to IMG@2 (frame 1)"
    assert not _can_attend(mask, 0, 6, 3), "CC@6 should NOT attend to CC@3 (frame 1)"
    assert     _can_attend(mask, 0, 6, 4), "CC@6 should attend to IMG@4 (frame 2)"
    assert     _can_attend(mask, 0, 6, 5), "CC@6 should attend to IMG@5 (frame 2)"
    assert     _can_attend(mask, 0, 6, 6), "CC@6 should attend to self"

    # -- TEXT@7 keeps normal causal (sees everything before it) --
    for j in range(8):
        assert _can_attend(mask, 0, 7, j), f"TEXT@7 should attend to pos {j}"

    print("PASSED: test_multi_frame")


def test_padding():
    """Padding tokens should be masked as keys.
    [IMG, IMG, CC, TEXT, PAD, PAD]
    """
    modality = torch.tensor([[I, I, C, T, P, P]])
    attn_2d = torch.tensor([[1, 1, 1, 1, 0, 0]])
    mask = build_cc_restricted_causal_mask(attn_2d, modality, torch.float32, torch.device("cpu"))

    # PAD positions should be masked for all queries
    for q in range(4):
        assert not _can_attend(mask, 0, q, 4), f"pos {q} should NOT attend to PAD@4"
        assert not _can_attend(mask, 0, q, 5), f"pos {q} should NOT attend to PAD@5"

    # CC@2 should attend to same-frame IMG
    assert     _can_attend(mask, 0, 2, 0), "CC@2 should attend to IMG@0"
    assert     _can_attend(mask, 0, 2, 1), "CC@2 should attend to IMG@1"
    assert     _can_attend(mask, 0, 2, 2), "CC@2 should attend to self"
    assert not _can_attend(mask, 0, 2, 3), "CC@2 should NOT attend to TEXT@3 (future)"

    print("PASSED: test_padding")


def test_batch():
    """Batch of 2 with different layouts.
    Batch 0: [TEXT, IMG, CC, TEXT]  (1 frame)
    Batch 1: [IMG, CC, IMG, CC]    (2 frames)
    """
    modality = torch.tensor([
        [T, I, C, T],
        [I, C, I, C],
    ])
    attn_2d = torch.ones(2, 4, dtype=torch.long)
    mask = build_cc_restricted_causal_mask(attn_2d, modality, torch.float32, torch.device("cpu"))

    # Batch 0: CC@2 → IMG@1 only (not TEXT@0)
    assert not _can_attend(mask, 0, 2, 0), "B0 CC@2 should NOT attend to TEXT@0"
    assert     _can_attend(mask, 0, 2, 1), "B0 CC@2 should attend to IMG@1"
    assert     _can_attend(mask, 0, 2, 2), "B0 CC@2 should attend to self"

    # Batch 1: CC@1 (frame 1) → IMG@0 only
    assert     _can_attend(mask, 1, 1, 0), "B1 CC@1 should attend to IMG@0 (frame 1)"
    assert     _can_attend(mask, 1, 1, 1), "B1 CC@1 should attend to self"

    # Batch 1: CC@3 (frame 2) → IMG@2 only, NOT frame 1
    assert not _can_attend(mask, 1, 3, 0), "B1 CC@3 should NOT attend to IMG@0 (frame 1)"
    assert not _can_attend(mask, 1, 3, 1), "B1 CC@3 should NOT attend to CC@1 (frame 1)"
    assert     _can_attend(mask, 1, 3, 2), "B1 CC@3 should attend to IMG@2 (frame 2)"
    assert     _can_attend(mask, 1, 3, 3), "B1 CC@3 should attend to self"

    print("PASSED: test_batch")


def test_no_cc_tokens():
    """When there are no CC tokens, mask should be standard causal."""
    modality = torch.tensor([[T, I, I, T]])
    attn_2d = torch.ones(1, 4, dtype=torch.long)
    mask = build_cc_restricted_causal_mask(attn_2d, modality, torch.float32, torch.device("cpu"))

    for i in range(4):
        for j in range(4):
            if j <= i:
                assert _can_attend(mask, 0, i, j), f"pos {i} should attend to pos {j}"
            else:
                assert not _can_attend(mask, 0, i, j), f"pos {i} should NOT attend to pos {j}"

    print("PASSED: test_no_cc_tokens")


def test_visual_mask_printout():
    """Print a visual representation for manual inspection.
    [TEXT, IMG, IMG, CC, IMG, IMG, CC, TEXT]
    """
    labels = ["T0", "I1", "I2", "C3", "I4", "I5", "C6", "T7"]
    modality = torch.tensor([[T, I, I, C, I, I, C, T]])
    attn_2d = torch.ones(1, 8, dtype=torch.long)
    mask = build_cc_restricted_causal_mask(attn_2d, modality, torch.float32, torch.device("cpu"))

    print("\n=== Visual Mask (. = attend, X = masked) ===")
    header = "Q\\K  " + "  ".join(f"{l:>3}" for l in labels)
    print(header)
    for i in range(8):
        row = f"{labels[i]:>3}  "
        for j in range(8):
            row += "  . " if _can_attend(mask, 0, i, j) else "  X "
        print(row)
    print()


if __name__ == "__main__":
    test_single_frame_basic()
    test_multi_frame()
    test_padding()
    test_batch()
    test_no_cc_tokens()
    test_visual_mask_printout()
    print("\nAll tests passed!")
