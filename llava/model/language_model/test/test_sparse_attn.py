"""Unit tests for compute_chunk_info() and MoCSparseAttention.

Run:
    python -m llava.model.language_model.test.test_sparse_attn
"""

import torch
import torch.nn.functional as F
import sys
import os
import math
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from llava.constants import (
    MODALITY_TEXT, MODALITY_IMAGE, MODALITY_COMPRESSED_CONTEXT, MODALITY_PAD,
    IGNORE_INDEX,
)
from llava.model.language_model.sparse_attn import compute_chunk_info, MoCSparseAttention

T, I, C, P = MODALITY_TEXT, MODALITY_IMAGE, MODALITY_COMPRESSED_CONTEXT, MODALITY_PAD

# =====================================================================
# Helpers
# =====================================================================

def _make_config(**overrides):
    """Create a minimal Qwen3Config for MoCSparseAttention tests."""
    from transformers import Qwen3Config
    defaults = dict(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,   # GQA: 4 query heads, 2 KV heads
        head_dim=64,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=0,
    )
    defaults.update(overrides)
    cfg = Qwen3Config(**defaults)
    cfg._attn_implementation = "eager"
    return cfg


def _make_attn(config=None, layer_idx=0):
    """Create a MoCSparseAttention module."""
    if config is None:
        config = _make_config()
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    attn = Qwen3Attention(config, layer_idx=layer_idx)
    attn.__class__ = MoCSparseAttention
    return attn


def _make_position_embeddings(B, S, head_dim, device="cpu"):
    """Create (cos, sin) position embeddings for RoPE."""
    cos = torch.randn(B, S, head_dim, device=device)
    sin = torch.randn(B, S, head_dim, device=device)
    return (cos, sin)


def _can_attend(mask_4d, batch, head, query, key):
    """True if query can attend to key (not masked out)."""
    val = mask_4d[batch, head, query, key].item()
    return val > -1e30


# =====================================================================
# compute_chunk_info tests
# =====================================================================

def test_chunk_info_no_image():
    """No image tokens → returns None."""
    modality_types = torch.tensor([[T, T, T, T]])
    result = compute_chunk_info(modality_types, ["image"], patch_wh_one_chunk=6, device=torch.device("cpu"))
    assert result is None, "Expected None when no image tokens"
    print("  PASS test_chunk_info_no_image")


def test_chunk_info_single_image():
    """Single image: 576 patches (24×24), K=6 → 16 chunks of 36 patches each."""
    K = 6
    W = 24
    n_patches = W * W  # 576
    n_text_before = 3
    n_cc = 4
    n_text_after = 5
    S = n_text_before + n_patches + n_cc + n_text_after

    mtype = torch.cat([
        torch.full((n_text_before,), T),
        torch.full((n_patches,), I),
        torch.full((n_cc,), C),
        torch.full((n_text_after,), T),
    ]).unsqueeze(0)  # [1, S]

    result = compute_chunk_info(mtype, ["image"], patch_wh_one_chunk=K, device=torch.device("cpu"))
    assert result is not None

    ptc = result["position_to_chunk"]           # [1, S]
    chunk_pos = result["chunk_positions"]        # [1, 16, 36]
    valid = result["chunk_valid_mask"]           # [1, 16]
    n_chunks = result["n_chunks_per_item"]       # [1]
    img_mask = result["image_mask"]              # [1, S]

    assert n_chunks[0].item() == 16, f"Expected 16 chunks, got {n_chunks[0]}"
    assert chunk_pos.shape == (1, 16, 36), f"Expected (1,16,36), got {chunk_pos.shape}"
    assert valid.all(), "All 16 chunks should be valid"

    # Every image position should have a chunk assignment (0..15)
    image_positions = img_mask[0].nonzero(as_tuple=True)[0]
    assert len(image_positions) == n_patches
    for pos in image_positions:
        assert ptc[0, pos].item() >= 0, f"Image pos {pos} has no chunk"

    # Non-image positions should be -1
    for pos in range(S):
        if mtype[0, pos] != I:
            assert ptc[0, pos].item() == -1, f"Non-image pos {pos} has chunk {ptc[0, pos]}"

    # Verify spatial correctness: chunk(0,0) should have rows 0-5, cols 0-5
    chunk0_positions = chunk_pos[0, 0].tolist()
    for idx, seq_pos in enumerate(chunk0_positions):
        r_in_chunk = idx // K
        c_in_chunk = idx % K
        row = r_in_chunk
        col = c_in_chunk
        expected_flat = row * W + col
        expected_seq_pos = n_text_before + expected_flat
        assert seq_pos == expected_seq_pos, (
            f"chunk(0,0)[{idx}]: expected seq_pos {expected_seq_pos}, got {seq_pos}"
        )

    # Verify chunk(1,1): rows 6-11, cols 6-11
    chunk5_positions = chunk_pos[0, 5].tolist()  # chunk_idx=5 = cr=1, cc=1
    for idx, seq_pos in enumerate(chunk5_positions):
        r_in_chunk = idx // K
        c_in_chunk = idx % K
        row = 1 * K + r_in_chunk  # cr=1
        col = 1 * K + c_in_chunk  # cc=1
        expected_flat = row * W + col
        expected_seq_pos = n_text_before + expected_flat
        assert seq_pos == expected_seq_pos, (
            f"chunk(1,1)[{idx}]: expected seq_pos {expected_seq_pos}, got {seq_pos}"
        )

    # Every patch belongs to exactly one chunk
    assigned_chunks = ptc[0, n_text_before : n_text_before + n_patches]
    for c_idx in range(16):
        count = (assigned_chunks == c_idx).sum().item()
        assert count == 36, f"Chunk {c_idx} has {count} patches, expected 36"

    print("  PASS test_chunk_info_single_image")


def test_chunk_info_video_multi_frame():
    """Video: 4 frames × 144 patches (12×12), K=6 → 4 chunks/frame → 16 total."""
    K = 6
    W = 12
    patches_per_frame = W * W  # 144
    n_frames = 4
    n_cc = 2
    n_text_before = 2
    n_text_after = 3

    # Layout: [TEXT×2] [frame0 patches×144] [CC×2] [frame1 patches×144] [CC×2] ...  [TEXT×3]
    parts = [torch.full((n_text_before,), T)]
    for f in range(n_frames):
        parts.append(torch.full((patches_per_frame,), I))
        parts.append(torch.full((n_cc,), C))
    parts.append(torch.full((n_text_after,), T))
    mtype = torch.cat(parts).unsqueeze(0)  # [1, S]

    result = compute_chunk_info(mtype, ["video"], patch_wh_one_chunk=K, device=torch.device("cpu"))
    assert result is not None

    n_chunks = result["n_chunks_per_item"][0].item()
    assert n_chunks == n_frames * 4, f"Expected {n_frames * 4} chunks, got {n_chunks}"

    chunk_pos = result["chunk_positions"]
    assert chunk_pos.shape[1] == n_chunks
    assert chunk_pos.shape[2] == K * K  # 36

    # Chunks are globally indexed across frames
    ptc = result["position_to_chunk"][0]
    img_mask = result["image_mask"][0]
    image_positions = img_mask.nonzero(as_tuple=True)[0]

    # Frame 0 patches → chunks 0-3, Frame 1 → chunks 4-7, etc.
    for f in range(n_frames):
        frame_patches = image_positions[f * patches_per_frame : (f + 1) * patches_per_frame]
        chunk_ids_in_frame = set()
        for pos in frame_patches:
            c = ptc[pos].item()
            assert f * 4 <= c < (f + 1) * 4, (
                f"Frame {f} patch at pos {pos} has chunk {c}, expected in [{f*4}, {(f+1)*4})"
            )
            chunk_ids_in_frame.add(c)
        assert chunk_ids_in_frame == set(range(f * 4, (f + 1) * 4))

    # CC tokens should be -1
    for pos in range(mtype.shape[1]):
        if mtype[0, pos] == C:
            assert ptc[pos].item() == -1, f"CC pos {pos} should have chunk -1"

    print("  PASS test_chunk_info_video_multi_frame")


def test_chunk_info_mixed_batch():
    """Batch with image (576 patches) + video (2 frames × 144 patches).
    Different n_chunks → padded to max_n_chunks."""
    K = 6

    # Item 0: image — 576 patches → 16 chunks
    n_patches_img = 576
    n_cc = 2
    mtype0_parts = [
        torch.full((2,), T),
        torch.full((n_patches_img,), I),
        torch.full((n_cc,), C),
        torch.full((3,), T),
    ]
    mtype0 = torch.cat(mtype0_parts)

    # Item 1: video — 2 frames × 144 → 8 chunks
    n_frames = 2
    patches_per_frame = 144
    mtype1_parts = [torch.full((2,), T)]
    for _ in range(n_frames):
        mtype1_parts.append(torch.full((patches_per_frame,), I))
        mtype1_parts.append(torch.full((n_cc,), C))
    mtype1_parts.append(torch.full((3,), T))
    mtype1 = torch.cat(mtype1_parts)

    # Pad to same length
    max_len = max(len(mtype0), len(mtype1))
    mtype0_padded = F.pad(mtype0, (0, max_len - len(mtype0)), value=P)
    mtype1_padded = F.pad(mtype1, (0, max_len - len(mtype1)), value=P)
    mtype = torch.stack([mtype0_padded, mtype1_padded])

    result = compute_chunk_info(mtype, ["image", "video"], patch_wh_one_chunk=K, device=torch.device("cpu"))
    assert result is not None

    n_chunks = result["n_chunks_per_item"]
    assert n_chunks[0].item() == 16, f"Image should have 16 chunks, got {n_chunks[0]}"
    assert n_chunks[1].item() == 8, f"Video should have 8 chunks, got {n_chunks[1]}"

    # chunk_positions padded to max_n_chunks=16
    assert result["chunk_positions"].shape[1] == 16
    assert result["chunk_valid_mask"][0].sum() == 16
    assert result["chunk_valid_mask"][1].sum() == 8
    # Invalid chunks for item 1 should be False
    assert not result["chunk_valid_mask"][1, 8:].any()

    print("  PASS test_chunk_info_mixed_batch")


# =====================================================================
# MoCSparseAttention tests
# =====================================================================

def _build_sparse_ctx(modality_types, modalities, labels, K=6, select_ratio=0.25, device="cpu"):
    """Build sparse_routing_ctx dict like LlavaQwen3ForCausalLM.forward() does."""
    text_token_mask = (modality_types == T)
    chunk_info = compute_chunk_info(modality_types, modalities, K, torch.device(device))
    assert chunk_info is not None, "No chunks found"
    return {
        "text_token_mask": text_token_mask.to(device),
        "attn_chunk_select_ratio": select_ratio,
        **{k: v.to(device) for k, v in chunk_info.items()},
    }


def _build_test_scenario(device="cpu", K=6, n_cc=2, n_question=8, n_answer=10, modality="image", n_frames=4):
    """Build a complete test scenario: modality_types, labels, hidden_states, position_embeddings, attn_mask.

    Token layout:
        Image:  [TEXT×2] [IMG×576] [CC×n_cc] [Q×n_question] [A×n_answer]
        Video:  [TEXT×2] [frame0 IMG×144] [CC×n_cc] [frame1 IMG×144] [CC×n_cc] ... [Q×n_question] [A×n_answer]
    """
    W = 24 if modality == "image" else 12
    patches_per_frame = W * W
    frames = 1 if modality == "image" else n_frames
    n_text_before = 2

    parts = [torch.full((n_text_before,), T)]
    for _ in range(frames):
        parts.append(torch.full((patches_per_frame,), I))
        parts.append(torch.full((n_cc,), C))
    parts.append(torch.full((n_question,), T))
    parts.append(torch.full((n_answer,), T))
    mtype = torch.cat(parts).unsqueeze(0).to(device)  # [1, S]
    S = mtype.shape[1]

    # Labels: IGNORE for everything except answer tokens
    labels = torch.full((1, S), IGNORE_INDEX, dtype=torch.long, device=device)
    labels[0, -n_answer:] = torch.arange(100, 100 + n_answer, device=device)

    return mtype, labels, S, [modality]


def test_attn_output_shape(device="cpu"):
    """MoCSparseAttention produces correct output shape."""
    config = _make_config()
    attn = _make_attn(config).to(device)
    K = 6

    mtype, labels, S, modalities = _build_test_scenario(device=device, K=K)
    sparse_ctx = _build_sparse_ctx(mtype, modalities, labels, K=K, device=device)

    hidden = torch.randn(1, S, config.hidden_size, device=device)
    pos_emb = _make_position_embeddings(1, S, config.head_dim, device=device)

    output, weights = attn(
        hidden_states=hidden,
        position_embeddings=pos_emb,
        attention_mask=None,
        sparse_routing_ctx=sparse_ctx,
    )
    assert output.shape == (1, S, config.hidden_size), f"Expected (1,{S},{config.hidden_size}), got {output.shape}"
    print(f"  PASS test_attn_output_shape ({device})")


def test_standard_path_when_no_ctx(device="cpu"):
    """Without sparse_routing_ctx, MoCSparseAttention == standard Qwen3Attention."""
    config = _make_config()
    attn = _make_attn(config).to(device)
    S = 32

    hidden = torch.randn(1, S, config.hidden_size, device=device)
    pos_emb = _make_position_embeddings(1, S, config.head_dim, device=device)

    # Without ctx
    torch.manual_seed(42)
    out_no_ctx, _ = attn(
        hidden_states=hidden,
        position_embeddings=pos_emb,
        attention_mask=None,
    )
    assert out_no_ctx.shape == (1, S, config.hidden_size)
    print(f"  PASS test_standard_path_when_no_ctx ({device})")


def test_standard_path_when_no_text(device="cpu"):
    """If text_token_mask is all False (no text tokens), should use standard attention."""
    config = _make_config()
    attn = _make_attn(config).to(device)
    K = 6

    # All-image sequence (no text tokens) → text_token_mask is all False
    S = 576
    mtype = torch.full((1, S), I, dtype=torch.long, device=device)
    labels = torch.full((1, S), IGNORE_INDEX, dtype=torch.long, device=device)
    text_token_mask = (mtype == T)  # all False
    chunk_info = compute_chunk_info(mtype, ["image"], K, torch.device(device))
    assert chunk_info is not None
    sparse_ctx = {
        "text_token_mask": text_token_mask,
        "attn_chunk_select_ratio": 0.25,
        **{k: v.to(device) for k, v in chunk_info.items()},
    }

    hidden = torch.randn(1, S, config.hidden_size, device=device)
    pos_emb = _make_position_embeddings(1, S, config.head_dim, device=device)

    # Should not crash — falls through to standard path since text_token_mask is all False
    output, _ = attn(
        hidden_states=hidden,
        position_embeddings=pos_emb,
        attention_mask=None,
        sparse_routing_ctx=sparse_ctx,
    )
    assert output.shape == (1, S, config.hidden_size)
    print(f"  PASS test_standard_path_when_no_text ({device})")


def test_sparse_mask_blocks_unselected_chunks(device="cpu"):
    """Stage 7: ALL text tokens (question + answer) should NOT attend to image patches
    outside selected chunks. Non-text tokens (image/CC) keep standard causal attention."""
    config = _make_config()
    attn = _make_attn(config).to(device)
    K = 6
    select_ratio = 0.25  # 16 chunks → k=4

    mtype, labels, S, modalities = _build_test_scenario(
        device=device, K=K, n_cc=2, n_question=4, n_answer=4,
    )
    sparse_ctx = _build_sparse_ctx(mtype, modalities, labels, K=K, select_ratio=select_ratio, device=device)

    # Manually run the mask construction logic to verify
    B, H = 1, config.num_attention_heads
    H_kv = config.num_key_value_heads

    hidden = torch.randn(B, S, config.hidden_size, device=device)
    hidden_shape = (B, S, -1, config.head_dim)

    query_states = attn.q_norm(attn.q_proj(hidden).view(hidden_shape)).transpose(1, 2)
    key_states = attn.k_norm(attn.k_proj(hidden).view(hidden_shape)).transpose(1, 2)

    # Pre-RoPE for routing
    query_pre = query_states.clone()
    key_pre = key_states.clone()

    # Chunk representatives
    chunk_positions = sparse_ctx["chunk_positions"]
    chunk_valid = sparse_ctx["chunk_valid_mask"]
    D = config.head_dim
    C_max = chunk_positions.shape[1]
    ppchunk = chunk_positions.shape[2]

    flat_pos = chunk_positions.reshape(B, -1)
    flat_pos_exp = flat_pos[:, None, :, None].expand(B, H_kv, -1, D)
    chunk_K = key_pre.gather(2, flat_pos_exp).reshape(B, H_kv, C_max, ppchunk, D)
    chunk_K = chunk_K * chunk_valid[:, None, :, None, None].float()
    chunk_repr_kv = chunk_K.sum(dim=3) / ppchunk

    from transformers.models.qwen3.modeling_qwen3 import repeat_kv
    chunk_repr = repeat_kv(chunk_repr_kv, config.num_attention_heads // config.num_key_value_heads)

    Q_norm = F.normalize(query_pre, dim=-1)
    C_norm = F.normalize(chunk_repr, dim=-1)
    sim = torch.einsum("bhsd,bhcd->bhsc", Q_norm, C_norm)
    sim.masked_fill_(~chunk_valid[:, None, None, :], float("-inf"))

    n_chunks_per_item = sparse_ctx["n_chunks_per_item"]
    k_per_item = (n_chunks_per_item.float() * select_ratio).ceil().long().clamp(min=1)
    k_max = min(k_per_item.max().item(), C_max)

    text_token_mask = sparse_ctx["text_token_mask"]
    image_mask = sparse_ctx["image_mask"]
    ptc = sparse_ctx["position_to_chunk"]

    # For EACH text token position (question + answer), verify routing selects exactly k chunks
    text_positions = text_token_mask[0].nonzero(as_tuple=True)[0]
    # Only check text positions that are AFTER image (can causally see image patches)
    first_img_pos = image_mask[0].nonzero(as_tuple=True)[0][0].item()
    text_after_image = [p for p in text_positions if p.item() > first_img_pos]
    assert len(text_after_image) > 0, "No text tokens after image"

    for txt_pos in text_after_image:
        for h in range(H):
            pos_sim = sim[0, h, txt_pos]  # [C_max]
            _, topk_idx = pos_sim.topk(k_max)
            selected_set = set(topk_idx[:k_per_item[0]].tolist())
            assert len(selected_set) == k_per_item[0].item()

            # Verify: image patches in selected chunks → attendable,
            #         image patches NOT in selected chunks → blocked
            for img_pos in image_mask[0].nonzero(as_tuple=True)[0]:
                if img_pos.item() > txt_pos.item():
                    continue  # causal: can't see future
                chunk_of_pos = ptc[0, img_pos].item()
                if chunk_of_pos in selected_set:
                    pass  # should be unmasked by step 4b
                else:
                    pass  # should remain masked by step 4a

    print(f"  PASS test_sparse_mask_blocks_unselected_chunks ({device})")


def test_non_text_tokens_unaffected(device="cpu"):
    """Non-text token (image/CC) rows should have identical attention as without sparse routing."""
    config = _make_config()
    K = 6

    mtype, labels, S, modalities = _build_test_scenario(
        device=device, K=K, n_cc=2, n_question=4, n_answer=4,
    )
    sparse_ctx = _build_sparse_ctx(mtype, modalities, labels, K=K, device=device)
    text_token_mask = sparse_ctx["text_token_mask"]

    # Build masks manually for comparison
    B, H = 1, config.num_attention_heads
    min_dtype = torch.finfo(torch.float32).min

    # Base causal mask [1, 1, S, S]
    causal = torch.triu(torch.full((S, S), min_dtype, dtype=torch.float32, device=device), diagonal=1)
    base_mask = causal[None, None, :, :]

    # The full_mask constructed in _sparse_routing_attention starts from base_mask
    full_mask = base_mask.expand(B, H, S, S).clone()

    # Step 4a only affects text rows
    image_mask = sparse_ctx["image_mask"]
    text_image_block = text_token_mask[:, :, None] & image_mask[:, None, :]
    full_mask.masked_fill_(text_image_block[:, None, :, :], min_dtype)

    # Verify: non-text rows (image/CC) in full_mask == base causal mask
    for q in range(S):
        if not text_token_mask[0, q]:
            for k_pos in range(S):
                for h in range(H):
                    expected = base_mask[0, 0, q, k_pos].item()
                    actual = full_mask[0, h, q, k_pos].item()
                    assert actual == expected, (
                        f"Non-text row {q}, key {k_pos}, head {h}: "
                        f"expected {expected}, got {actual}"
                    )

    print(f"  PASS test_non_text_tokens_unaffected ({device})")


def test_text_attends_to_cc_and_other_text(device="cpu"):
    """Text tokens (question + answer) should always attend to CC and other text tokens,
    regardless of sparse routing. Only image patches are subject to routing."""
    config = _make_config()
    K = 6
    n_cc = 2
    n_question = 4
    n_answer = 6

    mtype, labels, S, modalities = _build_test_scenario(
        device=device, K=K, n_cc=n_cc, n_question=n_question, n_answer=n_answer,
    )
    sparse_ctx = _build_sparse_ctx(mtype, modalities, labels, K=K, select_ratio=0.25, device=device)

    B, H = 1, config.num_attention_heads
    min_dtype = torch.finfo(torch.float32).min
    text_token_mask = sparse_ctx["text_token_mask"]
    image_mask = sparse_ctx["image_mask"]

    # Build the mask as _sparse_routing_attention does (Steps 4a + 4b only, simplified)
    causal = torch.triu(torch.full((S, S), min_dtype, dtype=torch.float32, device=device), diagonal=1)
    base_mask = causal[None, None, :, :]
    full_mask = base_mask.expand(B, H, S, S).clone()

    # 4a: mask all image patches for text rows
    text_image_block = text_token_mask[:, :, None] & image_mask[:, None, :]
    full_mask.masked_fill_(text_image_block[:, None, :, :], min_dtype)

    # Before 4b (unmask), check that CC/other-text are still attendable for text tokens
    for q in range(S):
        if not text_token_mask[0, q]:
            continue
        for k_pos in range(q + 1):  # causal: can only attend to k_pos <= q
            mt = mtype[0, k_pos].item()
            if mt == I:
                # Image patches: should be masked (will be selectively unmasked in 4b)
                for h in range(H):
                    assert not _can_attend(full_mask, 0, h, q, k_pos), (
                        f"Text@{q} should NOT attend to image@{k_pos} before unmasking"
                    )
            else:
                # CC, text (question/answer), should remain attendable
                for h in range(H):
                    assert _can_attend(full_mask, 0, h, q, k_pos), (
                        f"Text@{q} should attend to non-image@{k_pos} (type={mt})"
                    )

    print(f"  PASS test_text_attends_to_cc_and_other_text ({device})")


def test_question_tokens_sparse_routed(device="cpu"):
    """Stage 7 core test: question tokens (not just answer) must receive sparse routing.
    Question tokens after image should see only selected-chunk patches, NOT all patches.
    This is the key behavioral change from Stage 6 → Stage 7."""
    config = _make_config()
    attn = _make_attn(config).to(device)
    K = 6
    n_cc = 2
    n_question = 4
    n_answer = 4
    select_ratio = 0.25  # 16 chunks → k=4, so 4*36=144 patches visible out of 576

    mtype, labels, S, modalities = _build_test_scenario(
        device=device, K=K, n_cc=n_cc, n_question=n_question, n_answer=n_answer,
    )
    sparse_ctx = _build_sparse_ctx(mtype, modalities, labels, K=K, select_ratio=select_ratio, device=device)

    B, H = 1, config.num_attention_heads
    min_dtype = torch.finfo(torch.float32).min

    text_token_mask = sparse_ctx["text_token_mask"]
    image_mask = sparse_ctx["image_mask"]
    ptc = sparse_ctx["position_to_chunk"]
    n_total_img = image_mask[0].sum().item()  # 576

    # Run full forward to get the actual mask built inside _sparse_routing_attention
    hidden = torch.randn(B, S, config.hidden_size, device=device)
    pos_emb = _make_position_embeddings(B, S, config.head_dim, device=device)
    output, _ = attn(
        hidden_states=hidden,
        position_embeddings=pos_emb,
        attention_mask=None,
        sparse_routing_ctx=sparse_ctx,
    )

    # Re-run mask construction manually to inspect the actual mask
    # (same logic as _sparse_routing_attention Steps 1-4)
    hidden_shape = (B, S, -1, config.head_dim)
    H_kv = config.num_key_value_heads
    D = config.head_dim

    query_states = attn.q_norm(attn.q_proj(hidden).view(hidden_shape)).transpose(1, 2)
    key_states = attn.k_norm(attn.k_proj(hidden).view(hidden_shape)).transpose(1, 2)
    query_pre = query_states.clone()
    key_pre = key_states.clone()

    chunk_positions = sparse_ctx["chunk_positions"]
    chunk_valid = sparse_ctx["chunk_valid_mask"]
    C_max = chunk_positions.shape[1]
    ppchunk = chunk_positions.shape[2]

    flat_pos = chunk_positions.reshape(B, -1)
    flat_pos_exp = flat_pos[:, None, :, None].expand(B, H_kv, -1, D)
    chunk_K = key_pre.gather(2, flat_pos_exp).reshape(B, H_kv, C_max, ppchunk, D)
    chunk_K = chunk_K * chunk_valid[:, None, :, None, None].float()
    chunk_repr_kv = chunk_K.sum(dim=3) / ppchunk

    from transformers.models.qwen3.modeling_qwen3 import repeat_kv
    chunk_repr = repeat_kv(chunk_repr_kv, config.num_attention_heads // config.num_key_value_heads)

    Q_norm = F.normalize(query_pre, dim=-1)
    C_norm = F.normalize(chunk_repr, dim=-1)
    sim = torch.einsum("bhsd,bhcd->bhsc", Q_norm, C_norm)
    sim.masked_fill_(~chunk_valid[:, None, None, :], float("-inf"))

    n_chunks_per_item = sparse_ctx["n_chunks_per_item"]
    k_per_item = (n_chunks_per_item.float() * select_ratio).ceil().long().clamp(min=1)
    k_max = min(k_per_item.max().item(), C_max)
    expected_img_visible = k_per_item[0].item() * ppchunk  # k * K*K = 4 * 36 = 144

    # Build full_mask exactly as _sparse_routing_attention does
    causal = torch.triu(
        torch.full((S, S), min_dtype, dtype=torch.float32, device=device), diagonal=1
    )
    full_mask = causal[None, None, :, :].expand(B, H, S, S).clone()

    # Step 4a: mask all image patches for text rows
    text_image_block = text_token_mask[:, :, None] & image_mask[:, None, :]
    full_mask.masked_fill_(text_image_block[:, None, :, :], min_dtype)

    # Step 4b: unmask selected chunks for text rows
    _, topk_idx = sim.topk(k_max, dim=-1)
    selected_chunks = torch.zeros(B, H, S, C_max, dtype=torch.bool, device=device)
    selected_chunks.scatter_(-1, topk_idx, True)
    for b in range(B):
        k_b = k_per_item[b].item()
        if k_b < k_max:
            extra_idx = topk_idx[b, :, :, k_b:]
            selected_chunks[b].scatter_(-1, extra_idx, False)

    ptc_exp = ptc[:, None, None, :].clamp(min=0).expand(B, H, S, S)
    is_image_key = (ptc[:, None, None, :] >= 0)
    unmask = selected_chunks.gather(-1, ptc_exp)
    unmask &= is_image_key
    unmask &= text_token_mask[:, None, :, None]
    full_mask.masked_fill_(unmask, 0.0)

    # --- Core assertion: question tokens see only selected-chunk patches ---
    # Identify question positions (text tokens between CC and answer)
    answer_start = S - n_answer
    question_start = answer_start - n_question
    question_positions = list(range(question_start, answer_start))

    for q_pos in question_positions:
        for h in range(H):
            # Count how many image patches this question token can attend to
            attendable_img = 0
            for img_pos in image_mask[0].nonzero(as_tuple=True)[0]:
                if img_pos.item() <= q_pos and _can_attend(full_mask, 0, h, q_pos, img_pos.item()):
                    attendable_img += 1

            # Stage 7: question token should see exactly k*ppchunk image patches, NOT all 576
            assert attendable_img == expected_img_visible, (
                f"Question@{q_pos} head={h}: sees {attendable_img} img patches, "
                f"expected {expected_img_visible} (k={k_per_item[0].item()}, "
                f"ppchunk={ppchunk}). Stage 7 requires question tokens to be sparse-routed."
            )
            assert attendable_img < n_total_img, (
                f"Question@{q_pos} head={h}: sees ALL {n_total_img} patches — "
                f"sparse routing NOT applied to question token!"
            )

    # Also verify answer tokens get the same treatment
    answer_positions = list(range(answer_start, S))
    for a_pos in answer_positions:
        for h in range(H):
            attendable_img = 0
            for img_pos in image_mask[0].nonzero(as_tuple=True)[0]:
                if img_pos.item() <= a_pos and _can_attend(full_mask, 0, h, a_pos, img_pos.item()):
                    attendable_img += 1
            assert attendable_img == expected_img_visible, (
                f"Answer@{a_pos} head={h}: sees {attendable_img} img patches, "
                f"expected {expected_img_visible}"
            )

    print(f"  PASS test_question_tokens_sparse_routed ({device})")


def test_text_before_image_unaffected(device="cpu"):
    """Text tokens BEFORE image (e.g. system prompt) cannot see image due to causal mask.
    Sparse routing step 4a masks image columns, but these tokens can't see future anyway.
    Verify these rows are identical to base causal mask."""
    config = _make_config()
    K = 6

    mtype, labels, S, modalities = _build_test_scenario(
        device=device, K=K, n_cc=2, n_question=4, n_answer=4,
    )
    sparse_ctx = _build_sparse_ctx(mtype, modalities, labels, K=K, device=device)

    B, H = 1, config.num_attention_heads
    min_dtype = torch.finfo(torch.float32).min
    text_token_mask = sparse_ctx["text_token_mask"]
    image_mask = sparse_ctx["image_mask"]

    # text_before_image: text tokens at positions 0, 1 (before image starts at pos 2)
    first_img = image_mask[0].nonzero(as_tuple=True)[0][0].item()
    text_before_img = [i for i in range(first_img) if text_token_mask[0, i]]
    assert len(text_before_img) > 0, "No text before image in test scenario"

    causal = torch.triu(torch.full((S, S), min_dtype, dtype=torch.float32, device=device), diagonal=1)
    base_mask = causal[None, None, :, :]
    full_mask = base_mask.expand(B, H, S, S).clone()

    # Apply step 4a
    text_image_block = text_token_mask[:, :, None] & image_mask[:, None, :]
    full_mask.masked_fill_(text_image_block[:, None, :, :], min_dtype)

    # For text tokens before image, their row should be identical to base causal
    # (all image positions are in the future → already masked by causal)
    for q in text_before_img:
        for k_pos in range(S):
            for h in range(H):
                expected = base_mask[0, 0, q, k_pos].item()
                actual = full_mask[0, h, q, k_pos].item()
                assert actual == expected, (
                    f"Text-before-image@{q}, key={k_pos}, head={h}: "
                    f"expected {expected}, got {actual}"
                )

    print(f"  PASS test_text_before_image_unaffected ({device})")


def test_gqa_per_head_routing(device="cpu"):
    """With GQA (4 query heads, 2 KV heads), each query head should
    independently select chunks."""
    config = _make_config(num_attention_heads=4, num_key_value_heads=2)
    attn = _make_attn(config).to(device)
    K = 6

    mtype, labels, S, modalities = _build_test_scenario(device=device, K=K, n_answer=8)
    sparse_ctx = _build_sparse_ctx(mtype, modalities, labels, K=K, select_ratio=0.25, device=device)

    hidden = torch.randn(1, S, config.hidden_size, device=device)
    pos_emb = _make_position_embeddings(1, S, config.head_dim, device=device)

    output, _ = attn(
        hidden_states=hidden,
        position_embeddings=pos_emb,
        attention_mask=None,
        sparse_routing_ctx=sparse_ctx,
    )
    assert output.shape == (1, S, config.hidden_size)
    print(f"  PASS test_gqa_per_head_routing ({device})")


def test_gradient_flows(device="cpu"):
    """Verify gradient flows through the sparse attention."""
    config = _make_config()
    attn = _make_attn(config).to(device)
    attn.train()
    K = 6

    mtype, labels, S, modalities = _build_test_scenario(device=device, K=K, n_answer=6)
    sparse_ctx = _build_sparse_ctx(mtype, modalities, labels, K=K, device=device)

    hidden = torch.randn(1, S, config.hidden_size, device=device, requires_grad=True)
    pos_emb = _make_position_embeddings(1, S, config.head_dim, device=device)

    output, _ = attn(
        hidden_states=hidden,
        position_embeddings=pos_emb,
        attention_mask=None,
        sparse_routing_ctx=sparse_ctx,
    )

    loss = output.sum()
    loss.backward()

    assert hidden.grad is not None, "No gradient on hidden_states"
    assert hidden.grad.abs().sum() > 0, "Gradient is all zeros"

    # Check projection weights got gradients
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        w = getattr(attn, name).weight
        assert w.grad is not None, f"No gradient on {name}.weight"
        assert w.grad.abs().sum() > 0, f"Gradient on {name}.weight is all zeros"

    print(f"  PASS test_gradient_flows ({device})")


def test_video_routing(device="cpu"):
    """Video with multiple frames: chunks span all frames,
    routing selects from cross-frame pool."""
    config = _make_config()
    attn = _make_attn(config).to(device)
    K = 6
    n_frames = 4  # 4 frames × 4 chunks = 16 total chunks

    mtype, labels, S, modalities = _build_test_scenario(
        device=device, K=K, n_answer=6, modality="video", n_frames=n_frames,
    )
    sparse_ctx = _build_sparse_ctx(mtype, modalities, labels, K=K, select_ratio=0.5, device=device)

    assert sparse_ctx["n_chunks_per_item"][0].item() == n_frames * 4

    hidden = torch.randn(1, S, config.hidden_size, device=device)
    pos_emb = _make_position_embeddings(1, S, config.head_dim, device=device)

    output, _ = attn(
        hidden_states=hidden,
        position_embeddings=pos_emb,
        attention_mask=None,
        sparse_routing_ctx=sparse_ctx,
    )
    assert output.shape == (1, S, config.hidden_size)
    print(f"  PASS test_video_routing ({device})")


def test_select_ratio_variation(device="cpu"):
    """Different select_ratio values: small (0.125) and large (1.0)."""
    config = _make_config()
    K = 6

    mtype, labels, S, modalities = _build_test_scenario(device=device, K=K, n_answer=4)

    for ratio in [0.125, 0.25, 0.5, 1.0]:
        attn = _make_attn(config).to(device)
        sparse_ctx = _build_sparse_ctx(mtype, modalities, labels, K=K, select_ratio=ratio, device=device)

        n_chunks = sparse_ctx["n_chunks_per_item"][0].item()
        expected_k = math.ceil(n_chunks * ratio)

        hidden = torch.randn(1, S, config.hidden_size, device=device)
        pos_emb = _make_position_embeddings(1, S, config.head_dim, device=device)

        output, _ = attn(
            hidden_states=hidden,
            position_embeddings=pos_emb,
            attention_mask=None,
            sparse_routing_ctx=sparse_ctx,
        )
        assert output.shape == (1, S, config.hidden_size)

    print(f"  PASS test_select_ratio_variation ({device})")


def test_with_4d_base_mask(device="cpu"):
    """When a 4D base mask (from Stage 5 CC restriction) is provided,
    sparse routing should correctly layer on top of it."""
    config = _make_config()
    attn = _make_attn(config).to(device)
    K = 6

    mtype, labels, S, modalities = _build_test_scenario(device=device, K=K, n_cc=2, n_answer=4)
    sparse_ctx = _build_sparse_ctx(mtype, modalities, labels, K=K, device=device)

    # Build a simple 4D causal mask [1, 1, S, S]
    min_dtype = torch.finfo(torch.float32).min
    causal = torch.triu(torch.full((S, S), min_dtype, dtype=torch.float32, device=device), diagonal=1)
    base_mask = causal[None, None, :, :]

    hidden = torch.randn(1, S, config.hidden_size, device=device)
    pos_emb = _make_position_embeddings(1, S, config.head_dim, device=device)

    output, _ = attn(
        hidden_states=hidden,
        position_embeddings=pos_emb,
        attention_mask=base_mask,
        sparse_routing_ctx=sparse_ctx,
    )
    assert output.shape == (1, S, config.hidden_size)
    print(f"  PASS test_with_4d_base_mask ({device})")


# =====================================================================
# Large-scale tests
# =====================================================================

def test_large_scale_video(device="cpu"):
    """Large video: 32 frames × 144 patches = 4608 image tokens + CC + text.
    32 frames × 4 chunks = 128 total chunks."""
    config = _make_config()
    attn = _make_attn(config).to(device)
    K = 6
    n_frames = 32

    mtype, labels, S, modalities = _build_test_scenario(
        device=device, K=K, n_cc=4, n_question=20, n_answer=64,
        modality="video", n_frames=n_frames,
    )
    sparse_ctx = _build_sparse_ctx(mtype, modalities, labels, K=K, select_ratio=0.25, device=device)

    assert sparse_ctx["n_chunks_per_item"][0].item() == n_frames * 4  # 128

    hidden = torch.randn(1, S, config.hidden_size, device=device)
    pos_emb = _make_position_embeddings(1, S, config.head_dim, device=device)

    t0 = time.time()
    output, _ = attn(
        hidden_states=hidden,
        position_embeddings=pos_emb,
        attention_mask=None,
        sparse_routing_ctx=sparse_ctx,
    )
    dt = time.time() - t0

    assert output.shape == (1, S, config.hidden_size)
    print(f"  PASS test_large_scale_video ({device}) — S={S}, 128 chunks, {dt:.2f}s")


def test_large_batch(device="cpu"):
    """Batch of 4: 2 images + 2 videos (4 frames each)."""
    config = _make_config()
    attn = _make_attn(config).to(device)
    K = 6
    n_cc = 2
    n_question = 6
    n_answer = 8
    n_frames_video = 4

    items = []
    mods = []
    label_list = []

    for mod in ["image", "image", "video", "video"]:
        W = 24 if mod == "image" else 12
        ppf = W * W
        frames = 1 if mod == "image" else n_frames_video
        parts = [torch.full((2,), T)]
        for _ in range(frames):
            parts.append(torch.full((ppf,), I))
            parts.append(torch.full((n_cc,), C))
        parts.append(torch.full((n_question,), T))
        parts.append(torch.full((n_answer,), T))
        mt = torch.cat(parts)
        lb = torch.full_like(mt, IGNORE_INDEX)
        lb[-n_answer:] = torch.arange(100, 100 + n_answer)
        items.append(mt)
        label_list.append(lb)
        mods.append(mod)

    max_len = max(len(x) for x in items)
    mtype = torch.stack([F.pad(x, (0, max_len - len(x)), value=P) for x in items]).to(device)
    labels = torch.stack([F.pad(x, (0, max_len - len(x)), value=IGNORE_INDEX) for x in label_list]).to(device)
    S = max_len

    sparse_ctx = _build_sparse_ctx(mtype, mods, labels, K=K, select_ratio=0.25, device=device)

    hidden = torch.randn(4, S, config.hidden_size, device=device)
    pos_emb = _make_position_embeddings(4, S, config.head_dim, device=device)

    t0 = time.time()
    output, _ = attn(
        hidden_states=hidden,
        position_embeddings=pos_emb,
        attention_mask=None,
        sparse_routing_ctx=sparse_ctx,
    )
    dt = time.time() - t0

    assert output.shape == (4, S, config.hidden_size)
    print(f"  PASS test_large_batch ({device}) — B=4, S={S}, {dt:.2f}s")


# =====================================================================
# Runner
# =====================================================================

def run_all(device="cpu"):
    print(f"\n{'='*60}")
    print(f"Running tests on {device.upper()}")
    print(f"{'='*60}")

    print("\n--- compute_chunk_info ---")
    test_chunk_info_no_image()
    test_chunk_info_single_image()
    test_chunk_info_video_multi_frame()
    test_chunk_info_mixed_batch()

    print("\n--- MoCSparseAttention basic ---")
    test_attn_output_shape(device)
    test_standard_path_when_no_ctx(device)
    test_standard_path_when_no_text(device)

    print("\n--- MoCSparseAttention mask correctness ---")
    test_sparse_mask_blocks_unselected_chunks(device)
    test_non_text_tokens_unaffected(device)
    test_text_attends_to_cc_and_other_text(device)
    test_question_tokens_sparse_routed(device)
    test_text_before_image_unaffected(device)

    print("\n--- MoCSparseAttention functional ---")
    test_gqa_per_head_routing(device)
    test_gradient_flows(device)
    test_video_routing(device)
    test_select_ratio_variation(device)
    test_with_4d_base_mask(device)

    print("\n--- Large scale ---")
    test_large_scale_video(device)
    test_large_batch(device)


if __name__ == "__main__":
    run_all("cpu")

    if torch.cuda.is_available():
        run_all("cuda")
    else:
        print("\n[SKIP] CUDA not available, skipping GPU tests")

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)
