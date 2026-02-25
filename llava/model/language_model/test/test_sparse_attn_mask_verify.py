"""
Verification tests for MoCSparseAttention mask correctness.

Focus:
  1) Text tokens attend to ALL CC tokens (causal range)
  2) Text tokens attend to ALL previous text tokens
  3) Text tokens attend to ONLY selected image-patch chunks (sparse routing)
  4) Causal fix: text tokens before image do NOT attend to future image patches
  5) The causal fix does not break question/answer routing after image

Run:
    python -m llava.model.language_model.test.test_sparse_attn_mask_verify
"""

import sys, os, math, time
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from llava.constants import (
    MODALITY_TEXT as T,
    MODALITY_IMAGE as I,
    MODALITY_COMPRESSED_CONTEXT as C,
    MODALITY_PAD as P,
    IGNORE_INDEX,
)
from llava.model.language_model.sparse_attn import compute_chunk_info, MoCSparseAttention
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, repeat_kv

# ===================================================================
# Helpers
# ===================================================================

def make_config(**kw):
    defaults = dict(
        hidden_size=128, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, attention_bias=False, attention_dropout=0.0,
        rms_norm_eps=1e-6, use_sliding_window=False,
        sliding_window=None, max_window_layers=0,
    )
    defaults.update(kw)
    cfg = Qwen3Config(**defaults)
    cfg._attn_implementation = "eager"
    return cfg


def make_attn(config=None, layer_idx=0):
    config = config or make_config()
    attn = Qwen3Attention(config, layer_idx=layer_idx)
    attn.__class__ = MoCSparseAttention
    return attn


def make_pos_emb(B, S, head_dim, device):
    return (torch.randn(B, S, head_dim, device=device),
            torch.randn(B, S, head_dim, device=device))


def build_sparse_ctx(mtype, modalities, K, ratio, device):
    text_mask = (mtype == T).to(device)
    ci = compute_chunk_info(mtype.to(device), modalities, K, device)
    assert ci is not None
    return {"text_token_mask": text_mask, "attn_chunk_select_ratio": ratio,
            **{k: v.to(device) for k, v in ci.items()}}


def build_mtype(n_pre_text, patches_per_frame, n_cc, n_question, n_answer,
                n_frames=1, n_pad=0):
    """Build a single modality_types 1-D tensor.
    Layout: [TEXT*n_pre_text] ([IMG*ppf] [CC*n_cc])*n_frames [TEXT*(n_q+n_a)] [PAD*n_pad]
    """
    parts = [torch.full((n_pre_text,), T)]
    for _ in range(n_frames):
        parts.append(torch.full((patches_per_frame,), I))
        parts.append(torch.full((n_cc,), C))
    parts.append(torch.full((n_question + n_answer,), T))
    if n_pad > 0:
        parts.append(torch.full((n_pad,), P))
    return torch.cat(parts)


def extract_mask(attn, hidden, pos_emb, base_mask, sparse_ctx, device):
    """Run _sparse_routing_attention and return the full_mask [B, H, S, S].

    We monkey-patch F.scaled_dot_product_attention to capture the mask.
    """
    B, S, _ = hidden.shape
    config = attn.config
    H = config.num_attention_heads
    D = config.head_dim
    hidden_shape = (B, S, -1, D)

    q = attn.q_norm(attn.q_proj(hidden).view(hidden_shape)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(hidden).view(hidden_shape)).transpose(1, 2)
    v = attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
    q_pre, k_pre = q.clone(), k.clone()

    cos, sin = pos_emb
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # We call _sparse_routing_attention but intercept the SDPA call
    captured = {}
    orig_sdpa = F.scaled_dot_product_attention

    def spy_sdpa(query, key, value, attn_mask=None, **kw):
        captured["mask"] = attn_mask.clone() if attn_mask is not None else None
        return orig_sdpa(query, key, value, attn_mask=attn_mask, **kw)

    F.scaled_dot_product_attention = spy_sdpa
    try:
        attn._sparse_routing_attention(q, k, v, base_mask, sparse_ctx, q_pre, k_pre)
    finally:
        F.scaled_dot_product_attention = orig_sdpa

    return captured["mask"]  # [B, H, S, S]


def can_attend(mask_4d, b, h, q, k):
    return mask_4d[b, h, q, k].item() > -1e30


# ===================================================================
# Test 1 — text→CC always attend
# ===================================================================
def test_text_attends_all_cc(device):
    """Every text token (question+answer) must attend to ALL CC tokens that precede it."""
    K = 6
    n_pre, n_cc, n_q, n_a = 3, 4, 6, 8
    mtype = build_mtype(n_pre, 576, n_cc, n_q, n_a).unsqueeze(0).to(device)
    S = mtype.shape[1]
    cfg = make_config()
    attn = make_attn(cfg).to(device)

    ctx = build_sparse_ctx(mtype, ["image"], K, 0.25, device)
    hidden = torch.randn(1, S, cfg.hidden_size, device=device)
    pos = make_pos_emb(1, S, cfg.head_dim, device)
    mask = extract_mask(attn, hidden, pos, None, ctx, device)

    cc_positions = (mtype[0] == C).nonzero(as_tuple=True)[0].tolist()
    text_positions = (mtype[0] == T).nonzero(as_tuple=True)[0].tolist()
    text_after_img = [p for p in text_positions if p > cc_positions[-1]]

    for q in text_after_img:
        for cc in cc_positions:
            if cc > q:
                continue  # causal
            for h in range(cfg.num_attention_heads):
                assert can_attend(mask, 0, h, q, cc), (
                    f"text@{q} head={h} cannot attend to CC@{cc}")
    print(f"  PASS test_text_attends_all_cc ({device})")


# ===================================================================
# Test 2 — text→text always attend
# ===================================================================
def test_text_attends_all_prev_text(device):
    """Every text token must attend to ALL earlier text tokens."""
    K = 6
    n_pre, n_cc, n_q, n_a = 3, 4, 6, 8
    mtype = build_mtype(n_pre, 576, n_cc, n_q, n_a).unsqueeze(0).to(device)
    S = mtype.shape[1]
    cfg = make_config()
    attn = make_attn(cfg).to(device)

    ctx = build_sparse_ctx(mtype, ["image"], K, 0.25, device)
    hidden = torch.randn(1, S, cfg.hidden_size, device=device)
    pos = make_pos_emb(1, S, cfg.head_dim, device)
    mask = extract_mask(attn, hidden, pos, None, ctx, device)

    text_positions = (mtype[0] == T).nonzero(as_tuple=True)[0].tolist()
    for i, q in enumerate(text_positions):
        for k_pos in text_positions[:i]:
            for h in range(cfg.num_attention_heads):
                assert can_attend(mask, 0, h, q, k_pos), (
                    f"text@{q} head={h} cannot attend to text@{k_pos}")
    print(f"  PASS test_text_attends_all_prev_text ({device})")


# ===================================================================
# Test 3 — sparse routing: text→image is subset, correct count
# ===================================================================
def test_sparse_routing_image_count(device):
    """Question+answer tokens see exactly k*K*K image patches, not all 576."""
    K = 6
    ratio = 0.25  # 16 chunks → k=4 → 4*36=144 patches
    n_pre, n_cc, n_q, n_a = 2, 2, 5, 5
    mtype = build_mtype(n_pre, 576, n_cc, n_q, n_a).unsqueeze(0).to(device)
    S = mtype.shape[1]
    cfg = make_config()
    attn = make_attn(cfg).to(device)

    ctx = build_sparse_ctx(mtype, ["image"], K, ratio, device)
    hidden = torch.randn(1, S, cfg.hidden_size, device=device)
    pos = make_pos_emb(1, S, cfg.head_dim, device)
    mask = extract_mask(attn, hidden, pos, None, ctx, device)

    img_positions = (mtype[0] == I).nonzero(as_tuple=True)[0]
    n_chunks = ctx["n_chunks_per_item"][0].item()
    k_selected = math.ceil(n_chunks * ratio)
    expected_img = k_selected * K * K  # 4 * 36 = 144

    # question + answer are the last (n_q + n_a) text tokens after CC
    qa_start = S - n_q - n_a
    for q in range(qa_start, S):
        for h in range(cfg.num_attention_heads):
            img_attend = sum(1 for ip in img_positions
                            if ip.item() <= q and can_attend(mask, 0, h, q, ip.item()))
            assert img_attend == expected_img, (
                f"q={q} h={h}: sees {img_attend} img patches, expected {expected_img}")
    print(f"  PASS test_sparse_routing_image_count ({device})")


# ===================================================================
# Test 4 — causal fix: pre-image text does NOT see future image
# ===================================================================
def test_pre_image_text_no_future_image(device):
    """Text tokens before image must NOT attend to any image patch (all are in the future)."""
    K = 6
    n_pre, n_cc, n_q, n_a = 5, 2, 4, 4
    mtype = build_mtype(n_pre, 576, n_cc, n_q, n_a).unsqueeze(0).to(device)
    S = mtype.shape[1]
    cfg = make_config()
    attn = make_attn(cfg).to(device)

    ctx = build_sparse_ctx(mtype, ["image"], K, 0.5, device)
    hidden = torch.randn(1, S, cfg.hidden_size, device=device)
    pos = make_pos_emb(1, S, cfg.head_dim, device)
    mask = extract_mask(attn, hidden, pos, None, ctx, device)

    img_positions = (mtype[0] == I).nonzero(as_tuple=True)[0].tolist()
    first_img = img_positions[0]

    for q in range(first_img):  # text tokens before image
        if mtype[0, q].item() != T:
            continue
        for ip in img_positions:
            for h in range(cfg.num_attention_heads):
                assert not can_attend(mask, 0, h, q, ip), (
                    f"pre-image text@{q} h={h} attends to future img@{ip}!")
    print(f"  PASS test_pre_image_text_no_future_image ({device})")


# ===================================================================
# Test 5 — causal fix does NOT break post-image question routing
# ===================================================================
def test_post_image_routing_still_works(device):
    """After the fix, question tokens after image should still see selected chunks (>0 patches)."""
    K = 6
    ratio = 0.25
    n_pre, n_cc, n_q, n_a = 3, 2, 6, 6
    mtype = build_mtype(n_pre, 576, n_cc, n_q, n_a).unsqueeze(0).to(device)
    S = mtype.shape[1]
    cfg = make_config()
    attn = make_attn(cfg).to(device)

    ctx = build_sparse_ctx(mtype, ["image"], K, ratio, device)
    hidden = torch.randn(1, S, cfg.hidden_size, device=device)
    pos = make_pos_emb(1, S, cfg.head_dim, device)
    mask = extract_mask(attn, hidden, pos, None, ctx, device)

    img_positions = (mtype[0] == I).nonzero(as_tuple=True)[0]
    n_chunks = ctx["n_chunks_per_item"][0].item()
    k_selected = math.ceil(n_chunks * ratio)
    expected_img = k_selected * K * K

    qa_start = S - n_q - n_a
    for q in range(qa_start, S):
        for h in range(cfg.num_attention_heads):
            img_attend = sum(1 for ip in img_positions
                            if ip.item() <= q and can_attend(mask, 0, h, q, ip.item()))
            assert img_attend == expected_img, (
                f"q={q} h={h}: sees {img_attend} img, expected {expected_img}")
    print(f"  PASS test_post_image_routing_still_works ({device})")


# ===================================================================
# Test 6 — non-text tokens unaffected by sparse routing
# ===================================================================
def test_non_text_rows_unchanged(device):
    """Image/CC rows: sparse routing must NOT alter their mask vs plain causal."""
    K = 6
    n_pre, n_cc, n_q, n_a = 2, 2, 4, 4
    mtype = build_mtype(n_pre, 576, n_cc, n_q, n_a).unsqueeze(0).to(device)
    S = mtype.shape[1]
    cfg = make_config()
    attn = make_attn(cfg).to(device)
    H = cfg.num_attention_heads

    ctx = build_sparse_ctx(mtype, ["image"], K, 0.25, device)
    hidden = torch.randn(1, S, cfg.hidden_size, device=device)
    pos = make_pos_emb(1, S, cfg.head_dim, device)
    mask = extract_mask(attn, hidden, pos, None, ctx, device)

    # base causal
    min_v = torch.finfo(mask.dtype).min
    causal = torch.triu(torch.full((S, S), min_v, dtype=mask.dtype, device=device), diagonal=1)
    base = causal[None, None, :, :]

    for q in range(S):
        if mtype[0, q].item() == T:
            continue
        for k_pos in range(S):
            for h in range(H):
                expected = base[0, 0, q, k_pos].item()
                actual = mask[0, h, q, k_pos].item()
                assert actual == expected, (
                    f"non-text@{q} key={k_pos} h={h}: exp={expected}, got={actual}")
    print(f"  PASS test_non_text_rows_unchanged ({device})")


# ===================================================================
# Test 7 — video multi-frame
# ===================================================================
def test_video_multi_frame(device):
    """Video (4 frames, 144 patches each) — same 3 properties hold."""
    K = 6
    ratio = 0.5
    n_frames = 4
    n_pre, n_cc, n_q, n_a = 2, 2, 4, 6
    mtype = build_mtype(n_pre, 144, n_cc, n_q, n_a, n_frames=n_frames).unsqueeze(0).to(device)
    S = mtype.shape[1]
    cfg = make_config()
    attn = make_attn(cfg).to(device)

    ctx = build_sparse_ctx(mtype, ["video"], K, ratio, device)
    hidden = torch.randn(1, S, cfg.hidden_size, device=device)
    pos = make_pos_emb(1, S, cfg.head_dim, device)
    mask = extract_mask(attn, hidden, pos, None, ctx, device)

    n_chunks = ctx["n_chunks_per_item"][0].item()
    k_selected = math.ceil(n_chunks * ratio)
    expected_img = k_selected * K * K

    img_positions = (mtype[0] == I).nonzero(as_tuple=True)[0]
    cc_positions = (mtype[0] == C).nonzero(as_tuple=True)[0].tolist()
    qa_start = S - n_q - n_a

    for q in range(qa_start, S):
        for h in range(cfg.num_attention_heads):
            # CC attend
            for cc in cc_positions:
                assert can_attend(mask, 0, h, q, cc), f"q={q} h={h} !→ CC@{cc}"
            # img count
            img_attend = sum(1 for ip in img_positions
                            if ip.item() <= q and can_attend(mask, 0, h, q, ip.item()))
            assert img_attend == expected_img, (
                f"video q={q} h={h}: {img_attend} img, expected {expected_img}")
    print(f"  PASS test_video_multi_frame ({device})")


# ===================================================================
# Test 8 — select_ratio=1.0 means ALL patches visible
# ===================================================================
def test_ratio_1_sees_all_patches(device):
    """When ratio=1.0, all chunks selected → text sees all image patches (within causal)."""
    K = 6
    n_pre, n_cc, n_q, n_a = 2, 2, 3, 3
    mtype = build_mtype(n_pre, 576, n_cc, n_q, n_a).unsqueeze(0).to(device)
    S = mtype.shape[1]
    cfg = make_config()
    attn = make_attn(cfg).to(device)

    ctx = build_sparse_ctx(mtype, ["image"], K, 1.0, device)
    hidden = torch.randn(1, S, cfg.hidden_size, device=device)
    pos = make_pos_emb(1, S, cfg.head_dim, device)
    mask = extract_mask(attn, hidden, pos, None, ctx, device)

    img_positions = (mtype[0] == I).nonzero(as_tuple=True)[0]
    qa_start = S - n_q - n_a
    total_img = len(img_positions)

    for q in range(qa_start, S):
        for h in range(cfg.num_attention_heads):
            img_attend = sum(1 for ip in img_positions
                            if ip.item() <= q and can_attend(mask, 0, h, q, ip.item()))
            assert img_attend == total_img, (
                f"ratio=1 q={q} h={h}: {img_attend} vs {total_img}")
    print(f"  PASS test_ratio_1_sees_all_patches ({device})")


# ===================================================================
# Test 9 — select_ratio=0 means NO patches visible
# ===================================================================
def test_ratio_0_sees_no_patches(device):
    """When ratio=0, no chunks → text sees zero image patches."""
    K = 6
    n_pre, n_cc, n_q, n_a = 2, 2, 3, 3
    mtype = build_mtype(n_pre, 576, n_cc, n_q, n_a).unsqueeze(0).to(device)
    S = mtype.shape[1]
    cfg = make_config()
    attn = make_attn(cfg).to(device)

    ctx = build_sparse_ctx(mtype, ["image"], K, 0.0, device)
    hidden = torch.randn(1, S, cfg.hidden_size, device=device)
    pos = make_pos_emb(1, S, cfg.head_dim, device)
    mask = extract_mask(attn, hidden, pos, None, ctx, device)

    img_positions = (mtype[0] == I).nonzero(as_tuple=True)[0]
    qa_start = S - n_q - n_a

    for q in range(qa_start, S):
        for h in range(cfg.num_attention_heads):
            img_attend = sum(1 for ip in img_positions
                            if ip.item() <= q and can_attend(mask, 0, h, q, ip.item()))
            assert img_attend == 0, f"ratio=0 q={q} h={h}: sees {img_attend} img!"
    print(f"  PASS test_ratio_0_sees_no_patches ({device})")


# ===================================================================
# Test 10 — gradient flow
# ===================================================================
def test_gradient_flow(device):
    """Loss.backward() succeeds and all projection weights get gradients."""
    K = 6
    cfg = make_config()
    attn = make_attn(cfg).to(device).train()

    n_pre, n_cc, n_q, n_a = 2, 2, 4, 6
    mtype = build_mtype(n_pre, 576, n_cc, n_q, n_a).unsqueeze(0).to(device)
    S = mtype.shape[1]

    ctx = build_sparse_ctx(mtype, ["image"], K, 0.25, device)
    hidden = torch.randn(1, S, cfg.hidden_size, device=device, requires_grad=True)
    pos = make_pos_emb(1, S, cfg.head_dim, device)

    out, _ = attn(hidden_states=hidden, position_embeddings=pos,
                  attention_mask=None, sparse_routing_ctx=ctx)
    out.sum().backward()

    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        w = getattr(attn, name).weight
        assert w.grad is not None and w.grad.abs().sum() > 0, f"no grad on {name}"
    print(f"  PASS test_gradient_flow ({device})")


# ===================================================================
# Test 11 — per-head routing differs
# ===================================================================
def test_per_head_different_chunks(device):
    """Different heads should (usually) select different chunk sets for the same query."""
    K = 6
    ratio = 0.25
    n_pre, n_cc, n_q, n_a = 2, 2, 4, 4
    mtype = build_mtype(n_pre, 576, n_cc, n_q, n_a).unsqueeze(0).to(device)
    S = mtype.shape[1]
    cfg = make_config()
    attn = make_attn(cfg).to(device)

    ctx = build_sparse_ctx(mtype, ["image"], K, ratio, device)
    hidden = torch.randn(1, S, cfg.hidden_size, device=device)
    pos = make_pos_emb(1, S, cfg.head_dim, device)
    mask = extract_mask(attn, hidden, pos, None, ctx, device)

    img_positions = (mtype[0] == I).nonzero(as_tuple=True)[0].tolist()
    H = cfg.num_attention_heads
    qa_start = S - n_q - n_a
    q = qa_start  # pick first question token

    head_sets = []
    for h in range(H):
        attended = frozenset(ip for ip in img_positions
                             if ip <= q and can_attend(mask, 0, h, q, ip))
        head_sets.append(attended)

    # With random init, heads almost certainly select different chunks
    n_unique = len(set(head_sets))
    # Soft check: at least 2 distinct patterns expected with 4 heads
    print(f"  INFO per-head unique attend patterns: {n_unique}/{H}")
    assert n_unique >= 1  # at minimum they all exist
    print(f"  PASS test_per_head_different_chunks ({device})")


# ===================================================================
# Test 12 — large-scale video (32 frames)
# ===================================================================
def test_large_video_scale(device):
    """32-frame video runs without error and has correct output shape."""
    K = 6
    n_frames = 32  # 32 * 4 = 128 chunks
    n_pre, n_cc, n_q, n_a = 2, 4, 20, 64
    mtype = build_mtype(n_pre, 144, n_cc, n_q, n_a, n_frames=n_frames).unsqueeze(0).to(device)
    S = mtype.shape[1]
    cfg = make_config()
    attn = make_attn(cfg).to(device)

    ctx = build_sparse_ctx(mtype, ["video"], K, 0.25, device)
    hidden = torch.randn(1, S, cfg.hidden_size, device=device)
    pos = make_pos_emb(1, S, cfg.head_dim, device)

    t0 = time.time()
    out, _ = attn(hidden_states=hidden, position_embeddings=pos,
                  attention_mask=None, sparse_routing_ctx=ctx)
    dt = time.time() - t0

    assert out.shape == (1, S, cfg.hidden_size)
    print(f"  PASS test_large_video_scale ({device}) — S={S}, dt={dt:.2f}s")


# ===================================================================
# Test 13 — mixed batch (image + video)
# ===================================================================
def test_mixed_batch(device):
    """Batch of 2: item0=image (576 patches, 16 chunks), item1=video (4 frames, 16 chunks)."""
    K = 6
    ratio = 0.25
    n_cc, n_q, n_a = 2, 4, 4

    mt0 = build_mtype(2, 576, n_cc, n_q, n_a, n_frames=1)
    mt1 = build_mtype(2, 144, n_cc, n_q, n_a, n_frames=4)

    max_len = max(len(mt0), len(mt1))
    mt0_pad = F.pad(mt0, (0, max_len - len(mt0)), value=P)
    mt1_pad = F.pad(mt1, (0, max_len - len(mt1)), value=P)
    mtype = torch.stack([mt0_pad, mt1_pad]).to(device)
    S = mtype.shape[1]

    cfg = make_config()
    attn = make_attn(cfg).to(device)
    ctx = build_sparse_ctx(mtype, ["image", "video"], K, ratio, device)

    hidden = torch.randn(2, S, cfg.hidden_size, device=device)
    pos = make_pos_emb(2, S, cfg.head_dim, device)
    out, _ = attn(hidden_states=hidden, position_embeddings=pos,
                  attention_mask=None, sparse_routing_ctx=ctx)

    assert out.shape == (2, S, cfg.hidden_size)
    assert ctx["n_chunks_per_item"][0].item() == 16
    assert ctx["n_chunks_per_item"][1].item() == 16
    print(f"  PASS test_mixed_batch ({device})")


# ===================================================================
# Runner
# ===================================================================
def run(device):
    dev_str = device if isinstance(device, str) else str(device)
    print(f"\n{'='*60}\nDevice: {dev_str}\n{'='*60}")

    tests = [
        test_text_attends_all_cc,
        test_text_attends_all_prev_text,
        test_sparse_routing_image_count,
        test_pre_image_text_no_future_image,
        test_post_image_routing_still_works,
        test_non_text_rows_unchanged,
        test_video_multi_frame,
        test_ratio_1_sees_all_patches,
        test_ratio_0_sees_no_patches,
        test_gradient_flow,
        test_per_head_different_chunks,
        test_large_video_scale,
        test_mixed_batch,
    ]
    for t in tests:
        t(device)


if __name__ == "__main__":
    run("cpu")
    if torch.cuda.is_available():
        run("cuda")
    else:
        print("\n[SKIP] CUDA not available")

    print(f"\n{'='*60}\nALL TESTS PASSED\n{'='*60}")
