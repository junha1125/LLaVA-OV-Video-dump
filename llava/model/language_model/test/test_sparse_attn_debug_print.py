"""Debug print verification for MoCSparseAttention.

Run:
    python -m llava.model.language_model.test.test_sparse_attn_debug_print
"""

import torch
import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from llava.constants import MODALITY_TEXT as T, MODALITY_IMAGE as I, MODALITY_COMPRESSED_CONTEXT as C, IGNORE_INDEX
from llava.model.language_model.sparse_attn import compute_chunk_info, MoCSparseAttention
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention


def _make_attn(cfg):
    attn = Qwen3Attention(cfg, layer_idx=0)
    attn.__class__ = MoCSparseAttention
    return attn


cfg = Qwen3Config(hidden_size=256, num_attention_heads=4, num_key_value_heads=2,
                   head_dim=64, attention_bias=False, attention_dropout=0.0,
                   rms_norm_eps=1e-6, use_sliding_window=False, sliding_window=None, max_window_layers=0)
cfg._attn_implementation = 'eager'

K, n_cc, n_q, n_a = 6, 4, 8, 10

# === Image test ===
print('=== Image (576 patches, 16 chunks, ratio=0.25 -> k=4, ppchunk=36) ===')
parts_img = [torch.full((2,), T), torch.full((576,), I), torch.full((n_cc,), C),
             torch.full((n_q,), T), torch.full((n_a,), T)]
mtype_img = torch.cat(parts_img).unsqueeze(0)
S_img = mtype_img.shape[1]
labels_img = torch.full((1, S_img), IGNORE_INDEX, dtype=torch.long)
labels_img[0, -n_a:] = torch.arange(100, 100 + n_a)
chunk_info_img = compute_chunk_info(mtype_img, ['image'], K, torch.device('cpu'))
ctx_img = dict(text_token_mask=(mtype_img == T), attn_chunk_select_ratio=0.25, **chunk_info_img)

attn1 = _make_attn(cfg)
out1, _ = attn1(hidden_states=torch.randn(1, S_img, 256),
                position_embeddings=(torch.randn(1, S_img, 64), torch.randn(1, S_img, 64)),
                attention_mask=None, sparse_routing_ctx=ctx_img)

# === Video test ===
print()
print('=== Video (4frames x 144patches, 16 chunks, ratio=0.5 -> k=8) ===')
parts_vid = [torch.full((2,), T)]
for _ in range(4):
    parts_vid.append(torch.full((144,), I))
    parts_vid.append(torch.full((n_cc,), C))
parts_vid += [torch.full((n_q,), T), torch.full((n_a,), T)]
mtype_vid = torch.cat(parts_vid).unsqueeze(0)
S_vid = mtype_vid.shape[1]
labels_vid = torch.full((1, S_vid), IGNORE_INDEX, dtype=torch.long)
labels_vid[0, -n_a:] = torch.arange(100, 100 + n_a)
chunk_info_vid = compute_chunk_info(mtype_vid, ['video'], K, torch.device('cpu'))
ctx_vid = dict(text_token_mask=(mtype_vid == T), attn_chunk_select_ratio=0.5, **chunk_info_vid)

attn2 = _make_attn(cfg)
out2, _ = attn2(hidden_states=torch.randn(1, S_vid, 256),
                position_embeddings=(torch.randn(1, S_vid, 64), torch.randn(1, S_vid, 64)),
                attention_mask=None, sparse_routing_ctx=ctx_vid)

# === ratio=0 test (text tokens should see 0 image patches) ===
print()
print('=== Image ratio=0 (576 patches, 16 chunks, ratio=0 -> k=0, no image attend) ===')
ctx_zero = dict(text_token_mask=(mtype_img == T), attn_chunk_select_ratio=0, **chunk_info_img)
attn3 = _make_attn(cfg)
out3, _ = attn3(hidden_states=torch.randn(1, S_img, 256),
                position_embeddings=(torch.randn(1, S_img, 64), torch.randn(1, S_img, 64)),
                attention_mask=None, sparse_routing_ctx=ctx_zero)
