"""
Stage 7: Mixture-of-Contexts (MoC) Sparse Routing Attention

Per-layer, per-head, per-text-token sparse routing for image/video patches.
Text tokens (question + answer) selectively attend to top-k spatial chunks instead of all patches.
Non-text tokens keep standard causal attention (with CC restriction from Stage 5).
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Callable

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from llava.constants import MODALITY_IMAGE
from llava.utils import rank0_print


# ---------------------------------------------------------------------------
# Pre-computation: spatial chunk metadata
# ---------------------------------------------------------------------------

def compute_chunk_info(
    modality_types: torch.Tensor,
    modalities: List[str],
    patch_wh_one_chunk: int,
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """Pre-compute spatial chunk metadata from modality_types.

    이미지/비디오 패치를 K×K 크기의 spatial chunk로 분할하고,
    각 chunk가 시퀀스 상 어느 위치에 있는지를 기록한다.
    이 정보는 MoCSparseAttention에서 answer token이
    어떤 chunk를 attend할지 결정하는 데 사용된다.

    Chunk 분할 예시 (K=6):
        Image  (24×24 = 576 patches) → (24/6)×(24/6) = 4×4 = 16 chunks
        Video  (12×12 = 144 patches/frame) → (12/6)×(12/6) = 2×2 = 4 chunks/frame
        비디오 32프레임 → 총 32×4 = 128 chunks

    패치 레이아웃 (image, 24×24, K=6):
        chunk(0,0)  chunk(0,1)  chunk(0,2)  chunk(0,3)
        ┌──────────┬──────────┬──────────┬──────────┐
        │ rows 0-5 │ rows 0-5 │ rows 0-5 │ rows 0-5 │
        │ cols 0-5 │ cols 6-11│cols 12-17│cols 18-23│
        ├──────────┼──────────┼──────────┼──────────┤
        │chunk(1,0)│chunk(1,1)│  ...     │chunk(1,3)│
        ├──────────┼──────────┼──────────┼──────────┤
        │  ...     │  ...     │  ...     │  ...     │
        ├──────────┼──────────┼──────────┼──────────┤
        │chunk(3,0)│  ...     │  ...     │chunk(3,3)│
        └──────────┴──────────┴──────────┴──────────┘

    시퀀스 내 토큰 배치 (비디오, CC 포함):
        [frame0 patches ×144] [CC ×n] [frame1 patches ×144] [CC ×n] ...
        image_mask로 MODALITY_IMAGE만 추출하므로 CC 토큰은 자동 제외됨.
        각 프레임 내 패치는 raster order (row-major)로 배치되어 있으므로,
        image_pos[f*144 + row*12 + col] 이 frame f의 (row, col) 패치 위치.

    Args:
        modality_types: [B, S] 각 토큰의 modality type (TEXT/IMAGE/CC/PAD)
        modalities: 길이 B 리스트, 각 batch item이 "image" 또는 "video"
        patch_wh_one_chunk: K — chunk 한 변의 패치 수
        device: 출력 텐서의 device

    Returns:
        None if no image tokens, otherwise dict with:
            position_to_chunk  [B, S]                  시퀀스 위치 → chunk 인덱스 (-1 = 비이미지)
            chunk_positions    [B, max_n_chunks, K*K]   각 chunk에 속하는 시퀀스 위치들
            chunk_valid_mask   [B, max_n_chunks]        유효한 chunk인지 (배치 내 패딩 구분)
            n_chunks_per_item  [B]                      batch item별 총 chunk 수
            image_mask         [B, S]                   MODALITY_IMAGE 위치 boolean
    """
    B, S = modality_types.shape
    K = patch_wh_one_chunk

    image_mask = (modality_types == MODALITY_IMAGE)
    if not image_mask.any():
        return None

    position_to_chunk_list = []
    chunk_positions_list = []
    n_chunks_list = []

    for b in range(B):
        is_video = (modalities[b] == "video")
        patches_per_frame = 144 if is_video else 576  # 12×12 or 24×24
        W = 12 if is_video else 24                     # grid width = sqrt(patches_per_frame)

        # image_pos: MODALITY_IMAGE인 위치만 추출 (CC 토큰은 자동 제외)
        # 프레임 내 패치들은 연속적이고 raster order로 정렬됨
        image_pos = image_mask[b].nonzero(as_tuple=True)[0]
        n_patches = len(image_pos)

        # position_to_chunk: 시퀀스 위치 → chunk 인덱스 (-1 = 이미지 아님)
        ptc = torch.full((S,), -1, dtype=torch.long, device=device)

        if n_patches == 0:
            position_to_chunk_list.append(ptc)
            chunk_positions_list.append([])
            n_chunks_list.append(0)
            continue

        n_frames = n_patches // patches_per_frame
        assert n_patches == n_frames * patches_per_frame, (
            f"Batch {b}: {n_patches} image patches not divisible by "
            f"patches_per_frame={patches_per_frame} (modality={modalities[b]})"
        )

        n_chunks_per_dim = W // K  # 한 축의 chunk 수 (image: 4, video: 2)
        assert W % K == 0, f"Grid width {W} not divisible by chunk size K={K}"

        chunk_idx = 0  # 배치 item 내 전역 chunk 인덱스 (프레임 간 연속 번호)
        item_chunk_positions = []

        for f in range(n_frames):
            # 이 프레임의 패치 시퀀스 위치들 (raster order)
            frame_pos = image_pos[f * patches_per_frame : (f + 1) * patches_per_frame]

            # 2D 격자 위에서 K×K 블록 순회 (cr=chunk_row, cc=chunk_col)
            for cr in range(n_chunks_per_dim):
                for cc in range(n_chunks_per_dim):
                    positions = []
                    # chunk 내부의 K×K 패치를 raster order로 수집
                    for r in range(K):
                        for c in range(K):
                            row = cr * K + r     # 프레임 내 절대 행
                            col = cc * K + c     # 프레임 내 절대 열
                            flat_idx = row * W + col  # raster index
                            seq_pos = frame_pos[flat_idx].item()
                            positions.append(seq_pos)
                            ptc[seq_pos] = chunk_idx
                    item_chunk_positions.append(positions)
                    chunk_idx += 1

        position_to_chunk_list.append(ptc)
        chunk_positions_list.append(item_chunk_positions)
        n_chunks_list.append(chunk_idx)

    max_n_chunks = max(n_chunks_list) if n_chunks_list else 0
    if max_n_chunks == 0:
        return None

    # 배치 내 chunk 수가 다를 수 있으므로 max_n_chunks로 패딩하여 텐서화
    patches_per_chunk = K * K
    position_to_chunk = torch.stack(position_to_chunk_list)  # [B, S]

    chunk_positions = torch.zeros(
        B, max_n_chunks, patches_per_chunk, dtype=torch.long, device=device
    )
    chunk_valid_mask = torch.zeros(B, max_n_chunks, dtype=torch.bool, device=device)

    for b in range(B):
        for c, positions in enumerate(chunk_positions_list[b]):
            chunk_positions[b, c] = torch.tensor(
                positions, dtype=torch.long, device=device
            )
            chunk_valid_mask[b, c] = True
    # 패딩된 chunk (chunk_valid_mask=False)의 chunk_positions는 0으로 남음.
    # MoCSparseAttention에서 gather 시 position 0의 K가 수집되지만,
    # valid_mask로 zero-out 후 mean-pool하므로 결과에 영향 없음.
    # 또한 cosine similarity에서 -inf로 마스킹되어 top-k에 선택 불가.

    n_chunks_per_item = torch.tensor(n_chunks_list, dtype=torch.long, device=device)

    return {
        "position_to_chunk": position_to_chunk,
        "chunk_positions": chunk_positions,
        "chunk_valid_mask": chunk_valid_mask,
        "n_chunks_per_item": n_chunks_per_item,
        "image_mask": image_mask,
    }


# ---------------------------------------------------------------------------
# MoC Sparse Routing Attention
# ---------------------------------------------------------------------------

class MoCSparseAttention(Qwen3Attention):
    """Qwen3Attention with Mixture-of-Contexts sparse routing for text tokens.

    Behaviour per token type:
        Non-text tokens (image/CC) → standard attention (causal + CC restriction from Stage 5)
        Text tokens (question + answer) → attend to CC + other text tokens
                            + top-k spatial chunks (per-layer, per-head, per-token)
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        sparse_ctx = kwargs.pop("sparse_routing_ctx", None)

        # --- Q, K, V computation (identical to Qwen3Attention) ----------------
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Save pre-RoPE Q/K for routing similarity computation
        query_states_pre_rope = query_states
        key_states_pre_rope = key_states

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # --- Decide path ------------------------------------------------------
        should_route = (
            sparse_ctx is not None
            and sparse_ctx["text_token_mask"].any()
            and sparse_ctx["n_chunks_per_item"].sum() > 0
        )

        if not should_route:
            attn_output, attn_weights = self._standard_attention(
                query_states, key_states, value_states, attention_mask, **kwargs
            )
        else:
            attn_output, attn_weights = self._sparse_routing_attention(
                query_states, key_states, value_states, attention_mask, sparse_ctx,
                query_states_pre_rope, key_states_pre_rope,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    # ------------------------------------------------------------------
    # Standard attention (same logic as Qwen3Attention)
    # ------------------------------------------------------------------
    def _standard_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if (
                self.config._attn_implementation == "sdpa"
                and kwargs.get("output_attentions", False)
            ):
                pass  # fall back to eager
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

        return attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Sparse routing attention (text tokens: question + answer)
    # ------------------------------------------------------------------
    def _sparse_routing_attention(
        self,
        query_states: torch.Tensor,          # [B, H, S, D]  (post-RoPE)
        key_states: torch.Tensor,            # [B, H_kv, S, D]  (post-RoPE)
        value_states: torch.Tensor,          # [B, H_kv, S, D]
        attention_mask: Optional[torch.Tensor],  # [B, 1, S, S] or None
        sparse_ctx: Dict[str, torch.Tensor],
        query_pre_rope: torch.Tensor,        # [B, H, S, D]  (pre-RoPE)
        key_pre_rope: torch.Tensor,          # [B, H_kv, S, D]  (pre-RoPE)
    ) -> Tuple[torch.Tensor, None]:
        B, H, S, D = query_states.shape
        H_kv = key_states.shape[1]
        dtype = query_states.dtype
        device = query_states.device
        min_dtype = torch.finfo(dtype).min

        text_token_mask = sparse_ctx["text_token_mask"]         # [B, S]
        position_to_chunk = sparse_ctx["position_to_chunk"]   # [B, S]
        chunk_positions = sparse_ctx["chunk_positions"]       # [B, C_max, K*K]
        chunk_valid_mask = sparse_ctx["chunk_valid_mask"]     # [B, C_max]
        n_chunks_per_item = sparse_ctx["n_chunks_per_item"]   # [B]
        image_mask = sparse_ctx["image_mask"]                 # [B, S]
        select_ratio = sparse_ctx["attn_chunk_select_ratio"]

        C_max = chunk_positions.shape[1]                      # max chunks in batch
        ppchunk = chunk_positions.shape[2]                    # patches per chunk (K*K)

        # ---- Steps 1-3: routing (skipped when ratio=0) -------------------
        if select_ratio > 0:
            # ---- Step 1: chunk K representatives (mean-pool, pre-RoPE) ---
            flat_pos = chunk_positions.reshape(B, -1)             # [B, C_max * ppchunk]
            flat_pos_exp = flat_pos[:, None, :, None].expand(B, H_kv, -1, D)
            chunk_K = key_pre_rope.gather(2, flat_pos_exp)        # [B, H_kv, C_max*ppchunk, D]
            chunk_K = chunk_K.reshape(B, H_kv, C_max, ppchunk, D)

            # zero-out invalid chunks before mean-pool
            chunk_K = chunk_K * chunk_valid_mask[:, None, :, None, None].to(dtype)
            chunk_repr_kv = chunk_K.sum(dim=3) / ppchunk          # [B, H_kv, C_max, D]

            # expand to full query heads (GQA)
            chunk_repr = repeat_kv(chunk_repr_kv, self.num_key_value_groups)  # [B, H, C_max, D]

            # ---- Step 2: cosine similarity (pre-RoPE) --------------------
            Q_norm = F.normalize(query_pre_rope, dim=-1)          # [B, H, S, D]
            C_norm = F.normalize(chunk_repr, dim=-1)              # [B, H, C_max, D]
            sim = torch.einsum("bhsd,bhcd->bhsc", Q_norm, C_norm) # [B, H, S, C_max]

            # mask invalid chunks to -inf
            sim.masked_fill_(~chunk_valid_mask[:, None, None, :], float("-inf"))

            # ---- Step 3: top-k selection ---------------------------------
            k_per_item = (
                (n_chunks_per_item.float() * select_ratio).ceil().long().clamp(min=1)
            )                                                     # [B]
            k_max = min(k_per_item.max().item(), C_max)

            _, topk_idx = sim.topk(k_max, dim=-1)                # [B, H, S, k_max]

            # Build selected_chunks boolean mask [B, H, S, C_max]
            selected_chunks = torch.zeros(
                B, H, S, C_max, dtype=torch.bool, device=device
            )
            selected_chunks.scatter_(-1, topk_idx, True)

            # per-item k correction: zero out extra selections for items with k_i < k_max
            for b in range(B):
                k_b = k_per_item[b].item()
                if k_b < k_max:
                    extra_idx = topk_idx[b, :, :, k_b:]          # [H, S, k_max - k_b]
                    selected_chunks[b].scatter_(-1, extra_idx, False)

        # ---- Step 4: build attention mask --------------------------------
        # Ensure we have a concrete 4D causal mask
        if attention_mask is None:
            causal = torch.triu(
                torch.full((S, S), min_dtype, dtype=dtype, device=device),
                diagonal=1,
            )
            attention_mask = causal[None, None, :, :]         # [1, 1, S, S]

        # Expand base mask from [B, 1, S, S] to [B, H, S, S]
        full_mask = attention_mask.expand(B, H, S, S).clone()

        # 4a. Mask ALL image-patch columns for text-token rows (항상 실행)
        #     text_image[b, i, j] = text[b,i] & image[b,j]
        text_image_block = text_token_mask[:, :, None] & image_mask[:, None, :]  # [B, S, S]
        full_mask.masked_fill_(text_image_block[:, None, :, :], min_dtype)

        # 4b. Unmask selected-chunk patches for answer rows (ratio > 0일 때만)
        if select_ratio > 0:
            ptc = position_to_chunk[:, None, None, :]             # [B, 1, 1, S_key]
            ptc_clamped = ptc.clamp(min=0).expand(B, H, S, S)    # [B, H, S_q, S_key]
            is_image_key = (ptc >= 0)                             # [B, 1, 1, S_key]

            unmask = selected_chunks.gather(-1, ptc_clamped)      # [B, H, S, S]
            unmask &= is_image_key
            unmask &= text_token_mask[:, None, :, None]
            # causal bound: key position(col) <= query position(row)
            seq_idx = torch.arange(S, device=device)
            unmask &= (seq_idx[None, None, None, :] <= seq_idx[None, None, :, None])
            full_mask.masked_fill_(unmask, 0.0)
            del unmask

        # ---- Debug start: question/answer text tokens의 실제 attend 수 출력 ---- #
        if self.layer_idx == 0:
            text_positions = text_token_mask[0].nonzero(as_tuple=True)[0]
            n_img = image_mask[0].sum().item()
            k_str = f"k=0/{n_chunks_per_item[0].item()}" if select_ratio == 0 else \
                    f"k={k_per_item[0].item()}/{n_chunks_per_item[0].item()}"

            # question = image patch 직후 첫 2개 text tokens
            # image_mask만 사용하면 PAD에 영향받지 않음
            img_positions = image_mask[0].nonzero(as_tuple=True)[0]
            last_img = img_positions[-1].item() if len(img_positions) > 0 else -1
            question_cands = [p for p in text_positions if p.item() > last_img][:2]
            # answer: last 2 text tokens
            answer_cands = text_positions[-2:] if len(text_positions) >= 2 else text_positions

            lines = []
            for label, positions in [("question", question_cands), ("answer", answer_cands)]:
                for txt_pos in positions:
                    q = txt_pos.item()
                    attendable = (full_mask[0, :, q, :] > min_dtype + 1)  # [H, S]
                    per_head_counts = attendable.sum(dim=-1)              # [H]
                    img_attendable = attendable & image_mask[0:1]         # [H, S]
                    per_head_img = img_attendable.sum(dim=-1)             # [H]
                    lines.append(
                        f"  {label}[{q}]: total_attend(per head)={per_head_counts.tolist()}, "
                        f"img_patches(per head)={per_head_img.tolist()}"
                    )
            rank0_print(
                f"[MoC-Sparse L{self.layer_idx}] "
                f"img_patches={n_img}, "
                f"{k_str} chunks, "
                f"ppchunk={ppchunk}\n" + "\n".join(lines)
            )
        # ---- Debug end ---------------------------------------------------- #

        # ---- Step 5: SDPA ------------------------------------------------
        # Per-head masks [B, H, S, S] are not supported by flash/mem-efficient
        # backends. Force the math (naive) backend for correctness.
        key_states_full = repeat_kv(key_states, self.num_key_value_groups)
        value_states_full = repeat_kv(value_states, self.num_key_value_groups)

        query_states = query_states.contiguous()
        key_states_full = key_states_full.contiguous()
        value_states_full = value_states_full.contiguous()

        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states_full,
                value_states_full,
                attn_mask=full_mask,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                scale=self.scaling,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None
