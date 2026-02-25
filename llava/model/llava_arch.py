#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, MODALITY_TEXT, MODALITY_IMAGE, MODALITY_COMPRESSED_CONTEXT, MODALITY_PAD
from llava.model.visual_pe import VisualRotaryEmbedding

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(self.config, vision_cfg=self.vision_tower.config if hasattr(self.vision_tower, "config") else None)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

            if getattr(config, "add_visual_RoPE", False):
                self.visual_rope = VisualRotaryEmbedding(
                    hidden_size=config.hidden_size,
                    use_gate=getattr(config, "visual_rope_gate", False),
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", None)

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config if hasattr(vision_tower, "config") else None)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)

            if getattr(model_args, "add_visual_RoPE", False):
                self.visual_rope = VisualRotaryEmbedding(
                    hidden_size=self.config.hidden_size,
                    use_gate=getattr(model_args, "visual_rope_gate", False),
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            visual_rope_w = get_w(mm_projector_weights, "visual_rope")
            if visual_rope_w and hasattr(self, "visual_rope"):
                incompatible_keys = self.visual_rope.load_state_dict(visual_rope_w, strict=False)
                rank0_print(f"Loaded visual_rope weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")

        if getattr(self.config, "vision_tower_pretrained", None):
            vit_path = self.config.vision_tower_pretrained
            vit_blob = torch.load(vit_path, map_location="cpu")
            # 한 파일에 묶어 저장한 포맷 지원: {"vision_tower": state_dict, ...}
            if isinstance(vit_blob, dict) and "vision_tower" in vit_blob:
                vt_sd = vit_blob["vision_tower"]
            else:
                vt_sd = vit_blob  # 순수 state_dict 저장된 경우

            incompatible_vt = self.vision_tower.load_state_dict(vt_sd, strict=True)
            rank0_print(f"Loaded vision tower weights from {vit_path}. Incompatible keys: {incompatible_vt}")

            match = re.match(r"^mlp(\d+)x_gelu_(\d+)_(\d+)\+(\d+)$", self.config.mm_projector_type)
            if match:
                vt_pro = vit_blob["mm_projector"]
                mlp_depth = int(match.group(1))
                keys = []
                for i in range(mlp_depth):
                    lin_idx = i * 2  # Linear layer index
                    keys.extend([f"{lin_idx}.weight",f"{lin_idx}.bias"])
                mlp_sd = {k: vt_pro[k] for k in keys}
                module_slice_end = mlp_depth * 2 - 1  
                incompatible_pro = self.mm_projector[:module_slice_end].load_state_dict(mlp_sd, strict=True)
                rank0_print(f"Loaded projector weights from {vit_path}. Incompatible keys: {incompatible_pro}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature, stride=2):
        num_frames, num_tokens, num_dim = image_feature.shape
        height = width = int(math.sqrt(num_tokens))
        assert height * width == num_tokens, f"num_tokens={num_tokens} is not a perfect square (got h={height})"
        image_feature = image_feature.reshape(num_frames, height, width, num_dim)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.reshape(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_images_wan(self, images_list):
        """WAN: encode per-video frame groups preserving temporal structure."""
        wan_features = self.get_model().get_vision_tower()(images_list)
        # List[Tensor], each (T'_i, num_patches, vision_dim)
        split_sizes = [f.shape[0] for f in wan_features]
        all_feats = torch.cat(wan_features, dim=0)
        all_feats = self.get_model().mm_projector(all_feats)
        return list(torch.split(all_feats, split_sizes))

    def add_token_per_grid(self, image_feature):
        """
        For video features with 2D spatial pooling, add a special token at the end of each spatial grid row.
        [p p p p p p p p p p p p p]       [p p p p p p p p p p p p p \n]
        [p p p p p p p p p p p p p]       [p p p p p p p p p p p p p \n]
        [p p p p p p p p p p p p p]  →    [p p p p p p p p p p p p p \n]
        [p p p p p p p p p p p p p]       [p p p p p p p p p p p p p \n]
                13×13 = 169 tokens              13×14 = 182 tokens
        """
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None, frame_indices=None):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            self._modality_types = None
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        # === Compressed-Context embeddings (precompute once) ===
        num_compressed_context = getattr(self.config, "num_compressed_context", 0)
        if num_compressed_context > 0:
            cc_token_ids = torch.tensor(
                self.config.compressed_context_token_ids, device=input_ids.device
            )
            cc_embeds = self.get_model().embed_tokens(cc_token_ids)  # [num_cc, dim]

        # [DEBUG] Print frame_indices for debugging
        # if frame_indices is not None:
        #     for i, fi in enumerate(frame_indices):
        #         if fi is not None:
        #             print(f"[FrameIdx] batch_item={i}, modality={modalities[i]}, num_frames={len(fi)}, frame_indices={fi}")
        #         else:
        #             print(f"[FrameIdx] batch_item={i}, modality={modalities[i]}, frame_indices=None (image)")
        # [FrameIdx] batch_item=0, modality=image, frame_indices=None (image)
        # [FrameIdx] batch_item=1, modality=image, frame_indices=None (image)
        # [FrameIdx] batch_item=2, modality=video, num_frames=6, frame_indices=[0, 25, 50, 75, 100, 125]
        # [FrameIdx] batch_item=3, modality=video, num_frames=16, frame_indices=[0, 53, 107, 160, 214, 268, 321, 375, 428, 482, 536, 589, 643, 696, 750, 804]
        

        # ==========================================================
        # Branch A: images is a list or 5D tensor (multi-image / video)
        #   Each batch item may have multiple crops or video frames.
        # ==========================================================
        if type(images) is list or images.ndim == 5:
            # Ensure every image tensor is at least 4D: [num_crops, C, H, W]
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            # Identify which batch indices correspond to video
            video_idx_in_batch = [i for i, mod in enumerate(modalities) if mod == "video"]

            # Normalize all images to 4D and collect them
            images_4d = []
            for image in images:
                if image.ndim == 4:
                    images_4d.append(image)
                else:
                    images_4d.append(image.unsqueeze(0))

            # Concatenate all images across batch for a single encode pass
            split_sizes = [image.shape[0] for image in images_4d]
            concat_images = torch.cat(images_4d, dim=0)

            # Encode all images at once, then split back per batch item
            # Each element: [num_crops_or_frames, num_patches, hidden_dim]
            encoded_image_features = self.encode_images(concat_images)
            per_item_features = torch.split(encoded_image_features, split_sizes)

            # Apply 2D spatial pooling for video features (downsample patches)
            image_features = []
            for idx, image_feat in enumerate(per_item_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)

            for idx, feat in enumerate(image_features):
                expected = 144 if idx in video_idx_in_batch else 576
                patch_num_error = (f"Current version requires 576 patches for image and 144 patches for video frame.\n"
                                   f"Batch {idx}: expected {expected} patches, got {feat.shape[1]} (shape={feat.shape})")
                assert feat.shape[1] == expected, patch_num_error

            # Apply Visual RoPE (3D sinusoidal PE) to all visual features if enabled
            add_visual_RoPE = getattr(self.config, "add_visual_RoPE", False)
            if add_visual_RoPE:
                visual_rope = self.get_model().visual_rope
                for idx in range(len(image_features)):
                    fi = None
                    if frame_indices is not None and frame_indices[idx] is not None:
                        fi = frame_indices[idx]
                    image_features[idx] = visual_rope(image_features[idx], frame_indices=fi)

            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "spatial_unpad")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "grid")

            # ----- Merge patches based on configured strategy -----
            if mm_patch_merge_type == "flat":
                image_modality_types = []
                if num_compressed_context > 0:
                    new_features = []
                    for idx, x in enumerate(image_features):
                        if idx in video_idx_in_batch:
                            mtype_unit_img, mtype_repeat = x.shape[1], x.shape[0]
                            cc_exp = cc_embeds.unsqueeze(0).expand(x.shape[0], -1, -1)
                            x = torch.cat([x, cc_exp], dim=1)
                            new_features.append(x.flatten(0, 1))
                        else:
                            flat_patches = x.flatten(0, 1)
                            new_features.append(torch.cat([flat_patches, cc_embeds], dim=0))
                            mtype_unit_img, mtype_repeat = flat_patches.shape[0], 1
                        # modality type: [IMAGE * n_patches, CC * num_cc], repeated per frame (video) or once (image)
                        unit_mtype = torch.cat([
                            torch.full((mtype_unit_img,), MODALITY_IMAGE, dtype=torch.long, device=x.device),
                            torch.full((num_compressed_context,), MODALITY_COMPRESSED_CONTEXT, dtype=torch.long, device=x.device),
                        ])
                        image_modality_types.append(unit_mtype.repeat(mtype_repeat))
                    image_features = new_features
                else:
                    image_features = [x.flatten(0, 1) for x in image_features]
                    image_modality_types = [
                        torch.full((x.shape[0],), MODALITY_IMAGE, dtype=torch.long, device=x.device)
                        for x in image_features
                    ]

            elif mm_patch_merge_type.startswith("spatial"):
                raise NotImplementedError("Only spatial_unpad merge is implemented in this version. Other spatial merge strategies are not yet implemented.")
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):

                    if image_idx in video_idx_in_batch:
                        if add_visual_RoPE:
                            if num_compressed_context > 0:
                                cc_exp = cc_embeds.unsqueeze(0).expand(image_feature.shape[0], -1, -1)
                                image_feature = torch.cat([image_feature, cc_exp], dim=1)
                            image_feature = image_feature.flatten(0, 1)
                        else:
                            num_frames = image_feature.shape[0]
                            resize_h = int(math.sqrt(image_feature.shape[1]))
                            tokens_per_frame = resize_h * (resize_h + 1)
                            image_feature = self.add_token_per_grid(image_feature)
                            if num_compressed_context > 0:
                                chunks = image_feature.split(tokens_per_frame)
                                parts = []
                                for chunk in chunks:
                                    parts.append(chunk)
                                    parts.append(cc_embeds)
                                image_feature = torch.cat(parts, dim=0)
                        new_image_features.append(image_feature)

                    elif image_feature.shape[0] > 1:
                        # Multi-crop image: spatial unpadding merge
                        base_image_feature = image_feature[0]
                        crop_features = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        # Reshape crops into 2x2 spatial grid
                        #   [4, h*w, dim] -> [2, 2, h, w, dim]
                        crop_features = crop_features.view(2, 2, height, width, -1)

                        # Rearrange to [dim, 2h, 2w] layout
                        #   [2, 2, h, w, dim] -> [dim, 2, h, 2, w]
                        crop_features = crop_features.permute(4, 0, 2, 1, 3).contiguous()
                        #   [dim, 2, h, 2, w] -> [dim, 2h, 2w]
                        crop_features = crop_features.flatten(1, 2).flatten(2, 3)

                        # Remove padding from non-square images
                        crop_features = unpad_image(crop_features, image_sizes[image_idx])

                        # Append image_newline token column at the end of each row
                        # crop_features shape: [dim, H', W'] -> [dim, H', W'+1]
                        newline_col = self.model.image_newline[:, None, None].expand(
                            *crop_features.shape[:-1], 1
                        ).to(crop_features.device)
                        crop_features = torch.cat((crop_features, newline_col), dim=-1)

                        # Flatten spatial dims and transpose to [num_tokens, dim]
                        crop_features = crop_features.flatten(1, 2).transpose(0, 1)

                        # Prepend base image features
                        merged_feature = torch.cat((base_image_feature, crop_features), dim=0)
                        if num_compressed_context > 0:
                            merged_feature = torch.cat([merged_feature, cc_embeds], dim=0)
                        new_image_features.append(merged_feature)

                    else:
                        # Single image: just append one newline token
                        single_feature = image_feature[0]
                        single_feature = torch.cat(
                            (single_feature, self.model.image_newline[None]), dim=0
                        )
                        if num_compressed_context > 0:
                            single_feature = torch.cat([single_feature, cc_embeds], dim=0)
                        new_image_features.append(single_feature)

                image_features = new_image_features

        # ==========================================================
        # Branch B: simple 4D tensor — encode directly
        # ==========================================================
        else:
            raise NotImplementedError("llava/train/dataloader.py, llava/train/conversation_processor.py do not yeild simple 4D image tensors.")
            image_features = self.encode_images(images)
            if num_compressed_context > 0:
                num_patches = image_features.shape[1]
                cc_exp = cc_embeds.unsqueeze(0).expand(image_features.shape[0], -1, -1)
                image_features = torch.cat([image_features, cc_exp], dim=1)
                single_mtype = torch.cat([
                    torch.full((num_patches,), MODALITY_IMAGE, dtype=torch.long, device=image_features.device),
                    torch.full((num_compressed_context,), MODALITY_COMPRESSED_CONTEXT, dtype=torch.long, device=image_features.device),
                ])
                image_modality_types = [single_mtype for _ in range(image_features.shape[0])]
            else:
                single_mtype = torch.full((image_features.shape[1],), MODALITY_IMAGE, dtype=torch.long, device=image_features.device)
                image_modality_types = [single_mtype for _ in range(image_features.shape[0])]

        # ==========================================================
        # Step 1: Fill in defaults for None inputs (save originals)
        # ==========================================================
        orig_labels = labels
        orig_position_ids = position_ids
        orig_attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # ==========================================================
        # Step 2: Remove padding tokens using attention_mask
        #   After this, input_ids/labels become lists of 1D tensors
        #   (variable length per batch item, padding removed)
        # ==========================================================
        input_ids = [cur_ids[cur_mask] for cur_ids, cur_mask in zip(input_ids, attention_mask)]
        labels = [cur_lab[cur_mask] for cur_lab, cur_mask in zip(labels, attention_mask)]

        # ==========================================================
        # Step 3: Replace IMAGE_TOKEN_INDEX positions with actual
        #   image feature embeddings, for each batch item
        # ==========================================================
        new_input_embeds = []
        new_labels = []
        new_modality_types = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            # Note "num_images" means the number of <IMAGE> placeholders. 
            # Video = 1 <IMAGE> / One image in 1 conversation = 1 <IMAGE> / Two image in 1 conversation = 2 <IMAGE> 
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            if num_images == 0:
                # No image tokens in this sequence.
                # Cat zero-length image tensor to keep image encoder in
                # the compute graph for gradient flow.
                cur_image_features = image_features[cur_image_idx]
                text_embeds = self.get_model().embed_tokens(cur_input_ids)
                combined_embeds = torch.cat([text_embeds, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(combined_embeds)
                new_labels.append(labels[batch_idx])
                new_modality_types.append(
                    torch.full((combined_embeds.shape[0],), MODALITY_TEXT, dtype=torch.long, device=combined_embeds.device)
                )
                cur_image_idx += 1
                continue

            # Locate all IMAGE_TOKEN_INDEX positions.
            # Add sentinel -1 (start) and seq_len (end) so that slicing
            #   [boundaries[i]+1 : boundaries[i+1]]
            # gives each text-only segment between image tokens.
            #
            # Example: tokens = [A, B, <IMG>, C, D, <IMG>, E]
            #   image_positions = [2, 5]
            #   boundaries      = [-1, 2, 5, 7]
            #   text segments   = [0:2]=[A,B], [3:5]=[C,D], [6:7]=[E]
            image_positions = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            boundaries = [-1] + image_positions + [cur_input_ids.shape[0]]

            cur_labels = labels[batch_idx]
            text_id_segments = []
            text_label_segments = []
            for i in range(len(boundaries) - 1):
                seg_start = boundaries[i] + 1
                seg_end = boundaries[i + 1]
                text_id_segments.append(cur_input_ids[seg_start:seg_end])
                text_label_segments.append(cur_labels[seg_start:seg_end])

            # Embed all text tokens at once, then split back into segments
            segment_lengths = [seg.shape[0] for seg in text_label_segments]
            all_text_embeds = self.get_model().embed_tokens(torch.cat(text_id_segments))
            text_embed_segments = torch.split(all_text_embeds, segment_lengths, dim=0)

            # Interleave: [text_seg_0, img_feat_0, text_seg_1, img_feat_1, ..., text_seg_N]
            interleaved_embeds = []
            interleaved_labels = []
            interleaved_mtypes = []

            for i in range(num_images + 1):
                interleaved_embeds.append(text_embed_segments[i])
                interleaved_labels.append(text_label_segments[i])
                interleaved_mtypes.append(
                    torch.full((text_embed_segments[i].shape[0],), MODALITY_TEXT, dtype=torch.long, device=cur_labels.device)
                )

                if i < num_images:  # text segments = num_images+1, image segments = num_images, so skip image on the last iteration
                    try:
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_mtype = image_modality_types[cur_image_idx]
                    except IndexError: # in case that two <IMG> in a conversations, but only one image provided
                        cur_image_features = image_features[cur_image_idx - 1]
                        cur_image_mtype = image_modality_types[cur_image_idx - 1]
                    cur_image_idx += 1
                    interleaved_embeds.append(cur_image_features)
                    interleaved_mtypes.append(cur_image_mtype)
                    # Image tokens get IGNORE_INDEX label (no loss computed on them)
                    num_img_tokens = cur_image_features.shape[0]
                    img_labels = torch.full((num_img_tokens,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)
                    interleaved_labels.append(img_labels)

            interleaved_embeds = [x.to(self.device) for x in interleaved_embeds]
            new_input_embeds.append(torch.cat(interleaved_embeds))
            new_labels.append(torch.cat(interleaved_labels))
            new_modality_types.append(torch.cat(interleaved_mtypes))

        # ==========================================================
        # Step 4: Truncate to max length
        #   (image embeddings can make sequences exceed tokenizer max)
        # ==========================================================
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        new_modality_types = [x[:tokenizer_model_max_length] for x in new_modality_types]

        # ==========================================================
        # Step 5: Pad all sequences to same length, stack into batch
        # ==========================================================
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        padded_embeds = []
        padded_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        padded_modality_types = torch.full((batch_size, max_len), MODALITY_PAD, dtype=torch.long, device=new_labels[0].device)

        padding_side = getattr(self.config, "tokenizer_padding_side", "right")

        for i, (cur_embed, cur_label, cur_mtype) in enumerate(zip(new_input_embeds, new_labels, new_modality_types)):
            seq_len = cur_embed.shape[0]
            pad_len = max_len - seq_len
            embed_dim = cur_embed.shape[1]
            pad_tensor = torch.zeros((pad_len, embed_dim), dtype=cur_embed.dtype, device=cur_embed.device)

            if padding_side == "left":
                padded_embeds.append(torch.cat((pad_tensor, cur_embed), dim=0))
                if seq_len > 0:
                    padded_labels[i, -seq_len:] = cur_label
                    attention_mask[i, -seq_len:] = True
                    position_ids[i, -seq_len:] = torch.arange(0, seq_len, dtype=position_ids.dtype, device=position_ids.device)
                    padded_modality_types[i, -seq_len:] = cur_mtype
            else:
                padded_embeds.append(torch.cat((cur_embed, pad_tensor), dim=0))
                if seq_len > 0:
                    padded_labels[i, :seq_len] = cur_label
                    attention_mask[i, :seq_len] = True
                    position_ids[i, :seq_len] = torch.arange(0, seq_len, dtype=position_ids.dtype, device=position_ids.device)
                    padded_modality_types[i, :seq_len] = cur_mtype

        new_input_embeds = torch.stack(padded_embeds, dim=0)

        # ==========================================================
        # Step 6: Restore None for originally-None inputs
        # Look at "llava/train/train.py" Line 1320
        # In a most cases, `orig_attention_mask`` is not None. `orig_position_ids` is None.
        # ==========================================================
        if orig_labels is None:
            new_labels = None
        else:
            new_labels = padded_labels

        if orig_attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=orig_attention_mask.dtype)

        if orig_position_ids is None:
            position_ids = None

        # Optional: randomized position skipping during training
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add

        self._modality_types = padded_modality_types
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        pass
