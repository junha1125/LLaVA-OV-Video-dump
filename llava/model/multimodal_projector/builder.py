import torch
import torch.nn as nn
import re

from .pooler_projector import PoolerProjector


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class MLPConvCompressProjector(nn.Module):
    """2-layer MLP + GELU + Conv2d(k=3, s=2, p=1) for 2x spatial compression."""

    def __init__(self, mm_hidden_size, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.conv = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x: (N, num_patches, mm_hidden_size)
        x = self.mlp(x)  # (N, num_patches, hidden_size)
        N, L, D = x.shape
        H = W = int(round(L ** 0.5))
        assert H * W == L, f"num_patches={L} is not a perfect square"
        x = x.reshape(N, H, W, D).permute(0, 3, 1, 2)  # (N, D, H, W)
        x = self.conv(x)  # (N, D, H//2, W//2)
        x = x.permute(0, 2, 3, 1).reshape(N, -1, D)  # (N, (H//2)*(W//2), D)
        return x

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == "pooler":
        return PoolerProjector(config, kwargs["vision_cfg"])

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    mlp_gelu_resnet_match = re.match(r"^mlp(\d+)x_res(\d+)x_gelu$", projector_type)
    if mlp_gelu_resnet_match:
        mlp_depth = int(mlp_gelu_resnet_match.group(1))
        res_depth = int(mlp_gelu_resnet_match.group(2))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        for _ in range(res_depth):
            modules.append(SimpleResBlock(config.hidden_size))
        return nn.Sequential(*modules)

    mlp_gelu_dims_extra_match = re.match(r"^mlp(\d+)x_gelu_(\d+)_(\d+)\+(\d+)$", projector_type)
    if mlp_gelu_dims_extra_match:
        mlp_depth = int(mlp_gelu_dims_extra_match.group(1))
        in_dim = int(mlp_gelu_dims_extra_match.group(2))  # vision encoder's hidden size
        hid_dim = int(mlp_gelu_dims_extra_match.group(3)) # LLM(when VLM training)'s hidden size
        extra_linear = int(mlp_gelu_dims_extra_match.group(4))

        modules = [nn.Linear(in_dim, hid_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hid_dim, hid_dim))

        for i in range(extra_linear):
            modules.append(nn.GELU())
            out_dim = hid_dim if i < (extra_linear - 1) else config.hidden_size
            modules.append(nn.Linear(hid_dim, out_dim))

        return nn.Sequential(*modules)

    if projector_type == "mlp2x_gelu_conv2x":
        return MLPConvCompressProjector(config.mm_hidden_size, config.hidden_size)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")
