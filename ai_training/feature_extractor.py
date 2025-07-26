# feature_extractor.py
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, dilation=1):
        super().__init__()
        pad = (k // 2) * dilation
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, padding=pad, dilation=dilation)
        self.norm = nn.GroupNorm(8, out_c)  # size-agnostic, stable
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, c, k=3, dilation=1):
        super().__init__()
        self.block1 = ConvBlock(c, c, k, dilation)
        self.block2 = ConvBlock(c, c, k, dilation)
    def forward(self, x):
        return x + self.block2(self.block1(x))

def masked_global_avg(feat, mask):
    # feat: [B,C,H,W], mask: [B,1,H,W] (1 for valid)
    weighted = feat * mask
    denom = mask.sum(dim=(2,3), keepdim=True).clamp_min(1.0)
    return weighted.sum(dim=(2,3)) / denom.squeeze(-1).squeeze(-1)

class Match3FeaturesExtractor(BaseFeaturesExtractor):
    """
    Expects observation dict: {'board': (14,H,W), 'globals': (5,)}
    Returns  (cnn_out + mlp_out,)
    """
    def __init__(self, observation_space, cnn_out=256, glob_out=32):
        super().__init__(observation_space, features_dim=cnn_out + glob_out)

        board_shape = observation_space.spaces["board"].shape  # (14,H,W)
        self.board_ch = board_shape[0]

        # --- Board CNN ---
        self.stem = ConvBlock(self.board_ch, 64, k=3)
        self.body = nn.Sequential(
            ResidualBlock(64),
            ConvBlock(64, 128, k=3),
            ResidualBlock(128),
            ConvBlock(128, 128, k=5, dilation=2),  # larger RF
            ResidualBlock(128),
        )
        self.proj = nn.Linear(128, cnn_out)

        # --- Globals MLP ---
        self.gl_mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, glob_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs):
        board = obs["board"].float()  # SB3 casts to torch.FloatTensor already
        globs = obs["globals"].float()

        mask = board[:, 13:14]  # channel 13 is mask
        x = self.body(self.stem(board))
        pooled = masked_global_avg(x, mask)
        board_feat = self.proj(pooled)

        glob_feat = self.gl_mlp(globs)

        return torch.cat([board_feat, glob_feat], dim=1)
