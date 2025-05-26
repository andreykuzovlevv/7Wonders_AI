import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class RotInvMatch3Extractor(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space: gym.spaces.Dict,
                 n_gems: int = 8, 
                 pattern_k: int = 6, 
                 meta_channels: int = 6, 
                 fuse_channels: int = 64):
        """
        Rotation & mirror–invariant pattern extractor for a match-3 board.
        
        Args:
          observation_space: The observation space (Dict with 'board' and 'globals')
          n_gems: number of gem colours (default 8).
          pattern_k: number of base 5×5 filters learned *per* gem.
          meta_channels: non-gem channels (fragment + bonuses + background + mask).
          fuse_channels: number of channels in the fusion tower.
        """
        # Calculate the final feature size
        features_dim = fuse_channels + observation_space['globals'].shape[0]
        super().__init__(observation_space, features_dim)
        
        self.n_gems     = n_gems
        self.pattern_k  = pattern_k

        # 1) Base kernels: one (pattern_k × 5 × 5) set *per* gem colour
        self.base_kernels = nn.Parameter(
            torch.randn(n_gems, pattern_k, 5, 5) * 0.1
        )
        
        # 1b) Additional 3x3 kernels: one (pattern_k × 3 × 3) set *per* gem colour
        self.base_kernels_3x3 = nn.Parameter(
            torch.randn(n_gems, pattern_k, 3, 3) * 0.1
        )

        # 2) Meta conv for everything except the 8 gem planes
        self.meta_conv = nn.Sequential(
            nn.Conv2d(meta_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 3) Fuse gem + meta features (now with both 5x5 and 3x3 gem features)
        in_ch = n_gems * pattern_k * 2 + 32  # *2 for both 5x5 and 3x3 features
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, fuse_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fuse_channels, fuse_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 4) Global pooling → (B, fuse_channels)
        self.readout = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def _all_transforms_5x5(kernel: torch.Tensor) -> torch.Tensor:
        """
        Given a (5×5) kernel, return a Tensor (8,5,5)
        containing its 4 rotations × 2 mirror states.
        """
        outs = []
        for k in range(4):
            r = torch.rot90(kernel, k, [0,1])
            outs.append(r)
            outs.append(torch.flip(r, [1]))  # horizontal mirror
        return torch.stack(outs, 0)  # (8, 5, 5)
    
    @staticmethod
    def _all_transforms_3x3(kernel: torch.Tensor) -> torch.Tensor:
        """
        Given a (3×3) kernel, return a Tensor (8,3,3)
        containing its 4 rotations × 2 mirror states.
        """
        outs = []
        for k in range(4):
            r = torch.rot90(kernel, k, [0,1])
            outs.append(r)
            outs.append(torch.flip(r, [1]))  # horizontal mirror
        return torch.stack(outs, 0)  # (8, 3, 3)

    def forward(self, observations: dict) -> torch.Tensor:
        """
        observations: Dict with 'board' and 'globals' keys
                     'board': (B, 14, H, W) board tensor
                              channels 0–7 gems, 8 fragment, 9–11 bonuses,
                              12 background, 13 mask
                     'globals': (B, G) any global scalars
        Returns:
          (B, fuse_channels + G) feature vector
        """
        x = observations['board']
        global_feats = observations['globals']
        
        B, C, H, W = x.shape
        assert C == 14, f"Expected 14 channels, got {C}"

        # --- A) Rotation-invariant gem matching (5x5 kernels) --------------
        gem_planes = x[:, :self.n_gems]  # (B, 8, H, W)
        
        # Build 5x5 weight tensor
        all_w_5x5 = []
        for gem in range(self.n_gems):
            for p in range(self.pattern_k):
                base = self.base_kernels[gem, p]               # (5,5)
                trans = self._all_transforms_5x5(base)         # (8,5,5)
                all_w_5x5.append(trans)
        all_w_5x5 = torch.cat(all_w_5x5, dim=0)  # (8*pattern_k*8, 5, 5)
        all_w_5x5 = all_w_5x5.unsqueeze(1)       # (Nw,1,5,5)

        # Convolution with 5x5 kernels
        out_5x5 = F.conv2d(
            gem_planes,
            all_w_5x5,
            padding=2,
            groups=self.n_gems
        )  # (B, 8*pattern_k*8, H, W)

        # Reshape and max-over-transforms for 5x5
        out_5x5 = out_5x5.view(B, self.n_gems, self.pattern_k, 8, H, W)
        gem_feat_5x5, _ = out_5x5.max(dim=3)        # (B, n_gems, pattern_k, H, W)
        gem_feat_5x5 = gem_feat_5x5.view(B, -1, H, W)  # (B, n_gems*pattern_k, H, W)
        gem_feat_5x5 = F.relu(gem_feat_5x5)

        # --- A2) Rotation-invariant gem matching (3x3 kernels) -------------
        # Build 3x3 weight tensor
        all_w_3x3 = []
        for gem in range(self.n_gems):
            for p in range(self.pattern_k):
                base = self.base_kernels_3x3[gem, p]          # (3,3)
                trans = self._all_transforms_3x3(base)        # (8,3,3)
                all_w_3x3.append(trans)
        all_w_3x3 = torch.cat(all_w_3x3, dim=0)  # (8*pattern_k*8, 3, 3)
        all_w_3x3 = all_w_3x3.unsqueeze(1)       # (Nw,1,3,3)

        # Convolution with 3x3 kernels
        out_3x3 = F.conv2d(
            gem_planes,
            all_w_3x3,
            padding=1,
            groups=self.n_gems
        )  # (B, 8*pattern_k*8, H, W)

        # Reshape and max-over-transforms for 3x3
        out_3x3 = out_3x3.view(B, self.n_gems, self.pattern_k, 8, H, W)
        gem_feat_3x3, _ = out_3x3.max(dim=3)        # (B, n_gems, pattern_k, H, W)
        gem_feat_3x3 = gem_feat_3x3.view(B, -1, H, W)  # (B, n_gems*pattern_k, H, W)
        gem_feat_3x3 = F.relu(gem_feat_3x3)

        # --- B) Meta features (fragment, bonus, background, mask) ----------
        meta = self.meta_conv(x[:, self.n_gems:])  # (B,32,H,W)

        # --- C) Fuse & pool -------------------------------------------------
        # Combine both 5x5 and 3x3 gem features
        combined_gem_feat = torch.cat([gem_feat_5x5, gem_feat_3x3], dim=1)
        fused = self.fuse(torch.cat([combined_gem_feat, meta], dim=1))  # (B, fuse_ch, H, W)
        pooled = self.readout(fused).flatten(1)                # (B, fuse_ch)

        # --- D) Final feature vector ---------------------------------------
        return torch.cat([pooled, global_feats], dim=1)       # (B, fuse_ch+G)
