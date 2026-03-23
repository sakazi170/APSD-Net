import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Original Components (for reference)
# ============================================================================
class SobelHF3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Sobel X
        kx = torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            [[-2, 0, 2],
             [-4, 0, 4],
             [-2, 0, 2]],
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        ], dtype=torch.float32)

        # Sobel Y
        ky = torch.tensor([
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            [[-2, -4, -2],
             [0, 0, 0],
             [2, 4, 2]],
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]]
        ], dtype=torch.float32)

        # Sobel Z
        kz = torch.tensor([
            [[-1, -2, -1],
             [-2, -4, -2],
             [-1, -2, -1]],
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]]
        ], dtype=torch.float32)

        kx = kx / kx.abs().sum()
        ky = ky / ky.abs().sum()
        kz = kz / kz.abs().sum()

        kx = kx.unsqueeze(0).unsqueeze(0)
        ky = ky.unsqueeze(0).unsqueeze(0)
        kz = kz.unsqueeze(0).unsqueeze(0)

        def expand(k):
            return k.expand(channels, 1, 3, 3, 3).contiguous()

        self.register_buffer("Kx", expand(kx))
        self.register_buffer("Ky", expand(ky))
        self.register_buffer("Kz", expand(kz))

    def forward(self, x):
        x_pad = F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate')

        gx = F.conv3d(x_pad, self.Kx, groups=self.channels)
        gy = F.conv3d(x_pad, self.Ky, groups=self.channels)
        gz = F.conv3d(x_pad, self.Kz, groups=self.channels)

        hf = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + 1e-6)

        return hf


class LowHighFreqSeparator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.sobel_hf = SobelHF3D(channels)

    def forward(self, x):
        B, C, H, W, D = x.shape

        # Low-frequency: average pooling
        x_pad = F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate')
        kernel = torch.ones((C, 1, 3, 3, 3), device=x.device, dtype=x.dtype) / 27.
        lf = F.conv3d(x_pad, kernel, groups=C)

        # High-frequency: Sobel
        hf = self.sobel_hf(x)

        return lf, hf


class ChannelAttentionAlign(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, f_local, f_global):
        B, C, H, W, D = f_local.shape

        local_gap = f_local.mean(dim=[2, 3, 4])  # [B, C]
        weights = self.fc(local_gap).view(B, C, 1, 1, 1)

        return f_global * weights


# ============================================================================
# Original FLGFA (for comparison)
# ============================================================================
class FLGFA(nn.Module):
    """
    Original FLGFA with:
    - Frequency separation (LF + HF)
    - Channel attention alignment
    - Separate LF/HF processing
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Frequency separation
        self.freq_sep = LowHighFreqSeparator(channels)

        self.lf_norm = nn.GroupNorm(min(8, channels), channels)
        self.hf_norm = nn.GroupNorm(min(8, channels), channels)

        # Processing blocks
        self.lf_process = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True)
        )

        self.hf_process = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True)
        )

        # Alignment
        self.lf_align = ChannelAttentionAlign(channels)
        self.hf_align = ChannelAttentionAlign(channels)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True)
        )

        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, f_local, f_global):
        # 1. LF/HF decomposition
        lf_l, hf_l = self.freq_sep(f_local)
        lf_g, hf_g = self.freq_sep(f_global)

        # 2. Normalization
        lf_l = self.lf_norm(lf_l)
        lf_g = self.lf_norm(lf_g)
        hf_l = self.hf_norm(hf_l)
        hf_g = self.hf_norm(hf_g)

        # 3. Process
        lf_l = self.lf_process(lf_l)
        lf_g = self.lf_process(lf_g)
        hf_l = self.hf_process(hf_l)
        hf_g = self.hf_process(hf_g)

        # 4. Align global to local
        lf_g_aligned = self.lf_align(f_local=lf_l, f_global=lf_g)
        hf_g_aligned = self.hf_align(f_local=hf_l, f_global=hf_g)

        # 5. Fuse
        lf_fused = lf_l + lf_g_aligned
        hf_fused = hf_l + hf_g_aligned

        feat = self.fusion(torch.cat([lf_fused, hf_fused], dim=1))

        # 6. Residual with scaling
        return f_local + self.res_scale * feat


# ============================================================================
# ABLATION VERSION 1: No Frequency Separation (Spatial Domain Only)
# ============================================================================
class FLGFA_v1(nn.Module):
    """
    ABLATION: Spatial domain only (no frequency separation).
    - Removes: LowHighFreqSeparator, Sobel filtering
    - Keeps: Channel attention alignment, processing, fusion
    - Uses: Direct spatial features instead of LF/HF decomposition
    - Tests: Is frequency domain processing necessary?
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # No frequency separation - work directly on spatial features
        self.norm = nn.GroupNorm(min(8, channels), channels)

        # Single processing path (no LF/HF separation)
        self.process = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True)
        )

        # Channel attention alignment (kept from original)
        self.align = ChannelAttentionAlign(channels)

        # Fusion (simpler since no LF/HF)
        self.fusion = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True)
        )

        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, f_local, f_global):
        # 1. Normalize (no frequency separation)
        f_l = self.norm(f_local)
        f_g = self.norm(f_global)

        # 2. Process
        f_l = self.process(f_l)
        f_g = self.process(f_g)

        # 3. Align global to local using channel attention
        f_g_aligned = self.align(f_local=f_l, f_global=f_g)

        # 4. Fuse (simple addition + conv)
        f_fused = f_l + f_g_aligned
        feat = self.fusion(f_fused)

        # 5. Residual with scaling
        return f_local + self.res_scale * feat


# ============================================================================
# ABLATION VERSION 2: No Channel Attention Alignment
# ============================================================================
class FLGFA_v2(nn.Module):
    """
    ABLATION: No channel attention alignment.
    - Removes: ChannelAttentionAlign modules
    - Keeps: Frequency separation, LF/HF processing, fusion
    - Uses: Direct addition instead of attention-based alignment
    - Tests: Is channel attention critical for alignment?
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Frequency separation (kept from original)
        self.freq_sep = LowHighFreqSeparator(channels)

        self.lf_norm = nn.GroupNorm(min(8, channels), channels)
        self.hf_norm = nn.GroupNorm(min(8, channels), channels)

        # Processing blocks (kept from original)
        self.lf_process = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True)
        )

        self.hf_process = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True)
        )

        # No alignment modules - direct fusion

        # Fusion (kept from original)
        self.fusion = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True)
        )

        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, f_local, f_global):
        # 1. LF/HF decomposition
        lf_l, hf_l = self.freq_sep(f_local)
        lf_g, hf_g = self.freq_sep(f_global)

        # 2. Normalization
        lf_l = self.lf_norm(lf_l)
        lf_g = self.lf_norm(lf_g)
        hf_l = self.hf_norm(hf_l)
        hf_g = self.hf_norm(hf_g)

        # 3. Process
        lf_l = self.lf_process(lf_l)
        lf_g = self.lf_process(lf_g)
        hf_l = self.hf_process(hf_l)
        hf_g = self.hf_process(hf_g)

        # 4. Direct addition (no attention alignment)
        lf_fused = lf_l + lf_g
        hf_fused = hf_l + hf_g

        # 5. Fuse
        feat = self.fusion(torch.cat([lf_fused, hf_fused], dim=1))

        # 6. Residual with scaling
        return f_local + self.res_scale * feat


# ============================================================================
# ABLATION VERSION 3: Unified Frequency Processing (No LF/HF Separation)
# ============================================================================
class FLGFA_v3(nn.Module):
    """
    ABLATION: Unified frequency processing.
    - Removes: Separate LF and HF processing paths
    - Keeps: Frequency separation, channel attention alignment
    - Uses: Single unified processing for both frequencies
    - Tests: Does separating LF/HF processing matter?
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Frequency separation (kept from original)
        self.freq_sep = LowHighFreqSeparator(channels)

        # Reduction layer (2C -> C) - MUST be in __init__ not forward!
        self.reduce_local = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True)
        )

        self.reduce_global = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True)
        )

        self.norm = nn.GroupNorm(min(8, channels), channels)

        # Unified processing (no separate LF/HF paths)
        self.unified_process = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True)
        )

        # Channel attention alignment (kept from original)
        self.align = ChannelAttentionAlign(channels)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True)
        )

        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, f_local, f_global):
        # 1. LF/HF decomposition
        lf_l, hf_l = self.freq_sep(f_local)
        lf_g, hf_g = self.freq_sep(f_global)

        # 2. Concatenate LF and HF (instead of processing separately)
        freq_l = torch.cat([lf_l, hf_l], dim=1)  # [B, 2C, H, W, D]
        freq_g = torch.cat([lf_g, hf_g], dim=1)

        # 3. Reduce back to original channels (using proper layers)
        freq_l = self.reduce_local(freq_l)
        freq_g = self.reduce_global(freq_g)

        # 4. Normalize
        freq_l = self.norm(freq_l)
        freq_g = self.norm(freq_g)

        # 5. Unified processing (same path for both)
        freq_l = self.unified_process(freq_l)
        freq_g = self.unified_process(freq_g)

        # 6. Align global to local
        freq_g_aligned = self.align(f_local=freq_l, f_global=freq_g)

        # 7. Fuse
        freq_fused = freq_l + freq_g_aligned
        feat = self.fusion(torch.cat([freq_fused, freq_fused], dim=1))

        # 8. Residual with scaling
        return f_local + self.res_scale * feat
