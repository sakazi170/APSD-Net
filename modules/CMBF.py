import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableAttention(nn.Module):
    def __init__(self, channels, num_heads=8, num_points=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = channels // num_heads

        self.norm_query = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.query_proj = nn.Linear(channels, channels)

        self.offset_net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, num_heads * num_points * 3)
        )

        self.attn_weights_net = nn.Linear(channels, num_heads * num_points)
        self.out_proj = nn.Linear(channels, channels)

        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

    def forward(self, f_query, f_kv):
        B, C, H, W, D = f_query.shape
        N = H * W * D

        f_query_flat = f_query.flatten(2).transpose(1, 2)
        f_kv_flat = f_kv.flatten(2).transpose(1, 2)

        f_query_norm = self.norm_query(f_query_flat)
        f_kv_norm = self.norm_kv(f_kv_flat)

        Q = self.query_proj(f_query_norm)

        offsets = self.offset_net(Q)
        offsets = torch.clamp(offsets, -3.0, 3.0)
        offsets = offsets.view(B, N, self.num_heads, self.num_points, 3)

        attn_weights = self.attn_weights_net(Q)
        attn_weights = attn_weights.view(B, N, self.num_heads, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)

        sampled_values = self._deformable_sampling(f_kv, offsets, (H, W, D))

        out = torch.einsum('bnhp,bnhpc->bnhc', attn_weights, sampled_values)
        out = out.reshape(B, N, C)
        out = self.out_proj(out)
        out = out.transpose(1, 2).view(B, C, H, W, D)

        return out

    def _deformable_sampling(self, features, offsets, spatial_shape):
        B, C, H, W, D = features.shape
        _, N, num_heads, num_points, _ = offsets.shape
        head_dim = C // num_heads

        y_grid = torch.linspace(0, H - 1, H, device=features.device)
        x_grid = torch.linspace(0, W - 1, W, device=features.device)
        z_grid = torch.linspace(0, D - 1, D, device=features.device)
        grid_y, grid_x, grid_z = torch.meshgrid(y_grid, x_grid, z_grid, indexing='ij')
        base_grid = torch.stack([grid_y, grid_x, grid_z], dim=-1)
        base_grid = base_grid.view(-1, 3)
        base_grid = base_grid.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        base_grid = base_grid.expand(B, -1, num_heads, num_points, -1)

        sample_locs = base_grid + offsets

        sample_locs_norm = sample_locs.clone()
        sample_locs_norm[..., 0] = 2.0 * sample_locs[..., 0] / max(H - 1, 1) - 1.0
        sample_locs_norm[..., 1] = 2.0 * sample_locs[..., 1] / max(W - 1, 1) - 1.0
        sample_locs_norm[..., 2] = 2.0 * sample_locs[..., 2] / max(D - 1, 1) - 1.0

        features_perm = features.permute(0, 1, 4, 2, 3)

        sampled_all = []
        for h in range(num_heads):
            start_c = h * head_dim
            end_c = (h + 1) * head_dim
            features_head = features_perm[:, start_c:end_c]

            locs_head = sample_locs_norm[:, :, h, :, :]
            locs_head = locs_head.reshape(B, N * num_points, 1, 1, 3)

            sampled_head = F.grid_sample(
                features_head,
                locs_head,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )

            sampled_head = sampled_head.squeeze(-1).squeeze(-1)
            sampled_head = sampled_head.permute(0, 2, 1)
            sampled_head = sampled_head.view(B, N, num_points, head_dim)

            sampled_all.append(sampled_head)

        sampled = torch.stack(sampled_all, dim=2)

        return sampled


# ============================================================================
# StandardAttention (for CMBF_v3 - no deformable sampling)
# ============================================================================
class StandardAttention(nn.Module):
    """Standard multi-head attention without deformable sampling."""

    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_query = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)

        # Q, K, V projections
        self.qkv_proj = nn.Linear(channels, channels * 3)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, f_query, f_kv):
        B, C, H, W, D = f_query.shape
        N = H * W * D

        # Flatten to [B, N, C]
        f_query_flat = f_query.flatten(2).transpose(1, 2)
        f_kv_flat = f_kv.flatten(2).transpose(1, 2)

        # Normalize
        f_query_norm = self.norm_query(f_query_flat)
        f_kv_norm = self.norm_kv(f_kv_flat)

        # Generate Q from query, K and V from kv
        Q = self.qkv_proj(f_query_norm)[:, :, :C]
        K = self.qkv_proj(f_kv_norm)[:, :, C:2 * C]
        V = self.qkv_proj(f_kv_norm)[:, :, 2 * C:]

        # Reshape for multi-head attention
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = attn @ V  # [B, H, N, D]
        out = out.transpose(1, 2).reshape(B, N, C)

        # Output projection
        out = self.out_proj(out)

        # Reshape back to spatial
        out = out.transpose(1, 2).view(B, C, H, W, D)

        return out


# ============================================================================
# ABLATION VERSION 1: Local-to-Global Only (Unidirectional)
# ============================================================================
class CMBF_v1(nn.Module):
    """
    ABLATION: Local-to-Global attention only.
    - Keeps: f_local queries f_global
    - Removes: f_global queries f_local
    - Tests: Does local modality benefit more from global context?
    """

    def __init__(self, channels, num_heads=8, num_points=4):
        super().__init__()
        self.channels = channels

        # Only Local queries Global
        self.local_to_global = DeformableAttention(
            channels=channels,
            num_heads=num_heads,
            num_points=num_points
        )

        # Fusion conv (still concat but with local + global_original)
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_local, f_global):
        # Only Local-to-Global attention
        local_from_global = self.local_to_global(f_query=f_local, f_kv=f_global)

        # Enhance local with residual
        local_enhanced = f_local + local_from_global

        # Global is NOT enhanced (no attention applied)
        global_enhanced = f_global

        # Concat + Conv
        concat = torch.cat([local_enhanced, global_enhanced], dim=1)
        f_out = self.fusion_conv(concat)

        return f_out


# ============================================================================
# ABLATION VERSION 2: Global-to-Local Only (Unidirectional)
# ============================================================================
class CMBF_v2(nn.Module):
    """
    ABLATION: Global-to-Local attention only.
    - Keeps: f_global queries f_local
    - Removes: f_local queries f_global
    - Tests: Does global modality benefit more from local context?
    """

    def __init__(self, channels, num_heads=8, num_points=4):
        super().__init__()
        self.channels = channels

        # Only Global queries Local
        self.global_to_local = DeformableAttention(
            channels=channels,
            num_heads=num_heads,
            num_points=num_points
        )

        # Fusion conv
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_local, f_global):
        # Only Global-to-Local attention
        global_from_local = self.global_to_local(f_query=f_global, f_kv=f_local)

        # Local is NOT enhanced (no attention applied)
        local_enhanced = f_local

        # Enhance global with residual
        global_enhanced = f_global + global_from_local

        # Concat + Conv
        concat = torch.cat([local_enhanced, global_enhanced], dim=1)
        f_out = self.fusion_conv(concat)

        return f_out


# ============================================================================
# ABLATION VERSION 3: No Deformable Sampling (Standard Attention)
# ============================================================================
class CMBF_v3(nn.Module):
    """
    ABLATION: Bidirectional but with standard attention (no deformable sampling).
    - Keeps: Bidirectional attention
    - Removes: Deformable sampling (num_points, offset prediction)
    - Uses: Standard multi-head attention
    - Tests: Is deformable sampling necessary for performance?
    """

    def __init__(self, channels, num_heads=8, num_points=4):
        super().__init__()
        self.channels = channels
        # Note: num_points is ignored in this version

        # Bidirectional Standard Attention
        self.local_to_global = StandardAttention(
            channels=channels,
            num_heads=num_heads
        )

        self.global_to_local = StandardAttention(
            channels=channels,
            num_heads=num_heads
        )

        # Fusion conv
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_local, f_global):
        # Bidirectional Standard Attention
        local_from_global = self.local_to_global(f_query=f_local, f_kv=f_global)
        global_from_local = self.global_to_local(f_query=f_global, f_kv=f_local)

        # Enhance with residual
        local_enhanced = f_local + local_from_global
        global_enhanced = f_global + global_from_local

        # Concat + Conv
        concat = torch.cat([local_enhanced, global_enhanced], dim=1)
        f_out = self.fusion_conv(concat)

        return f_out


# ============================================================================
# Original CMBF (for comparison)
# ============================================================================
class CMBF(nn.Module):
    """
    Original CMBF with:
    - Bidirectional deformable attention
    - Full deformable sampling with num_points
    """

    def __init__(self, channels, num_heads=8, num_points=4):
        super().__init__()
        self.channels = channels

        self.local_to_global = DeformableAttention(
            channels=channels,
            num_heads=num_heads,
            num_points=num_points
        )

        self.global_to_local = DeformableAttention(
            channels=channels,
            num_heads=num_heads,
            num_points=num_points
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_local, f_global):
        local_from_global = self.local_to_global(f_query=f_local, f_kv=f_global)
        global_from_local = self.global_to_local(f_query=f_global, f_kv=f_local)

        local_enhanced = f_local + local_from_global
        global_enhanced = f_global + global_from_local

        concat = torch.cat([local_enhanced, global_enhanced], dim=1)
        f_out = self.fusion_conv(concat)

        return f_out

