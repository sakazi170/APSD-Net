import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()

        # Depthwise: Spatial convolution per channel
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_channels, bias=bias)

        # Pointwise: Channel mixing (1×1×1 conv)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1,
                                   padding=0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class RCB(nn.Module):
    """Residual Convolutional Block with Depthwise Separable Convolutions"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv3D(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv3D(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True))

        # Skip Connection
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(8, out_channels))  # Changed from BatchNorm3d
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return out

class TumorAwareAttention3D(nn.Module):

    def __init__(self, dim, num_heads=4, window_size=None, dropout=0.1):
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Conv3d(dim, dim * 3, 1)
        self.proj = nn.Conv3d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)  # Only used after softmax

        # Learnable temperature parameter for attention
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x, tpm):
        B, C, D, H, W = x.shape

        qkv = self.qkv(x)

        if self.window_size is None:
            out = self._global_attention(qkv, tpm, B, C, D, H, W)
        else:
            out = self._window_attention(qkv, tpm, B, C, D, H, W)

        out = self.proj(out)
        # No dropout here
        return out

    def _global_attention(self, qkv, tpm, B, C, D, H, W):
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, D * H * W)
        q, k, v = qkv.unbind(1)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale * torch.clamp(self.temperature, min=0.1, max=2.0)

        # TPM Application
        tpm = torch.clamp(tpm, min=0.01, max=1.0)
        tpm_flat = tpm.reshape(B, 1, 1, D * H * W)
        tpm_i = tpm_flat.unsqueeze(-1)   # Query position
        tpm_j = tpm_flat.unsqueeze(-2)   # Key position
        tpm_interaction = (tpm_i + tpm_j) / 2
        tpm_interaction = tpm_interaction.squeeze(2)

        attn = attn * tpm_interaction
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)  # ✅ Only dropout here

        v = v.transpose(-2, -1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, D * H * W, C)
        out = out.transpose(1, 2).reshape(B, C, D, H, W)

        return out

    def _window_attention(self, qkv, tpm, B, C, D, H, W):
        ws = self.window_size

        pad_d = (ws - D % ws) % ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            qkv = F.pad(qkv, (0, pad_w, 0, pad_h, 0, pad_d))
            tpm = F.pad(tpm, (0, pad_w, 0, pad_h, 0, pad_d))

        D_pad, H_pad, W_pad = qkv.shape[2:]

        qkv = qkv.reshape(B, 3 * C,
                          D_pad // ws, ws,
                          H_pad // ws, ws,
                          W_pad // ws, ws)
        qkv = qkv.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        num_windows = (D_pad // ws) * (H_pad // ws) * (W_pad // ws)
        qkv = qkv.reshape(B * num_windows, 3 * C, ws, ws, ws)

        tpm_windows = tpm.reshape(B, 1,
                                  D_pad // ws, ws,
                                  H_pad // ws, ws,
                                  W_pad // ws, ws)
        tpm_windows = tpm_windows.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        tpm_windows = tpm_windows.reshape(B * num_windows, 1, ws, ws, ws)
        tpm_windows = torch.clamp(tpm_windows, min=0.01, max=1.0)

        qkv = qkv.reshape(B * num_windows, 3, self.num_heads, self.head_dim, ws * ws * ws)
        q, k, v = qkv.unbind(1)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale * torch.clamp(self.temperature, min=0.1, max=2.0)

        # TPM Application
        tpm_flat = tpm_windows.reshape(B * num_windows, 1, 1, ws * ws * ws)
        tpm_i = tpm_flat.unsqueeze(-1)
        tpm_j = tpm_flat.unsqueeze(-2)
        tpm_interaction = (tpm_i + tpm_j) / 2
        tpm_interaction = tpm_interaction.squeeze(2)

        attn = attn * tpm_interaction
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)  # ✅ Only dropout here

        v = v.transpose(-2, -1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B * num_windows, ws * ws * ws, C)
        out = out.transpose(1, 2).reshape(B * num_windows, C, ws, ws, ws)

        out = out.reshape(B, D_pad // ws, H_pad // ws, W_pad // ws,
                          C, ws, ws, ws)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        out = out.reshape(B, C, D_pad, H_pad, W_pad)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            out = out[:, :, :D, :H, :W]

        return out


class TAT(nn.Module):
    """
    TAT: Tumor-Aware Transformer Block
    """

    def __init__(self, in_channels, out_channels, window_size, layer_id, num_heads=4, dropout=0.1):
        super().__init__()

        self.window_size = window_size
        self.layer_id = layer_id

        # Channel projection if needed
        self.proj = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # TPM Generation (No dropout here)
        self.tpm_gen = nn.Sequential(
            nn.Conv3d(out_channels, out_channels // 2, 3, padding=1),
            nn.GroupNorm(4, out_channels // 2),
            nn.ReLU(),
            nn.Conv3d(out_channels // 2, 1, 1),
            nn.Sigmoid()
        )

        # Tumor-Aware Attention
        self.tumor_aware_attn = TumorAwareAttention3D(
            dim=out_channels,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout
        )

        # MLP (Only 1 dropout at end)
        self.mlp = nn.Sequential(
            nn.Conv3d(out_channels, out_channels * 4, 1),
            nn.GELU(),
            nn.Conv3d(out_channels * 4, out_channels, 1),
            nn.Dropout3d(dropout)  # ✅ Only 1 dropout
        )

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

    def forward(self, x):
        x = self.proj(x)

        # Generate TPM
        tpm = self.tpm_gen(x)

        # Attention
        residual = x
        x = self.norm1(x)
        x = self.tumor_aware_attn(x, tpm)
        x = x + residual

        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual

        return x


class StandardAttention3D(nn.Module):
    """
    Standard 3D Multi-Head Attention (without tumor probability modulation)
    Supports both window-based (local) and global attention
    """

    def __init__(self, dim, num_heads=4, window_size=None, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Conv3d(dim, dim * 3, 1)
        self.proj = nn.Conv3d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)  # Only used after softmax

        # Learnable temperature parameter for attention
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, C, D, H, W = x.shape

        qkv = self.qkv(x)

        if self.window_size is None:
            out = self._global_attention(qkv, B, C, D, H, W)
        else:
            out = self._window_attention(qkv, B, C, D, H, W)

        out = self.proj(out)
        # No dropout here
        return out

    def _global_attention(self, qkv, B, C, D, H, W):
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, D * H * W)
        q, k, v = qkv.unbind(1)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale * torch.clamp(self.temperature, min=0.1, max=2.0)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)  # ✅ Only dropout here

        v = v.transpose(-2, -1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, D * H * W, C)
        out = out.transpose(1, 2).reshape(B, C, D, H, W)

        return out

    def _window_attention(self, qkv, B, C, D, H, W):
        ws = self.window_size

        pad_d = (ws - D % ws) % ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            qkv = F.pad(qkv, (0, pad_w, 0, pad_h, 0, pad_d))

        D_pad, H_pad, W_pad = qkv.shape[2:]

        qkv = qkv.reshape(B, 3 * C,
                          D_pad // ws, ws,
                          H_pad // ws, ws,
                          W_pad // ws, ws)
        qkv = qkv.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        num_windows = (D_pad // ws) * (H_pad // ws) * (W_pad // ws)
        qkv = qkv.reshape(B * num_windows, 3 * C, ws, ws, ws)

        qkv = qkv.reshape(B * num_windows, 3, self.num_heads, self.head_dim, ws * ws * ws)
        q, k, v = qkv.unbind(1)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale * torch.clamp(self.temperature, min=0.1, max=2.0)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)  # ✅ Only dropout here

        v = v.transpose(-2, -1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B * num_windows, ws * ws * ws, C)
        out = out.transpose(1, 2).reshape(B * num_windows, C, ws, ws, ws)

        out = out.reshape(B, D_pad // ws, H_pad // ws, W_pad // ws,
                          C, ws, ws, ws)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        out = out.reshape(B, C, D_pad, H_pad, W_pad)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            out = out[:, :, :D, :H, :W]

        return out


class VIT(nn.Module):
    """
    Standard Vision Transformer Block (without TPM)
    Stabilized for batch_size=1 to match TAT for fair comparison
    """

    def __init__(self, in_channels, out_channels, window_size, layer_id, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.window_size = window_size
        self.layer_id = layer_id

        # Channel projection if needed
        self.proj = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # Multi-Head Attention
        self.attention = StandardAttention3D(
            dim=out_channels,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout
        )

        # MLP (Only 1 dropout at end)
        mlp_hidden_dim = int(out_channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv3d(out_channels, mlp_hidden_dim, 1),
            nn.GELU(),
            nn.Conv3d(mlp_hidden_dim, out_channels, 1),
            nn.Dropout3d(dropout)  # ✅ Only 1 dropout
        )

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

    def forward(self, x):
        x = self.proj(x)

        # Attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = x + residual

        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual

        return x