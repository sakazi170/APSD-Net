import torch
import torch.nn as nn
from modules.blocks import RCB, VIT

class AuxiliaryEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        # ========== Separate Stems for each modality ==========
        self.stem_t1 = self._make_stem(1, 16)
        self.stem_t1ce = self._make_stem(1, 16)
        self.stem_flair = self._make_stem(1, 16)

        # Fusion: 48 (3×16) → 16
        self.fusion = nn.Sequential(
            nn.Conv3d(48, 16, kernel_size=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True)
        )

        # ========== Transformer Layers ==========
        self.layer1 = nn.Sequential(VIT(16, 32, window_size=8, layer_id=1, dropout=0.1))
        self.pool1 = nn.MaxPool3d(2, 2)

        self.layer2 = nn.Sequential(VIT(32, 64, window_size=8, layer_id=2, dropout=0.1))
        self.pool2 = nn.MaxPool3d(2, 2)

        self.layer3 = nn.Sequential(VIT(64, 128, window_size=None, layer_id=3, dropout=0.1))
        self.pool3 = nn.MaxPool3d(2, 2)

        self.layer4 = nn.Sequential(VIT(128, 256, window_size=None, layer_id=4, dropout=0.1))

    def _make_stem(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, t1, t1ce, flair):
        # Stem processing
        f_t1 = self.stem_t1(t1)
        f_t1ce = self.stem_t1ce(t1ce)
        f_flair = self.stem_flair(flair)

        # Fuse modalities
        x = torch.cat([f_t1, f_t1ce, f_flair], dim=1)
        x = self.fusion(x)

        # Encoder layers
        g1 = self.layer1(x)  # [B, 32, 64, 64, 64]
        x = self.pool1(g1)

        g2 = self.layer2(x)  # [B, 64, 32, 32, 32]
        x = self.pool2(g2)

        g3 = self.layer3(x)  # [B, 128, 16, 16, 16]
        x = self.pool3(g3)

        g4 = self.layer4(x)  # [B, 256, 8, 8, 8] (bottleneck)

        return [g1, g2, g3, g4]


class LeadingEncoder(nn.Module):

    def __init__(self, in_ch=1):
        super().__init__()

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True)
        )

        # RCB layers (no FreqLGFA)
        self.layer1 = nn.Sequential(RCB(16, 32))
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer2 = nn.Sequential(RCB(32, 64))
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer3 = nn.Sequential(RCB(64, 128))
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer4 = nn.Sequential(RCB(128, 256))

    def forward(self, x, g1=None, g2=None, g3=None):

        f0 = self.stem(x)

        # Layer 1 with simple addition
        f1 = self.layer1(f0)
        if g1 is not None:
            f1 = f1 + g1  # Simple addition
        x = self.pool1(f1)

        # Layer 2 with simple addition
        f2 = self.layer2(x)
        if g2 is not None:
            f2 = f2 + g2  # Simple addition
        x = self.pool2(f2)

        # Layer 3 with simple addition
        f3 = self.layer3(x)
        if g3 is not None:
            f3 = f3 + g3  # Simple addition
        x = self.pool3(f3)

        f4 = self.layer4(x)  # [B, 256, 8, 8, 8]

        return [f0, f1, f2, f3, f4]


class BottleneckFusion(nn.Module):
    """
    Bottleneck fusion using concatenation + 1x1 conv.
    Simple baseline fusion module.
    """

    def __init__(self, channels=256):
        super().__init__()

        # Concatenate and reduce: 512 → 256
        self.fusion = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_local, f_global):

        x = torch.cat([f_local, f_global], dim=1)  # [B, 512, D, H, W]
        return self.fusion(x)  # [B, 256, D, H, W]


class Decoder(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        # Stage 4: 256 (fused) + 128 (f3 skip) → 128
        self.up4 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.rcb4 = RCB(128 + 128, 128)

        # Stage 3: 128 → 64
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.rcb3 = RCB(64 + 64, 64)

        # Stage 2: 64 → 32
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.rcb2 = RCB(32 + 32, 32)

        # Stage 1: 32 → 16
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.up_f0 = nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2)
        self.rcb1 = RCB(16 + 16, 16)

        self.final_conv = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, bottleneck_fused, f0, f1, f2, f3):

        x = self.up4(bottleneck_fused)
        x = torch.cat([x, f3], dim=1)
        x = self.rcb4(x)

        x = self.up3(x)
        x = torch.cat([x, f2], dim=1)
        x = self.rcb3(x)

        x = self.up2(x)
        x = torch.cat([x, f1], dim=1)
        x = self.rcb2(x)

        x = self.up1(x)
        f0_up = self.up_f0(f0)
        x = torch.cat([x, f0_up], dim=1)
        x = self.rcb1(x)

        x = self.final_conv(x)

        return x


class BTS_Baseline_VIT(nn.Module):


    def __init__(self, patch_size_d=None, patch_size_h=None, patch_size_w=None,
                 in_channels=1, num_classes=4):
        super().__init__()

        # Encoders
        self.leading_encoder = LeadingEncoder(in_ch=in_channels)
        self.auxiliary_encoder = AuxiliaryEncoder()

        # Bottleneck fusion (concatenation + 1x1 conv)
        self.bottleneck_fusion = BottleneckFusion(channels=256)

        # Decoder
        self.decoder = Decoder(num_classes=num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, t1, t1ce, t2, flair):
        # Auxiliary encoder
        g1, g2, g3, g4 = self.auxiliary_encoder(t1, t1ce, flair)

        # Leading encoder with simple addition at layers 1, 2, 3
        f0, f1, f2, f3, f4 = self.leading_encoder(t2, g1=g1, g2=g2, g3=g3)

        # Bottleneck fusion (concatenation + 1x1 conv)
        bottleneck_fused = self.bottleneck_fusion(f_local=f4, f_global=g4)
        # bottleneck_fused: [B, 256, 8, 8, 8]

        # Decoder
        output = self.decoder(bottleneck_fused, f0, f1, f2, f3)
        # output: [B, num_classes, 128, 128, 128]

        return output


class unet(nn.Module):
    """
    Complete 3D U-Net with Encoder + Decoder in single class
    Takes 4 modalities (T1, T1ce, T2, FLAIR) and uses RCB blocks
    Consistent with baseline1 architecture
    """

    def __init__(self, patch_size_d=None, patch_size_h=None, patch_size_w=None,
                 in_channels=4, num_classes=4):
        super().__init__()

        # ======================== ENCODER ========================
        # Initial stem: 4 → 16 (stride=2, consistent with baseline1)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True)
        )

        # Encoder layers with RCB
        self.enc1 = RCB(16, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = RCB(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = RCB(64, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc4 = RCB(128, 256)  # Bottleneck (no pool after)

        # ======================== DECODER ========================
        # Stage 4: 8³ → 16³
        self.up4 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec4 = RCB(128 + 128, 128)

        # Stage 3: 16³ → 32³
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec3 = RCB(64 + 64, 64)

        # Stage 2: 32³ → 64³
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = RCB(32 + 32, 32)

        # Stage 1: 64³ → 128³ (with f0 skip, both need upsampling)
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.up_f0 = nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2)
        self.dec1 = RCB(16 + 16, 16)

        # Final output
        self.final_conv = nn.Conv3d(16, num_classes, kernel_size=1)

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, t1, t1ce, t2, flair):
        # Concatenate all 4 modalities at input
        x = torch.cat([t1, t1ce, t2, flair], dim=1)  # [B, 4, 128, 128, 128]

        # ======================== ENCODER ========================
        f0 = self.stem(x)           # [B, 16, 64, 64, 64]

        f1 = self.enc1(f0)          # [B, 32, 64, 64, 64]
        x = self.pool1(f1)          # [B, 32, 32, 32, 32]

        f2 = self.enc2(x)           # [B, 64, 32, 32, 32]
        x = self.pool2(f2)          # [B, 64, 16, 16, 16]

        f3 = self.enc3(x)           # [B, 128, 16, 16, 16]
        x = self.pool3(f3)          # [B, 128, 8, 8, 8]

        x = self.enc4(x)            # [B, 256, 8, 8, 8] - Bottleneck

        # ======================== DECODER ========================
        # Stage 4: 8³ → 16³
        x = self.up4(x)                     # [B, 128, 16, 16, 16]
        x = torch.cat([x, f3], dim=1)       # [B, 256, 16, 16, 16]
        x = self.dec4(x)                    # [B, 128, 16, 16, 16]

        # Stage 3: 16³ → 32³
        x = self.up3(x)                     # [B, 64, 32, 32, 32]
        x = torch.cat([x, f2], dim=1)       # [B, 128, 32, 32, 32]
        x = self.dec3(x)                    # [B, 64, 32, 32, 32]

        # Stage 2: 32³ → 64³
        x = self.up2(x)                     # [B, 32, 64, 64, 64]
        x = torch.cat([x, f1], dim=1)       # [B, 64, 64, 64, 64]
        x = self.dec2(x)                    # [B, 32, 64, 64, 64]

        # Stage 1: 64³ → 128³ (with upsampled f0)
        x = self.up1(x)                     # [B, 16, 128, 128, 128]
        f0_up = self.up_f0(f0)              # [B, 16, 128, 128, 128]
        x = torch.cat([x, f0_up], dim=1)    # [B, 32, 128, 128, 128]
        x = self.dec1(x)                    # [B, 16, 128, 128, 128]

        # Final prediction
        out = self.final_conv(x)            # [B, num_classes, 128, 128, 128]

        return out


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def test_model():
    """Test the model with dummy data"""
    print("=" * 80)
    print("Testing Brain Tumor Segmentation Model")
    print("=" * 80)

    num_classes = 4
    model = unet(128,128,128, in_channels=1, num_classes=num_classes)

    total_params, trainable_params = count_parameters(model)
    print(f"\nModel Statistics:")
    print(f"  - Total parameters: {total_params / 1e6:.3f}M")
    print(f"  - Trainable parameters: {trainable_params / 1e6:.3f}M")
    print(f"  - Model size (approx): {total_params * 4 / (1024 ** 2):.3f} MB")


if __name__ == "__main__":
    test_model()