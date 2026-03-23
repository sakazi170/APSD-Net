import torch
import torch.nn as nn
from modules.blocks import RCB, TAT
from modules.FLGFA import FLGFA, FLGFA_v1, FLGFA_v2, FLGFA_v3


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
        self.layer1 = nn.Sequential(TAT(16, 32, window_size=8, layer_id=1, dropout=0.1))
        self.pool1 = nn.MaxPool3d(2, 2)

        self.layer2 = nn.Sequential(TAT(32, 64, window_size=8, layer_id=2, dropout=0.1))
        self.pool2 = nn.MaxPool3d(2, 2)

        self.layer3 = nn.Sequential(TAT(64, 128, window_size=None, layer_id=3, dropout=0.1))
        self.pool3 = nn.MaxPool3d(2, 2)

        self.layer4 = nn.Sequential(TAT(128, 256, window_size=None, layer_id=4, dropout=0.1))

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

        # RCB layers
        self.layer1 = nn.Sequential(RCB(16, 32))
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer2 = nn.Sequential(RCB(32, 64))
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer3 = nn.Sequential(RCB(64, 128))
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer4 = nn.Sequential(RCB(128, 256))

        # FreqLGFA modules (applied BEFORE pooling - high resolution)
        self.freq_lgfa1 = FLGFA(channels=32)
        self.freq_lgfa2 = FLGFA(channels=64)
        self.freq_lgfa3 = FLGFA(channels=128)

    def forward(self, x, g1=None, g2=None, g3=None):

        f0 = self.stem(x)

        # Layer 1
        f1 = self.layer1(f0)
        if g1 is not None:
            f1 = self.freq_lgfa1(f_local=f1, f_global=g1)
        x = self.pool1(f1)

        # Layer 2
        f2 = self.layer2(x)
        if g2 is not None:
            f2 = self.freq_lgfa2(f_local=f2, f_global=g2)
        x = self.pool2(f2)

        # Layer 3
        f3 = self.layer3(x)
        if g3 is not None:
            f3 = self.freq_lgfa3(f_local=f3, f_global=g3)
        x = self.pool3(f3)

        f4 = self.layer4(x)  # [B, 256, 8, 8, 8]

        return [f0, f1, f2, f3, f4]

class LeadingEncoder1(nn.Module):

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

        # RCB layers
        self.layer1 = nn.Sequential(RCB(16, 32))
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer2 = nn.Sequential(RCB(32, 64))
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer3 = nn.Sequential(RCB(64, 128))
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer4 = nn.Sequential(RCB(128, 256))

        # FreqLGFA modules (applied BEFORE pooling - high resolution)
        self.freq_lgfa1 = FLGFA_v1(channels=32)
        self.freq_lgfa2 = FLGFA_v1(channels=64)
        self.freq_lgfa3 = FLGFA_v1(channels=128)

    def forward(self, x, g1=None, g2=None, g3=None):

        f0 = self.stem(x)

        # Layer 1
        f1 = self.layer1(f0)
        if g1 is not None:
            f1 = self.freq_lgfa1(f_local=f1, f_global=g1)
        x = self.pool1(f1)

        # Layer 2
        f2 = self.layer2(x)
        if g2 is not None:
            f2 = self.freq_lgfa2(f_local=f2, f_global=g2)
        x = self.pool2(f2)

        # Layer 3
        f3 = self.layer3(x)
        if g3 is not None:
            f3 = self.freq_lgfa3(f_local=f3, f_global=g3)
        x = self.pool3(f3)

        f4 = self.layer4(x)  # [B, 256, 8, 8, 8]
        return [f0, f1, f2, f3, f4]


class LeadingEncoder2(nn.Module):

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

        # RCB layers
        self.layer1 = nn.Sequential(RCB(16, 32))
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer2 = nn.Sequential(RCB(32, 64))
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer3 = nn.Sequential(RCB(64, 128))
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer4 = nn.Sequential(RCB(128, 256))

        # FreqLGFA modules (applied BEFORE pooling - high resolution)
        self.freq_lgfa1 = FLGFA_v2(channels=32)
        self.freq_lgfa2 = FLGFA_v2(channels=64)
        self.freq_lgfa3 = FLGFA_v2(channels=128)

    def forward(self, x, g1=None, g2=None, g3=None):

        f0 = self.stem(x)

        # Layer 1
        f1 = self.layer1(f0)
        if g1 is not None:
            f1 = self.freq_lgfa1(f_local=f1, f_global=g1)
        x = self.pool1(f1)

        # Layer 2
        f2 = self.layer2(x)
        if g2 is not None:
            f2 = self.freq_lgfa2(f_local=f2, f_global=g2)
        x = self.pool2(f2)

        # Layer 3
        f3 = self.layer3(x)
        if g3 is not None:
            f3 = self.freq_lgfa3(f_local=f3, f_global=g3)
        x = self.pool3(f3)

        f4 = self.layer4(x)  # [B, 256, 8, 8, 8]

        return [f0, f1, f2, f3, f4]

class LeadingEncoder3(nn.Module):

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

        # RCB layers
        self.layer1 = nn.Sequential(RCB(16, 32))
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer2 = nn.Sequential(RCB(32, 64))
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer3 = nn.Sequential(RCB(64, 128))
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer4 = nn.Sequential(RCB(128, 256))

        # FreqLGFA modules (applied BEFORE pooling - high resolution)
        self.freq_lgfa1 = FLGFA_v3(channels=32)
        self.freq_lgfa2 = FLGFA_v3(channels=64)
        self.freq_lgfa3 = FLGFA_v3(channels=128)

    def forward(self, x, g1=None, g2=None, g3=None):

        f0 = self.stem(x)

        # Layer 1
        f1 = self.layer1(f0)
        if g1 is not None:
            f1 = self.freq_lgfa1(f_local=f1, f_global=g1)
        x = self.pool1(f1)

        # Layer 2
        f2 = self.layer2(x)
        if g2 is not None:
            f2 = self.freq_lgfa2(f_local=f2, f_global=g2)
        x = self.pool2(f2)

        # Layer 3
        f3 = self.layer3(x)
        if g3 is not None:
            f3 = self.freq_lgfa3(f_local=f3, f_global=g3)
        x = self.pool3(f3)

        f4 = self.layer4(x)  # [B, 256, 8, 8, 8]

        return [f0, f1, f2, f3, f4]


class BottleneckFusion(nn.Module):

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


class BTS_FLGFA(nn.Module):

    def __init__(self, patch_size_d=None, patch_size_h=None, patch_size_w=None,
                 in_channels=1, num_classes=4):
        super().__init__()

        self.leading_encoder = LeadingEncoder(in_ch=in_channels)
        self.auxiliary_encoder = AuxiliaryEncoder()
        self.bottleneck_fusion = BottleneckFusion(channels=256)
        self.decoder = Decoder(num_classes=num_classes)
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

        # Leading encoder with FreqLGFA fusion at layers 1, 2, 3
        f0, f1, f2, f3, f4 = self.leading_encoder(t2, g1=g1, g2=g2, g3=g3)
        # Bottleneck fusion (concatenation + 1x1 conv)
        bottleneck_fused = self.bottleneck_fusion(f_local=f4, f_global=g4)
        # bottleneck_fused: [B, 256, 8, 8, 8]
        output = self.decoder(bottleneck_fused, f0, f1, f2, f3)
        # output: [B, num_classes, 128, 128, 128]
        return output


class BTS_FLGFA1(nn.Module):

    def __init__(self, patch_size_d=None, patch_size_h=None, patch_size_w=None,
                 in_channels=1, num_classes=4):
        super().__init__()

        self.leading_encoder = LeadingEncoder1(in_ch=in_channels)
        self.auxiliary_encoder = AuxiliaryEncoder()
        self.bottleneck_fusion = BottleneckFusion(channels=256)
        self.decoder = Decoder(num_classes=num_classes)
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

        # Leading encoder with FreqLGFA fusion at layers 1, 2, 3
        f0, f1, f2, f3, f4 = self.leading_encoder(t2, g1=g1, g2=g2, g3=g3)
        # Bottleneck fusion (concatenation + 1x1 conv)
        bottleneck_fused = self.bottleneck_fusion(f_local=f4, f_global=g4)
        # bottleneck_fused: [B, 256, 8, 8, 8]
        output = self.decoder(bottleneck_fused, f0, f1, f2, f3)
        # output: [B, num_classes, 128, 128, 128]
        return output


class BTS_FLGFA2(nn.Module):

    def __init__(self, patch_size_d=None, patch_size_h=None, patch_size_w=None,
                 in_channels=1, num_classes=4):
        super().__init__()

        self.leading_encoder = LeadingEncoder2(in_ch=in_channels)
        self.auxiliary_encoder = AuxiliaryEncoder()
        self.bottleneck_fusion = BottleneckFusion(channels=256)
        self.decoder = Decoder(num_classes=num_classes)
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

        # Leading encoder with FreqLGFA fusion at layers 1, 2, 3
        f0, f1, f2, f3, f4 = self.leading_encoder(t2, g1=g1, g2=g2, g3=g3)
        # Bottleneck fusion (concatenation + 1x1 conv)
        bottleneck_fused = self.bottleneck_fusion(f_local=f4, f_global=g4)
        # bottleneck_fused: [B, 256, 8, 8, 8]
        output = self.decoder(bottleneck_fused, f0, f1, f2, f3)
        # output: [B, num_classes, 128, 128, 128]
        return output


class BTS_FLGFA3(nn.Module):

    def __init__(self, patch_size_d=None, patch_size_h=None, patch_size_w=None,
                 in_channels=1, num_classes=4):
        super().__init__()

        self.leading_encoder = LeadingEncoder3(in_ch=in_channels)
        self.auxiliary_encoder = AuxiliaryEncoder()
        self.bottleneck_fusion = BottleneckFusion(channels=256)
        self.decoder = Decoder(num_classes=num_classes)
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

        # Leading encoder with FreqLGFA fusion at layers 1, 2, 3
        f0, f1, f2, f3, f4 = self.leading_encoder(t2, g1=g1, g2=g2, g3=g3)
        # Bottleneck fusion (concatenation + 1x1 conv)
        bottleneck_fused = self.bottleneck_fusion(f_local=f4, f_global=g4)
        # bottleneck_fused: [B, 256, 8, 8, 8]
        output = self.decoder(bottleneck_fused, f0, f1, f2, f3)
        # output: [B, num_classes, 128, 128, 128]
        return output



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
    model = BTS_FLGFA(128,128,128, in_channels=1, num_classes=num_classes)

    total_params, trainable_params = count_parameters(model)
    print(f"\nModel Statistics:")
    print(f"  - Total parameters: {total_params / 1e6:.3f}M")
    print(f"  - Trainable parameters: {trainable_params / 1e6:.3f}M")
    print(f"  - Model size (approx): {total_params * 4 / (1024 ** 2):.3f} MB")


if __name__ == "__main__":
    test_model()