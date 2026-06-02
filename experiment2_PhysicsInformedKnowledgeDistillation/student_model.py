"""
Student U-Net for Physics-Guided Knowledge Distillation (Approach 2).

Mirrors the teacher ImprovedPhysicsInformedUNet conceptually:
  - Residual encoder (strides 2, 2, (1,2))           ← same layout as teacher
  - RSS CNN encoder → global feature (B, d)           ← same role as teacher
  - Single 8-head cross-attention bottleneck           ← replaces teacher's 339M FC + 5-layer transformer
  - Residual decoder with skip connections             ← same skip structure as teacher
  - Global residual + final conv                       ← identical to teacher

The only two architectural changes (per the proposal):
  1. Standard Conv2d → depthwise-separable (DW-sep) convs
  2. 339M FC bottleneck + 5-layer transformer → single MHA cross-attention

I/O contract matches teacher exactly:
  initial_channel : (B, 32, 4, 576)
  rss_map         : (B,  2, 30, 30)
  output          : (B, 32, 4, 576)

Three presets (all verified during planning, teacher = 358.9M params):
  light    — d=1280, widths=(1024,2048)  → ~35.98M params  (10.0x)
  moderate — d=768,  widths=(768,1536)   → ~17.83M params  (20.1x)
  extreme  — d=640,  widths=(512,1024)   →  ~9.29M params  (38.7x)
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

PresetName = Literal["light", "moderate", "extreme"]

STUDENT_PRESETS: dict[PresetName, dict] = {
    "light":    {"d_bottleneck": 1280, "widths": (1024, 2048)},
    "moderate": {"d_bottleneck": 768,  "widths": (768,  1536)},
    "extreme":  {"d_bottleneck": 640,  "widths": (512,  1024)},
}


# ── Building blocks ──────────────────────────────────────────────────────────

class DWSepDownBlock(nn.Module):
    """
    Depthwise-separable encoder block with residual.

    main: DW-conv (strided, groups=in_ch) → PW-conv (1×1) → GroupNorm
    skip: 1×1 strided conv → GroupNorm
    act:  SiLU

    Single DW+PW pass (no extra full conv) keeps param counts predictable.
    The residual branch provides the skip gradient path.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int | tuple = 1):
        super().__init__()
        s = (stride, stride) if isinstance(stride, int) else stride
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=s, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=s, bias=False),
            nn.GroupNorm(8, out_ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.shortcut(x))


class DWSepUpBlock(nn.Module):
    """
    Depthwise-separable decoder block: bilinear upsample → DW-sep conv + residual.
    Bilinear upsample avoids checkerboard artifacts and is efficient on MPS.
    Single DW+PW pass (no extra full conv) for predictable param counts.
    """

    def __init__(self, in_ch: int, out_ch: int, scale_factor: float | tuple):
        super().__init__()
        sf = scale_factor
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=sf, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch),
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=sf, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.shortcut(x))


# ── Student U-Net ─────────────────────────────────────────────────────────────

class StudentUNet(nn.Module):
    """
    Scalable student U-Net for PG-KD.

    Conceptual mirror of ImprovedPhysicsInformedUNet (teacher):
      encoder →  RSS encoder + cross-attention  →  decoder → output
    with depthwise-separable convs and single-layer cross-attention bottleneck.

    Args:
        widths        : (w1, w2) — encoder channel widths for stage 1 and 2.
        d_bottleneck  : bottleneck dim d; enc3 output is (B, d, 1, 72).
        rss_in_ch     : RSS map input channels (2 = grayscale + dBm).
        return_features : when True, forward() also returns the cross-attn
                          map (B, 72, d) for the xattn distillation term.
    """

    def __init__(
        self,
        widths: tuple[int, int] = (512, 1024),
        d_bottleneck: int = 640,
        rss_in_ch: int = 2,
    ):
        super().__init__()
        w1, w2 = widths
        d = d_bottleneck
        self.d_bottleneck = d

        # ── Encoder (same strides as teacher: 2, 2, (1,2)) ──────────────────
        self.enc1 = DWSepDownBlock(32, w1, stride=2)           # (B, w1, 2, 288)
        self.enc2 = DWSepDownBlock(w1, w2, stride=2)           # (B, w2, 1, 144)
        self.enc3 = DWSepDownBlock(w2, d,  stride=(1, 2))      # (B,  d, 1,  72)

        # ── RSS encoder (same role as teacher's rss_encoder) ─────────────────
        # Input (B, 2, 30, 30) → (B, d, 1, 1) → (B, d)
        self.rss_encoder = nn.Sequential(
            nn.Conv2d(rss_in_ch, 32, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),                           # (B, 32, 15, 15)
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),                           # (B, 64, 7, 7)
            nn.Conv2d(64, d, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),              # (B, d, 1, 1)
        )

        # ── Single cross-attention bottleneck (replaces 339M FC + 5-layer TF) ─
        # query: encoder spatial sequence (B, 72, d)
        # key/value: RSS summary token   (B,  1, d)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=8, dropout=0.1, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d)

        # ── Skip-connection 1×1 convs (mirror teacher's skip_conv1/skip_conv2) ─
        self.skip_conv2 = nn.Sequential(nn.Conv2d(w2, w2, 1, bias=False), nn.GroupNorm(8, w2))
        self.skip_conv1 = nn.Sequential(nn.Conv2d(w1, w1, 1, bias=False), nn.GroupNorm(8, w1))

        # ── Decoder with skip connections ────────────────────────────────────
        # dec1: (B, d, 1, 72) → upsample (1,2) → (B, w2, 1, 144) + cat(enc2) → (B, 2*w2, 1, 144)
        self.dec1 = DWSepUpBlock(d, w2, scale_factor=(1.0, 2.0))
        # dec2: (B, 2*w2, 1, 144) → upsample 2 → (B, w1, 2, 288) + cat(enc1) → (B, 2*w1, 2, 288)
        self.dec2 = DWSepUpBlock(2 * w2, w1, scale_factor=2.0)
        # dec3: (B, 2*w1, 2, 288) → upsample 2 → (B, 32, 4, 576)
        self.dec3 = DWSepUpBlock(2 * w1, 32, scale_factor=2.0)

        # ── Output (mirrors teacher's final_conv + global_residual) ──────────
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 1),
        )
        self.global_residual = nn.Conv2d(32, 32, 1)

    def forward(
        self,
        initial_channel: torch.Tensor,
        rss_map: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            initial_channel : (B, 32, 4, 576)
            rss_map         : (B, 2, 30, 30)
            return_features : when True, also returns cross-attn map (B, 72, d)
                              needed for the xattn feature-distillation loss

        Returns:
            output          : (B, 32, 4, 576)
            xattn_feat      : (B, 72, d)  — only when return_features=True
        """
        B = initial_channel.shape[0]

        # Global residual (mirrors teacher)
        input_res = self.global_residual(initial_channel)

        # Encoder with skip connection storage
        e1 = self.enc1(initial_channel)  # (B, w1, 2, 288)
        e2 = self.enc2(e1)               # (B, w2, 1, 144)
        e3 = self.enc3(e2)               # (B,  d, 1,  72)

        # RSS encoder: (B, 2, 30, 30) → (B, d, 1, 1) → (B, d) → (B, 1, d)
        rss_feat = self.rss_encoder(rss_map).view(B, self.d_bottleneck)
        rss_kv = rss_feat.unsqueeze(1)   # (B, 1, d) — key/value for cross-attn

        # Reshape bottleneck to spatial sequence for cross-attention
        query = e3.reshape(B, self.d_bottleneck, 72).permute(0, 2, 1)  # (B, 72, d)

        # Cross-attention: each of the 72 spatial positions attends to RSS summary
        attended, _ = self.cross_attn(query, rss_kv, rss_kv)           # (B, 72, d)
        attended = self.attn_norm(query + attended)                      # pre-norm residual

        # Reshape back to 2-D spatial and fuse with encoder residual
        processed = attended.permute(0, 2, 1).reshape(B, self.d_bottleneck, 1, 72)
        processed = processed + e3  # encoder residual (mirrors teacher's e3 add)

        # Decoder with skip connections
        d1 = self.dec1(processed)                                        # (B, w2, 1, 144)
        d1 = torch.cat([d1, self.skip_conv2(e2)], dim=1)                # (B, 2*w2, 1, 144)

        d2 = self.dec2(d1)                                               # (B, w1, 2, 288)
        d2 = torch.cat([d2, self.skip_conv1(e1)], dim=1)                # (B, 2*w1, 2, 288)

        d3 = self.dec3(d2)                                               # (B, 32, 4, 576)

        output = self.final_conv(d3) + input_res                         # (B, 32, 4, 576)

        if return_features:
            return output, attended   # attended: (B, 72, d)
        return output


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_student(preset: PresetName) -> StudentUNet:
    cfg = STUDENT_PRESETS[preset]
    return StudentUNet(widths=cfg["widths"], d_bottleneck=cfg["d_bottleneck"])


if __name__ == "__main__":
    teacher_params = 358_878_624

    for name, cfg in STUDENT_PRESETS.items():
        m = StudentUNet(**cfg)
        p = count_parameters(m)
        ratio = teacher_params / p
        x = torch.randn(2, 32, 4, 576)
        r = torch.randn(2, 2, 30, 30)
        out, feat = m(x, r, return_features=True)
        print(
            f"[{name:8s}] d={cfg['d_bottleneck']:4d} widths={cfg['widths']} "
            f"params={p/1e6:.2f}M  ratio={ratio:.1f}x "
            f"out={tuple(out.shape)} feat={tuple(feat.shape)}"
        )
