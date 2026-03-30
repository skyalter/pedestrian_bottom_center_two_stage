"""
BottomCenterNet — 精度优先的底部中心点回归模型

精度提升来源（对比 FastBottomCenterModel）：
  1. 输入 256×256（原 128×128），4倍空间分辨率 → 亚像素定位更准
  2. 完整 MobileNetV2 通道宽度（最宽 320ch，原 96ch）
  3. 深层 InvResidual 加入 SE 注意力，聚焦有效特征区域
  4. 更大回归头（512→256→2，原 256→128→2）

延迟目标：~5ms GPU / ~30ms CPU（原 0.7ms / 3ms）
参数量：~3.6M（原 ~280K）
"""

import time
import torch
import torch.nn as nn


def _make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=(kernel_size - 1) // 2,
                              groups=groups, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力（轻量，reduction=4）"""
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, ch // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x)).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class InvertedResidual(nn.Module):
    """
    MobileNetV2 风格 Inverted Residual Block，可选 SE 注意力。
      PW expand → DW 3×3 → PW linear (→ SE)
    """
    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, expand_ratio: int = 6,
                 use_se: bool = False):
        super().__init__()
        self.use_res = (stride == 1 and in_ch == out_ch)
        hidden = _make_divisible(in_ch * expand_ratio)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, hidden, 1))
        layers += [
            ConvBNAct(hidden, hidden, 3, stride=stride, groups=hidden),
            ConvBNAct(hidden, out_ch, 1, act=False),
        ]
        self.conv = nn.Sequential(*layers)
        self.se   = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.se(self.conv(x))
        return x + out if self.use_res else out


class BottomCenterNet(nn.Module):
    """
    行人底部中心点直接回归模型（精度优先）。

    推荐输入尺寸：256×256
    参数量：~3.6M
    GPU 延迟：~5ms（batch=1）
    CPU 延迟：~30ms

    输出：(B, 2) 归一化坐标 [0, 1]
        output[:, 0] = x_norm = x_pixel / img_width
        output[:, 1] = y_norm = y_pixel / img_height

    解码（推理时）：
        x_in   = x_norm * W_in
        y_in   = y_norm * H_in
        x_orig = (x_in - dw) / ratio   # 去除 letterbox padding
        y_orig = (y_in - dh) / ratio
    """

    def __init__(self, num_classes: int = 1, dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_classes  # 保留 API 兼容字段

        # ── Backbone：完整 MobileNetV2 通道宽度 + 深层 SE ──────────────
        # 输入 3×256×256
        self.features = nn.Sequential(
            # stem
            ConvBNAct(3, 32, 3, stride=2),                                 # 128×128, 32ch

            # stage1: t=1, c=16, n=1, s=1
            InvertedResidual(32,  16,  stride=1, expand_ratio=1),          # 128×128, 16ch

            # stage2: t=6, c=24, n=2, s=2
            InvertedResidual(16,  24,  stride=2, expand_ratio=6),          # 64×64,   24ch
            InvertedResidual(24,  24,  stride=1, expand_ratio=6),          # 64×64,   24ch

            # stage3: t=6, c=32, n=3, s=2
            InvertedResidual(24,  32,  stride=2, expand_ratio=6),          # 32×32,   32ch
            InvertedResidual(32,  32,  stride=1, expand_ratio=6),          # 32×32,   32ch
            InvertedResidual(32,  32,  stride=1, expand_ratio=6),          # 32×32,   32ch

            # stage4: t=6, c=64, n=4, s=2
            InvertedResidual(32,  64,  stride=2, expand_ratio=6),          # 16×16,   64ch
            InvertedResidual(64,  64,  stride=1, expand_ratio=6),          # 16×16,   64ch
            InvertedResidual(64,  64,  stride=1, expand_ratio=6),          # 16×16,   64ch
            InvertedResidual(64,  64,  stride=1, expand_ratio=6),          # 16×16,   64ch

            # stage5: t=6, c=96, n=3, s=1  ← SE 启用
            InvertedResidual(64,  96,  stride=1, expand_ratio=6, use_se=True),   # 16×16, 96ch
            InvertedResidual(96,  96,  stride=1, expand_ratio=6, use_se=True),   # 16×16, 96ch
            InvertedResidual(96,  96,  stride=1, expand_ratio=6, use_se=True),   # 16×16, 96ch

            # stage6: t=6, c=160, n=3, s=2  ← SE 启用
            InvertedResidual(96,  160, stride=2, expand_ratio=6, use_se=True),   # 8×8,  160ch
            InvertedResidual(160, 160, stride=1, expand_ratio=6, use_se=True),   # 8×8,  160ch
            InvertedResidual(160, 160, stride=1, expand_ratio=6, use_se=True),   # 8×8,  160ch

            # stage7: t=6, c=320, n=1, s=1  ← SE 启用
            InvertedResidual(160, 320, stride=1, expand_ratio=6, use_se=True),   # 8×8,  320ch

            # conv head
            ConvBNAct(320, 512, 1),                                        # 8×8,  512ch
        )

        # ── 全局池化 + 回归头 ─────────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, 2),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)  推荐 H=W=256
        Returns:
            (B, 2)  [x_norm, y_norm] ∈ [0, 1]
        """
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# ── 快速测试 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = BottomCenterNet().eval()

    n = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n:,} ({n/1e6:.3f}M)")

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    print(f"Input: {x.shape}  →  Output: {out.shape}  values: {out[0].tolist()}")

    runs = 200
    with torch.no_grad():
        for _ in range(20):
            model(x)
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        t1 = time.perf_counter()
    print(f"CPU avg latency: {(t1 - t0) / runs * 1000:.2f} ms/img")

    if torch.cuda.is_available():
        model_gpu = model.cuda()
        x_gpu = x.cuda()
        with torch.no_grad():
            for _ in range(50):
                model_gpu(x_gpu)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                model_gpu(x_gpu)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        print(f"GPU avg latency: {(t1 - t0) / runs * 1000:.2f} ms/img")
