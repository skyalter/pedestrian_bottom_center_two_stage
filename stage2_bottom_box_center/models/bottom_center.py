import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------- Utils --------
class LayerNorm2d(nn.Module):
    """LayerNorm over channel dimension for 2D feature maps (B, C, H, W)."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, H, W, C) -> LN -> (B, C, H, W)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DropPath(nn.Module):
    """Stochastic Depth; drop paths per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand.floor_()
        return x / keep_prob * rand


# -------- ResNet34 stem & basic blocks (for early stages) --------
def conv3x3(in_ch, out_ch, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, groups=groups, dilation=dilation)

class BasicBlock(nn.Module):
    """ResNet-34 BasicBlock (for early stages C2/C3)."""
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = norm_layer(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = norm_layer(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# -------- ConvNeXt-style block (for later stages) --------
class ConvNeXtBlock(nn.Module):
    """
    Depthwise 7x7 -> LayerNorm -> 1x1 (expand) -> GELU -> 1x1 (project) + DropPath + residual.
    """
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1.0, expansion: int = 4):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise
        self.ln = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, dim * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * expansion, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dw(x)
        x = self.ln(x)
        x = self.pw2(self.act(self.pw1(x)))
        # layer scale
        x = x * self.gamma.view(1, -1, 1, 1)
        x = shortcut + self.drop_path(x)
        return x


# -------- ResNet34(ConvNeXt 化) Backbone --------
class ResNet34_ConvNeXtBackbone(nn.Module):
    """
    Outputs feature maps at strides 4, 8, 16, 32 (C2, C3, C4, C5).
    Early stages: ResNet BasicBlock, Later: ConvNeXtBlock.
    """
    def __init__(self, drop_path_rate: float = 0.1, norm_layer=nn.BatchNorm2d, in_chans: int = 3):
        super().__init__()
        self.inplanes = 64
        self.norm_layer = norm_layer

        # Stem: 7x7 s=2 + BN + ReLU + 3x3 maxpool s=2 => stride 4, 64x64 for 256x256 input
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet34 layers (block counts: 3,4,6,3)
        self.layer1 = self._make_layer(BasicBlock, 64,  3, stride=1)  # C2: stride 4
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)  # C3: stride 8

        # Later stages use ConvNeXt-style blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 6 + 3)]  # 9 blocks across C4(6)+C5(3)
        self.layer3 = self._make_convnext_layer(256, blocks=6, stride=2, dp_list=dpr[:6])  # C4: stride 16
        self.layer4 = self._make_convnext_layer(512, blocks=3, stride=2, dp_list=dpr[6:])  # C5: stride 32

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self.norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, norm_layer))
        return nn.Sequential(*layers)

    def _make_convnext_layer(self, planes, blocks, stride, dp_list):
        # downsample conv to change stride & channels
        layer = []
        layer.append(nn.Sequential(
            nn.Conv2d(self.inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=True),
            LayerNorm2d(planes)
        ))
        self.inplanes = planes
        for i in range(blocks):
            layer.append(ConvNeXtBlock(dim=planes, drop_path=dp_list[i], expansion=4))
        return nn.Sequential(*layer)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, LayerNorm2d):
                nn.init.ones_(m.ln.weight)
                nn.init.zeros_(m.ln.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))     # /2
        x = self.maxpool(x)                        # /4  => C1
        c2 = self.layer1(x)                        # /4
        c3 = self.layer2(c2)                       # /8
        c4 = self.layer3(c3)                       # /16
        c5 = self.layer4(c4)                       # /32
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}


# -------- Lightweight FPN (top-down) to stride=4 --------
class LightweightFPN(nn.Module):
    """
    Fuse C5(1/32), C4(1/16), C3(1/8), C2(1/4) -> produce P2 at stride=4.
    """
    def __init__(self, in_channels: Dict[str, int], out_ch: int = 256, refine_ch: int = 256):
        super().__init__()
        self.lateral_c5 = nn.Conv2d(in_channels["c5"], out_ch, 1)
        self.lateral_c4 = nn.Conv2d(in_channels["c4"], out_ch, 1)
        self.lateral_c3 = nn.Conv2d(in_channels["c3"], out_ch, 1)
        self.lateral_c2 = nn.Conv2d(in_channels["c2"], out_ch, 1)

        self.smooth_p4 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.smooth_p3 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.smooth_p2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # optional light HR-style two-way refinement between P3 and P2
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, refine_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(refine_ch, out_ch, 3, padding=1),
        )

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        c2, c3, c4, c5 = feats["c2"], feats["c3"], feats["c4"], feats["c5"]
        p5 = self.lateral_c5(c5)
        p4 = self._upsample_add(p5, self.lateral_c4(c4))
        p4 = self.smooth_p4(p4)
        p3 = self._upsample_add(p4, self.lateral_c3(c3))
        p3 = self.smooth_p3(p3)
        p2 = self._upsample_add(p3, self.lateral_c2(c2))
        p2 = self.smooth_p2(p2)

        # light refine at P2
        p2 = p2 + self.refine(p2)
        return p2  # stride=4 feature (H/4, W/4)

    @staticmethod
    def _upsample_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # upsample x to y's spatial size and add
        return F.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=False) + y


# -------- Heads: Heatmap (C ch) + Offset (2 ch) --------
class Head(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, act: str = "gelu"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.norm = LayerNorm2d(mid_ch)
        self.act = nn.GELU() if act == "gelu" else nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class BottomCenterModel(nn.Module):
    """
    Full model:
    - Backbone: ResNet34 (ConvNeXt-style later) -> C2, C3, C4, C5
    - Neck: Lightweight FPN -> stride=4 feature (P2)
    - Heads: heatmap (C ch) + offset (2 ch), both at H/4 x W/4
    """
    def __init__(self,
                 num_classes: int,
                 offset_out_ch: int = 2,
                 fpn_out_ch: int = 256,
                 head_mid_ch: int = 128,
                 drop_path_rate: float = 0.1):
        super().__init__()
        self.backbone = ResNet34_ConvNeXtBackbone(drop_path_rate=drop_path_rate)

        in_channels = {"c2": 64, "c3": 128, "c4": 256, "c5": 512}
        self.fpn = LightweightFPN(in_channels=in_channels, out_ch=fpn_out_ch, refine_ch=fpn_out_ch)

        self.heatmap_head = Head(fpn_out_ch, head_mid_ch, num_classes, act="gelu")
        self.offset_head  = Head(fpn_out_ch, head_mid_ch, offset_out_ch,  act="gelu")

        with torch.no_grad():
            self.heatmap_head.conv2.bias.fill_(-2.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        p2 = self.fpn(feats)                    # stride=4, (B,256,H/4,W/4)
        heatmap = self.heatmap_head(p2)         # (B, C, H/4, W/4)
        offset  = self.offset_head(p2)          # (B, 2, H/4, W/4)
        return heatmap, offset



# -------- quick test --------
if __name__ == "__main__":
    model = BottomCenterModel()
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        hm, off = model(x)
    print("Input:", x.shape)
    print("Heatmap:", hm.shape, "(expect: 1 x C x 64 x 64)")
    print("Offset :", off.shape, "(expect: 1 x 2 x 64 x 64)")
