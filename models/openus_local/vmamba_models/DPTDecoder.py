import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        x = self.relu(x)
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        return x + residual


class FeatureFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res_conv_unit1 = ResidualConvUnit(channels)
        self.res_conv_unit2 = ResidualConvUnit(channels)

    def forward(self, x, skip=None):
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.res_conv_unit1(skip)
        x = self.res_conv_unit2(x)
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class DPTDecoder(nn.Module):
    """
    DPT-style segmentation decoder for OpenUS/VMamba multiscale features.

    Expected input is the four feature maps returned by
    Backbone_DINOv2_VSSM_2(seg_head=True):
    [B, 96, H/4, W/4], [B, 192, H/8, W/8],
    [B, 384, H/16, W/16], [B, 768, H/32, W/32].
    """

    def __init__(
        self,
        in_channels=(96, 192, 384, 768),
        features=256,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()
        self.projects = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, features, kernel_size=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False),
                )
                for in_ch in in_channels
            ]
        )

        self.refinenet4 = FeatureFusionBlock(features)
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet1 = FeatureFusionBlock(features)

        self.head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=False),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(features, num_classes, kernel_size=1),
        )

    def forward(self, features):
        if not isinstance(features, (list, tuple)) or len(features) != 4:
            raise ValueError("DPTDecoder expects a list/tuple of four feature maps.")

        layer_1, layer_2, layer_3, layer_4 = [
            project(feature) for project, feature in zip(self.projects, features)
        ]

        path_4 = self.refinenet4(layer_4)
        path_3 = self.refinenet3(path_4, layer_3)
        path_2 = self.refinenet2(path_3, layer_2)
        path_1 = self.refinenet1(path_2, layer_1)

        return self.head(path_1)
