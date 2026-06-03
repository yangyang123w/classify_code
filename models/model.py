import torch
import torch.nn as nn

from models.dinov3 import DINOv3FeatureBackbone
from models.fetalclip import FetalCLIPFeatureBackbone
from models.openus import OpenUSFeatureBackbone
from models.usfm import USFMFeatureBackbone


def build_backbone(args) -> nn.Module:
    name = args.backbone.lower()
    if name == "usfm":
        return USFMFeatureBackbone(args.usfm_weight, freeze=args.freeze_backbone)
    if name == "fetalclip":
        return FetalCLIPFeatureBackbone(args.fetalclip_weight, freeze=args.freeze_backbone)
    if name == "dinov3":
        return DINOv3FeatureBackbone(args.dinov3_weight, freeze=args.freeze_backbone)
    if name == "openus":
        return OpenUSFeatureBackbone(args.openus_weight, freeze=args.freeze_backbone)
    raise ValueError(f"Unsupported backbone: {args.backbone}")


class BackboneClassifier(nn.Module):
    def __init__(self, args, num_classes: int) -> None:
        super().__init__()
        self.backbone = build_backbone(args)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(self.backbone.out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if feats.dim() == 3:
            feats = feats[:, 1:, :] if feats.shape[1] > 1 else feats
            feats = feats.mean(dim=1)
        elif feats.dim() == 4:
            feats = self.pool(feats).flatten(1)
        else:
            raise RuntimeError(f"Unsupported feature shape: {tuple(feats.shape)}")
        return self.classifier(self.dropout(feats))
