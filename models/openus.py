from pathlib import Path

import torch
import torch.nn as nn


PREFIXES = (
    "module.backbone.",
    "module.",
    "backbone.",
    "teacher.",
    "student.",
)

DEFAULT_CHECKPOINT_KEY = "teacher"


def _strip_prefix(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def load_openus_checkpoint(model: nn.Module, weight_path: str, checkpoint_key: str = DEFAULT_CHECKPOINT_KEY) -> dict:
    raw = torch.load(weight_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and checkpoint_key in raw:
        state_dict = raw[checkpoint_key]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    elif isinstance(raw, dict):
        state_dict = raw
    else:
        raise ValueError(f"Unsupported OpenUS checkpoint format: {type(raw)}")

    state_dict = {
        _strip_prefix(key): value
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }
    model_state = model.state_dict()
    matched = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    msg = model.load_state_dict(matched, strict=False)
    return {
        "checkpoint_path": weight_path,
        "checkpoint_key": checkpoint_key,
        "matched_keys": len(matched),
        "model_keys": len(model_state),
        "missing_keys": list(msg.missing_keys),
        "unexpected_keys": list(msg.unexpected_keys),
    }


class OpenUSFeatureBackbone(nn.Module):
    def __init__(
        self,
        weight_path: str,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        from models.openus_local.vmamba_models.dino_vmamba import Backbone_DINOv2_VSSM_2

        self.backbone = Backbone_DINOv2_VSSM_2(
            pretrained=None,
            masked_im_modeling=False,
            training=False,
        )
        self.load_info = load_openus_checkpoint(self.backbone, weight_path)
        self.out_dim = int(self.backbone.dims[-1])
        self.freeze = bool(freeze)
        if self.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("OpenUS VMamba backbone requires CUDA input.")
        return self.backbone(x)
