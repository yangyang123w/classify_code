from pathlib import Path

import torch
import torch.nn as nn


SKIP_SUBSTRINGS = (
    "dino_head",
    "ibot_head",
    "head.last_layer",
    "head.prototypes",
    "prototypes",
    "classifier",
    "seg_head",
)

PREFIXES = (
    "module.",
    "teacher.",
    "student.",
    "model.",
    "backbone.",
    "encoder.",
    "network.",
)


def _looks_like_state_dict(obj) -> bool:
    return isinstance(obj, dict) and any(isinstance(v, torch.Tensor) for v in obj.values())


def _strip_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _extract_state_dict(raw_obj) -> dict:
    candidates = []
    if _looks_like_state_dict(raw_obj):
        candidates.append(raw_obj)
    if isinstance(raw_obj, dict):
        for key in ("teacher", "student", "state_dict", "model", "backbone"):
            value = raw_obj.get(key)
            if _looks_like_state_dict(value):
                candidates.append(value)
    if not candidates:
        raise ValueError("Cannot find a valid DINOv3 state_dict in checkpoint.")

    merged = {}
    for state_dict in candidates:
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            if any(skip in key for skip in SKIP_SUBSTRINGS):
                continue
            merged[_strip_prefixes(key)] = value
    return merged


def load_dinov3_checkpoint(model: nn.Module, checkpoint_path: str, strict: bool = True) -> dict:
    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(raw)
    model_state = model.state_dict()
    matched = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    skipped = [key for key in state_dict if key not in matched]
    msg = model.load_state_dict(matched, strict=False)
    info = {
        "checkpoint_path": checkpoint_path,
        "matched_keys": len(matched),
        "model_keys": len(model_state),
        "missing_keys": list(msg.missing_keys),
        "unexpected_keys": list(msg.unexpected_keys),
        "skipped_keys": skipped[:200],
    }
    if strict and not matched:
        raise RuntimeError(f"No DINOv3 checkpoint keys matched: {checkpoint_path}")
    return info


class DINOv3FeatureBackbone(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        from models.dinov3_local.dinov3.models.vision_transformer import DinoVisionTransformer

        self.backbone = DinoVisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            ffn_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-5,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=0,
            mask_k_bias=False,
            pos_embed_rope_base=100.0,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2.0,
            pos_embed_rope_dtype="fp32",
        )
        self.backbone.init_weights()
        self.load_info = load_dinov3_checkpoint(self.backbone, checkpoint_path, strict=True)
        self.out_indices = (2, 5, 8, 11)
        self.out_dim = 768
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
        feats = self.backbone.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=True,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        return list(feats)[-1]
