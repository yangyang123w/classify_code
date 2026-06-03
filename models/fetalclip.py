# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_FETALCLIP_CONFIG = {
    "embed_dim": 768,
    "vision_cfg": {
        "image_size": 224,
        "layers": 24,
        "width": 1024,
        "patch_size": 14,
    },
    "text_cfg": {
        "context_length": 117,
        "vocab_size": 49408,
        "width": 768,
        "heads": 12,
        "layers": 12,
    },
}


def _clean_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        for prefix in ["module.", "model.", "clip_model."]:
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v
    return cleaned


def load_fetalclip_official(weight_path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> nn.Module:
    """Load FetalCLIP in the same way as the official repo.

    Official quick start:
        open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config
        open_clip.create_model_and_transforms("FetalCLIP", pretrained=PATH_FETALCLIP_WEIGHT)
    """
    try:
        import open_clip
    except ImportError as e:
        raise ImportError("缺少 open_clip，请先安装: pip install open-clip-torch==2.26.1") from e

    weight_path = Path(weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"FetalCLIP 权重不存在: {weight_path}")

    open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = DEFAULT_FETALCLIP_CONFIG

    try:
        model, _, _ = open_clip.create_model_and_transforms("FetalCLIP", pretrained=str(weight_path))
    except Exception as first_error:
        # Some local environments have open_clip path-loading quirks. Fall back to manual load.
        model, _, _ = open_clip.create_model_and_transforms("FetalCLIP", pretrained=None)
        ckpt = torch.load(str(weight_path), map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
        state = _clean_state_dict(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("[FetalCLIP] open_clip pretrained path load failed, used manual state_dict fallback.")
        print(f"[FetalCLIP] original error: {repr(first_error)}")
        print(f"[FetalCLIP] missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()
    model.to(device)
    return model


class FetalCLIPVisionBackbone(nn.Module):
    """Dense patch-token extractor from the official FetalCLIP image encoder.

    FetalCLIP official image encoder is OpenCLIP ViT-L/14 with image_size=224.
    This wrapper returns patch tokens reshaped as feature maps: [B, C, 16, 16].
    """

    def __init__(self,
                 weight_path: Union[str, Path],
                 freeze: bool = True,
                 unfreeze_last_n_blocks: int = 0,
                 input_size: int = 224):
        super().__init__()
        self.clip_model = load_fetalclip_official(weight_path, device="cpu")
        self.visual = self.clip_model.visual
        self.freeze = bool(freeze)
        self.input_size = int(input_size)
        self.width = self._infer_width()
        self.patch_size = self._infer_patch_size()

        # We only use the image encoder for this downstream keypoint task.
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        if not self.freeze:
            for p in self.visual.parameters():
                p.requires_grad_(True)

        if self.freeze and unfreeze_last_n_blocks > 0:
            self.unfreeze_last_blocks(unfreeze_last_n_blocks)
            self.freeze = False

    def _infer_width(self) -> int:
        if hasattr(self.visual, "conv1"):
            return int(self.visual.conv1.out_channels)
        if hasattr(self.visual, "trunk") and hasattr(self.visual.trunk, "embed_dim"):
            return int(self.visual.trunk.embed_dim)
        return 1024

    def _infer_patch_size(self) -> int:
        if hasattr(self.visual, "conv1"):
            k = self.visual.conv1.kernel_size
            return int(k[0] if isinstance(k, tuple) else k)
        if hasattr(self.visual, "patch_size"):
            p = self.visual.patch_size
            return int(p[0] if isinstance(p, tuple) else p)
        return 14

    def unfreeze_last_blocks(self, n: int) -> None:
        """Unfreeze the last n transformer residual blocks of the visual encoder."""
        blocks = None
        if hasattr(self.visual, "transformer") and hasattr(self.visual.transformer, "resblocks"):
            blocks = self.visual.transformer.resblocks
        elif hasattr(self.visual, "trunk") and hasattr(self.visual.trunk, "blocks"):
            blocks = self.visual.trunk.blocks
        if blocks is None:
            print("[FetalCLIP] 未找到 transformer blocks，无法按层解冻；保持 backbone 冻结。")
            return
        for blk in list(blocks)[-int(n):]:
            for p in blk.parameters():
                p.requires_grad_(True)
        # Norm layers around tokens are safe to train when last blocks are unfrozen.
        for name in ["ln_pre", "ln_post", "norm"]:
            if hasattr(self.visual, name):
                for p in getattr(self.visual, name).parameters():
                    p.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.visual.eval()
            self.clip_model.eval()
        return self

    def _interpolate_pos_embed(self, pos: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        # pos: [1+N, C] or [1,1+N,C]
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        if pos.shape[1] == 1 + grid_h * grid_w:
            return pos
        cls_pos = pos[:, :1]
        patch_pos = pos[:, 1:]
        old = int(patch_pos.shape[1] ** 0.5)
        patch_pos = patch_pos.reshape(1, old, old, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def _extract_tokens_openclip_vit(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        v = self.visual
        x = v.conv1(x)  # [B, C, H/patch, W/patch]
        b, c, gh, gw = x.shape
        x = x.reshape(b, c, gh * gw).permute(0, 2, 1)  # [B, N, C]

        cls = v.class_embedding.to(dtype=x.dtype, device=x.device)
        cls = cls.view(1, 1, -1).expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if hasattr(v, "positional_embedding") and v.positional_embedding is not None:
            pos = v.positional_embedding.to(dtype=x.dtype, device=x.device)
            pos = self._interpolate_pos_embed(pos, gh, gw)
            x = x + pos

        if hasattr(v, "patch_dropout"):
            x = v.patch_dropout(x)
        if hasattr(v, "ln_pre"):
            x = v.ln_pre(x)

        # OpenCLIP Transformer normally accepts [B, L, C]. Older variants may expect [L, B, C].
        try:
            x = v.transformer(x)
        except RuntimeError:
            x = v.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)

        if hasattr(v, "ln_post") and v.ln_post is not None:
            try:
                x = v.ln_post(x)
            except Exception:
                pass
        tokens = x[:, 1:, :]
        feat = tokens.transpose(1, 2).reshape(b, -1, gh, gw).contiguous()
        return feat, (gh, gw)

    def _extract_tokens_generic(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        v = self.visual
        # Some timm-like visual trunks expose forward_features.
        out = v.forward_features(x)
        if isinstance(out, dict):
            out = out.get("x_norm_patchtokens", out.get("tokens", out.get("last_hidden_state", None)))
        if out is None:
            raise RuntimeError("visual.forward_features 返回格式无法解析。")
        if out.dim() == 4:
            return out, (out.shape[-2], out.shape[-1])
        if out.dim() == 3:
            # Remove class token if token count is 1+square.
            b, n, c = out.shape
            if int((n - 1) ** 0.5) ** 2 == n - 1:
                out = out[:, 1:, :]
                n = n - 1
            g = int(n ** 0.5)
            return out.transpose(1, 2).reshape(b, c, g, g).contiguous(), (g, g)
        raise RuntimeError(f"无法解析 visual features shape: {tuple(out.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                return self._forward_impl(x)
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.visual, "conv1") and hasattr(self.visual, "class_embedding") and hasattr(self.visual, "transformer"):
            feat, _ = self._extract_tokens_openclip_vit(x)
        elif hasattr(self.visual, "forward_features"):
            feat, _ = self._extract_tokens_generic(x)
        else:
            raise RuntimeError("当前 open_clip visual 类型不支持密集 token 提取，请检查 open_clip 版本。")
        return feat


class FetalCLIPFeatureBackbone(nn.Module):
    def __init__(self, weight_path: Union[str, Path], freeze: bool = True) -> None:
        super().__init__()
        self.backbone = FetalCLIPVisionBackbone(
            weight_path=weight_path,
            freeze=freeze,
            unfreeze_last_n_blocks=0,
            input_size=224,
        )
        self.out_dim = int(getattr(self.backbone, "width", 1024))
        self.freeze = bool(freeze)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
