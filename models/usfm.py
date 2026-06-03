from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


@dataclass
class PretrainedLoadInfo:
    path: str
    matched_keys: int
    missing_keys: list[str]
    unexpected_keys: list[str]
    shape_mismatch_keys: list[str]
    loaded_parameter_names: set[str]


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        window_size: tuple[int, int] | None = None,
        attn_head_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim or dim // num_heads
        all_head_dim = head_dim * num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.window_size = window_size
        if window_size is not None:
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, num_heads))
            self.register_buffer('relative_position_index', self._build_relative_position_index(window_size))
        else:
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def _build_relative_position_index(window_size: tuple[int, int]) -> torch.Tensor:
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2,
            dtype=relative_coords.dtype,
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        relative_position_index[0:, 0] = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 1
        relative_position_index[0, 0] = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 2
        return relative_position_index

    def forward(self, x: torch.Tensor, rel_pos_bias: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.relative_position_bias_table is not None and self.relative_position_index is not None:
            bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1,
                -1,
            )
            attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        window_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x: torch.Tensor, rel_pos_bias: torch.Tensor | None = None) -> torch.Tensor:
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        x = self.proj(x)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return x, (height, width)


class RelativePositionBias(nn.Module):
    def __init__(self, window_size: tuple[int, int], num_heads: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, num_heads))
        self.register_buffer('relative_position_index', Attention._build_relative_position_index(window_size))

    def forward(self) -> torch.Tensor:
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1,
            -1,
        )
        return bias.permute(2, 0, 1).contiguous()


class OfficialHVITBackbone4Seg(nn.Module):
    """Dependency-light implementation of openmedlab/USFM's HVITBackbone4Seg."""

    def __init__(
        self,
        img_size: int = 448,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        init_values: float | None = 0.1,
        use_abs_pos_emb: bool = False,
        use_rel_pos_bias: bool = True,
        use_shared_rel_pos_bias: bool = False,
        out_indices: Sequence[int] = (3, 5, 7, 11),
        apply_fpn: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.apply_fpn = apply_fpn
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.out_indices = tuple(int(i) for i in out_indices)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(self.patch_embed.patch_shape, num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                )
                for i in range(depth)
            ]
        )

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.fpn2 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif patch_size == 8:
            self.fpn1 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            self.fpn2 = nn.Identity()
            self.fpn3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
            self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=4, stride=4))
        else:
            raise ValueError('Official USFM adapter supports patch_size 8 or 16.')

        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self) -> None:
        for layer_id, layer in enumerate(self.blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_num_layers(self) -> int:
        return len(self.blocks)

    def no_weight_decay(self) -> set[str]:
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        batch_size = x.shape[0]
        x, (height, width) = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        features: list[torch.Tensor] = []
        for idx, block in enumerate(self.blocks):
            x = block(x, rel_pos_bias=rel_pos_bias)
            if idx in self.out_indices:
                feat = x[:, 1:, :].permute(0, 2, 1).reshape(batch_size, -1, height, width)
                features.append(feat.contiguous())

        if len(features) != len(self.out_indices):
            raise RuntimeError(f'Official USFM produced {len(features)} features, expected {len(self.out_indices)}.')
        if self.apply_fpn:
            features = [op(feat) for op, feat in zip([self.fpn1, self.fpn2, self.fpn3, self.fpn4], features)]
        return features

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.forward_features(x)


def _candidate_state_dicts(checkpoint: object) -> Iterable[dict[str, torch.Tensor]]:
    if isinstance(checkpoint, dict):
        for key in ('model', 'state_dict', 'module', 'teacher', 'student'):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                yield value
        if all(isinstance(k, str) for k in checkpoint.keys()):
            yield checkpoint


def _strip_known_prefixes(key: str) -> str:
    prefixes = (
        'module.backbone.',
        'model.backbone.',
        'backbone.',
        'module.encoder.',
        'model.encoder.',
        'encoder.',
        'module.',
        'model.',
    )
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _interpolate_relative_position_bias(value: torch.Tensor, target: torch.Tensor, patch_shape: tuple[int, int]) -> torch.Tensor:
    if value.ndim != 2 or target.ndim != 2 or value.shape[1] != target.shape[1]:
        return value
    dst_num_pos, num_heads = target.shape
    dst_extra = dst_num_pos - (2 * patch_shape[0] - 1) * (2 * patch_shape[1] - 1)
    if dst_extra < 0 or value.shape[0] <= dst_extra:
        return value
    src_num_pos = value.shape[0] - dst_extra
    src_size = int(src_num_pos**0.5)
    dst_size = int((dst_num_pos - dst_extra) ** 0.5)
    if src_size * src_size != src_num_pos or dst_size * dst_size != dst_num_pos - dst_extra:
        return value
    if src_size == dst_size:
        return value

    rel_tokens = value[:-dst_extra].transpose(0, 1).reshape(1, num_heads, src_size, src_size)
    rel_tokens = F.interpolate(rel_tokens, size=(dst_size, dst_size), mode='bicubic', align_corners=False)
    rel_tokens = rel_tokens.reshape(num_heads, -1).transpose(0, 1)
    extra_tokens = value[-dst_extra:] if dst_extra else value.new_empty((0, num_heads))
    return torch.cat((rel_tokens, extra_tokens), dim=0)


def _remap_pretrained_keys(model: OfficialHVITBackbone4Seg, checkpoint_model: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {_strip_known_prefixes(key): value for key, value in checkpoint_model.items()}
    shared_key = 'rel_pos_bias.relative_position_bias_table'
    if model.use_rel_pos_bias and shared_key in remapped:
        shared = remapped.pop(shared_key)
        for idx in range(model.get_num_layers()):
            remapped[f'blocks.{idx}.attn.relative_position_bias_table'] = shared.clone()

    for key in list(remapped.keys()):
        if 'relative_position_index' in key:
            remapped.pop(key)

    model_state = model.state_dict()
    for key, value in list(remapped.items()):
        if key in model_state and 'relative_position_bias_table' in key and value.shape != model_state[key].shape:
            remapped[key] = _interpolate_relative_position_bias(value, model_state[key], model.patch_embed.patch_shape)
    return remapped


class OfficialUSFMBackbone(nn.Module):
    """Official openmedlab/USFM backbone adapter for the local DPT-like decoder."""

    def __init__(
        self,
        img_size: int = 448,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.1,
        init_values: float | None = 0.1,
        out_indices: Sequence[int] = (3, 5, 7, 11),
        pretrained_path: str = '',
        require_pretrained: bool = True,
        freeze: bool = True,
        freeze_loaded_only: bool = False,
        apply_fpn: bool = False,
        use_abs_pos_emb: bool = False,
        use_rel_pos_bias: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = OfficialHVITBackbone4Seg(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            use_abs_pos_emb=use_abs_pos_emb,
            use_rel_pos_bias=use_rel_pos_bias,
            out_indices=out_indices,
            apply_fpn=apply_fpn,
        )
        self.pretrained_load_info: PretrainedLoadInfo | None = None
        self.freeze = bool(freeze)
        self.freeze_loaded_only = bool(freeze_loaded_only)

        if pretrained_path:
            self.pretrained_load_info = self.load_pretrained(pretrained_path)
        elif require_pretrained:
            raise FileNotFoundError('model.usfm.pretrained_path is required for official USFM.')

        self.apply_freeze()

    def load_pretrained(self, pretrained_path: str) -> PretrainedLoadInfo:
        path = Path(pretrained_path)
        if not path.is_file():
            raise FileNotFoundError(
                f'Official USFM pretrained weight not found: {path}. '
                'Download USFM_latest.pth from openmedlab/USFM and place it at this path.'
            )

        checkpoint = torch.load(path, map_location='cpu')
        model_state = self.backbone.state_dict()
        best_state: dict[str, torch.Tensor] | None = None
        best_matched = -1
        best_shape_mismatch: list[str] = []
        for candidate in _candidate_state_dicts(checkpoint):
            remapped = _remap_pretrained_keys(self.backbone, candidate)
            matched = 0
            shape_mismatch = []
            for key, value in remapped.items():
                if key in model_state:
                    if model_state[key].shape == value.shape:
                        matched += 1
                    else:
                        shape_mismatch.append(key)
            if matched > best_matched:
                best_state = remapped
                best_matched = matched
                best_shape_mismatch = shape_mismatch

        if best_state is None:
            raise RuntimeError(f'No usable state_dict found in {path}.')

        loadable = {
            key: value
            for key, value in best_state.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        missing_keys = [key for key in model_state if key not in loadable]
        unexpected_keys = [key for key in best_state if key not in model_state]
        self.backbone.load_state_dict(loadable, strict=False)

        parameter_names = set(dict(self.backbone.named_parameters()).keys())
        loaded_parameter_names = {key for key in loadable if key in parameter_names}
        return PretrainedLoadInfo(
            path=str(path),
            matched_keys=len(loadable),
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            shape_mismatch_keys=best_shape_mismatch,
            loaded_parameter_names=loaded_parameter_names,
        )

    def apply_freeze(self) -> None:
        if not self.freeze:
            return
        if self.freeze_loaded_only and self.pretrained_load_info is not None:
            frozen_names = self.pretrained_load_info.loaded_parameter_names
            for name, param in self.backbone.named_parameters():
                if name in frozen_names:
                    param.requires_grad = False
            return
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone(x)


class USFMFeatureBackbone(nn.Module):
    def __init__(self, weight_path: str, freeze: bool = True) -> None:
        super().__init__()
        self.backbone = OfficialUSFMBackbone(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.1,
            init_values=0.1,
            out_indices=(3, 5, 7, 11),
            pretrained_path=weight_path,
            require_pretrained=True,
            freeze=freeze,
            freeze_loaded_only=False,
            apply_fpn=False,
            use_abs_pos_emb=False,
            use_rel_pos_bias=True,
        )
        self.out_dim = 768
        self.freeze = bool(freeze)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return feats[-1] if isinstance(feats, (list, tuple)) else feats
