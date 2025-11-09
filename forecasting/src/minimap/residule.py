from dataclasses import dataclass, field
from typing import Any

import torch
from konductor.data import get_dataset_properties
from konductor.init import ExperimentInitConfig
from konductor.models import MODEL_REGISTRY
from konductor.models._pytorch import TorchModelConfig
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock

from .common import MinimapTarget


@dataclass
class BaseConfig(TorchModelConfig):
    in_layers: MinimapTarget
    input_layer_names: list[str]
    history_len: int = 8
    target: MinimapTarget = MinimapTarget.BOTH

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0) -> Any:
        config.model[0].args["input_layer_names"] = get_dataset_properties(config)[
            "minimap_ch_names"
        ]
        return super().from_config(config, idx)

    def __post_init__(self):
        if isinstance(self.target, str):
            self.target = MinimapTarget[self.target.upper()]
        if isinstance(self.in_layers, str):
            self.in_layers = MinimapTarget[self.in_layers.upper()]

    @property
    def future_len(self) -> int:
        return 9 - self.history_len

    @property
    def is_logit_output(self):
        return False


def _get_layer_indices(layers: list[str], target: MinimapTarget):
    return [layers.index(n) for n in MinimapTarget.names(target)]


class ResiduleConvV1(nn.Module):
    """
    Ultra barebones temporal model that channel-wise stacks self/enemy
    and predicts the residual from the previous frame and always stays at
    high resolution to ensure fidelity.
    """

    @property
    def future_len(self):
        return 9 - self.history_len

    @property
    def is_logit_output(self):
        return False

    def __init__(
        self,
        history_len: int,
        hidden_chs: list[int],
        kernel_sizes: list[int],
        in_layers: MinimapTarget,
        target: MinimapTarget,
        input_layer_names: list[str],
    ) -> None:
        super().__init__()
        assert len(hidden_chs) == len(kernel_sizes)
        self.in_layers = [
            input_layer_names.index(n) for n in MinimapTarget.names(in_layers)
        ]
        self.out_layers = [
            input_layer_names.index(n) for n in MinimapTarget.names(target)
        ]
        self.history_len = history_len

        modules = []
        in_ch = history_len * len(self.in_layers)
        for out_ch, kernel in zip(hidden_chs, kernel_sizes):
            modules.append(self._make_layer(in_ch, out_ch, kernel))
            in_ch = out_ch
        modules.append(nn.Conv2d(in_ch, len(self.out_layers), 1))
        self.model = nn.Sequential(*modules)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, kernel_size: int):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, inputs: dict[str, Tensor]):
        """"""
        minimaps = inputs["minimaps"][:, : self.history_len, self.in_layers]
        minimap_stack = minimaps.reshape(
            -1, self.history_len * len(self.in_layers), *minimaps.shape[-2:]
        )
        residule = self.model(minimap_stack)
        residule = F.tanh(residule)

        last_minimap = inputs["minimaps"][:, self.history_len - 1, self.out_layers]
        # Ensure prediction between 0 and 1 and unsqueeze time dimension
        prediction = torch.clamp(last_minimap + residule, 0, 1).unsqueeze(1)
        return prediction


@dataclass
@MODEL_REGISTRY.register_module("residule-conv-v1")
class ResiduleConvV1Cfg(BaseConfig):
    hidden_chs: list[int] = field(kw_only=True)
    kernel_sizes: list[int] = field(kw_only=True)

    def get_instance(self, *args, **kwargs) -> Any:
        return self.init_auto_filter(ResiduleConvV1)


# Backcompat
MODEL_REGISTRY.register_module("conv-forecast-v3", ResiduleConvV1Cfg)


class ResiduleConvV2(nn.Module):
    @property
    def future_len(self):
        return 9 - self.history_len

    @property
    def is_logit_output(self):
        return False

    def __init__(
        self,
        history_len: int,
        hidden_chs: list[int],
        num_blocks: list[int],
        in_layers: MinimapTarget,
        target: MinimapTarget,
        input_layer_names: list[str],
    ) -> None:
        super().__init__()
        assert len(hidden_chs) - 1 == len(num_blocks)
        self.in_layers = _get_layer_indices(input_layer_names, in_layers)
        self.out_layers = _get_layer_indices(input_layer_names, target)
        self.history_len = history_len

        in_ch = history_len * len(self.in_layers)
        modules: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_ch, hidden_chs[0], 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(hidden_chs[0]),
                nn.ReLU(),
                nn.MaxPool2d(3, 2, 1),
            )
        ]
        for in_ch, out_ch, num_block in zip(hidden_chs, hidden_chs[1:], num_blocks):
            modules.append(self._make_block(in_ch, out_ch, num_block))

        # Upsample to original resolution and apply final layer as depthwise-separable
        modules.append(
            nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv2d(out_ch, out_ch, 5, padding=2, groups=out_ch),
                nn.Conv2d(out_ch, self.future_len * len(self.out_layers), 1),
            )
        )
        self.blocks = nn.ModuleList(modules)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)  # Make identity

    @staticmethod
    def _make_block(in_ch: int, out_ch: int, num_block: int):
        downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False), nn.BatchNorm2d(out_ch)
        )
        block = [BasicBlock(in_ch, out_ch, downsample=downsample)]
        for _ in range(1, num_block):
            block.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*block)

    def forward(self, inputs: dict[str, Tensor]):
        minimaps = inputs["minimaps"][:, : self.history_len, self.in_layers]
        size = minimaps.shape[-2:]
        residule: Tensor = minimaps.reshape(
            -1, self.history_len * len(self.in_layers), *size
        )
        for block in self.blocks:
            residule = block(residule)
        residule = F.tanh(residule)

        last_minimap = inputs["minimaps"][
            :, self.history_len - 1, None, self.out_layers
        ]
        residule = residule.reshape(-1, self.future_len, len(self.out_layers), *size)
        # Ensure prediction between 0 and 1
        prediction = torch.clamp(last_minimap + residule, 0, 1)
        return prediction


@dataclass
@MODEL_REGISTRY.register_module("residule-conv-v2")
class ResiduleConvV2Cfg(BaseConfig):
    hidden_chs: list[int] = field(kw_only=True)
    num_blocks: list[int] = field(kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        assert len(self.hidden_chs) - 1 == len(self.num_blocks)

    def get_instance(self, *args, **kwargs) -> Any:
        return self.init_auto_filter(ResiduleConvV2)


def _make_conv_block(in_channels: int, out_channels: int):
    """Double pump conv->norm->relu"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 2, 1),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(),
    )


class UpsampleBlock(nn.Module):
    def __init__(self, up_channels: int, skip_channels: int) -> None:
        super().__init__()
        self.transpose_conv = nn.ConvTranspose2d(up_channels, skip_channels, 2, 2)
        self.module = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, 3, 1, 1),
            nn.InstanceNorm2d(skip_channels),
            nn.LeakyReLU(),
            nn.Conv2d(skip_channels, skip_channels, 3, 1, 1),
            nn.InstanceNorm2d(skip_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x: Tensor, x_skip: Tensor) -> Tensor:
        x = self.transpose_conv(x)
        x = self.module(x + x_skip)
        return x


class ResiduleUnet(nn.Module):
    """Based on nnUnet https://arxiv.org/pdf/2110.03352v2.pdf"""

    @property
    def future_len(self):
        return 9 - self.history_len

    @property
    def is_logit_output(self):
        return self.disable_residule

    def __init__(
        self,
        history_len: int,
        in_layers: MinimapTarget,
        target: MinimapTarget,
        hidden_chs: list[int],
        deep_supervision: bool,
        include_heightmap: bool,
        input_layer_names: list[str],
        disable_residule: bool,
    ) -> None:
        super().__init__()
        self.in_layers = _get_layer_indices(input_layer_names, in_layers)
        self.out_layers = _get_layer_indices(input_layer_names, target)
        self.history_len = history_len
        self.deep_supervision = deep_supervision
        self.disable_residule = disable_residule
        in_ch = history_len * len(self.in_layers)

        if include_heightmap:
            self.height_map_idx = input_layer_names.index("heightMap")
            in_ch += 1
        else:
            self.height_map_idx = None

        self.input_module = nn.Sequential(
            nn.Conv2d(in_ch, hidden_chs[0], 3, padding=1, bias=False),
            nn.Conv2d(hidden_chs[0], hidden_chs[0], 3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_chs[0]),
        )
        self.downsamples = nn.ModuleList(
            _make_conv_block(*args) for args in zip(hidden_chs[:-2], hidden_chs[1:])
        )
        self.bottleneck = _make_conv_block(hidden_chs[-2], hidden_chs[-1])
        self.upsamples = nn.ModuleList(
            UpsampleBlock(*args)
            for args in zip(
                reversed(hidden_chs[1:]),
                reversed(hidden_chs[:-1]),
            )
        )
        out_ch = self.future_len * len(self.out_layers)
        out_mods = [nn.Conv2d(hidden_chs[0], out_ch, 1)]
        if deep_supervision:
            out_mods.append(nn.Conv2d(hidden_chs[1], out_ch, 1))
            out_mods.append(nn.Conv2d(hidden_chs[2], out_ch, 1))
        self.linear_proj = nn.ModuleList(out_mods)

    def forward_aux(self, inputs: Tensor) -> list[Tensor]:
        """"""
        out = self.input_module(inputs)
        encoder_outputs = [out]
        for downsample in self.downsamples:
            out = downsample(out)
            encoder_outputs.append(out)
        out = self.bottleneck(out)

        decoder_outputs = []
        for upsample, skip in zip(self.upsamples, reversed(encoder_outputs)):
            out = upsample(out, skip)
            decoder_outputs.append(out)

        out: list[Tensor] = [self.linear_proj[0](out)]
        if self.training and self.deep_supervision:
            for i, decoder_out in enumerate(decoder_outputs[-3:-1][::-1], 1):
                out.append(self.linear_proj[i](decoder_out))

        # Transform to BTCHW
        out = [
            o.reshape(-1, self.future_len, len(self.out_layers), *o.shape[-2:])
            for o in out
        ]

        if not self.disable_residule:
            out = [F.tanh(o) for o in out]

        return out

    def forward(self, inputs: dict[str, Tensor]):
        minimaps = inputs["minimaps"][:, : self.history_len, self.in_layers]
        # Flatten to B[TC]HW
        minimaps = minimaps.reshape(
            -1, self.history_len * len(self.in_layers), *minimaps.shape[-2:]
        )
        if self.height_map_idx is not None:
            heightmap = inputs["minimaps"][:, 0, [self.height_map_idx]]
            heightmap = (heightmap - 127) / 128  # Normalize 0-255 -> ~[-1,1]
            minimaps = torch.cat([minimaps, heightmap], dim=1)
        residules = self.forward_aux(minimaps)

        if self.disable_residule:
            return residules[0] if len(residules) == 1 else residules

        last_minimap = inputs["minimaps"][:, self.history_len - 1, self.out_layers]

        preds: list[Tensor] = []
        for residule in residules:
            last_resize = F.interpolate(
                last_minimap, size=residule.shape[-2:]
            ).unsqueeze(1)
            # Ensure prediction between 0 and 1
            preds.append(torch.clamp(last_resize + residule, 0, 1))

        return preds[0] if len(preds) == 1 else preds


@dataclass
@MODEL_REGISTRY.register_module("residule-unet")
class ResiduleUnetCfg(BaseConfig):
    hidden_chs: list[int] = field(kw_only=True)
    deep_supervision: bool = False
    include_heightmap: bool = False
    disable_residule: bool = False

    @property
    def is_logit_output(self):
        return self.disable_residule

    def get_instance(self, *args, **kwargs) -> Any:
        return self.init_auto_filter(ResiduleUnet)
