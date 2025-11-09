from dataclasses import dataclass, field
from typing import Any

import torch
from konductor.data import get_dataset_properties
from konductor.init import ModuleInitConfig
from konductor.models import MODEL_REGISTRY, ExperimentInitConfig
from konductor.models._pytorch import TorchModelConfig
from torch import Tensor, nn

from .transformer import TransformerDecoderV1


@dataclass
class BaseConfig(TorchModelConfig):
    image_enc: ModuleInitConfig | None = None
    scalar_enc: ModuleInitConfig | None = None
    decoder: ModuleInitConfig = field(kw_only=True)
    dropout: float = 0.0

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0) -> Any:
        props = get_dataset_properties(config)
        model_cfg = config.model[idx].args

        if image_enc := model_cfg.get("image_enc", None):
            if "in_ch" not in image_enc["args"]:
                image_enc["args"]["in_ch"] = props["image_ch"]

        if scalar_enc := model_cfg.get("scalar_enc", None):
            if "in_ch" not in scalar_enc["args"]:
                scalar_enc["args"]["in_ch"] = props["scalar_ch"]
            if scalar_enc["type"] == "scalar-v2":
                scalar_enc["args"]["timerange"] = props["timepoints"]

        if model_cfg["decoder"]["type"] == "transformer-v1":
            model_cfg["decoder"]["args"]["max_time"] = props["timepoints"].max

        assert image_enc or scalar_enc, "At least Image or Scalar encoder required"

        return super().from_config(config, idx)

    def __post_init__(self):
        if isinstance(self.image_enc, dict):
            self.image_enc = ModuleInitConfig(**self.image_enc)
        if isinstance(self.scalar_enc, dict):
            self.scalar_enc = ModuleInitConfig(**self.scalar_enc)
        if isinstance(self.decoder, dict):
            self.decoder = ModuleInitConfig(**self.decoder)

    def _get_modules(self):
        """Construct and return image encoder, scalar encoder and decoder modules"""

        def get_mod_ch(conf: ModuleInitConfig | None):
            """Get module and channels if not None"""
            if conf is None:
                return None, 0
            model = MODEL_REGISTRY[conf.type](**conf.args, dropout=self.dropout)
            return model, model.out_ch

        image_enc, image_ch = get_mod_ch(self.image_enc)
        scalar_enc, scalar_ch = get_mod_ch(self.scalar_enc)

        decoder = MODEL_REGISTRY[self.decoder.type](
            in_ch=image_ch + scalar_ch, **self.decoder.args
        )
        return image_enc, scalar_enc, decoder


class SnapshotPredictor(nn.Module):
    """Make basic prediction of game outcome based on single snapshot observation"""

    def __init__(
        self,
        image_enc: nn.Module | None,
        scalar_enc: nn.Module | None,
        decoder: nn.Module,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_enc = image_enc
        self.scalar_enc = scalar_enc
        self.decoder = decoder
        self.dropout = nn.Dropout(dropout)
        self.is_logit_output = self.decoder.is_logit_output

    def forward(self, step_data: dict[str, Tensor]) -> Tensor:
        """Step data features should be [B,T,...]"""
        # Merge batch and time dimension for minimaps
        feats: list[Tensor] = []
        if self.image_enc is not None:
            feats.append(self.image_enc(step_data["minimaps"].flatten(0, 1)))

        if self.scalar_enc is not None:
            # Process scalar features per timestep
            scalar_feats = [
                self.scalar_enc(step_data["scalars"][:, tidx])
                for tidx in range(step_data["scalars"].shape[1])
            ]
            # Make same shape as image feats
            feats.append(torch.stack(scalar_feats, dim=1).flatten(0, 1))

        # Cat feats if more than 1, or remove list dimension
        all_feats = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]

        all_feats = self.dropout(all_feats)

        result = self.decoder(all_feats).reshape(step_data["win"].shape[0], -1)

        return result


@dataclass
@MODEL_REGISTRY.register_module("snapshot-prediction")
class SnapshotConfig(BaseConfig):
    """Basic snapshot model configuration"""

    def get_instance(self, *args, **kwargs) -> Any:
        image_enc, scalar_enc, decoder = self._get_modules()
        return SnapshotPredictor(image_enc, scalar_enc, decoder, self.dropout)


class SequencePredictor(nn.Module):
    """
    Predict the outcome of a game depending based on a small sequence of observations.
    The decoder should handle unraveling a set of stacked features
    """

    def __init__(
        self,
        image_enc: nn.Module | None,
        scalar_enc: nn.Module | None,
        decoder: nn.Module,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_enc = image_enc
        self.scalar_enc = scalar_enc
        self.decoder = decoder
        self.dropout = nn.Dropout(dropout)
        self.is_logit_output = self.decoder.is_logit_output

    def forward(self, step_data: dict[str, Tensor]) -> Tensor:
        """"""
        batch_sz, n_timestep = step_data["scalars"].shape[:2]
        feats: list[Tensor] = []
        if self.image_enc is not None:
            # Merge batch and time dimension for minimaps
            image_feats = self.image_enc(step_data["minimaps"].flatten(0, 1))
            feats.append(image_feats.reshape(batch_sz, n_timestep, -1))

        if self.scalar_enc is not None:
            # Process scalar features per timestep
            scalar_feats = [
                self.scalar_enc(step_data["scalars"][:, tidx])
                for tidx in range(n_timestep)
            ]
            # Make same shape as image feats
            feats.append(torch.stack(scalar_feats, dim=1))

        # Stack image and scalar features, decode, then reshape to [B, T, 1]
        all_feats = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]

        all_feats = self.dropout(all_feats)

        if isinstance(self.decoder, TransformerDecoderV1):
            outputs = self.decoder(all_feats, step_data["scalars"][..., -1])
        else:
            outputs = self.decoder(all_feats)

        return outputs


@dataclass
@MODEL_REGISTRY.register_module("sequence-prediction")
class SequenceConfig(BaseConfig):
    """Basic snapshot model configuration"""

    def get_instance(self, *args, **kwargs) -> Any:
        image_enc, scalar_enc, decoder = self._get_modules()
        return SequencePredictor(image_enc, scalar_enc, decoder, dropout=self.dropout)
