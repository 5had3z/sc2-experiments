import enum
from dataclasses import dataclass
from typing import Any

from konductor.data import get_dataset_properties
from konductor.init import ModuleInitConfig
from konductor.models import ExperimentInitConfig
from konductor.models._pytorch import TorchModelConfig


class MinimapTarget(enum.Enum):
    SELF = enum.auto()
    ENEMY = enum.auto()
    BOTH = enum.auto()

    @staticmethod
    def names(target: "MinimapTarget"):
        """Index of target(s) in minimap feature layer stack"""
        match target:
            case MinimapTarget.SELF:
                return ["self"]
            case MinimapTarget.ENEMY:
                return ["enemy"]
            case MinimapTarget.BOTH:
                return ["self", "enemy"]


@dataclass
class BaseConfig(TorchModelConfig):
    encoder: ModuleInitConfig
    temporal: ModuleInitConfig
    decoder: ModuleInitConfig
    input_layer_names: list[str]
    history_len: int = 8
    target: MinimapTarget = MinimapTarget.BOTH

    @property
    def future_len(self) -> int:
        return 9 - self.history_len

    @property
    def is_logit_output(self):
        return True

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0) -> Any:
        props = get_dataset_properties(config)
        model_cfg = config.model[idx].args
        model_cfg["encoder"]["args"]["in_ch"] = props["image_ch"]
        model_cfg["input_layer_names"] = props["minimap_ch_names"]
        return super().from_config(config, idx)

    def __post_init__(self):
        if isinstance(self.encoder, dict):
            self.encoder = ModuleInitConfig(**self.encoder)
        if isinstance(self.temporal, dict):
            self.temporal = ModuleInitConfig(**self.temporal)
        if isinstance(self.decoder, dict):
            self.decoder = ModuleInitConfig(**self.decoder)
        if isinstance(self.target, str):
            self.target = MinimapTarget[self.target.upper()]
