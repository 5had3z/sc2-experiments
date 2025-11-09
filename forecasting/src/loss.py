"""Losses"""

from dataclasses import dataclass

import torch
from konductor.data import get_dataset_properties
from konductor.init import ExperimentInitConfig
from konductor.losses import REGISTRY, LossConfig
from konductor.models import get_model_config
from torch import Tensor, nn
from torch.nn import functional as F

from .minimap.common import BaseConfig as MinimapModelCfg
from .minimap.common import MinimapTarget
from .utils import get_valid_sequence_mask


class WinBCE(nn.Module):
    def __init__(self, pred_is_logit: bool) -> None:
        super().__init__()
        self.loss_fn = (
            F.binary_cross_entropy_with_logits
            if pred_is_logit
            else F.binary_cross_entropy
        )

    def forward(self, pred: Tensor, data: dict[str, Tensor]) -> dict[str, Tensor]:
        win_repeat = data["win"].unsqueeze(-1).repeat(1, pred.shape[1])
        loss = self.loss_fn(pred, win_repeat, reduction="none")
        loss *= data["valid"]
        return {"win-bce": loss.mean()}


@dataclass
@REGISTRY.register_module("win-bce")
class WinBCECfg(LossConfig):
    model_output_logits: bool = True

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int, **kwargs):
        model = get_model_config(config=config).get_instance()
        if model.is_logit_output:  # check Truth and None
            config.criterion[idx].args["model_output_logits"] = model.is_logit_output
        return super().from_config(config, idx, **kwargs)

    def get_instance(self, *args, **kwargs):
        return WinBCE(self.model_output_logits)


class MinimapLoss(nn.Module):
    """
    Base minimap loss that deals with masking and motion weighting, pixel-wise
    loss must be defined and the name of loss assigned in the derived class
    """

    def __init__(
        self,
        history_len: int,
        motion_weight: float | None,
        motion_version: int,
        target_ch: list[int],
        pred_is_logit: bool,
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.motion_weight = motion_weight
        self.target_ch = target_ch
        self.pred_is_logit = pred_is_logit
        if self.motion_weight is not None:
            assert self.motion_weight > 1, f"{motion_weight=}"
        self._get_motion_weight = [
            self._get_motion_weight_v1,
            self._get_motion_weight_v2,
        ][motion_version - 1]

    @property
    def _key(self) -> str:
        """Name of the loss function"""
        raise NotImplementedError()

    def _loss_fn(self, preds: Tensor, target: Tensor) -> Tensor:
        """Returns pixel-wise loss between preds and target"""
        raise NotImplementedError()

    def _get_motion_weight_v1(self, minimaps: Tensor) -> Tensor:
        """
        Calculate pixel-wise motion weight based on a change between two frames
        """
        assert self.motion_weight is not None
        prev = minimaps[:, self.history_len - 1 : -1]
        nxt = minimaps[:, self.history_len :]
        mask = torch.ones_like(nxt)
        mask[prev != nxt] = self.motion_weight
        return mask

    def _get_motion_weight_v2(self, minimaps: Tensor) -> Tensor:
        """
        Calculate pixel-wise motion weight based on if that pixel has been
        occupied for the entire sequence duration
        """
        assert self.motion_weight is not None
        not_static_units = torch.sum(minimaps, dim=1) != minimaps.shape[1]
        mask = torch.ones_like(minimaps[:, 0])
        mask[not_static_units] = self.motion_weight
        return mask.unsqueeze(1)

    def forward(
        self, predictions: Tensor, targets: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        target_minimap = targets["minimaps"][:, :, self.target_ch]
        loss_mask = self._loss_fn(predictions, target_minimap[:, self.history_len :])

        if self.motion_weight is not None:
            loss_mask *= self._get_motion_weight(target_minimap)

        loss_sequence = loss_mask.mean(dim=(-1, -2, -3))

        if "valid" in targets:
            valid_seq = get_valid_sequence_mask(targets["valid"], self.history_len + 1)
            loss_sequence *= valid_seq

        return {self._key: loss_sequence.mean()}


@dataclass
class MinimapCfg(LossConfig):
    history_len: int
    pred_is_logit: bool
    target_ch: list[int]
    motion_weight: float | None = None
    motion_version: int = 1

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int, **kwargs):
        model_cfg: MinimapModelCfg = get_model_config(config)
        minimap_layers: list[str] = get_dataset_properties(config)["minimap_ch_names"]
        config.criterion[idx].args["history_len"] = model_cfg.history_len
        config.criterion[idx].args["target_ch"] = [
            minimap_layers.index(n) for n in MinimapTarget.names(model_cfg.target)
        ]
        config.criterion[idx].args["pred_is_logit"] = model_cfg.is_logit_output
        return super().from_config(config, idx, **kwargs)


class MinimapBCE(MinimapLoss):
    @property
    def _key(self):
        return "minimap-bce"

    def _loss_fn(self, preds: Tensor, target: Tensor) -> Tensor:
        if self.pred_is_logit:
            return F.binary_cross_entropy_with_logits(preds, target, reduction="none")
        return F.binary_cross_entropy(preds, target, reduction="none")


@dataclass
@REGISTRY.register_module("minimap-bce")
class MinimapBCECfg(MinimapCfg):
    def get_instance(self, *args, **kwargs):
        return self.init_auto_filter(MinimapBCE)


def focal_loss(pred: Tensor, target: Tensor, eps: float, gamma: float, alpha: float):
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    prob = pred.sigmoid()
    p_t = prob * target + (1 - prob) * (1 - target)
    loss = bce_loss * ((1 - p_t + eps) ** gamma)

    if alpha > 0:
        loss *= alpha * target + (1 - alpha) * (1 - target)

    return loss


focal_loss_jit: torch.ScriptFunction = torch.jit.script(focal_loss)


class MinimapFocal(MinimapLoss):
    def __init__(
        self,
        history_len: int,
        alpha: float,
        gamma: float,
        motion_weight: float | None,
        target_ch: list[int],
        pred_is_logit: bool,
    ) -> None:
        assert pred_is_logit is False, "Focal loss incompatible with sigmoid output"
        super().__init__(history_len, motion_weight, target_ch, pred_is_logit)
        self.alpha = alpha
        self.gamma = gamma

    @property
    def _key(self):
        return "minimap-focal"

    def _loss_fn(self, pred: Tensor, target: Tensor):
        return focal_loss_jit(
            pred, target, torch.finfo(pred.dtype).eps, self.gamma, self.alpha
        )


@dataclass
@REGISTRY.register_module("minimap-focal")
class MinimapFocalCfg(MinimapCfg):
    alpha: float = 0.75
    gamma: float = 0.2

    def get_instance(self, *args, **kwargs):
        return self.init_auto_filter(MinimapFocal)
