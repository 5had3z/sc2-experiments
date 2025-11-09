"""
Statistics for game outcome prediction
"""

import itertools
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
from torch import Tensor
from konductor.data import get_dataset_properties
from konductor.models import get_model_config
from konductor.init import ExperimentInitConfig
from konductor.metadata.base_statistic import Statistic, STATISTICS_REGISTRY

from .minimap.common import MinimapTarget, BaseConfig as MinimapModelCfg


@dataclass
class Confusion:
    """Batch-Last Confusion Array Shape: (thresh, batchidx)"""

    @classmethod
    def preallocate(cls, batch: int, thresholds: int, device=None):
        data = torch.empty((thresholds, batch), dtype=torch.float32, device=device)
        return cls(data, data.clone(), data.clone(), data.clone())

    tp: Tensor
    fp: Tensor
    tn: Tensor
    fn: Tensor

    @property
    def device(self):
        """Device tensors are currently on"""
        return self.tp.device


def _div_no_nan(a: Tensor, b: Tensor) -> Tensor:
    """Divide and set nan/inf values to zero"""
    c = a / b
    c[~torch.isfinite(c)] = 0
    return c


@STATISTICS_REGISTRY.register_module("binary-acc")
class BinaryAcc(Statistic):
    @classmethod
    def from_config(cls, cfg: ExperimentInitConfig, **extras):
        data_cfg = get_dataset_properties(cfg)
        timepoints = data_cfg["timepoints"].arange()
        model = get_model_config(cfg).get_instance()
        should_sigmoid = model.is_logit_output
        return cls(timepoints=timepoints, should_sigmoid=should_sigmoid, **extras)

    def __init__(
        self,
        timepoints: Sequence[int],
        should_sigmoid: bool = True,
        keep_batch: bool = False,
    ) -> None:
        """Initialize the win prediction statistics calculator

        Args:
            timepoints (Sequence[int] | None, optional): Sequence of timepoints
            in minutes where to sample the accuracy of the model. If None given
            defaults to every 2min up to 30min.
        """
        self.timepoints = timepoints
        self.should_sigmoid = should_sigmoid
        self.keep_batch = keep_batch

    def get_keys(self) -> list[str]:
        return [f"binary_acc_{t}" for t in self.timepoints]

    def __call__(
        self, predictions: Tensor, targets: dict[str, Tensor]
    ) -> dict[str, float | Tensor]:
        """Calculate Binary accuracy of the win prediction closest to the
           previously specified timepoints

        Args:
            predictions (dict[str, Tensor]): Outputs of the model
            targets (dict[str, Tensor]): Loaded data, should contain win/loss

        Returns:
            dict[str, float]: Dictionary of Binary Accuracy at each timepoint
            (until end of replay)
        """
        pred = torch.sigmoid(predictions) if self.should_sigmoid else predictions

        result: dict[str, float | Tensor] = {}
        for idx, key in enumerate(self.get_keys()):
            res = self.calculate_binary_accuracy(
                pred[:, idx], targets["win"], targets["valid"][:, idx], self.keep_batch
            )
            if res is not None:
                result[key] = res
        return result

    @staticmethod
    def _calculate_binary_accuracy_batch(
        predictions: Tensor, targets: Tensor, valid_mask: Tensor
    ) -> Tensor:
        valid_predictions = predictions > 0.5
        valid_targets = targets.bool()
        correct_predictions = valid_predictions == valid_targets

        total_valid_samples = valid_predictions.sum().item()

        if total_valid_samples > 0:
            accuracy = correct_predictions
        else:
            accuracy = torch.zeros(
                correct_predictions.shape,
                dtype=torch.bool,
                device=predictions.device,
            )

        return accuracy

    @staticmethod
    def calculate_binary_accuracy(
        predictions: Tensor, targets: Tensor, valid_mask: Tensor, keep_batch: bool
    ) -> float | Tensor | None:
        """Calculate binary accuracy considering the valid mask

        Args:
            predictions (Tensor): Predicted values [0, 1]
            targets (Tensor): Actual target values (0 or 1)
            valid_mask (Tensor): Mask indicating whether the data is valid

        Returns:
            float: Binary accuracy
        """
        if keep_batch:
            return BinaryAcc._calculate_binary_accuracy_batch(
                predictions, targets, valid_mask
            )

        valid_predictions = predictions[valid_mask] > 0.5
        valid_targets = targets[valid_mask].bool()

        correct_predictions = (valid_predictions == valid_targets).sum().item()
        total_valid_samples = valid_mask.sum().item()

        if total_valid_samples > 0:
            return correct_predictions / total_valid_samples
        else:
            return None  # Handle the case when there are no valid samples


@STATISTICS_REGISTRY.register_module("win-auc")
class WinAUC(Statistic):
    """AUC of win prediction performance at several timepoints (min) in the game"""

    @classmethod
    def from_config(cls, cfg: ExperimentInitConfig, **extras):
        data_cfg = get_dataset_properties(cfg)
        if "timepoints" in data_cfg:
            timepoints = data_cfg["timepoints"].arange()
        else:
            timepoints = None
        model = get_model_config(cfg).get_instance()
        should_sigmoid = model.is_logit_output
        return cls(timepoints=timepoints, should_sigmoid=should_sigmoid, **extras)

    def get_keys(self) -> list[str]:
        return [f"auc_{t}" for t in self.timepoints]

    def __init__(
        self,
        should_sigmoid: bool = True,
        timepoints: Sequence[int] | None = None,
        auc_thresholds: int = 100,
    ) -> None:
        """Initialize the win prediction statistics calculator

        Args:
            timepoints (Sequence[int] | None, optional): Sequence of timepoints
            in minutes where to sample the accuracy of the model. If None given
            defaults to every 2min up to 30min.
        """
        self.timepoints = torch.arange(2, 32, 2) if timepoints is None else timepoints
        self.auc_thresholds = self.make_thresholds(auc_thresholds)
        self.should_sigmoid = should_sigmoid

    @staticmethod
    def make_thresholds(count) -> Tensor:
        # ensure 0,0 -> 1,1 with 1 and 0 thresholds
        # thresholds = np.concatenate(
        #     [
        #         np.linspace(1, 0.8, 21),
        #         np.linspace(0.7, 0.3, 5),
        #         np.linspace(0.20, 0, 21),
        #     ]
        # )

        thresh = torch.linspace(0, 1, count, dtype=torch.float32)
        # Go beyond 0,1 to capture float rounding issues
        thresh[0] = -torch.finfo(thresh.dtype).eps
        thresh[-1] = 1 + torch.finfo(thresh.dtype).eps
        return thresh

    def calculate_confusion(
        self, pred: Tensor, target: Tensor, valid: Tensor
    ) -> Confusion:
        """Calculate confusion matrix from prediction and target"""
        target_binary = target.bool()
        conf = Confusion.preallocate(
            pred.shape[0], self.auc_thresholds.shape[0], pred.device
        )

        # Thresholds should ordered 0 -> 1
        for idx, threshold in enumerate(self.auc_thresholds):
            pred_binary: Tensor = pred > threshold
            conf.fn[idx] = (~pred_binary & target_binary & valid).sum()
            conf.tp[idx] = (pred_binary & target_binary & valid).sum()
            conf.fp[idx] = (pred_binary & ~target_binary & valid).sum()
            conf.tn[idx] = (~pred_binary & ~target_binary & valid).sum()

        return conf

    def interpolate_pr_auc(self, confusion: Confusion) -> Tensor:
        """From Keras PR AUC Interpolation"""
        zero_ = torch.tensor(0, device=confusion.device)

        dtp = confusion.tp[:-1] - confusion.tp[1:]
        p = confusion.tp + confusion.fp
        dp = p[:-1] - p[1:]
        prec_slope = _div_no_nan(dtp, torch.maximum(dp, zero_))
        intercept = confusion.tp[1:] - prec_slope * p[1:]

        safe_p_ratio = torch.where(
            torch.logical_and(p[:-1] > 0, p[1:] > 0),
            _div_no_nan(p[:-1], torch.maximum(p[1:], zero_)),
            torch.ones_like(p[1:]),
        )

        pr_auc_increment = _div_no_nan(
            prec_slope * (dtp + intercept * torch.log(safe_p_ratio)),
            torch.maximum(confusion.tp[1:] + confusion.fn[1:], zero_),
        )

        return pr_auc_increment.sum(dim=0).cpu()

    def calculate_auc(self, pred: Tensor, target: Tensor, valid: Tensor):
        if not torch.any(valid):
            return None
        conf = self.calculate_confusion(pred, target, valid)
        auc = self.interpolate_pr_auc(conf)
        return auc.mean().item()

    def __call__(
        self, predictions: Tensor, targets: dict[str, Tensor]
    ) -> dict[str, float]:
        """Calculate AUC of the win prediction closest to the previously
           specified timepoints

        Args:
            predictions (dict[str, Tensor]): Outputs of the model
            targets (dict[str, Tensor]): Loaded data, should contain win/loss

        Returns:
            dict[str, float]: Dictionary of AUC at each timepoint (until end of replay)
        """
        result: dict[str, float] = {}
        pred_sig = torch.sigmoid(predictions) if self.should_sigmoid else predictions
        for idx, key in enumerate(self.get_keys()):
            res = self.calculate_auc(
                pred_sig[:, idx], targets["win"], targets["valid"][:, idx]
            )
            if res is not None:
                result[key] = res
        return result


@STATISTICS_REGISTRY.register_module("minimap-soft-iou")
class MinimapSoftIoU(Statistic):
    @classmethod
    def from_config(cls, cfg: ExperimentInitConfig, **extras):
        model_cfg: MinimapModelCfg = get_model_config(cfg)
        data_props = get_dataset_properties(cfg)

        if model_cfg.future_len > 1:
            timepoints = [
                float(i * data_props["step_sec"])
                for i in range(1, model_cfg.future_len + 1)
            ]
        else:
            timepoints = None

        target_ch = [
            data_props["minimap_ch_names"].index(n)
            for n in MinimapTarget.names(model_cfg.target)
        ]

        return cls(
            model_cfg.history_len,
            model_cfg.target,
            target_ch,
            timepoints,
            model_cfg.is_logit_output,
        )

    def get_keys(self) -> list[str]:
        keys = ["motion_soft_iou", "soft_iou", "diff_soft_iou"]

        names = MinimapTarget.names(self.target)
        keys = [f"{p}_{n}" for p, n in itertools.product(keys, names)]

        if self.timepoints is not None:
            keys = [f"{p}_{n}" for p, n in itertools.product(keys, self.timepoints)]

        return keys

    def __init__(
        self,
        sequence_len: int,
        target: MinimapTarget,
        target_ch: list[int],
        timepoints: list[float] | None = None,
        should_sigmoid: bool = True,
        keep_batch: bool = False,
    ) -> None:
        super().__init__()
        self.target = target
        self.target_ch = target_ch
        self.sequence_len = sequence_len
        self.should_sigmoid = should_sigmoid
        self.timepoints = timepoints
        self.keep_batch = keep_batch

    @staticmethod
    def calculate_soft_iou(
        pred: Tensor, target: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        """Calculates iou for binary mask over hw axis"""
        if mask is None:
            mask = torch.ones_like(pred)
        soft_intersection = (pred * target * mask).sum(dim=(-1, -2))
        soft_union = ((pred + target - pred * target) * mask).sum(dim=(-1, -2))
        soft_iou = soft_intersection / soft_union
        return soft_iou

    def __call__(
        self, predictions: Tensor, targets: dict[str, Tensor]
    ) -> Dict[str, float | Tensor]:
        target_minimaps = targets["minimaps"][:, :, self.target_ch]
        next_minimap = target_minimaps[:, self.sequence_len :]
        static_unit_mask = torch.sum(target_minimaps, dim=1) != target_minimaps.shape[1]
        diff_frame_mask = (
            target_minimaps[:, self.sequence_len - 1, None] != next_minimap
        )

        if self.should_sigmoid:
            predictions = torch.sigmoid(predictions)

        results: dict[str, float | Tensor] = {}
        for t_idx in range(next_minimap.shape[1]):
            soft_iou = MinimapSoftIoU.calculate_soft_iou(
                predictions[:, t_idx], next_minimap[:, t_idx]
            )

            motion_soft_iou = MinimapSoftIoU.calculate_soft_iou(
                predictions[:, t_idx],
                next_minimap[:, t_idx],
                static_unit_mask,
            )

            diff_soft_iou = MinimapSoftIoU.calculate_soft_iou(
                predictions[:, t_idx],
                next_minimap[:, t_idx],
                diff_frame_mask[:, t_idx],
            )

            postfix = "" if self.timepoints is None else f"_{self.timepoints[t_idx]}"
            for ch_idx, name in enumerate(MinimapTarget.names(self.target)):
                results[f"soft_iou_{name}{postfix}"] = soft_iou[:, ch_idx]
                results[f"motion_soft_iou_{name}{postfix}"] = motion_soft_iou[:, ch_idx]
                results[f"diff_soft_iou_{name}{postfix}"] = diff_soft_iou[:, ch_idx]

        # TODO Mask out accuracy contrib of parts with invalid sequences
        # valid_mask = get_valid_sequence_mask(targets["valid"], self.sequence_len)

        if not self.keep_batch:
            results = {k: v.nanmean().item() for k, v in results.items()}

        return results
