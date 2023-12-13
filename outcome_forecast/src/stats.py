"""
Statistics for game outcome prediction
"""
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from konductor.data import get_dataset_properties
from konductor.init import ExperimentInitConfig
from konductor.metadata.base_statistic import Statistic, STATISTICS_REGISTRY


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
        if "timepoints" in data_cfg:
            timepoints = data_cfg["timepoints"].arange()
        else:
            timepoints = None
        return cls(timepoints=timepoints, **extras)

    def __init__(
        self, timepoints: Sequence[int] | None = None, auc_thresholds: int = 100
    ) -> None:
        """Initialize the win prediciton statistics calculator

        Args:
            timepoints (Sequence[int] | None, optional): Sequence of timepoints
            in minutes where to sample the accuracy of the model. If None given
            defaults to every 2min up to 30min.
        """
        self.timepoints = torch.arange(2, 32, 2) if timepoints is None else timepoints

    def get_keys(self) -> list[str]:
        return [f"binary_acc_{t}" for t in self.timepoints]

    def __call__(
        self, predictions: Tensor, targets: dict[str, Tensor]
    ) -> dict[str, float]:
        """Calculate Binary accuracy of the win prediction closest to the
           previously specified timepoints

        Args:
            predictions (dict[str, Tensor]): Outputs of the model
            targets (dict[str, Tensor]): Loaded data, should contain win/loss

        Returns:
            dict[str, float]: Dictionary of Binary Accuracy at each timepoint
            (until end of replay)
        """
        result: dict[str, float] = {}
        pred_sig = torch.sigmoid(predictions)
        for idx, key in enumerate(self.get_keys()):
            res = self.calculate_binary_accuracy(
                pred_sig[:, idx], targets["win"], targets["valid"][:, idx]
            )
            result[key] = res
        return result

    @staticmethod
    def calculate_binary_accuracy(
        predictions: Tensor, targets: Tensor, valid_mask: Tensor
    ) -> float:
        """Calculate binary accuracy considering the valid mask

        Args:
            predictions (Tensor): Predicted values [0, 1]
            targets (Tensor): Actual target values (0 or 1)
            valid_mask (Tensor): Mask indicating whether the data is valid

        Returns:
            float: Binary accuracy
        """
        valid_predictions = predictions[valid_mask] > 0.5
        valid_targets = targets[valid_mask].bool()

        correct_predictions = (valid_predictions == valid_targets).sum().item()
        total_valid_samples = valid_mask.sum().item()

        if total_valid_samples > 0:
            accuracy = correct_predictions / total_valid_samples
        else:
            accuracy = 0.0  # Handle the case when there are no valid samples

        return accuracy


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
        return cls(timepoints=timepoints, **extras)

    def get_keys(self) -> list[str]:
        return [f"auc_{t}" for t in self.timepoints]

    def __init__(
        self, timepoints: Sequence[int] | None = None, auc_thresholds: int = 100
    ) -> None:
        """Initialize the win prediciton statistics calculator

        Args:
            timepoints (Sequence[int] | None, optional): Sequence of timepoints
            in minutes where to sample the accuracy of the model. If None given
            defaults to every 2min up to 30min.
        """
        self.timepoints = torch.arange(2, 32, 2) if timepoints is None else timepoints
        self.auc_thresholds = self.make_thresholds(auc_thresholds)

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
        pred_sig = torch.sigmoid(predictions)
        for idx, key in enumerate(self.get_keys()):
            res = self.calculate_auc(
                pred_sig[:, idx], targets["win"], targets["valid"][:, idx]
            )
            if res is not None:
                result[key] = res
        return result
