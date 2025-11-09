"""Common helper functions for evaluation"""

from pathlib import Path

import torch
import pandas as pd
from torch import nn, Tensor
from konductor.data import Split
from konductor.config import ExperimentEvalConfig
from konductor.utilities.pbar import LivePbar, IntervalPbar


def get_pbar(total: int, desc: str = "", live: bool = True):
    """Get live or interval (fraction=0.1) progress bar depending on live flag"""
    return (
        LivePbar(total, desc) if live else IntervalPbar(total, fraction=0.1, desc=desc)
    )


def metadata_to_str(metadata: Tensor) -> list[str]:
    return ["".join(chr(x) for x in sublist) for sublist in metadata.cpu()]


def load_model_checkpoint(
    exp_config: ExperimentEvalConfig, filename: str = "latest.pt"
):
    """Load model from checkpoint in experiment directory"""
    model: nn.Module = exp_config.model[0].get_instance()
    ckpt = torch.load(exp_config.exp_path / filename)["model"]
    model.load_state_dict(ckpt)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model


def add_metadata_to_dataset_config(exp_config: ExperimentEvalConfig):
    """Get dataloader that also returns metadata (unique id associated with sample)"""
    dataset_cfg = exp_config.dataset[0]
    if hasattr(dataset_cfg, "keys"):
        if "metadata" not in dataset_cfg.keys:
            dataset_cfg.keys.append("metadata")
    else:
        dataset_cfg.metadata = True  # Need to add metadata list of keys to yield


def setup_eval_model_and_dataloader(
    run_path: Path, workers: int, batch_size: int | None = None, **loader_kwargs
):
    """Read experiment config from run path and create model and dataloader"""
    exp_config = ExperimentEvalConfig.from_run(run_path)

    # AMP isn't enabled during eval
    if "amp" in exp_config.init.trainer:
        del exp_config.init.trainer["amp"]

    model = load_model_checkpoint(exp_config)
    add_metadata_to_dataset_config(exp_config)
    exp_config.set_workers_and_prefetch(workers, **loader_kwargs)
    if batch_size is not None:
        exp_config.set_batch_size(batch_size, Split.VAL)
    dataloader = exp_config.get_dataloader(Split.VAL)

    return exp_config, model, dataloader


def write_outcome_prediction(
    data: dict[str, Tensor], preds: Tensor, gidx: int, df: pd.DataFrame
):
    """Write the predicted outcome of the game over its duration"""
    replay_names = metadata_to_str(data["metadata"])
    for bidx in range(preds.shape[0]):
        df_idx = gidx + bidx
        if df_idx >= df.size:
            break

        row = df.iloc[gidx + bidx]
        row["replay"] = replay_names[bidx][:-1]
        row["playerId"] = int(replay_names[bidx][-1])
        row["outcome"] = bool(data["win"][bidx].item())
        for tidx in range(preds.shape[1]):
            if not data["valid"][bidx, tidx].item():
                continue
            col = df.columns[tidx + 3]
            row[col] = preds[bidx, tidx].sigmoid().item()
