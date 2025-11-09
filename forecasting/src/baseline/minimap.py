import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
import yaml
from konductor.data import DatasetInitConfig, Split, make_from_init_config
from konductor.data.dali import DaliLoaderConfig
from konductor.metadata.database import Database, Metadata, get_sqlite_uri
from konductor.metadata.database.interface import DEFAULT_SQLITE_FILENAME
from konductor.utilities.metadata import update_metadata_entry
from konductor.utilities.pbar import IntervalPbar
from torch import Tensor

from ..data.base_dataset import SC2DatasetCfg
from ..sqlite_utils import write_entry
from ..stats import MinimapSoftIoU, MinimapTarget

app = typer.Typer()

EVAL_BATCH_SIZE = 64


def get_dataset(config_path: Path):
    """
    From configuration file create dataloader.
    Configuration file can be of a full experiment format, or just the data configuration.
    ```
    dataset:
      - type: foo
        args: {}
    model:
       ...
    ```
    *OR*
    ```
    type: foo
    args: {}
    ```
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "dataset" in config:
        config = config["dataset"][0]

    init_cfg = DatasetInitConfig.from_dict(config)
    dataset_cfg = make_from_init_config(init_cfg)

    return dataset_cfg


def evaluate_trivial_prediction(dataset: SC2DatasetCfg) -> dict[str, float]:
    """Trivially predict the next frame with the previous frame"""
    sequence_len = 9
    history_len = 6
    future_len = sequence_len - history_len
    assert dataset.minimap_ch_names is not None
    target_ch = [
        dataset.minimap_ch_names.index(n)
        for n in MinimapTarget.names(MinimapTarget.BOTH)
    ]
    evaluator = MinimapSoftIoU(
        history_len,
        MinimapTarget.BOTH,
        target_ch,
        timepoints=list(range(3, 10, 3)),
        should_sigmoid=False,
    )
    assert isinstance(dataset.val_loader, DaliLoaderConfig)
    dataset.val_loader.batch_size = EVAL_BATCH_SIZE
    dataset.val_loader.workers = 8
    dataset.val_loader.py_workers = 6
    dataloader = dataset.get_dataloader(Split.VAL)
    results: dict[str, list[float]] = {k: [] for k in evaluator.get_keys()}
    with IntervalPbar(total=len(dataloader), fraction=0.01) as pbar:
        for sample in dataloader:
            # Unwrap dali pipe list
            if isinstance(sample, list):
                sample = sample[0]

            # Prediction is last minimap of history length
            prediction: Tensor = sample["minimaps"][:, history_len - 1, None, target_ch]
            # This is repeated for the future forecasts
            prediction = prediction.repeat(1, future_len, 1, 1, 1)
            result = evaluator(prediction, sample)
            for k, v in result.items():
                results[k].append(v)
            pbar.update(1)

    # Calculate average and remove soft_iou_ from key
    return {
        k.replace("soft_iou_", ""): np.nansum(v) / np.isfinite(v).sum()
        for k, v in results.items()
    }


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="Dataloader configuration to use")],
    workspace: Annotated[Path, typer.Option(help="Workspace with Results Database")],
):
    """Basic minimap baseline where the previous frame is used to predict the next frame"""
    dataset = get_dataset(config)
    assert isinstance(dataset, SC2DatasetCfg)
    results = evaluate_trivial_prediction(dataset)
    results["iteration"] = 0  # Add dummy iteration

    print(f"Writing results: {results}")
    exp_hash = "baseline"

    with closing(Database(get_sqlite_uri(workspace))) as db:
        update_metadata_entry(
            Metadata(hash=exp_hash, brief="current frame predicts next"), db
        )
        db.commit()

    with closing(sqlite3.connect(workspace / DEFAULT_SQLITE_FILENAME)) as db:
        cursor = db.cursor()
        write_entry(cursor, "sequence_soft_iou_2", exp_hash, results)
        db.commit()
