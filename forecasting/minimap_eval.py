#!/usr/bin/env python3
"""Tool for gathering and formatting minimap results"""
import itertools
import random
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated

import pandas as pd
import torch
import typer
from konductor.data import Split
from konductor.metadata.database.metadata import Metadata
from konductor.metadata.database.interface import (
    get_sqlite_uri,
    DEFAULT_SQLITE_FILENAME,
)
from konductor.metadata.loggers import AverageMeter
from konductor.utilities.metadata import (
    update_database,
    update_database_entry,
    Database,
)
from pyarrow import parquet as pq
from src.baseline.minimap import EVAL_BATCH_SIZE
from src.data.base_dataset import SC2DatasetCfg
from src.eval_helpers import get_pbar, setup_eval_model_and_dataloader
from src.stats import MinimapModelCfg, MinimapSoftIoU
from src.visualisation import write_minimap_forecast_results
from torch import Tensor

from .utils.sqlite_utils import create_table, write_entry

app = typer.Typer()


_TIME_RANGE = range(3, 10, 3)


def _make_pq_to_db():
    base = {
        "soft_iou_self": "self_3",
        "soft_iou_enemy": "enemy_3",
        "motion_soft_iou_self": "motion_self_3",
        "motion_soft_iou_enemy": "motion_enemy_3",
        "diff_soft_iou_self": "diff_self_3",
        "diff_soft_iou_enemy": "diff_enemy_3",
    }
    for ts, name in itertools.product(_TIME_RANGE, ["self", "enemy"]):
        ts_dict = {
            f"soft_iou_{name}_{float(ts)}": f"{name}_{int(ts)}",
            f"motion_soft_iou_{name}_{float(ts)}": f"motion_{name}_{int(ts)}",
            f"diff_soft_iou_{name}_{float(ts)}": f"diff_{name}_{int(ts)}",
        }
        base.update(ts_dict)
    return base


_PQ_TO_DB = _make_pq_to_db()


def transform_soft_iou_to_db_format(data: pd.DataFrame) -> dict[str, float | int]:
    """
    Grab the last iteration from the parquet data and transform to database dictionary input format.
    For the multi-frame minimap experiments, since the next frame is 3.0 sec, we also add this to
    common results table.
    """
    iteration = data["iteration"].max()
    average = data.query(f"iteration == {iteration}").mean()
    transformed: dict[str, float | int] = {"iteration": int(iteration)}

    for pq_key, db_key in _PQ_TO_DB.items():
        if pq_key in average:
            transformed[db_key] = average[pq_key].item()

    return transformed


@app.command()
def gather_minimap_soft_iou(workspace: Annotated[Path, typer.Option()] = Path.cwd()):
    """Gather soft iou for each of the minimap experiments and save to analysis table"""
    update_database(workspace, get_sqlite_uri(workspace))

    db_handle = sqlite3.connect(workspace / DEFAULT_SQLITE_FILENAME)
    cur = db_handle.cursor()
    table_name = "sequence_soft_iou"
    table_spec = {"iteration": "INTEGER"}
    for ts in _TIME_RANGE:
        table_spec.update(
            {
                f"self_{ts}": "FLOAT",
                f"enemy_{ts}": "FLOAT",
                f"motion_self_{ts}": "FLOAT",
                f"motion_enemy_{ts}": "FLOAT",
            }
        )
    create_table(cur, table_name, table_spec)

    for exp_run in filter(lambda x: x.is_dir(), workspace.iterdir()):
        parquet_filename = exp_run / "val_minimap-soft-iou.parquet"
        if not parquet_filename.exists():
            continue
        data: pd.DataFrame = pq.read_table(parquet_filename).to_pandas()
        results = transform_soft_iou_to_db_format(data)
        write_entry(cur, table_name, exp_run.name, results)

    db_handle.commit()


def make_sequence_2_table(cursor: sqlite3.Cursor):
    """Make sequence2 table if not already exists"""
    table_spec = {"iteration": "INTEGER"}
    for prefix, name, ts in itertools.product(
        ["", "motion_", "diff_"], ["self", "enemy"], _TIME_RANGE
    ):
        table_spec[f"{prefix}{name}_{ts}"] = "FLOAT"
    create_table(cursor, "sequence_soft_iou_2", table_spec)


@app.command()
@torch.inference_mode()
def run(
    run_path: Annotated[Path, typer.Option()],
    workers: Annotated[int, typer.Option()] = 4,
    live_pbar: Annotated[bool, typer.Option()] = False,
):
    """Re-run evaluation with a model and write the results to the common database"""
    db_path = run_path.parent / DEFAULT_SQLITE_FILENAME
    with closing(Database(get_sqlite_uri(db_path))) as db_handle:
        meta = Metadata.from_yaml(run_path / "metadata.yaml")
        update_database_entry(meta, db_handle)
        db_handle.commit()

    exp_config, model, dataloader = setup_eval_model_and_dataloader(
        run_path, split=Split.VAL, workers=workers, batch_size=EVAL_BATCH_SIZE
    )
    metric = MinimapSoftIoU.from_config(exp_config.init)
    meter = AverageMeter()

    with get_pbar(total=len(dataloader), desc="Evaluating", live=live_pbar) as pbar:
        for sample in dataloader:
            sample = sample[0]
            preds = model(sample)
            perf = metric(preds, sample)
            meter.add(perf)
            pbar.update(1)

    db_format = {_PQ_TO_DB[k]: v for k, v in meter.results().items()}
    db_format["iteration"] = meta.iteration
    with closing(sqlite3.connect(db_path)) as db_handle:
        cur = db_handle.cursor()
        write_entry(cur, "sequence_soft_iou_2", run_path.name, db_format)
        db_handle.commit()


@app.command()
def run_all(
    workspace: Annotated[Path, typer.Option()],
    workers: Annotated[int, typer.Option()] = 4,
    live_pbar: Annotated[bool, typer.Option()] = False,
):
    """Re-run evaluation over all experiments in workspace and write to database"""
    with closing(sqlite3.connect(workspace / DEFAULT_SQLITE_FILENAME)) as db_handle:
        cur = db_handle.cursor()
        make_sequence_2_table(cur)
        existing = {
            res[0]
            for res in cur.execute("SELECT hash FROM sequence_soft_iou_2;").fetchall()
        }

    def run_filt(run_dir: Path):
        res = (run_dir / "latest.pt").exists()
        res &= run_dir.name not in existing
        return res

    exps = list(filter(run_filt, workspace.iterdir()))
    for idx, exp in enumerate(exps, 1):
        try:
            run(exp, workers, live_pbar)
        except Exception as err:
            print(f"Failed {exp.name} with error: {err}")
        else:
            print(f"Processed {exp.name} ({idx} of {len(exps)})")


@app.command()
@torch.inference_mode()
def visualise_minimap_forecast(
    run_path: Annotated[Path, typer.Option()],
    workers: Annotated[int, typer.Option()] = 4,
    batch_size: Annotated[int, typer.Option()] = 16,
    split: Annotated[Split, typer.Option()] = Split.VAL,
    n_samples: Annotated[int, typer.Option()] = 16,
):
    """Write images or minimap forecast for konduct review image viewer."""
    exp_config, model, dataloader = setup_eval_model_and_dataloader(
        run_path, split=split, workers=workers, batch_size=batch_size
    )

    model_cfg = exp_config.model[0]
    assert isinstance(model_cfg, MinimapModelCfg)
    data_cfg = exp_config.dataset[0]
    assert isinstance(data_cfg, SC2DatasetCfg)
    assert data_cfg.minimap_ch_names is not None

    if model_cfg.future_len > 1:
        step_sec = data_cfg.properties["step_sec"]
        timepoints = [float(i * step_sec) for i in range(1, model_cfg.future_len + 1)]
    else:
        timepoints = None

    random.seed(0)  # Fix random seed for time_idx sampling

    outdir = exp_config.exp_path / "images"
    outdir.mkdir(exist_ok=True)
    with get_pbar(
        total=n_samples, desc="Generating Minimap Predictions", live=True
    ) as pbar:
        for sample_ in dataloader:
            sample: dict[str, Tensor] = sample_[0]
            preds: Tensor = model(sample)
            if model.is_logit_output:
                preds = preds.sigmoid()
            write_minimap_forecast_results(
                preds,
                sample,
                outdir,
                timepoints,
                model_cfg.target,
                data_cfg.minimap_ch_names,
            )
            pbar.update(preds.shape[0])
            if pbar.n >= n_samples:
                break

    with open(outdir / "samples.txt", "r", encoding="utf-8") as f:
        filenames = f.readlines()
    filenames = sorted(set(filenames))
    with open(outdir / "samples.txt", "w", encoding="utf-8") as f:
        f.write("".join(filenames))


if __name__ == "__main__":
    app()
