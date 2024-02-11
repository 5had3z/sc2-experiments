#!/usr/bin/env python3
"""Tool for gathering and formatting results"""
import sqlite3
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import torch
import typer
from konductor.data import Split
from konductor.metadata.database import Metadata
from konductor.metadata.database.sqlite import DEFAULT_FILENAME, SQLiteDB
from konductor.metadata.loggers import ParquetLogger
from konductor.utilities.metadata import update_database
from konductor.utilities.pbar import LivePbar
from pyarrow import parquet as pq
from src.stats import BinaryAcc
from torch import Tensor
from utils.eval_helpers import (
    get_dataloader_with_metadata,
    setup_eval_model_and_dataloader,
    write_minimap_forecast_results,
    metadata_to_str,
)

app = typer.Typer()


class TimePoint:
    __slots__ = "value"

    def __init__(self, value: float):
        self.value = value

    def as_db_key(self):
        return "t_" + str(self.value).replace(".", "_")

    def as_pq_key(self):
        return "binary_acc_" + str(self.value)

    def as_float(self):
        return float(self.value)


def transform_latest_to_db_format(
    data: pd.DataFrame, time_points: list[TimePoint]
) -> dict[str, float]:
    """Grab the last iteration from the parquet
    data and transform to database dictionary input format"""
    iteration = data["iteration"].max()
    average = data.query(f"iteration == {iteration}").mean()
    transformed = {
        t.as_db_key(): float(average[t.as_pq_key()])
        for t in time_points
        if t.as_pq_key() in data.columns
    }
    transformed["iteration"] = int(iteration)
    return transformed


@app.command()
def gather_ml_binary_accuracy(workspace: Annotated[Path, typer.Option()] = Path.cwd()):
    """Add Binary Accuracy to database table (and update metadata at the same time)"""
    update_database(
        workspace, "sqlite", f'{{"path": "{workspace / DEFAULT_FILENAME}"}}'
    )

    db_handle = SQLiteDB(workspace / DEFAULT_FILENAME)
    time_points = [TimePoint(t) for t in np.arange(0, 20, 0.5)]
    table_name = "binary_accuracy"
    db_format = {"iteration": "INTEGER"}
    db_format.update({t.as_db_key(): "FLOAT" for t in time_points})
    db_handle.create_table(table_name, db_format)

    for exp_run in filter(lambda x: x.is_dir(), workspace.iterdir()):
        parquet_filename = exp_run / "val_binary-acc.parquet"
        if not parquet_filename.exists():
            continue
        data: pd.DataFrame = pq.read_table(parquet_filename).to_pandas()
        results = transform_latest_to_db_format(data, time_points)
        db_handle.write(table_name, exp_run.name, results)

    db_handle.commit()
    db_handle.con.close()


# -------------------------------------------------------------------
# ------- Re-Running evaluation and saving to a separate file -------
# -------------------------------------------------------------------


@app.command()
def evaluate(
    run_path: Annotated[Path, typer.Option()],
    outdir: Annotated[str, typer.Option()],
    batch_size: Annotated[Optional[int], typer.Option()] = None,
):
    """Run validation and save results new subdirectory"""
    exp_config, model, dataloader = setup_eval_model_and_dataloader(
        run_path, batch_size=batch_size
    )
    binary_acc = BinaryAcc.from_config(exp_config)

    outpath = run_path / outdir
    outpath.mkdir(exist_ok=True)
    logger = ParquetLogger(outpath)

    meta = Metadata.from_yaml(run_path / "metadata.yaml")

    with LivePbar(total=len(dataloader)) as pbar, torch.inference_mode():
        for sample in dataloader:
            if isinstance(sample, list):
                sample = sample[0]
            preds = model(sample)
            results = binary_acc(preds, sample)
            logger(Split.VAL, meta.iteration, results)
            pbar.update(1)

    logger.flush()


@app.command()
def evaluate_percent(
    run_path: Annotated[Path, typer.Option()],
    outdir: Annotated[str, typer.Option()],
    database: Annotated[Path, typer.Option()],
    num_buckets: Annotated[int, typer.Option()] = 50,
    batch_size: Annotated[Optional[int], typer.Option()] = None,
    workers: Annotated[Optional[int], typer.Option()] = None,
):
    """Run validation and save results new subdirectory"""
    conn = sqlite3.connect(str(database))
    cursor = conn.cursor()

    exp_config, model, _ = setup_eval_model_and_dataloader(
        run_path, batch_size=batch_size, workers=workers
    )
    dataloader = get_dataloader_with_metadata(exp_config)

    binary_acc = BinaryAcc.from_config(exp_config)
    binary_acc.keep_batch = True

    outpath = run_path / outdir
    outpath.mkdir(exist_ok=True)

    total_results = torch.zeros((2, num_buckets), device="cuda")
    interval = 100 / num_buckets

    with LivePbar(total=len(dataloader)) as pbar, torch.inference_mode():
        for sample in dataloader:
            if isinstance(sample, list):
                sample = sample[0]
            preds = model(sample)
            results = binary_acc(preds, sample)

            metadata = metadata_to_str(sample["metadata"])

            replayHash, playerId = [x[:-1] for x in metadata], [x[-1] for x in metadata]
            query = "SELECT game_length FROM 'game_data' where " + " OR ".join(
                [
                    f'(replayHash = "{rh}" AND playerId = {pId})'
                    for rh, pId in zip(replayHash, playerId)
                ]
            )
            cursor.execute(query)
            game_length = torch.tensor(
                [x[0] for x in cursor.fetchall()], device=preds.device
            )
            game_length_mins = game_length / 22.4 / 60

            for idx, k in enumerate(results.keys()):
                time_point = float(k.split("_")[-1])
                percent = 100 * time_point / game_length_mins

                idx_mask = (percent // interval).type(torch.int64)
                mask = torch.logical_and(
                    sample["valid"][:, idx].type(torch.bool), percent <= 100
                )

                corrects = results[k][mask]

                values, counts = torch.unique(
                    idx_mask[mask][corrects.type(torch.bool)], return_counts=True
                )
                total_results[0, values] += counts

                values, counts = torch.unique(idx_mask[mask], return_counts=True)
                total_results[1, values] += counts

            pbar.update(1)

    np.savetxt(
        outpath / f"game_length_results_{num_buckets}",
        total_results.cpu().numpy(),
        delimiter=",",
    )


@app.command()
def evaluate_all(
    workspace: Annotated[Path, typer.Option()],
    outdir: Annotated[str, typer.Option()],
    batch_size: Annotated[Optional[int], typer] = None,
):
    """Run validaiton and save to a subdirectory"""

    def is_valid_run(path: Path):
        return path.is_dir() and (path / "latest.pt").exists()

    for run in filter(is_valid_run, workspace.iterdir()):
        print(f"Doing {run}")
        evaluate(run, outdir, batch_size)


@app.command()
def evaluate_all_percent(
    workspace: Annotated[Path, typer.Option()],
    outdir: Annotated[str, typer.Option()],
    database: Annotated[Path, typer.Option()],
    num_buckets: Annotated[int, typer.Option()] = 50,
    batch_size: Annotated[Optional[int], typer.Option()] = None,
    workers: Annotated[Optional[int], typer.Option()] = None,
):
    """Run validaiton and save to a subdirectory"""

    def is_valid_run(path: Path):
        return path.is_dir() and (path / "latest.pt").exists()

    for run in filter(is_valid_run, workspace.iterdir()):
        print(f"Doing {run}")
        evaluate_percent(run, outdir, database, num_buckets, batch_size, workers)


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
    exp_config, model, _ = setup_eval_model_and_dataloader(
        run_path, split=split, workers=workers, batch_size=batch_size
    )
    dataloader = get_dataloader_with_metadata(exp_config)

    outdir = exp_config.exp_path / "images"
    outdir.mkdir(exist_ok=True)
    with LivePbar(total=n_samples, desc="Generating Images") as pbar:
        for sample_ in dataloader:
            sample: dict[str, Tensor] = sample_[0]
            preds: Tensor = model(sample)
            write_minimap_forecast_results(preds, sample, outdir)
            pbar.update(preds.shape[0])
            if pbar.n >= n_samples:
                break


if __name__ == "__main__":
    app()
