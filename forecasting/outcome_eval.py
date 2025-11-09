#!/usr/bin/env python3
"""Tool for gathering and formatting outcome results"""
import sqlite3
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import torch
import typer
from konductor.data import Split
from konductor.metadata.database import Metadata
from konductor.metadata.database.interface import DEFAULT_SQLITE_FILENAME
from konductor.metadata.loggers import ParquetLogger
from konductor.utilities.metadata import update_database
from konductor.utilities.pbar import IntervalPbar, LivePbar
from pyarrow import parquet as pq
from src.eval_helpers import setup_eval_model_and_dataloader, write_outcome_prediction
from src.stats import BinaryAcc
from src.utils import TimeRange
from src.visualisation import metadata_to_str
from torch import Tensor

from .utils.sqlite_utils import create_table, write_entry

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


def transform_outcome_to_db_format(
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
        workspace, f"sqlite:////{workspace.resolve()/DEFAULT_SQLITE_FILENAME}"
    )

    db_handle = sqlite3.connect(workspace / DEFAULT_SQLITE_FILENAME)
    cursor = db_handle.cursor()

    time_points = [TimePoint(t) for t in np.arange(0, 20, 0.5)]
    table_name = "binary_accuracy"
    db_format = {"iteration": "INTEGER"}
    db_format.update({t.as_db_key(): "FLOAT" for t in time_points})
    create_table(cursor, table_name, db_format)

    for exp_run in filter(lambda x: x.is_dir(), workspace.iterdir()):
        parquet_filename = exp_run / "val_binary-acc.parquet"
        if not parquet_filename.exists():
            continue
        data: pd.DataFrame = pq.read_table(parquet_filename).to_pandas()
        results = transform_outcome_to_db_format(data, time_points)
        write_entry(cursor, table_name, exp_run.name, results)

    cursor.close()
    db_handle.commit()
    db_handle.close()


# -------------------------------------------------------------------
# ------- Re-Running evaluation and saving to a separate file -------
# -------------------------------------------------------------------


@app.command()
def evaluate(
    run_path: Annotated[Path, typer.Option()],
    outdir: Annotated[str, typer.Option()],
    batch_size: Annotated[Optional[int], typer.Option()] = None,
    workers: Annotated[int, typer.Option()] = 4,
    py_workers: Annotated[int, typer.Option()] = 2,
    source_prefetch: Annotated[int, typer.Option()] = 2,
    pipe_prefetch: Annotated[int, typer.Option()] = 2,
):
    """Run validation and save results new subdirectory"""
    exp_config, model, dataloader = setup_eval_model_and_dataloader(
        run_path,
        workers,
        batch_size,
        py_workers=py_workers,
        source_prefetch=source_prefetch,
        prefetch=pipe_prefetch,
    )

    if batch_size is None:
        batch_size_ = exp_config.get_batch_size(Split.VAL)
        assert isinstance(batch_size_, int)
        batch_size = batch_size_

    binary_acc = BinaryAcc.from_config(exp_config.init)
    binary_acc.keep_batch = True

    outpath = run_path / outdir
    outpath.mkdir(exist_ok=True)
    logger = ParquetLogger(outpath)
    logger.add_topic("binary-acc", binary_acc.get_keys())

    meta = Metadata.from_yaml(run_path / "metadata.yaml")
    with IntervalPbar(
        total=len(dataloader), fraction=0.1, desc="Evaluating Model..."
    ) as pbar, torch.inference_mode():
        for sample in dataloader:
            if isinstance(sample, list):
                sample = sample[0]
            preds = model(sample)
            results = binary_acc(preds, sample)
            for i in range(batch_size):
                results_ = {}
                for j, k in enumerate(results.keys()):
                    valid = sample["valid"][i][j]
                    if valid:
                        results_[k] = results[k][i]
                logger(Split.VAL, meta.iteration, results_, category="binary-acc")

            pbar.update(1)

    logger.flush()


@app.command()
def evaluate_percent(
    run_path: Annotated[Path, typer.Option()],
    outdir: Annotated[str, typer.Option()],
    database: Annotated[Path, typer.Option()],
    num_buckets: Annotated[int, typer.Option()] = 50,
    batch_size: Annotated[Optional[int], typer.Option()] = None,
    workers: Annotated[int, typer.Option()] = 8,
):
    """Run validation and save results new subdirectory"""
    conn = sqlite3.connect(str(database))
    cursor = conn.cursor()
    exp_config, model, dataloader = setup_eval_model_and_dataloader(
        run_path, workers, batch_size, prefetch=4, py_workers=3, source_prefetch=3
    )

    binary_acc = BinaryAcc.from_config(exp_config.init)
    binary_acc.keep_batch = True

    outpath = run_path / outdir
    outpath.mkdir(exist_ok=True)

    total_results = torch.zeros((2, num_buckets), device="cuda")
    interval = 100 / num_buckets

    with IntervalPbar(
        total=len(dataloader),
        fraction=0.1,
        desc="Predicting Game Outcome By Duration...",
    ) as pbar, torch.inference_mode():
        for sample in dataloader:
            if isinstance(sample, list):
                sample = sample[0]
            preds = model(sample)
            results = binary_acc(preds, sample)

            metadata = metadata_to_str(sample["metadata"])

            replayHash, playerId = [x[:-1] for x in metadata], [x[-1] for x in metadata]
            query = (
                "SELECT replayHash, game_length FROM 'game_data' where "
                + " OR ".join(
                    [
                        f'(replayHash = "{rh}" AND playerId = {pId})'
                        for rh, pId in zip(replayHash, playerId)
                    ]
                )
            )
            cursor.execute(query)
            rg_map = {x[0]: x[1] for x in cursor.fetchall()}
            game_length = torch.tensor(
                [rg_map[rh] for rh in replayHash], device=preds.device
            )
            game_length_mins = game_length / 22.4 / 60

            for idx, k in enumerate(results.keys()):
                time_point = float(k.split("_")[-1])
                percent = 100 * time_point / game_length_mins

                mask = torch.logical_and(
                    sample["valid"][:, idx].type(torch.bool), percent < 100
                )

                corrects = results[k][mask]

                if mask.sum() > 0:
                    idx_mask = (percent // interval).type(torch.int64)

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
    workers: Annotated[int, typer.Option()] = 4,
    dali_py_workers: Annotated[int, typer.Option()] = 2,
    dali_external_prefetch: Annotated[int, typer.Option()] = 2,
    dali_pipe_prefetch: Annotated[int, typer.Option()] = 2,
):
    """Run validation and save to a subdirectory"""

    def is_valid_run(path: Path):
        return path.is_dir() and (path / "latest.pt").exists()

    for run in filter(is_valid_run, workspace.iterdir()):
        print(f"Doing {run}")
        try:
            evaluate(
                run,
                outdir,
                batch_size,
                workers,
                dali_py_workers,
                dali_external_prefetch,
                dali_pipe_prefetch,
            )
        except Exception as e:
            print(e)
            print(
                f"{run} did an oops oh well, anyway..moving on like nothing happened."
            )
            print(
                """⠂⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⡾⣟⣿⠿⠛⠛⠋⡉⠍⡐⢠⠠⡐⡈⠌⣉⠙⠛⠛⠽⣿⣛⢽⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣶⣻⠟⠛⠉⡀⡀⠆⡁⠆⡁⠆⡐⢁⠂⠔⡁⠊⢄⠊⠔⡠⢂⠀⠉⠛⢾⣝⢶⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                    ⠀⠀⠀⠀⠀⠀⠀⣠⠖⠉⡉⠁⠄⡌⠂⢄⠁⠒⡀⠒⡈⠔⠠⠌⡐⠢⢠⠁⠆⢨⠐⠄⠢⠡⠌⠰⢀⠈⠛⢮⣷⣆⠀⠀⠀⠀⠀⠀⡀⠀
                    ⠀⠀⠀⠀⠀⣠⣾⡗⠁⢂⠐⡉⠐⠠⢁⠂⠌⠡⠐⠡⠈⠌⣁⠂⠄⡑⠠⠌⠒⠠⠌⡐⢁⠢⢈⠁⢢⢈⠐⡀⠉⠻⣷⣄⠀⠀⠀⠀⠀⠀
                    ⠀⠀⠀⢀⣾⡽⠋⡀⠎⠠⠁⢄⣡⡾⠀⠌⢂⠡⢃⠡⢉⡐⠈⠻⣷⣤⣁⣂⣡⠈⠔⡈⠄⠒⡈⠌⣀⠢⢁⠤⢁⠂⠒⢾⢧⡀⠀⠈⠀⠀
                    ⠀⠀⣠⣿⡟⠁⡐⢠⣨⣤⡷⠿⠋⠠⢁⡘⢀⠢⠄⠂⠅⡠⢁⠂⠄⡈⢉⠉⡉⢁⠂⡐⠌⠡⢐⠂⠄⢂⠄⠢⢐⠈⡐⠈⠡⣳⡀⠀⠀⠀
                    ⢠⣴⣿⠟⢠⠂⠄⢋⠉⠠⠐⠠⢈⠄⠡⠐⡀⠢⢈⠒⠠⢁⠢⢈⢐⣤⣦⡶⠷⠾⠶⢦⣌⡐⠈⠄⡉⠤⢈⡐⠨⠐⠄⡁⢂⠡⠹⡄⠀⠀
                    ⣿⢟⡾⠀⠂⠌⡐⠠⠈⠄⡁⠂⠄⡈⠆⣁⠂⡑⠠⢈⢂⡁⢂⣶⣿⣵⣶⣶⡶⠀⠀⠀⠈⠛⢾⣄⠄⠡⠂⠤⢁⡘⢀⠒⠠⢀⠡⠀⠀⠀
                    ⣿⣾⠃⡈⢐⣠⡶⠷⢛⣷⣺⣷⣤⡐⠈⢄⠒⣈⠁⢂⠤⢘⣵⣿⣿⢿⣿⠏⠀⠀⠀⠀⠀⠀⠀⠉⢷⣆⠡⠌⢠⠐⠂⠌⢒⠠⠀⠀⢣⡄
                    ⣿⡇⠐⢠⡾⠋⢀⣴⣿⣿⣿⡟⠉⠻⣆⡀⠒⡄⠨⠄⠒⣼⣿⣿⢯⣿⣿⠀⠀⠀⠂⠄⠀⠀⠀⠀⠀⠹⣦⠐⠠⠌⢂⢁⠢⠄⠡⠀⠰⣿
                    ⢻⡅⢨⡿⠁⢠⣿⣿⣿⡿⣽⣿⠀⠀⠙⣧⠐⠠⠑⡈⣼⣿⣿⣯⣿⣟⣿⣧⡀⠀⠀⠀⣤⡀⠀⠀⠀⠀⠹⣧⠐⡈⠄⢂⠰⢈⠐⡀⡆⡿
                    ⢻⡄⣿⠁⠀⣼⣿⣿⡿⣽⣿⢿⣷⣄⣀⣹⣇⠠⠑⡄⣿⣿⣿⣾⣯⣿⢯⣿⣿⣿⣷⣿⣿⠄⠀⠀⠀⠀⠀⢹⡇⠠⠈⠄⠂⡄⠡⠀⡇⡇
                    ⣹⢾⡇⠀⠀⣿⣿⣿⡿⣿⣽⣿⣻⣿⣿⣿⣿⣀⠣⢰⣿⣿⡟⠉⠛⢿⣿⣯⣿⣿⣿⣿⣿⠀⠐⠀⠀⡐⠀⠰⣟⠠⢁⠊⡐⠠⢁⠂⡄⣇
                    ⣿⡞⡇⠀⠀⢻⣿⣿⠿⠿⣿⣾⣟⣷⣿⣿⡏⣷⢉⡚⣧⢻⣧⡀⠀⣸⣿⣿⣿⣿⣿⣿⠇⠀⠠⠀⡀⢀⠀⢘⣏⠀⢂⡐⢈⠁⢂⠀⠐⣿
                    ⣿⣧⡷⠀⠀⠈⣿⣧⡀⠀⣼⣿⣿⣿⣿⡿⠁⣿⠢⠉⣿⡀⠻⣿⣿⣿⣿⣿⣿⣿⡿⠃⠀⠀⠁⠀⠀⠀⠀⢸⡇⠈⢄⡐⠠⢊⠀⠀⢸⠷
                    ⣿⣹⣿⠀⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⠟⠁⠀⣿⠀⠁⠘⣧⠀⠀⠙⠛⠛⠛⠛⠁⣠⣀⠀⠀⠀⠀⠀⠀⢀⣾⠃⠈⡀⠠⢁⠂⢨⣴⡏⠁
                    ⠉⢿⡽⣆⠀⠀⠀⢀⡶⣯⠉⠛⠉⠀⠀⠀⢰⡏⠀⠌⠐⠘⣧⠀⠀⠀⠀⠀⠀⣾⠉⠉⢷⡄⠀⠀⠀⠀⣾⠁⢄⠣⠌⡑⢠⣸⢟⡞⠀⠀
                    ⠀⠸⣿⡹⣦⡀⣠⡟⠁⢸⡇⠀⠀⠀⠀⢠⡟⠀⠠⠁⠂⢀⠘⢳⣄⠀⠀⠀⠀⣿⡀⢀⠀⠛⢦⣄⡀⣼⠇⡈⢆⡑⢂⡁⢢⢯⡟⠀⠀⠀
                    ⠀⠀⣾⣿⣿⠿⠋⢀⠐⣺⣧⣤⠶⠶⠶⠛⠈⠳⣶⠋⠉⠁⠉⠀⠉⠛⠛⠛⠒⠛⠻⠀⠄⢂⠀⠈⠙⠻⣦⡈⠔⡈⠄⣼⣻⡟⠈⠀⠀⠀
                    ⣬⣿⣻⠏⠁⡀⠤⠀⠄⠀⡀⠀⡀⢀⠀⠄⣐⣠⠿⢦⣄⣂⣈⡀⢁⡠⠀⠄⢂⠐⢀⠈⠠⠀⠌⡐⠠⢀⠈⠻⣦⡘⣛⣽⠋⠀⠀⠀⠀⠀
                    ⣿⣷⠏⠀⠐⡀⠂⠌⢀⠂⠄⠁⢰⡶⠾⠛⠋⠁⡀⢀⠈⠉⠁⠉⠉⢹⡇⠀⠂⠌⡀⠄⠁⡐⠠⠀⡁⠂⠄⠀⠉⣿⣻⡇⠀⠀⠀⠀⠀⠀
                    ⣿⡇⠀⠄⠒⡀⢁⠂⠠⢈⠀⠌⣼⡗⠀⠐⠠⠁⡐⠂⠠⠁⡘⠠⠐⠈⢿⡄⠐⠠⢀⠂⠁⡐⠀⡁⠐⠈⡐⠈⢀⢸⣿⡇⠀⠀⠀⠀⠀⠀
                    ⣼⡅⠀⠌⠐⡀⢂⠀⠡⢀⠂⠀⣽⣿⣶⢤⣆⣐⠀⠀⡁⠠⠀⠄⠁⠠⠘⣧⠈⠐⠠⢀⠁⠄⡐⢀⠁⠂⠄⠁⢀⣾⣻⡇⠀⠀⠀⠀⠀⠀
                    ⣿⣷⡀⠈⠐⡀⢂⠈⠄⠠⠀⢡⣿⣿⠉⠳⠾⢭⣿⣻⣶⣶⡶⠦⡼⠴⢦⣾⣄⠉⠀⠂⠈⡀⠐⠀⠈⠠⣐⣨⣾⣻⠏⠀⠀⠀⠀⠀⠀⠀
                    ⠿⣶⣿⣤⣴⣤⣤⣤⣦⣴⣬⢾⣷⡏⠀⠀⠀⢀⢀⡉⢉⠉⠙⣛⢛⡟⣻⣿⣽⣿⠛⠶⠷⡶⠳⠟⣿⣿⣯⣽⣿⡉⠀⠀⠀⠀⠀⠀⠀⠀
                    """
            )
            continue


def and_filter(*funcs, items):
    return filter(lambda x: all(f(x) for f in funcs), items)


@app.command()
def evaluate_all_percent(
    workspace: Annotated[Path, typer.Option()],
    outdir: Annotated[str, typer.Option()],
    database: Annotated[Path, typer.Option()],
    num_buckets: Annotated[int, typer.Option()] = 50,
    batch_size: Annotated[Optional[int], typer.Option()] = None,
    workers: Annotated[int, typer.Option()] = 8,
):
    """Run validation and save to a subdirectory"""

    def is_valid_run(path: Path):
        return path.is_dir() and (path / "latest.pt").exists()

    def no_results_exist(path: Path):
        return not (path / outdir / "game_length_results_50").exists()

    for run in and_filter(is_valid_run, no_results_exist, items=workspace.iterdir()):
        print(f"Doing {run}")
        try:
            evaluate_percent(run, outdir, database, num_buckets, batch_size, workers)
        except Exception as e:
            print(e)
            print(f"failed for {run}")


@app.command()
@torch.inference_mode()
def single_replay_analysis(
    run_path: Annotated[Path, typer.Option()],
    workers: Annotated[int, typer.Option()] = 4,
    batch_size: Annotated[int, typer.Option()] = 16,
    split: Annotated[Split, typer.Option()] = Split.VAL,
    n_samples: Annotated[int, typer.Option()] = 16,
):
    """Record and plot the single replay outcome prediction over the duration of the replay"""
    exp_config, model, dataloader = setup_eval_model_and_dataloader(
        run_path, split=split, workers=workers, batch_size=batch_size
    )

    timepoints: TimeRange = exp_config.dataset[0].properties["timepoints"]

    results = pd.DataFrame(
        index=pd.RangeIndex(0, n_samples),
        columns=["replay", "playerId", "outcome"]
        + [str(t.item()) for t in timepoints.arange()],
    )

    with LivePbar(total=n_samples, desc="Predicting Replay Outcomes...") as pbar:
        for sample_ in dataloader:
            sample: dict[str, Tensor] = sample_[0]
            preds: Tensor = model(sample)
            write_outcome_prediction(sample, preds, pbar.n, results)
            pbar.update(preds.shape[0])
            if pbar.n >= n_samples:
                break

    results.to_csv(run_path / "outcome_prediction.csv")


@app.command()
def single_replay_analysis_all(
    workspace: Annotated[Path, typer.Option()],
    workers: Annotated[int, typer.Option()] = 4,
    batch_size: Annotated[int, typer.Option()] = 16,
    split: Annotated[Split, typer.Option()] = Split.VAL,
    n_samples: Annotated[int, typer.Option()] = 16,
):
    """Run single-replay-analysis for all experiments in a workspace"""

    def is_trained_experiment(path: Path):
        """Must be trained experiment if checkpoint exists"""
        return (path / "latest.pt").exists()

    experiments = list(filter(is_trained_experiment, workspace.iterdir()))
    for idx, item in enumerate(experiments, 1):
        try:
            single_replay_analysis(item, workers, batch_size, split, n_samples)
        except RuntimeError as err:
            print(f"Failed to run experiment {item.stem} with error: {err}")
        print(f"Finished {idx} of {len(experiments)} experiments")
