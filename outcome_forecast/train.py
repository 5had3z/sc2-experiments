import logging
from functools import partial
from pathlib import Path
from typing import Optional

import src
import torch
import typer
import yaml
from torch import Tensor
from konductor.metadata.loggers import TBLogger, ParquetLogger, MultiWriter
from konductor.init import ExperimentInitConfig, ModuleInitConfig
from konductor.metadata import DataManager
from konductor.trainer.pytorch import (
    PyTorchTrainer,
    PyTorchTrainerConfig,
    PyTorchTrainerModules,
)
from konductor.utilities import comm
from konductor.utilities.pbar import PbarType, pbar_wrapper
from typing_extensions import Annotated


class Trainer(PyTorchTrainer):
    """Specialize for prediciton"""

    def data_transform(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if torch.cuda.is_available():
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                data = {k: d.cuda(non_blocking=True) for k, d in data.items()}
            stream.synchronize()

        mask = data["valid"].sum(axis=1) > self.modules.trainloader.dataset.min_index

        return {k: d[mask, ...] for k, d in data.items()}


app = typer.Typer()


@app.command()
def main(
    workspace: Annotated[Path, typer.Option()],
    epoch: Annotated[int, typer.Option()],
    remote: Annotated[Optional[Path], typer.Option()] = None,
    run_hash: Annotated[Optional[str], typer.Option()] = None,
    config_file: Annotated[Optional[Path], typer.Option()] = None,
    workers: Annotated[int, typer.Option()] = 4,
    pbar: Annotated[bool, typer.Option()] = False,
    brief: Annotated[Optional[str], typer.Option()] = None,
):
    """Run training"""
    if config_file is not None:
        assert run_hash is None, "config-file or run-hash should be exclusively set"
        exp_cfg = ExperimentInitConfig.from_config(workspace, config_file)
    elif run_hash is not None:
        exp_cfg = ExperimentInitConfig.from_run(workspace / run_hash)
    else:
        raise RuntimeError("Either config-file or run-hash should be set")
    exp_cfg.set_workers(workers)

    if remote is not None:
        with open(remote, "r", encoding="utf-8") as file:
            remote_cfg = yaml.safe_load(file)
        exp_cfg.remote_sync = ModuleInitConfig(**remote_cfg)

    train_modules = PyTorchTrainerModules.from_config(exp_cfg)

    data_manager = DataManager.default_build(
        exp_cfg,
        train_modules.get_checkpointables(),
        {
            "win-auc": src.stats.WinAUC.from_config(exp_cfg),
            "binary-acc": src.stats.BinaryAcc.from_config(exp_cfg),
        },
        MultiWriter([TBLogger(exp_cfg.work_dir), ParquetLogger(exp_cfg.work_dir)]),
    )

    if brief is not None:
        data_manager.metadata.brief = brief

    trainer_cfg = PyTorchTrainerConfig()
    if pbar and comm.get_local_rank() == 0:
        trainer_cfg.pbar = partial(pbar_wrapper, pbar_type=PbarType.LIVE)
    elif comm.get_local_rank() == 0:
        trainer_cfg.pbar = partial(
            pbar_wrapper, pbar_type=PbarType.INTERVAL, fraction=0.1
        )

    trainer = Trainer(trainer_cfg, train_modules, data_manager)
    trainer.train(epoch=epoch)


if __name__ == "__main__":
    if torch.cuda.is_available():
        comm.initialize()
    torch.set_float32_matmul_precision("high")
    logging.basicConfig(
        format=(
            "%(asctime)s-RANK:{comm.get_local_rank()}-"
            "%(levelname)s-%(name)s: %(message)s"
        ),
        level=logging.INFO,
        force=True,
    )
    app()
