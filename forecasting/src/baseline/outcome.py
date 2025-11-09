import os
import timeit
from enum import Enum
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import typer
import yaml
from konductor.utilities.pbar import IntervalPbar, LivePbar
from sc2_serializer.sampler import SQLSampler
from sklearn import preprocessing, svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader
from typing_extensions import Annotated

from ..data.torch_dataset import SC2ReplayOutcome, Split
from ..utils import TimeRange

try:
    import xgboost
except ImportError:
    xgboost = None


app = typer.Typer()

current_script_path = Path(__file__).resolve().parent


class Model(Enum):
    XGBOOST = "xgboost"
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"
    MLP = "mlp"


def get_model(m: Model, **config):
    def inner(random_state: int):
        match m:
            case Model.XGBOOST:
                if xgboost is None:
                    raise ImportError("Cannot Use XGBoost, as it is not installed :(")
                return xgboost.XGBClassifier(
                    **config,
                    random_state=random_state,
                )
            case Model.MLP:
                return MLPClassifier(**config, random_state=random_state)
            case Model.SVM:
                return svm.SVC(**config, random_state=random_state)
            case Model.LOGISTIC_REGRESSION:
                return LogisticRegression(**config, random_state=random_state)

    return inner


@app.command()
def fit_model(
    model: Annotated[Model, typer.Option()] = Model.XGBOOST,
    yaml_config: Annotated[Path, typer.Option()] = current_script_path
    / "cfg"
    / "baseline.yml",
    recreate_split: Annotated[bool, typer.Option()] = False,
    tmp_workspace: Annotated[Path, typer.Option()] = Path("./processed_data"),
    save_plots: Annotated[bool, typer.Option()] = True,
    workers: Annotated[int, typer.Option()] = 8,
    liveProgress: Annotated[bool, typer.Option()] = True,
):
    assert yaml_config.exists()
    with yaml_config.open(mode="r") as file:
        yaml_data = yaml.safe_load(file)

    time = TimeRange(**yaml_data["timepoints"])
    sampler = SQLSampler(
        yaml_data["database"],
        Path(os.environ["DATAPATH"]),
        yaml_data["sql_filters"],
        0.8,
        Split.TRAIN,
    )
    dataset = SC2ReplayOutcome(sampler, time, ["scalars"])

    dataloader = DataLoader(dataset, batch_size=8, num_workers=workers)

    if recreate_split:
        for time_step, current_time in enumerate(time.arange()):
            file = tmp_workspace / f"{time_step}_x.npy"
            wins = tmp_workspace / f"{time_step}_y.npy"
            if file.exists():
                os.remove(file)
                os.remove(wins)

        pbar_type = LivePbar if liveProgress else IntervalPbar
        pbar_kwargs = {"total": len(dataloader) * len(time), "desc": "recreate"}
        if not liveProgress:
            pbar_kwargs["fraction"] = 0.1
        with pbar_type(**pbar_kwargs) as pbar:
            for sample in dataloader:
                for time_step, current_time in enumerate(time.arange()):
                    file = tmp_workspace / f"{time_step}_x.npy"
                    wins = tmp_workspace / f"{time_step}_y.npy"
                    mask_ = sample["valid"][:, time_step]
                    x = sample["scalars"][:, time_step, :][mask_, :]
                    y = sample["win"][mask_]

                    if mask_.sum() == 0:
                        pbar.update(len(time) - time_step)
                        break

                    if file.exists():
                        data = np.load(file)
                        data = np.concatenate((data, x))

                        wins_array = np.load(wins)
                        wins_array = np.concatenate((wins_array, y))
                        np.save(str(file), data)
                        np.save(str(wins), wins_array)
                    else:
                        np.save(str(file), x)
                        np.save(str(wins), y)

                    pbar.update(1)

        del dataloader
        del dataset

    results = []
    std_dev = []
    _m = get_model(model, **yaml_data[model.name])

    for time_step, current_time in enumerate(time.arange()):
        print(f"Running current_time: {current_time}")
        file = tmp_workspace / f"{time_step}_x.npy"
        wins = tmp_workspace / f"{time_step}_y.npy"

        if not (file.exists() and wins.exists()):
            break

        x = np.load(file)
        y = np.load(wins)
        print(f"for timestep = {time_step}, shape of data is {x.shape}")
        X = preprocessing.StandardScaler().fit(x).transform(x)
        random_state = 22
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        print("Starting fit...")
        t1 = timeit.default_timer()

        model_ = _m(random_state)

        model_scores = cross_val_score(model_, X, y, cv=cv, n_jobs=-1)
        t2 = timeit.default_timer()
        results.append(model_scores.mean())
        std_dev.append(model_scores.std())
        print(f"Took {t2 - t1}s")
        print(
            f"{model.name}: "
            f"{model_scores.mean():.4f} accuracy with a "
            f"standard deviation of {model_scores.std():.4f}"
        )

    if save_plots:
        plt.plot((time.arange() * 60).tolist(), results, label="Results")

        plt.fill_between(
            (time.arange() * 60).tolist(),
            [result - std for result, std in zip(results, std_dev)],
            [result + std for result, std in zip(results, std_dev)],
            alpha=0.2,
            label="Standard Deviation",
        )

        # Add labels and title
        plt.xlabel("Time (s)")
        plt.ylabel("Accuracry")
        plt.title(f"Accuracy of {model.name}")

        # Add legend
        plt.legend()
        plt.savefig(f"{tmp_workspace}/{model.name}.png")

    return results, std_dev


@app.command()
def fit_all(
    yaml_config: Annotated[Path, typer.Option()] = current_script_path
    / "cfg"
    / "baseline.yml",
    recreate_split: Annotated[bool, typer.Option()] = False,
    tmp_workspace: Annotated[Path, typer.Option()] = Path("./processed_npy_files"),
    save_plots: Annotated[bool, typer.Option()] = True,
    skip: Annotated[List[str] | None, typer.Option()] = None,
    workers: Annotated[int, typer.Option()] = 8,
):
    results = {}
    for m in Model:
        try:
            if skip is not None and m.name in skip:
                continue
            results[m.name] = fit_model(
                model=m,
                yaml_config=yaml_config,
                recreate_split=recreate_split,
                tmp_workspace=tmp_workspace,
                save_plots=False,
                workers=workers,
            )

            recreate_split = False
        except ImportError as e:
            print(e)
            continue

    if save_plots:
        assert yaml_config.exists()
        with yaml_config.open(mode="r") as file:
            yaml_data = yaml.safe_load(file)

        time = TimeRange(**yaml_data["timepoints"])

        for name, (results, std_dev) in results.items():
            line_color = np.random.rand(
                3,
            )

            plt.plot(
                (time.arange() * 60).tolist(), results, label=name, color=line_color
            )

            plt.fill_between(
                (time.arange() * 60).tolist(),
                [result - std for result, std in zip(results, std_dev)],
                [result + std for result, std in zip(results, std_dev)],
                alpha=0.2,
                color=line_color,
                label=f"{name}_Standard Deviation",
            )

        # Add labels and title
        plt.xlabel("Time (s)")
        plt.ylabel("Accuracry")

        # Add legend
        plt.legend()

        plt.savefig(f"{tmp_workspace}/all.png")
