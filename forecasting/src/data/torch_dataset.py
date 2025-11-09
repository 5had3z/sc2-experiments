import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np
import torch
from konductor.data import DATASET_REGISTRY, Split
from konductor.data._pytorch.dataloader import TorchLoaderConfig
from sc2_serializer import Result, get_database_and_parser
from sc2_serializer.sampler import ReplaySampler
from torch.utils.data import Dataset

from ..utils import TimeRange
from .base_dataset import (
    SAMPLER_REGISTRY,
    SC2FolderCfg,
    SC2SamplerCfg,
    find_closest_indices,
)


class SC2ReplayBase(Dataset):
    """Standard Base StarCraft II Pytorch Dataset
    Should be inherited from with `process_replay` overwritten with what you want to do.
    """

    def __init__(
        self,
        sampler: ReplaySampler,
        features: list[str],
        minimap_layers: list[str] | None = None,
        metadata: bool = False,
    ):
        super().__init__()
        self.sampler = sampler
        self.features = features
        self.metadata = metadata
        self.logger = logging.getLogger("replay-dataset")
        if features is not None:
            self.db_handle, self.parser = get_database_and_parser(
                parse_units="units" in features,
                parse_minimaps="minimaps" in features,
            )
        else:
            self.db_handle, self.parser = get_database_and_parser(
                parse_units=True, parse_minimaps=True
            )

        if minimap_layers is not None:
            self.parser.setMinimapFeatures(minimap_layers)

    def __len__(self) -> int:
        return len(self.sampler)

    def process_replay(self):
        raise NotImplementedError

    def __getitem__(self, index: int):
        replay_file, replay_idx = self.sampler.sample(index)
        if not self.db_handle.load(replay_file):
            raise FileNotFoundError(replay_file)
        replay_data = self.db_handle.getEntry(replay_idx)
        self.parser.parse_replay(replay_data)
        outputs = self.process_replay()

        if self.metadata:
            outputs["metadata"] = self.parser.info.replayHash + str(
                self.parser.info.playerId
            )

        return outputs


class SC2ReplayOutcome(SC2ReplayBase):
    """StarCraftII dataset geared towards outcome prediction"""

    def __init__(
        self,
        sampler: ReplaySampler,
        timepoints: TimeRange,
        features: list[str],
        metadata: bool = False,
        minimap_layers: list[str] | None = None,
        min_game_time: float | None = None,
    ) -> None:
        super().__init__(sampler, features, minimap_layers, metadata)

        _loop_per_min = 22.4 * 60
        self._target_game_loops = (timepoints.arange() * _loop_per_min).to(torch.int)

        if min_game_time is None:
            self.min_index = None
        else:
            difference_array = torch.absolute(timepoints.arange() - min_game_time)
            self.min_index = difference_array.argmin()

    def process_replay(self):
        """Process replay data currently in parser into dictionary of
        features and game outcome"""
        try:
            test_sample: dict[str, Any] = self.parser.sample_all(0)
        except (RuntimeError, IndexError) as err:
            raise RuntimeError(f"Parse failure for {self.parser.info}") from err

        # Find the indices to sample at based on recorded gamesteps
        sample_indices = find_closest_indices(
            self.parser.data.gameStep, self._target_game_loops
        )

        if (sample_indices == -1).all():
            print(f"No valid samples in {self.parser.info.replayHash}")

        outputs_list = {
            k: []
            for k in (test_sample.keys() if self.features is None else self.features)
        }

        for idx in sample_indices:
            if idx == -1:
                sample = {k: np.zeros_like(test_sample[k]) for k in outputs_list}
            else:
                sample = self.parser.sample_all(int(idx.item()))
            for k in outputs_list:
                outputs_list[k].append(sample[k])

        outputs = {
            "win": torch.as_tensor(
                self.parser.info.playerResult == Result.Win, dtype=torch.float32
            ),
            "valid": (sample_indices != -1).to(torch.bool),
        }
        for k in outputs_list:
            outputs[k] = torch.stack([torch.as_tensor(o) for o in outputs_list[k]])

        return outputs


@dataclass
class SC2ReplayBaseConfig(SC2SamplerCfg):
    """Pytorch Variant of SC2 Dataloader"""

    train_loader: TorchLoaderConfig
    val_loader: TorchLoaderConfig

    def get_cls(self):
        raise NotImplementedError

    def get_dataloader(self, split: Split) -> Any:
        sampler = SAMPLER_REGISTRY[self.sampler_cfg.type](
            is_train=split is Split.TRAIN,
            train_ratio=self.train_ratio,
            replays_path=self.basepath,
            **self.sampler_cfg.args,
        )
        dataset = self.init_auto_filter(
            self.get_cls(), known_unused=SC2SamplerCfg._known_unused, sampler=sampler
        )
        match split:
            case Split.TRAIN:
                return self.train_loader.get_instance(dataset)
            case Split.VAL | Split.TEST:
                return self.val_loader.get_instance(dataset)
            case _:
                raise RuntimeError(f"How did I get here with {split=}")


@dataclass
@DATASET_REGISTRY.register_module("sc2-replay-outcome")
class OutcomeDatasetConfig(SC2ReplayBaseConfig):
    """Configuration for outcome dataset"""

    timepoints: TimeRange = field(
        default_factory=lambda: TimeRange(0, 30, 2)
    )  # Minutes
    min_game_time: float = 5.0  # Minutes

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.timepoints, dict):
            self.timepoints = TimeRange(**self.timepoints)

    def get_cls(self):
        return SC2ReplayOutcome


class FolderDataset(Dataset):
    """
    Basic folder dataset (basepath/split/sample.npz) which contains numpy
    files that contain the expected format to train on directly
    """

    def __init__(self, basepath: Path, split: Split) -> None:
        super().__init__()
        self.folder = basepath / split.name.lower()
        assert self.folder.exists(), f"Root folder doesn't exist: {basepath}"
        self.files = [f.name for f in self.folder.iterdir()]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        data = np.load(self.folder / self.files[index], allow_pickle=True)

        def transform(x: np.ndarray):
            return str(x) if "str" in x.dtype.name else torch.tensor(x)

        try:
            out_data = {k: transform(v) for k, v in data.items()}
        except BadZipFile as e:
            raise RuntimeError(f"Got bad data from {self.files[index]}") from e

        return out_data


@dataclass
@DATASET_REGISTRY.register_module("folder-dataset")
class FolderDatasetConfig(SC2FolderCfg):
    """Configuration for pre-processed numpy file dataset"""

    # Dataloader type we want to use
    train_loader: TorchLoaderConfig
    val_loader: TorchLoaderConfig

    def get_dataloader(self, split: Split) -> Any:
        dataset = FolderDataset(self.basepath, split)
        match split:
            case Split.TRAIN:
                return self.train_loader.get_instance(dataset)
            case Split.VAL | Split.TEST:
                return self.val_loader.get_instance(dataset)
            case _:
                raise RuntimeError(f"How did I get here with {split=}")
