from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from konductor.data import DATASET_REGISTRY, DatasetConfig, ExperimentInitConfig, Split
from konductor.data._pytorch.dataloader import DataloaderV1Config
from sc2_replay_reader import (
    GAME_INFO_FILE,
    ReplayDatabase,
    ReplayParser,
    Result,
    setReplayDBLoggingLevel,
    spdlog_lvl,
)
from torch.utils.data import Dataset

from .utils import upper_bound


class SC2Replay(Dataset):
    def __init__(
        self,
        basepath: Path,
        split: Split,
        train_ratio: float,
        features: set[str] | None,
    ) -> None:
        super().__init__()
        self.features = features
        self.db_handle = ReplayDatabase()
        self.parser = ReplayParser(GAME_INFO_FILE)

        setReplayDBLoggingLevel(spdlog_lvl.warn)

        if basepath.is_file():
            self.replays = [basepath]
        else:
            self.replays = list(basepath.glob("*.SC2Replays"))
            assert len(self.replays) > 0, f"No .SC2Replays found in {basepath}"

        replays_per_file = torch.empty([len(self.replays) + 1], dtype=torch.int)
        replays_per_file[0] = 0
        for idx, replay in enumerate(self.replays, start=1):
            self.db_handle.open(replay)
            replays_per_file[idx] = self.db_handle.size()

        self._accumulated_replays = torch.cumsum(replays_per_file, 0)
        self.n_replays = int(self._accumulated_replays[-1].item())
        self.n_replays *= train_ratio if split is Split.TRAIN else 1 - train_ratio
        self.n_replays = int(self.n_replays)
        assert self.n_replays > 0, "No replays in dataset"

    def __len__(self) -> int:
        return self.n_replays

    def __getitem__(self, index: int):
        file_index = upper_bound(self._accumulated_replays, index)
        self.db_handle.open(self.replays[file_index])
        db_index = index - int(self._accumulated_replays[file_index].item())
        assert (  # This should hold if calculation checks out
            db_index < self.db_handle.size()
        ), f"{db_index} exceeds {self.db_handle.size()}"

        self.parser.parse_replay(self.db_handle.getEntry(db_index))

        outputs = self.parser.sample(0)
        if self.features is not None:
            outputs = {k: torch.as_tensor(outputs[k]) for k in self.features}
        outputs["win"] = torch.as_tensor(
            self.parser.data.playerResult == Result.Win, dtype=torch.float32
        ).unsqueeze(0)

        return outputs


@dataclass
@DATASET_REGISTRY.register_module("sc2-replay")
class SC2ReplayConfig(DatasetConfig):
    # Dataloader type we want to use
    train_loader: DataloaderV1Config
    val_loader: DataloaderV1Config

    features: set[str] | None = None
    train_ratio: float = 0.8  # Portion of all data to use for training

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0):
        return super().from_config(config, idx)

    def __post_init__(self):
        assert 0 < self.train_ratio < 1, f"Failed: 0<{self.train_ratio=}<1"
        # If features is not None, ensure that it is a set
        if self.features is not None and not isinstance(self.features, set):
            self.features = set(self.features)

    @property
    def properties(self) -> Dict[str, Any]:
        ret = {"image_ch": 10, "scalar_ch": 28}
        ret.update(self.__dict__)
        return ret

    def get_instance(self, split: Split) -> Any:
        known_unused = {"train_loader", "val_loader", "basepath"}
        dataset = self.init_auto_filter(
            SC2Replay, known_unused=known_unused, split=split
        )
        match split:
            case Split.TRAIN:
                return self.train_loader.get_instance(dataset)
            case Split.VAL | Split.TEST:
                return self.train_loader.get_instance(dataset)
            case _:
                raise RuntimeError(f"How did I get here with {split=}")
