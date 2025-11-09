import random
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import torch
from konductor.data import (
    DATASET_REGISTRY,
    ExperimentInitConfig,
    ModuleInitConfig,
    Split,
)
from konductor.data.dali import DALI_AUGMENTATIONS, DALIExternalSource, DaliLoaderConfig
from nvidia.dali import fn
from nvidia.dali.data_node import DataNode
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.types import DALIDataType
from sc2_serializer import (
    ReplayDatabase,
    ReplayParser,
    get_database_and_parser,
    set_replay_database_logger_level,
    spdlog_lvl,
)
from sc2_serializer.sampler import ReplaySampler
from torch import Tensor

from .base_dataset import (
    SC2FolderCfg,
    SC2SamplerCfg,
    SAMPLER_REGISTRY,
    find_closest_indices,
)


def _min_to_game_step(t: float):
    """Convert time (min) to game step"""
    return int(t * 22.4 * 60)


class DaliFolderDataset(DALIExternalSource):
    """Data source for folder full of preprocessed numpy files"""

    def __init__(
        self,
        path: Path,
        split: Split,
        features: list[str],
        batch_size: int,
        shard_id: int,
        num_shards: int,
        random_shuffle: bool,
        yields_batch: bool = False,
        file_suffix: str = "",
        load_other_player: bool = False,
    ) -> None:
        super().__init__(batch_size, shard_id, num_shards, random_shuffle, yields_batch)
        self.split = split
        self.features = features
        self.folder = path / split.name.lower()
        self.file_suffix = file_suffix
        self.load_other_player = load_other_player

        if not self.folder.exists():
            raise FileNotFoundError(self.folder)

        self.files: list[str] = []

    def _post_init(self):
        file_list = (
            self.folder.parent / f"{self.split.lower()}-list{self.file_suffix}.txt"
        )
        with open(file_list, "r", encoding="utf-8") as f:
            self.files = [s.strip() for s in f.readlines()]
        super()._post_init()

    def __len__(self) -> int:
        return len(self.files)

    def get_data(self, index: int):
        data = np.load(self.folder / self.files[index], allow_pickle=True)
        if self.load_other_player:
            base_name = self.files[index][:-5]
            alt_idx = int(self.files[index][-5]) % 2 + 1
            other_data = np.load(
                self.folder / f"{base_name}{alt_idx}.npz", allow_pickle=True
            )

        try:
            out_data = tuple(
                (
                    np.array(
                        [ord(c) for c in data["metadata"].tolist()], dtype=np.uint8
                    )
                    if k == "metadata"
                    else data[k]
                )
                for k in self.features
            )
            if self.load_other_player:
                x = []
                for d, k in zip(out_data, self.features):
                    if k == "win":
                        x.append(np.array(d == np.zeros_like(d), dtype=d.dtype))
                    elif d.dtype == bool:
                        x.append(np.logical_and(other_data[k], d))
                    elif k == "metadata":
                        x.append(d)
                    else:
                        x.append(np.concatenate((d, other_data[k]), axis=1))
                out_data = tuple(x)
        except BadZipFile as e:
            raise RuntimeError(f"Got bad data from {self.files[index]}") from e

        return out_data


@dataclass
@DATASET_REGISTRY.register_module("dali-folder")
class DaliFolderDatasetConfig(SC2FolderCfg):
    """Configuration for numpy folder dataset"""

    train_loader: DaliLoaderConfig
    val_loader: DaliLoaderConfig
    fp16_out: bool = False
    prefetch_queue_depth: int = 4
    file_suffix: str = ""
    load_other_player: bool = False

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0):
        if "amp" in config.trainer:
            config.data[idx].dataset.args["fp16_out"] = True
        return super().from_config(config, idx)

    def _make_source(self, split: Split) -> DaliFolderDataset:
        loader = self.train_loader if split is Split.TRAIN else self.val_loader
        pipe_kwargs = loader.pipe_kwargs()
        del pipe_kwargs["prefetch_queue_depth"]  # Use config specific key
        source = self.init_auto_filter(
            DaliFolderDataset, path=self.basepath, split=split, **pipe_kwargs
        )
        return source

    def _get_size(self, split: Split):
        inst = self._make_source(split)
        inst._post_init()
        return inst.num_iterations * inst.batch_size

    def get_dataloader(self, split: Split) -> Any:
        loader = self.train_loader if split is Split.TRAIN else self.val_loader
        pipeline = sc2_data_pipeline(
            source=self._make_source(split),
            source_prefetch=self.prefetch_queue_depth,
            keys=self.features,
            fp16_out=self.fp16_out,
            **loader.pipe_kwargs(),
        )
        size = self._get_size(split)
        return loader.get_instance(pipeline, output_map=self.features, size=size)


class DaliReplayClipDataset(DALIExternalSource):
    """Data source for loading clips from a replay file"""

    def __init__(
        self,
        sampler_cfg: ModuleInitConfig,
        start_min: float,
        end_min: float,
        step_sec: float,
        clip_len: int,
        features: list[str],
        batch_size: int,
        shard_id: int,
        num_shards: int,
        random_shuffle: bool,
        yields_batch: bool = False,
        metadata: bool = False,
        minimap_layers: list[str] | None = None,
        valid_clip_file: Path | None = None,
    ) -> None:
        super().__init__(
            batch_size,
            shard_id,
            num_shards,
            random_shuffle,
            yields_batch,
        )
        self.sampler_cfg = sampler_cfg
        self.sampler: ReplaySampler | None = None
        self.start_step = _min_to_game_step(start_min)
        self.end_step = _min_to_game_step(end_min)
        self.step_size = int(step_sec * 22.4)
        self.clip_len = clip_len + 1  # Need to yield frame after history
        self.metadata = metadata
        self.features = features
        self.valid_clip_file = valid_clip_file

        self.db_handle: ReplayDatabase | None = None
        self.parser: ReplayParser | None = None
        self.minimap_layers = minimap_layers
        self.yields_batch = yields_batch
        self.valid_indices: None | np.ndarray = None

    def _post_init(self):
        self.sampler = SAMPLER_REGISTRY[self.sampler_cfg.type](**self.sampler_cfg.args)
        self.db_handle, self.parser = get_database_and_parser(
            parse_units="units" in self.features,
            parse_minimaps="minimaps" in self.features,
        )
        if self.minimap_layers is not None:
            self.parser.setMinimapFeatures(self.minimap_layers)
        set_replay_database_logger_level(spdlog_lvl.warn)
        return super()._post_init()

    def __len__(self) -> int:
        assert self.sampler is not None, "Must be _post_init()"
        return len(self.sampler)

    def sample_start_time(self, game_steps: list[int]) -> int:
        """Sample a start time that is within self.timepoints and the current replay"""
        start = max(self.start_step, game_steps[0])
        end = min(self.end_step, game_steps[-1] - self.clip_len)
        return random.randint(start, end)

    def get_sequence_no_mask_file(self):
        """
        Randomly sample a start time from the replay to make the sequence,
        will throw after 100 attempts if a valid sequence can't be found
        """
        assert self.parser is not None
        sample_indices = torch.full([self.clip_len], -1, dtype=torch.int32)
        attempts = 0
        while (sample_indices == -1).any():
            sample_indices = self.get_sample_indices_from_start(
                self.sample_start_time(self.parser.data.gameStep)
            )
            attempts += 1
            if attempts > 100:
                raise RuntimeError(
                    "Maximum iteration attempt exceeded for replay: "
                    f"{self.parser.info.replayHash}, {self.parser.info.playerId}"
                )
        return sample_indices

    def load_valid_indices(self, shuffle: bool = True):
        """
        Read the valid sequence file and find the replay hash and player id
        """
        assert self.valid_clip_file is not None
        assert self.parser is not None
        filters = [
            ("replayHash", "==", self.parser.info.replayHash),
            ("playerId", "==", self.parser.info.playerId),
        ]
        valid_data = pd.read_parquet(self.valid_clip_file, filters=filters)
        if valid_data.size == 0:
            raise KeyError(
                f"Can't find replayHash {self.parser.info.replayHash} and "
                f"playerId {self.parser.info.playerId} in {self.valid_clip_file}"
            )
        mask_data = np.frombuffer(valid_data["validMask"].iat[0].encode("utf-8"), "i1")
        self.valid_indices = np.argwhere(mask_data == ord("1"))
        if self.valid_indices.size == 0:
            raise ValueError(
                f"Literally no valid sequences in {self.parser.info.replayHash}, "
                f"{self.parser.info.playerId}"
            )
        if shuffle:
            np.random.shuffle(self.valid_indices)

    def get_sequence_with_mask_file(self, offset: int, retry_depth: int):
        """Get the valid sequence with an offset from the randomly shuffled set"""
        assert self.valid_indices is not None
        if not self.random_shuffle:  # consistently sample ~evenly along the indices
            offset *= max(len(self.valid_indices) // self.batch_size, 1)

        sample_indices = self.get_sample_indices_from_start(
            self.parser.data.gameStep[
                self.valid_indices[offset % len(self.valid_indices)].item()
            ]
        )
        if (sample_indices == -1).any():
            if retry_depth > 128:  # You gotta be kidding me
                raise RuntimeError(
                    f"Got invalid sample {self.valid_indices[offset]} from mask at "
                    f"{self.parser.info.replayHash}, {self.parser.info.playerId}"
                )
            return self.get_sequence_with_mask_file(offset + 1, retry_depth + 1)
        return sample_indices

    def get_sample_indices_from_start(self, start_idx: int):
        """Get the replay indices to create the sequence"""
        end_idx = start_idx + self.clip_len * self.step_size
        sample_indices = find_closest_indices(
            self.parser.data.gameStep,
            range(start_idx, end_idx, self.step_size),
        )
        return sample_indices

    def get_sample_indices(self, offset: int = 0):
        """Get the indices of the replay to sample"""
        return (
            self.get_sequence_no_mask_file()
            if self.valid_clip_file is None
            else self.get_sequence_with_mask_file(offset, 0)
        )

    def process_replay(self, sample_indices: Tensor):
        """Load self.features from numpy file at sample_indices

        Args:
            sample_indicies (Tensor): indices to sample from the replay

        Returns
            list[np.ndarray]: Sampled data from the replay in order of self.features
        """
        assert self.parser is not None
        outputs_list = {k: [] for k in self.features}

        for idx in sample_indices:
            sample = self.parser.sample_all(int(idx.item()))
            for k in outputs_list:
                outputs_list[k].append(sample[k])

        outputs = [np.stack(outputs_list[k]) for k in outputs_list]

        if self.metadata:
            _str = self.parser.info.replayHash + str(self.parser.info.playerId)
            outputs.append(np.array([ord(c) for c in _str], dtype=np.uint8))

        return outputs

    def get_data(self, index: int):
        assert self.sampler is not None
        assert self.db_handle is not None
        assert self.parser is not None
        # LSP doesn't recognise 'assert all(m is not None for m in ...)'
        replay_file, replay_idx = self.sampler.sample(index)
        if not self.db_handle.load(replay_file):
            raise FileNotFoundError(replay_file)
        replay_data = self.db_handle.getEntry(replay_idx)
        self.parser.parse_replay(replay_data)

        if self.valid_clip_file is not None:
            self.load_valid_indices(shuffle=self.random_shuffle)

        if self.yields_batch:
            samples = [
                self.process_replay(self.get_sample_indices(i))
                for i in range(self.batch_size)
            ]
            # Batch each of the properties
            outputs = [
                np.stack([s[idx] for s in samples]) for idx in range(len(samples[0]))
            ]
        else:
            outputs = self.process_replay(self.get_sample_indices())
        return outputs


@dataclass
@DATASET_REGISTRY.register_module("dali-replay-clip")
class DaliReplayClipConfig(SC2SamplerCfg):
    """Dataset configuration for loading clips from a replay file"""

    train_loader: DaliLoaderConfig
    val_loader: DaliLoaderConfig
    start_min: float = 0
    end_min: float = 120
    step_sec: float = 3
    clip_len: int = 9
    fp16_out: bool = False
    yields_batch: bool = False
    prefetch_queue_depth: int = 4

    # Precalculated valid start indices of clip to yield, this is calculated from the
    # basepath and the filename is calculated replay_mask_{int(step_sec*22.4)}_{clip_len+1}
    # the sampler configuration for this experiment should be strictly equal or a subset
    # of the sampler configuration used to generate the clip file
    valid_clip_file: Path | None = None

    precalculated_clips: bool = False

    minimap_ch_names: list[str] | None = field(init=False)

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0):
        if "amp" in config.trainer:
            config.dataset[idx].args["fp16_out"] = True
        return super().from_config(config, idx)

    def __post_init__(self):
        super().__post_init__()
        # Precalculated sequence should be in the root directory of the data
        # self.valid_clip_file is deprecated in favour of just a flag that enables it
        if self.valid_clip_file is not None or self.precalculated_clips:
            self.valid_clip_file = self.basepath
            if not isinstance(self.valid_clip_file, Path):
                self.valid_clip_file = Path(self.valid_clip_file)
            self.valid_clip_file /= (
                f"replay_mask_{int(self.step_sec*22.4)}_{self.clip_len+1}.parquet"
            )
            if not self.valid_clip_file.exists():
                raise FileNotFoundError(
                    f"Valid clip file not found: {self.valid_clip_file}"
                )

    def _make_source(self, split: Split) -> DaliReplayClipDataset:
        loader = self.train_loader if split is Split.TRAIN else self.val_loader
        sampler_cfg = deepcopy(self.sampler_cfg)
        sampler_cfg.args["is_train"] = split is Split.TRAIN
        sampler_cfg.args["train_ratio"] = self.train_ratio
        sampler_cfg.args["replays_path"] = self.basepath
        source = self.init_auto_filter(
            DaliReplayClipDataset, sampler_cfg=sampler_cfg, **loader.pipe_kwargs()
        )
        return source

    def _get_size(self, split: Split):
        inst = self._make_source(split)
        inst._post_init()
        return inst.num_iterations * inst.batch_size

    def get_dataloader(self, split: Split) -> Any:
        loader = self.train_loader if split is Split.TRAIN else self.val_loader
        features = deepcopy(self.features)
        if self.metadata:
            features.append("metadata")

        pipeline = sc2_data_pipeline(
            source=self._make_source(split),
            source_prefetch=self.prefetch_queue_depth,
            keys=features,
            fp16_out=self.fp16_out,
            **loader.pipe_kwargs(),
        )
        size = self._get_size(split)

        return loader.get_instance(pipeline, output_map=features, size=size)


@dataclass
class FeatureType:
    dtype: DALIDataType
    ndim: int
    layout: str
    should_cast: bool


_DTYPES = {
    "win": FeatureType(DALIDataType.FLOAT, 0, "", False),
    "valid": FeatureType(DALIDataType.BOOL, 1, "", False),
    "metadata": FeatureType(DALIDataType.UINT8, 1, "", False),
    "scalars": FeatureType(DALIDataType.FLOAT, 2, "", False),
    "minimaps": FeatureType(DALIDataType.FLOAT, 4, "FCHW", True),
}


def apply_minimap_augs(minimaps: DataNode, augs: list[ModuleInitConfig]):
    """Apply list of augmentations to minimaps"""
    for aug in augs:
        minimaps = DALI_AUGMENTATIONS[aug.type](minimaps, **aug.args)
    return minimaps


@pipeline_def(py_start_method="spawn", enable_conditionals=True)
def sc2_data_pipeline(
    shard_id: int,
    num_shards: int,
    random_shuffle: bool,
    source: DALIExternalSource,
    source_prefetch: int,
    keys: list[str],
    fp16_out: bool,
    augmentations: list[ModuleInitConfig],
):
    """Create pipeline"""
    outputs = fn.external_source(
        source=source,
        num_outputs=len(keys),
        parallel=True,
        batch=source.yields_batch,
        batch_info=source.yields_batch,
        dtype=[_DTYPES[k].dtype for k in keys],
        ndim=[_DTYPES[k].ndim for k in keys],
        layout=[_DTYPES[k].layout for k in keys],
        prefetch_queue_depth=source_prefetch,
    )

    def transform(data: DataNode, key: str):
        """Move data to gpu and cast to fp16 if enabled"""
        data = data.gpu()
        if (
            _DTYPES[key].dtype == DALIDataType.FLOAT
            and _DTYPES[key].should_cast
            and fp16_out
        ):
            return fn.cast(data, dtype=DALIDataType.FLOAT16)
        if key == "metadata":
            return fn.pad(data, fill_value=0)
        return data

    if len(augmentations) != 0:
        minimap_idx = keys.index("minimaps")
        outputs[minimap_idx] = apply_minimap_augs(
            outputs[minimap_idx].gpu(), augmentations
        )

    outputs = [transform(o, k) for o, k in zip(outputs, keys)]

    return tuple(outputs)
