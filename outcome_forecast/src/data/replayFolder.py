from dataclasses import dataclass
from pathlib import Path

import torch
from konductor.data import DATASET_REGISTRY

from .baseDataset import SC2ReplayBase, SC2ReplayConfigBase

from .utils import upper_bound


class SC2Replay(SC2ReplayBase):
    """Run over all replays in folder of databases"""

    def load_files(self, basepath: Path):
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
        self.init_split_params(int(self._accumulated_replays[-1].item()))

    def __getitem__(self, index: int):
        file_index = upper_bound(self._accumulated_replays, self.start_idx + index)
        db_index = index - int(self._accumulated_replays[file_index].item())
        self.load_to_parser(self.replays[file_index], db_index)
        return self.process_replay()


@dataclass
@DATASET_REGISTRY.register_module("sc2-replay")
class SC2ReplayConfig(SC2ReplayConfigBase):
    def get_class(self):
        return SC2Replay
