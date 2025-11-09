"""General testing of dataloading"""

import os
from pathlib import Path

from sc2_serializer.sampler import SQLSampler
from src.data.torch_dataset import SC2ReplayOutcome, TimeRange


def test_iterating():
    sampler = SQLSampler(
        "$ENV:gamedata.db",
        Path(os.environ["DATAPATH"]),
        [
            "game_length > 6720",
            "read_success = 1",
            "parse_success = 1",
            "number_game_step > 1024",
            "playerAPM > 100",
        ],
        train_ratio=0.8,
        is_train=True,
    )
    dataset = SC2ReplayOutcome(
        sampler,
        TimeRange(2, 30, 0.5),
        features=["minimaps", "scalars"],
    )
    for idx, sample in enumerate(dataset):
        if idx == 100:
            break
        print(sample["win"])


if __name__ == "__main__":
    test_iterating()
