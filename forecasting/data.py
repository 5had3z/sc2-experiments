#!/usr/bin/env python3
"""Data preprocessing utilities"""

import os
import concurrent.futures as fut
from pathlib import Path
from typing import Annotated, Callable

import torch
import numpy as np
import typer
import yaml
import pandas as pd
from pyarrow import parquet as pq
from konductor.data import make_from_init_config, Split
from konductor.init import DatasetInitConfig
from konductor.registry import Registry
from konductor.utilities.pbar import IntervalPbar, LivePbar
from sc2_serializer import (
    set_replay_database_logger_level,
    spdlog_lvl,
    ReplayDataScalarOnlyDatabase,
)
from sc2_serializer.sampler import SQLSampler
from src.data.base_dataset import find_closest_indices
from src.utils import StrEnum
from torch import Tensor

try:
    from ffmpegcv import VideoWriter, FFmpegWriter
except ImportError:
    VideoWriter = None
    FFmpegWriter = None

app = typer.Typer()


def get_dataset_config_for_data_transform(config_file: Path, workers: int):
    """Returns configured dataset and original configuration dictionary"""
    with open(config_file, "r", encoding="utf-8") as f:
        loaded_dict = yaml.safe_load(f)
        if "dataset" in loaded_dict:  # Unwrap if normal experiment config
            loaded_dict = loaded_dict["dataset"]

    init_config = DatasetInitConfig.from_dict(loaded_dict)
    dataset_cfg = make_from_init_config(init_config)

    # Return metadata to save as filename
    dataset_cfg.metadata = True
    # Set dataloader workers
    dataset_cfg.train_loader.workers = workers
    dataset_cfg.val_loader.workers = workers
    # Enforce batch size of 1
    dataset_cfg.train_loader.batch_size = 1
    dataset_cfg.val_loader.batch_size = 1

    return dataset_cfg, loaded_dict


def save_dataset_configuration(config_dict: dict[str, object], outfolder: Path):
    """Saves configuration to outfolder for traceability"""
    with open(outfolder / "generation-config.yml", "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)


def make_pbar(total: int, desc: str, live: bool):
    """Make pbar live or fraction depending on flag"""
    pbar_type = LivePbar if live else IntervalPbar
    pbar_kwargs = {"total": total, "desc": desc}
    if not live:
        pbar_kwargs["fraction"] = 0.1
    return pbar_type(**pbar_kwargs)


def convert_to_numpy_files(outfolder: Path, dataloader, live: bool):
    """Run conversion and write to outfolder"""
    with make_pbar(len(dataloader), outfolder.stem, live) as pbar:
        for sample in dataloader:
            # Unwrap batch dim and change torch tensor to numpy array
            formatted: dict[str, str | np.ndarray] = {}
            for k, v in sample.items():
                v = v[0]
                formatted[k] = v if isinstance(v, str) else v.numpy()
            np.savez_compressed(outfolder / formatted["metadata"], **formatted)
            pbar.update(1)


def write_numpy_file_list(folder: Path):
    """Write newline separated filenames inside folder to
    parent directory as {folder.stem}-list.txt"""
    filelines = [f"{f.name}\n" for f in folder.iterdir()]
    with open(folder.parent / f"{folder.stem}-list.txt", "w", encoding="utf-8") as f:
        f.writelines(filelines)


@app.command()
def make_numpy_subset(
    config: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    workers: Annotated[int, typer.Option()] = 4,
    live: Annotated[bool, typer.Option(help="Use live pbar")] = False,
):
    """
    Make a subfolder dataset of numpy files from configuration
    for less expensive dataloading for training. When training with
    a small amount of data from a full replay, it is probably worth
    it to make a small subset of the required data, rather than loading
    all data and discarding the vast majority of it.

    Args:
        config (Path): Path to data configuration yaml
        output (Path): Root folder to save new dataset
        workers (int): Number of dataloader workers to use
    """
    dataset_cfg, loaded_dict = get_dataset_config_for_data_transform(config, workers)

    # Create output root and copy configuration for traceability
    output.mkdir(exist_ok=True)
    save_dataset_configuration(loaded_dict, output)

    for split in [Split.TRAIN, Split.VAL]:
        dataloader = dataset_cfg.get_dataloader(split)
        outsubfolder = output / split.name.lower()
        outsubfolder.mkdir(exist_ok=True)
        convert_to_numpy_files(outsubfolder, dataloader, live)
        write_numpy_file_list(outsubfolder)


WRITER_REGISTRY = Registry("video-writer")


@WRITER_REGISTRY.register_module()
def self_enemy_heightmap(minimap_seq: Tensor, writer: FFmpegWriter):
    """Write video with three channels [self, enemy, heightmap]"""
    heightmap = 0
    self_idx = 1
    enemy_idx = -1
    for minimap in minimap_seq:
        frame_data = np.zeros([writer.height, writer.width, 3], dtype=np.uint8)
        frame_data[..., 0] = (255 * minimap[self_idx]).to(torch.uint8).cpu().numpy()
        frame_data[..., 1] = (255 * minimap[enemy_idx]).to(torch.uint8).cpu().numpy()
        frame_data[..., 2] = minimap[heightmap].to(torch.uint8).cpu().numpy()
        writer.write(frame_data)


WriterType = StrEnum("WriterType", list(WRITER_REGISTRY.module_dict))


WriterFn = Callable[[Tensor, FFmpegWriter], None]


def convert_minimaps_to_videos(
    outfolder: Path, dataloader, live: bool, writer: WriterFn
):
    """Write each minimap sequence as a video file"""
    with make_pbar(len(dataloader), outfolder.stem, live) as pbar:
        for sample_ in dataloader:
            sample = sample_[0] if isinstance(sample_, list) else sample_
            outpath: Path = outfolder / (sample["metadata"][0] + ".mp4")
            sz = sample["minimaps"].shape[-2:]
            with VideoWriter(outpath, "h264", 6, sz) as w:
                writer(sample["minimaps"][0], w)
            pbar.update(1)


@app.command()
def make_minimap_videos(
    config: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    writer: Annotated[WriterType, typer.Option()],
    workers: Annotated[int, typer.Option()] = 4,
    live: Annotated[bool, typer.Option(help="Use live pbar")] = False,
):
    """
    Reading a few frames from a SC2Replay is a bit wasteful, perhaps
    using the native DALI videoreader could perhaps be faster?
    Note: Data becomes huge, sampling small batch from same game works fine.
    """
    dataset_cfg, loaded_dict = get_dataset_config_for_data_transform(config, workers)

    output.mkdir(exist_ok=True)
    save_dataset_configuration(loaded_dict, output)

    writer_func: WriterFn = WRITER_REGISTRY[writer]

    for split in [Split.TRAIN, Split.VAL]:
        dataloader = dataset_cfg.get_dataloader(split)
        outsubfolder = output / split.name.lower()
        outsubfolder.mkdir(exist_ok=True)
        convert_minimaps_to_videos(outsubfolder, dataloader, live, writer_func)


def get_valid_start_mask(game_step: list[int], stride: int, length: int):
    """Gather all the start indices of length with stride in a replay"""
    valid_starts = np.zeros(len(game_step), dtype=bool)
    for start_idx, start_val in enumerate(game_step):
        end_val = start_val + stride * length
        step_indices = find_closest_indices(
            game_step[start_idx:], range(start_val, end_val, stride)
        )
        valid_starts[start_idx] = (step_indices != -1).all()
    return valid_starts


def get_sampler_config(conf_file):
    """Extract sql sampler configuration from file"""
    with open(conf_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Unwrap nested configuration if necessary
    if "dataset" in config:
        config = config["dataset"][0]["args"]
    if "sampler_cfg" in config:
        assert config["sampler_cfg"]["type"] == "sql"
        config = config["sampler_cfg"]["args"]

    return config


def sampler_from_config(conf_file: Path) -> SQLSampler:
    """Create SQLSampler from config file"""
    config = get_sampler_config(conf_file)
    return SQLSampler(
        replays_path=Path(os.environ["DATAPATH"]),
        is_train=True,
        train_ratio=1.0,
        **config,
    )


def get_partition_range(total_len: int, part_idx: int, num_part: int):
    """Get the range to process corresponding to the partition index"""
    chunk_size = total_len // num_part
    start_idx = part_idx * chunk_size
    end_idx = total_len if part_idx + 1 == num_part else start_idx + chunk_size
    return range(start_idx, end_idx)


def gather_valid_stride_data(
    sampler: SQLSampler, stride: int, sequence_len: int, live: bool, indices: range
):
    """Gather replays from sampler"""
    print(f"Running over {indices.start} to {indices.stop} of {len(sampler)}")

    # Pandas won't accept this being a dataframe, so using dict of series first
    mask_data = {
        "replayHash": pd.Series(
            name="replayHash", index=indices, dtype=pd.StringDtype(storage="pyarrow")
        ),
        "validMask": pd.Series(
            name="validMask", index=indices, dtype=pd.StringDtype(storage="pyarrow")
        ),
        "playerId": pd.Series(name="validMask", index=indices, dtype=pd.UInt8Dtype()),
    }

    db = ReplayDataScalarOnlyDatabase()
    with make_pbar(len(indices), "Creating Masks", live) as pbar:
        for sample_idx in indices:
            path, sidx = sampler.sample(sample_idx)
            assert db.load(path), f"Failed to load {path}"
            replay = db.getEntry(sidx)
            valid_mask = get_valid_start_mask(
                replay.data.gameStep, stride, sequence_len
            )
            write_idx = sample_idx - indices.start
            mask_data["replayHash"].iat[write_idx] = replay.header.replayHash
            mask_data["playerId"].iat[write_idx] = replay.header.playerId
            mask_data["validMask"].iat[write_idx] = "".join(
                str(x.item()) for x in valid_mask.astype(np.uint8)
            )
            pbar.update(1)

    return pd.DataFrame(mask_data)


def run_valid_stride_creation(
    sampler: SQLSampler,
    output: Path,
    stride: int,
    sequence_len: int,
    indices: range,
    live: bool,
):
    """Run valid stride creation and write to file"""
    mask_data = gather_valid_stride_data(sampler, stride, sequence_len, live, indices)
    # If shard then add shard start idx to filename
    if indices.start != 0 or indices.stop != len(sampler):
        output = output.with_name(f"{output.stem}_{indices.start}.parquet")
    mask_data.to_parquet(output, compression=None)


def get_subsequences(total_len: int, n_sequences: int):
    """Get all the subsequence ranges for n_sequences"""
    return [get_partition_range(total_len, i, n_sequences) for i in range(n_sequences)]


@app.command()
def create_valid_stride(
    config: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    step_sec: Annotated[float, typer.Option()],
    sequence_len: Annotated[int, typer.Option()],
    workers: Annotated[int, typer.Option()] = 1,
    live: Annotated[bool, typer.Option(help="Use live pbar")] = False,
):
    """
    Find valid strides and write to file. This can be sampled directly rather than randomly
    sampling the replay until we find a good subsequence when dataloading.
    """
    stride = int(step_sec * 22.4)
    sampler = sampler_from_config(config)
    total_len = len(sampler)
    assert output.is_dir(), "Output should be folder"
    output /= f"replay_mask_{stride}_{sequence_len}.parquet"
    if workers > 1:
        with fut.ProcessPoolExecutor(workers) as ctx:
            work = [
                ctx.submit(
                    run_valid_stride_creation,
                    sampler,
                    output,
                    stride,
                    sequence_len,
                    part,
                    live=False,
                )
                for part in get_subsequences(total_len, workers)
            ]
            fut.wait(work)
        merge_valid_stride(output.parent, step_sec, sequence_len, clean=True)
    elif "POD_NAME" in os.environ:  # Get partition from k8s pod index and replicas
        part = get_partition_range(
            total_len,
            int(os.environ["POD_NAME"].split("-")[-1]),
            int(os.environ["REPLICAS"]),
        )
        run_valid_stride_creation(sampler, output, stride, sequence_len, part, live)
    else:  # Run over entire dataset on one thread
        run_valid_stride_creation(
            sampler, output, stride, sequence_len, range(total_len), live
        )

    # Log the configuration used to generate parquet file
    config_dict = get_sampler_config(config)
    with open(output.with_suffix(".yml"), "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)


@app.command()
def merge_valid_stride(
    path: Annotated[Path, typer.Option()],
    step_sec: Annotated[float, typer.Option()],
    sequence_len: Annotated[int, typer.Option()],
    clean: Annotated[bool, typer.Option(help="Remove shards when done")] = False,
):
    """Merges replay mask shards to single file"""
    filestem = f"replay_mask_{int(step_sec * 22.4)}_{sequence_len}"
    shards = list(path.glob(f"{filestem}_*.parquet"))
    if len(shards) == 0:
        raise FileNotFoundError(f"No shards matching {filestem} in {path}")

    # Sort by start index of shard
    shards = sorted(shards, key=lambda p: int(p.stem.split("_")[-1]))

    schema = pq.read_schema(shards[0])

    with pq.ParquetWriter(path / f"{filestem}.parquet", schema) as writer:
        with LivePbar(total=len(shards), desc="Merging files") as pbar:
            for shard in shards:
                data = pq.read_table(shard)
                writer.write_table(data)
                pbar.update(1)

    if clean:
        for f in shards:
            f.unlink()


@app.command()
def mask_analysis(pqfile: Path, invalid_thresh: int = 48):
    """Check how many valid subsequences exist in each replay"""
    data = pd.read_parquet(pqfile)
    intMask = data["validMasks"].map(
        lambda x: np.frombuffer(x.encode("utf-8"), "i1") - invalid_thresh
    )
    numValidPerReplay = intMask.map(np.sum)
    numInvalidReplays = numValidPerReplay[numValidPerReplay <= 0].sum()
    print(f"Number of invalid replays: {numInvalidReplays}")


if __name__ == "__main__":
    set_replay_database_logger_level(spdlog_lvl.warn)
    app()
