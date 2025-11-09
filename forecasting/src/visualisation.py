import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .eval_helpers import metadata_to_str
from .minimap.common import MinimapTarget
from .stats import MinimapSoftIoU


def create_score_frame(pred: Tensor, target: Tensor) -> np.ndarray:
    """Create an rgb frame showing the ground truth and predicted frames"""
    ctor_kwargs = {"device": pred.device, "dtype": torch.uint8}
    bgr_frame = torch.full([*pred.shape, 3], 255, **ctor_kwargs)

    # Red for false negatives
    bgr_frame[target == 1] = torch.tensor((0, 0, 200), **ctor_kwargs)

    mask = target == 0
    rg = (255 * pred).to(torch.uint8)[mask]
    b = torch.zeros_like(rg)
    # subtract rg from prediction
    bgr_frame[mask] -= torch.stack([b, rg, rg], dim=-1)

    tp_thresh = 0.5
    # Blue for false positives
    bgr_frame[pred > tp_thresh] = torch.tensor((255, 0, 0), **ctor_kwargs)
    # Green for true positives
    mask = (pred > tp_thresh) & (target == 1)
    rb = ((1 - pred) * 255).to(torch.uint8)[mask]
    g = 200 * torch.ones_like(rb)
    bgr_frame[mask] = torch.stack([rb, g, rb], dim=-1)

    bgr_frame = cv2.resize(
        bgr_frame.cpu().numpy(), (480, 480), interpolation=cv2.INTER_NEAREST
    )

    soft_iou = MinimapSoftIoU.calculate_soft_iou(pred, target)
    cv2.putText(
        bgr_frame, f"{soft_iou=:.2f}", (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0)
    )

    return bgr_frame


def write_minimaps(
    pred: Tensor,
    target: Tensor,
    last: Tensor,
    folder: Path,
    prefix: str,
    out_type: MinimapTarget,
    postfix: str = "",
):
    """Write visualization results to disk"""
    predFolder = folder / "pred"
    dataFolder = folder / "data"

    for idx, name in enumerate(MinimapTarget.names(out_type)):
        cv2.imwrite(
            str(predFolder / f"{prefix}_{name}{postfix}.png"),
            (255 * (1 - pred[idx])).to(torch.uint8).cpu().numpy(),
        )
        cv2.imwrite(
            str(dataFolder / f"{prefix}_{name}{postfix}.png"),
            (255 * (1 - target[idx])).to(torch.uint8).cpu().numpy(),
        )

        cv2.imwrite(
            str(predFolder / f"{prefix}_diff_{name}{postfix}.png"),
            create_score_frame(pred[idx], target[idx]),
        )
        cv2.imwrite(
            str(dataFolder / f"{prefix}_diff_{name}{postfix}.png"),
            create_score_frame(last[idx], target[idx]),
        )


def get_upper_left_coord(idx: int, grid_size: int, tile_size: int):
    """
    Get pixel coordinates of upper left of tile at index
    Returns (h,w)
    """
    row, col = divmod(idx, grid_size)
    assert row <= grid_size and col <= grid_size, f"Tile {idx=} out of bounds"
    return row * tile_size, col * tile_size


def write_tiled_sequence(
    data: Tensor, end_idx: int, seq_len: int, folder: Path, prefix: str
):
    """Write the historicalsequence and target frame"""
    dataFolder = folder / "data"
    im_size = 1024
    grid_size = int(math.ceil(math.sqrt(seq_len)))
    tile_size = im_size // grid_size
    data = (255 * (1 - data)).to(torch.uint8)
    for ch_idx, name in enumerate(["self", "enemy"]):
        base_image = torch.full(
            [im_size, im_size], 255, dtype=torch.uint8, device=data.device
        )
        for idx, t_idx in enumerate(range(end_idx + 1 - seq_len, end_idx + 1)):
            data_time: Tensor = F.interpolate(
                data[t_idx, ch_idx, None, None],
                size=(tile_size, tile_size),
                mode="nearest",
            )
            px_y, px_x = get_upper_left_coord(idx, grid_size, tile_size)
            base_image[px_y : px_y + tile_size, px_x : px_x + tile_size] = data_time[
                0, 0
            ]
        cv2.imwrite(
            str(dataFolder / f"{prefix}_{name}_seq.png"),
            base_image.cpu().numpy(),
        )


def write_gradient_sequence(
    data: Tensor,
    end_idx: int,
    seq_len: int,
    folder: Path,
    prefix: str,
    layers: MinimapTarget,
):
    """Display a sequence of frames as a ghost trail to indicate motion of units over time"""
    px_vals = torch.linspace(200, 0, seq_len, dtype=torch.uint8, device=data.device)
    dataFolder = folder / "data"
    data = data.to(torch.bool)
    for ch_idx, name in enumerate(MinimapTarget.names(layers)):
        base_image = torch.full(
            data.shape[-2:], 255, dtype=torch.uint8, device=data.device
        )
        for px_val, t_idx in zip(px_vals, range(end_idx - seq_len + 1, end_idx + 1)):
            base_image[data[t_idx, ch_idx]] = px_val
        cv2.imwrite(
            str(dataFolder / f"{prefix}_{name}_seq.png"), base_image.cpu().numpy()
        )


def write_minimap_forecast_results(
    preds: Tensor,
    data: dict[str, Tensor],
    outdir: Path,
    timepoints: list[float] | None,
    out_type: MinimapTarget,
    ch_names: list[str],
):
    """
    Write minimap forecast predictions for konduct review image viewer.
    The history sequence length and thus association between prediction and
    ground truth time indices is inferred from the time dimension of the
    prediction which is full_sequence - history_length.
    """
    predFolder = outdir / "pred"
    predFolder.mkdir(exist_ok=True)
    dataFolder = outdir / "data"
    dataFolder.mkdir(exist_ok=True)

    metadata = metadata_to_str(data["metadata"])
    if all(m == metadata[0] for m in metadata):  # If batch from same replay append idx
        metadata = [m + str(i) for i, m in enumerate(metadata)]

    layer_idxs = [ch_names.index(n) for n in MinimapTarget.names(out_type)]
    targets = data["minimaps"][:, :, layer_idxs]
    history_len = targets.shape[1] - preds.shape[1]

    for bidx in range(preds.shape[0]):
        prefix = metadata[bidx]
        write_gradient_sequence(
            targets[bidx], history_len, history_len + 1, outdir, prefix, out_type
        )

        for t_idx in range(preds.shape[1]):
            write_minimaps(
                preds[bidx, t_idx],
                targets[bidx, history_len + t_idx],
                targets[bidx, history_len - 1],
                outdir,
                prefix,
                out_type,
                "" if timepoints is None else f"_{timepoints[t_idx]}",
            )

    with open(outdir / "samples.txt", "a") as f:
        f.writelines(m + "\n" for m in metadata)
