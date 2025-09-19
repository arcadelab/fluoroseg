from typing import Optional, Tuple
import numpy as np
import torch


from pycocotools import mask as mask_utils
from . import coco_utils


def idx_to_points(rle: dict[str, list[int]], idx: np.ndarray) -> Tuple[int, int]:
    """Convert an index into an rle mask to a point in image space.

    Args:
        rle (dict[str, list[int]]): The RLE mask, with size in (H,W) format.
        idx (int): The index into the flattened mask.

    Returns:
        Tuple[int, int]: The point corresponding to the index, in (x,y) (columns, rows).
    """

    # [off, on, off, on, ...]
    counts = np.array(rle["counts"])
    indices = np.cumsum(counts)

    # Cumulative sum of the counts of just the mask
    sum_along_mask = np.concatenate([[0], np.cumsum(counts[1::2])])
    insertion_idx_in_mask = np.searchsorted(sum_along_mask, idx, side="right") - 1

    insertion_sum = sum_along_mask[insertion_idx_in_mask]
    offset = idx - insertion_sum  # how much the point needs to be offset from the insetion_idx
    insertion_idx_in_indices = 2 * insertion_idx_in_mask
    idx_in_image = indices[insertion_idx_in_indices] + offset
    y, x = np.unravel_index(idx_in_image, (rle["size"][0], rle["size"][1]))
    return np.stack([y, x], axis=1)


def sample_point(mask: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    """Sample a point from the given mask.

    Args:
        mask: The mask to sample from.

    Returns:
        The coordinates of the sampled point (in x,y).
    """
    # Sample the point uniformly from the area of the mask. Then figure out which point that is
    # based on the RLE.
    n_ = 1 if n is None else n

    mask = mask.astype(np.uint8)
    compressed_rle = mask_utils.encode(np.asfortranarray(mask))
    area = mask_utils.area(compressed_rle)
    rle = coco_utils.mask_to_rle(mask)

    indices = np.random.randint(area, size=n_)
    points = idx_to_points(rle, indices)

    if n is None:
        return points[0]
    return points


def sample_point_torch(mask: torch.Tensor) -> torch.Tensor:
    """Sample a point from the given mask.

    Args:
        mask: The mask to sample from.

    Returns:
        The coordinates of the sampled point (in x,y).
    """
    # Sample the point uniformly from the area of the mask. Then figure out which point that is
    # based on the RLE.
    coords = torch.nonzero(mask)  # (n, 2)
    idx = torch.randint(high=coords.shape[0], size=(1,))
    return coords[idx][0]


def sample_box(
    box: np.ndarray, original_size: tuple[int, int], max_shift_px: int = 20
) -> np.ndarray:
    """Sample boxes.

    Args:
        box: The ground truth box to sample from (xyxy) in range [0,1].

    Returns:
        box: The sampled box (in xyxy).
    """

    max_x_shift = max_shift_px / original_size[1]
    max_y_shift = max_shift_px / original_size[0]
    min_shift = [-max_x_shift, -max_y_shift, -max_x_shift, -max_y_shift]
    max_shift = [max_x_shift, max_y_shift, max_x_shift, max_y_shift]

    shift = np.random.normal(0, 0.1, size=4)
    shift = np.clip(shift, min_shift, max_shift)
    box = box + shift
    box = np.clip(box, 0, 1)
    return box


def sample_box_torch(
    box: torch.Tensor, original_size: tuple[int, int], max_shift_px: int = 20
) -> torch.Tensor:
    """Sample boxes."""

    max_x_shift = max_shift_px / original_size[1]
    max_y_shift = max_shift_px / original_size[0]
    min_shift = torch.tensor([-max_x_shift, -max_y_shift, -max_x_shift, -max_y_shift])
    max_shift = torch.tensor([max_x_shift, max_y_shift, max_x_shift, max_y_shift])

    shift = torch.normal(0, 0.1, size=(4,))
    shift = torch.clamp(shift, min_shift, max_shift)
    shift = shift.to(box.device)
    box = box + shift
    box = torch.clamp(box, 0, 1)
    return box
