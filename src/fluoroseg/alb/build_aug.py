import numpy as np
import albumentations as A
from PIL import Image
import logging

from .dropout import CoarseDropout, Dropout
from .gaussian_contrast import gaussian_contrast
from .window import window, adjustable_window, mixture_window
from .neglog import neglog

log = logging.getLogger(__name__)


def build_aug_dcm(
    image_size: int | tuple[int, int], bbox: bool = False, mixture: bool = False
) -> A.Compose:
    """Build transformation pipeline for real images, from a raw pixel array."""
    if isinstance(image_size, int):
        h = w = image_size
    else:
        h, w = image_size

    compose_kwargs = dict()
    if bbox:
        compose_kwargs["bbox_params"] = A.BboxParams(
            format="albumentations",  # xyxy, normalized to [0, 1]
            min_area=0,
            min_visibility=0,
            label_fields=["bbox_indices"],
        )

    if mixture:
        return A.Compose(
            [
                mixture_window(keep_original=True, model="kmeans"),
                A.LongestMaxSize(max_size=max(h, w), always_apply=True),
            ],
            **compose_kwargs,
        )

    else:
        return A.Compose(
            [
                window(0.0, 1.0, convert=False),
                A.LongestMaxSize(max_size=max(h, w), always_apply=True),
            ],
            **compose_kwargs,
        )


def build_aug_cxr(image_size: int | tuple[int, int], bbox: bool = False) -> A.Compose:
    """Build transformation pipeline for real images, from a raw pixel array."""
    if isinstance(image_size, int):
        h = w = image_size
    else:
        h, w = image_size

    compose_kwargs = dict()
    if bbox:
        compose_kwargs["bbox_params"] = A.BboxParams(
            format="albumentations",  # xyxy, normalized to [0, 1]
            min_area=0,
            min_visibility=0,
            label_fields=["bbox_indices"],
        )
    return A.Compose(
        [
            # neglog(),
            window(0.0, 1.0, convert=False),
            # mixture_window(keep_original=True, model="kmeans"),
            # clahe,
            # adjustable_window(0.1, 0.9, y_lo=0.1, y_hi=0.9, quantile=True),
            # A.InvertImg(always_apply=True),
            # A.InvertImg(always_apply=True),
            A.Resize(h, w, always_apply=True),
        ],
        **compose_kwargs,
    )


# Mean
def build_augmentation(
    image_size: tuple[int, int], train: bool = True, bbox: bool = False
) -> A.Compose:
    """

    Args:
        image_size: The size to resize images to. If None, no resizing is done.
        train: Whether to build an augmentation for training or testing. If True, the wrapped
            function is used to get the training augmentations.
        bbox: Whether the dataset contains bounding boxes.

    """
    h, w = image_size

    compose_kwargs = dict()
    if bbox:
        compose_kwargs.update(
            bbox_params=A.BboxParams(
                format="albumentations",  # xyxy, normalized to [0, 1]
                min_area=0,
                min_visibility=0,
                label_fields=["bbox_indices"],
            ),
        )

    if not train:
        # For the simulated validation set
        return A.Compose(
            [
                neglog(),
                A.InvertImg(always_apply=True),
                mixture_window(keep_original=True, model="kmeans"),
                A.InvertImg(always_apply=True),
                A.Resize(h, w, always_apply=True),
            ],
            **compose_kwargs,
        )

    clahe = A.Sequential(
        [
            A.FromFloat(max_value=255, dtype="uint8", always_apply=True),
            A.CLAHE(clip_limit=(4, 6), tile_grid_size=(8, 12), always_apply=True),
            A.ToFloat(max_value=255, always_apply=True),
        ],
        p=1,
    )

    intensity_transforms = A.SomeOf(
        [
            A.OneOf(
                [
                    A.GaussianBlur((3, 5)),
                    A.MotionBlur(blur_limit=(3, 5)),
                    A.MedianBlur(blur_limit=5),
                ],
            ),
            A.OneOf(
                [
                    A.Sharpen(alpha=(0.2, 0.5)),
                    A.Emboss(alpha=(0.2, 0.5)),
                ],
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(0, 0), contrast_limit=(-2, 2)
            ),  # Fine
            gaussian_contrast(alpha=(0.6, 1.4), sigma=(0.1, 0.5), max_value=1),
            A.OneOf(
                [
                    A.RandomFog(
                        fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08
                    ),
                    A.RandomRain(rain_type="drizzle", drop_width=1, blur_value=1),
                ],
            ),
            A.OneOf(
                [
                    Dropout(dropout_prob=0.05),
                    CoarseDropout(
                        max_holes=20,
                        max_height=64,
                        max_width=64,
                        min_holes=8,
                        min_height=8,
                        min_width=8,
                        p=0.8,
                    ),
                ],
                p=3,
            ),
            A.ChannelShuffle(p=0.5),
        ],
        n=np.random.randint(2, 5),
        replace=False,
    )

    return A.Compose(
        [
            neglog(),
            A.InvertImg(always_apply=True),
            mixture_window(keep_original=True, model="kmeans"),
            A.InvertImg(p=0.5),
            A.OneOf(
                [clahe, intensity_transforms],
            ),
            A.Resize(h, w, always_apply=True),
        ],
        **compose_kwargs,
    )
