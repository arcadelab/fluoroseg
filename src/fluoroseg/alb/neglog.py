import numpy as np
import albumentations as A


def neglog_fn(images: np.ndarray, epsilon: float = 0.001) -> np.ndarray:
    """Take the negative log transform of an intensity image.

    Args:
        image (np.ndarray): [H,W,C] array of intensity images.
        epsilon (float, optional): positive offset from 0 before taking the logarithm.

    Returns:
        np.ndarray: the image or images after a negative log transform.
    """

    # shift image to avoid invalid values
    images += images.min(axis=(0, 1), keepdims=True) + epsilon

    # negative log transform
    images = -np.log(images)

    return images


def neglog(epsilon: float = 0.001) -> A.Lambda:
    """Take the negative log transform of an intensity image.

    Args:
    """

    def f_image(images: np.ndarray, **kwargs) -> np.ndarray:
        return neglog_fn(images, epsilon)

    def f_id(x, **kwargs):
        return x

    return A.Lambda(
        image=f_image,
        mask=f_id,
        keypoints=f_id,
        bboxes=f_id,
        name="neglog",
    )
