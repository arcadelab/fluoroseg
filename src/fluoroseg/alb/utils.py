from typing import TypeVar
import numpy as np


T = TypeVar("T", bound=np.number)
SampleArg = tuple[T, T] | T


def sample(x: SampleArg) -> T:
    return np.random.uniform(x[0], x[1]) if isinstance(x, tuple) else x
