from typing import Optional, Any
from pathlib import Path
import numpy as np
import json


def jsonable(obj: Any):
    """Convert obj to a JSON-ready container or object.
    Args:
        obj ([type]):
    """
    if obj is None:
        return "null"
    elif isinstance(obj, (str, float, int, complex)):
        return obj
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(map(jsonable, obj))
    elif isinstance(obj, dict):
        return dict(jsonable(list(obj.items())))
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "__array__"):
        return np.array(obj).tolist()
    else:
        raise ValueError(f"Unknown type for JSON: {type(obj)}")


def save_json(path: str, obj: Any, indent: Optional[int] = None):
    obj = jsonable(obj)
    with open(path, "w") as file:
        json.dump(obj, file, sort_keys=True, indent=indent)


def load_json(path: str) -> Any:
    with open(path, "r") as file:
        out = json.load(file)
    return out
