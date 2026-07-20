"""Compatibility aliases for checkpoints written with NumPy 2.x."""

from __future__ import annotations

import sys

import numpy as np


def install_numpy_pickle_aliases() -> None:
    """Allow NumPy 1.x to unpickle arrays stored under `numpy._core`."""
    if not hasattr(np, "_core"):
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
