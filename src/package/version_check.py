from __future__ import annotations

"""Helpers to enforce consistent library versions across scripts."""

import warnings
from typing import Tuple

from sklearn import __version__ as sklearn_version

EXPECTED_SKLEARN: Tuple[int, int] = (1, 6)


def ensure_sklearn_version(expected: Tuple[int, int] = EXPECTED_SKLEARN, strict: bool = False) -> None:
    """Warn or raise if scikit-learn major/minor does not match the expected tuple."""
    try:
        major, minor, *_ = sklearn_version.split(".")
        current = (int(major), int(minor))
    except Exception:
        current = (-1, -1)

    if current != expected:
        msg = f"Expected scikit-learn {expected[0]}.{expected[1]}.x, found {sklearn_version}"
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg)
