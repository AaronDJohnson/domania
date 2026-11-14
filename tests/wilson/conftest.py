"""Package-wide test configuration for ``domania.wilson``.

Puts ``tests/oracle`` on ``sys.path`` so the frozen oracle imports as the
top-level package ``vendored_2f7136c`` (never as ``wdm``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ORACLE_DIR = Path(__file__).resolve().parent / "oracle"
if str(ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(ORACLE_DIR))


@pytest.fixture
def rng():
    return np.random.default_rng(20261011)


#: The out-of-repo oracle trees live under the directory this variable names; unset, or
#: without the expected subdirectory, the tests that need them skip.
ORACLE_ROOT_ENV = "DOMANIA_ORACLE_ROOT"


def _oracle_paths(*subpaths: str) -> list[Path]:
    """``$DOMANIA_ORACLE_ROOT/<subpath>`` for each ``subpath``; skips the test unless the
    variable is set and every path exists.  ``test_grid_bands`` and
    ``test_evolutionary_kernels`` import it (``from .conftest import _oracle_paths``)."""
    root = os.environ.get(ORACLE_ROOT_ENV)
    if not root:
        pytest.skip(f"{ORACLE_ROOT_ENV} is not set: point it at the directory holding "
                    + " and ".join(subpaths))
    paths = [Path(root).expanduser().resolve() / s for s in subpaths]
    missing = [s for s, p in zip(subpaths, paths, strict=True) if not p.exists()]
    if missing:
        pytest.skip(f"{ORACLE_ROOT_ENV}={root} has no " + " and no ".join(missing))
    return paths
