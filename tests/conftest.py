"""Test configuration for domania: the rng fixture."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(20261012)
