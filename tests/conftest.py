"""Test configuration for domainia: the NumPy-only lane switch and the backend / rng fixtures."""

from __future__ import annotations

import importlib.abc
import os
import sys

import numpy as np
import pytest


class _HideJax(importlib.abc.MetaPathFinder):
    """NumPy-only lane: with WDM_TESTS_HIDE_JAX=1 jax cannot be imported, so every JAX
    test skips and every NumPy test must still pass (as packages/wdm/tests)."""

    def find_spec(self, name, path=None, target=None):  # noqa: ARG002
        if name.split(".")[0] in ("jax", "jaxlib"):
            raise ModuleNotFoundError(f"{name} hidden by WDM_TESTS_HIDE_JAX=1 (NumPy-only lane)",
                                      name=name)
        return None


if os.environ.get("WDM_TESTS_HIDE_JAX") == "1" and not any(
        isinstance(f, _HideJax) or type(f).__name__ == "_HideJax" for f in sys.meta_path):
    sys.meta_path.insert(0, _HideJax())


def _jax_or_skip():
    jax = pytest.importorskip("jax", reason="jax not installed (NumPy-only lane)")
    from domainia.wilson.backend import enable_x64

    enable_x64()
    return jax


@pytest.fixture
def jax_be():
    """The JAX backend with float64 enabled; skips when jax is not importable."""
    _jax_or_skip()
    from domainia.wilson.backend import jax_backend

    return jax_backend()


@pytest.fixture(params=["numpy", "jax"])
def backend(request):
    """Both backends (the JAX leg skips when jax is not importable)."""
    if request.param == "jax":
        _jax_or_skip()
        from domainia.wilson.backend import jax_backend

        return jax_backend()
    from domainia.wilson.backend import NUMPY

    return NUMPY


@pytest.fixture
def rng():
    return np.random.default_rng(20261012)
