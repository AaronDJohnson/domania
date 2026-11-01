"""Every window consumer samples the window it is given, and every admissible window
gives an orthonormal Wilson basis with block-tridiagonal stationary covariance."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import domainia
from domainia import wilson
from domainia.testing import ALT_WINDOW, smoothstep
from domainia.wilson import MEYER, Meyer, RampWindow

from .meyer_cases import psd, wilson_cases

#: Case-name prefixes whose kernels take ``window=`` (extended task by task).
THREADED = ("transform.", "matrix.")
#: Source files (relative to src/domainia) whose every consumer call must pass ``window=``.
SCANNED = ("wilson/transform.py", "wilson/matrix.py")

#: Every function that samples the window, directly or through another consumer.
CONSUMERS = frozenset({
    "_gather_bands", "forward", "forward_many", "forward_time", "forward_time_many", "inverse",
    "inverse_many", "inverse_time", "inverse_time_many", "base_atom_spectra", "atom_spectrum",
    "band_atoms", "forward_bands", "inverse_bands",
    "build_W", "active_W", "build_C_wdm", "build_C_wdm_active",
    "_slab_K", "build_lambda_blocks_full_active", "build_K_full_active", "psd_from_K",
    "apply_lambda_from_psd", "psd_from_lambda_blocks", "_forward", "_inverse", "_forward_wdm",
    "_inverse_wdm",
    "band_averaged_variance", "evolutionary_band_variance", "empirical_variance",
    "energy_fraction", "apply_time_mask", "wrap_fractions", "wrap_cut_columns", "exact_covariance",
    "eq45_plan",
    "gaussian", "gaussian_from_func", "loglike_gaussian", "loglike_gaussian_many",
    "dense_W", "wdm_congruence", "eq45_dense", "_dense_atoms",
})
WINDOWS = [MEYER, Meyer(1), Meyer(8), RampWindow(smoothstep, "smoothstep"), ALT_WINDOW]
GRIDS = [(8, 16, 1.0), (16, 8, 0.25), (4, 32, 2.0)]


class CountingWindow:
    """A third-party Window (not a built-in) that counts its ``sample`` calls."""

    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    @property
    def name(self) -> str:
        return f"counting({self.inner.name})"

    def sample(self, Nf, Nt, dt):
        self.calls += 1
        return self.inner.sample(Nf, Nt, dt)


def _missing_window(path: Path) -> list[tuple[int, str]]:
    """``(line, name)`` of every consumer call without ``window=``/``base=``, and every
    ``phi_centered`` call."""
    bad = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        if name == "phi_centered" or (
                name in CONSUMERS and not any(k.arg in ("window", "base") for k in node.keywords)):
            bad.append((node.lineno, name))
    return bad


@pytest.mark.parametrize("rel", SCANNED)
def test_no_kernel_call_drops_the_window(rel):
    assert _missing_window(Path(domainia.__file__).parent / rel) == []


_CASES = [c for c in wilson_cases(wilson, 8, 16, 1.0) if c[0].startswith(THREADED)]


@pytest.mark.parametrize("case", [c for _, c in _CASES], ids=[n for n, _ in _CASES])
def test_every_kernel_samples_the_window_it_is_given(case):
    w = CountingWindow(ALT_WINDOW)
    case({"window": w})
    assert w.calls > 0


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_every_admissible_window_gives_an_orthonormal_basis(window, Nf, Nt, dt):
    A = wilson.matrix.active_W(Nf, Nt, dt, window=window)
    np.testing.assert_allclose(A @ A.T, dt * np.eye(A.shape[0]), rtol=0, atol=1e-12 * dt)


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_time_round_trip_and_energy(window, Nf, Nt, dt):
    x = np.random.default_rng(7).standard_normal(Nf * Nt)
    w = wilson.forward_time(x, Nf, Nt, dt, window=window)
    np.testing.assert_allclose(wilson.inverse_time(w, Nf, Nt, dt, window=window), x, rtol=0, atol=1e-12)
    assert float(np.sum(w ** 2)) == pytest.approx(dt * float(np.sum(x ** 2)), rel=1e-12)
    many = wilson.forward_time_many(np.stack([x, -x]), Nf, Nt, dt, window=window)
    np.testing.assert_allclose(many[0], w, rtol=0, atol=1e-13 * float(np.max(np.abs(w))))


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
def test_stationary_covariance_is_block_tridiagonal(window):
    Nf, Nt, dt = 8, 16, 1.0
    C = wilson.matrix.build_C_wdm(Nf, Nt, dt, psd, window=window).reshape(Nf + 1, Nt, Nf + 1, Nt)
    far = max(float(np.max(np.abs(C[m, :, k, :])))
              for m in range(Nf + 1) for k in range(Nf + 1) if abs(m - k) > 1)
    assert far <= 1e-12 * float(np.max(np.abs(C)))


def test_a_different_window_gives_a_different_basis():
    x = np.random.default_rng(8).standard_normal(128)
    a = wilson.forward_time(x, 8, 16, 1.0)
    b = wilson.forward_time(x, 8, 16, 1.0, window=ALT_WINDOW)
    assert float(np.max(np.abs(a - b))) > 1e-3 * float(np.max(np.abs(a)))
