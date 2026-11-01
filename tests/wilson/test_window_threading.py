"""Every window consumer samples, validates and computes with the window it is given, and
every admissible window gives an orthonormal Wilson basis with block-tridiagonal stationary
covariance."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import domainia
from domainia import wilson
from domainia.testing import ALT_WINDOW, smoothstep
from domainia.wilson import MEYER, Meyer, RampWindow, WindowError

from .meyer_cases import THREADED, psd, wilson_cases

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
#: Names that hard-code a window inside a kernel: the default instance and the constructors.
HARD_CODED = frozenset({"MEYER", "Meyer", "RampWindow", "SampledWindow"})
WINDOWS = [MEYER, Meyer(1), Meyer(8), RampWindow(smoothstep, "smoothstep"), ALT_WINDOW]
GRIDS = [(8, 16, 1.0), (16, 8, 0.25), (4, 32, 2.0)]
#: Relative tolerance (to the reference's peak) of the definition-level comparisons.
RTOL = 1e-13
#: atom_spectrum's phase ramp exp(-2 pi i k (n - a) / Nt) rounds its argument at every bin,
#: so its error grows with the bin index: worst seen on GRIDS 6.1e-14 relative (the others
#: stay below 1.3e-15).
RTOL_ATOM = 1e-12

T, M = wilson.transform, wilson.matrix


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


class InadmissibleWindow:
    """A third-party Window whose samples are 1.01 x ALT_WINDOW's: real, even and compactly
    supported, but not a partition of unity."""

    name = "alt-times-1.01"

    def sample(self, Nf, Nt, dt):
        return 1.01 * ALT_WINDOW.sample(Nf, Nt, dt)


def _callee(node: ast.Call) -> str | None:
    f = node.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def _hard_coded(value: ast.expr) -> bool:
    """A literal, or an expression that names MEYER or a window constructor anywhere
    (``MEYER``, ``wilson.MEYER``, ``Meyer(8)``, ``window if a == 0 else MEYER``)."""
    return isinstance(value, ast.Constant) or any(
        (isinstance(n, ast.Name) and n.id in HARD_CODED)
        or (isinstance(n, ast.Attribute) and n.attr in HARD_CODED) for n in ast.walk(value))


def _window_violations(source: str) -> list[tuple[int, str, str]]:
    """``(line, callee, reason)`` of every call in ``source`` that loses or bypasses the window.

    * a consumer call without ``window=`` (``base=`` excuses ``atom_spectrum`` only: the base
      carries the window it was built with), or with a hard-coded ``window=``;
    * ``sample_window`` of a hard-coded window;
    * any ``.sample(...)`` (it bypasses ``sample_window``'s validation) and any ``phi_centered``.

    Any other ``window=`` expression passes (``window=window``, ``window=self.dst.window``).
    """
    bad = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = _callee(node)
        kw = {k.arg: k.value for k in node.keywords}
        if name == "phi_centered":
            bad.append((node.lineno, name, "samples Meyer directly"))
        elif name == "sample" and isinstance(node.func, ast.Attribute):
            bad.append((node.lineno, name, "bypasses sample_window's validation"))
        elif name == "sample_window":
            win = node.args[0] if node.args else kw.get("window")
            if win is None or _hard_coded(win):
                bad.append((node.lineno, name, "samples a hard-coded window"))
        elif name in CONSUMERS:
            if "window" in kw:
                if _hard_coded(kw["window"]):
                    bad.append((node.lineno, name, "passes a hard-coded window"))
            elif not (name == "atom_spectrum" and "base" in kw):
                bad.append((node.lineno, name, "drops the window"))
    return sorted(bad)


def _missing_window(path: Path) -> list[tuple[int, str, str]]:
    """:func:`_window_violations` of a source file."""
    return _window_violations(path.read_text())


@pytest.mark.parametrize("rel", SCANNED)
def test_no_kernel_call_drops_the_window(rel):
    assert _missing_window(Path(domainia.__file__).parent / rel) == []


@pytest.mark.parametrize("source, flagged", [
    ("forward(x, Nf, Nt, dt, window=window)", False),
    ("forward(x, Nf, Nt, dt, be=be, window=self.dst.window)", False),
    ("sample_window(window, Nf, Nt, dt)", False),
    ("sample_window(self.window, Nf, Nt, dt)", False),
    ("atom_spectrum(m, n, Nf, Nt, dt, base=base)", False),
    ("np.fft.rfft(x)", False),
    ("forward(x, Nf, Nt, dt)", True),
    ("forward(x, Nf, Nt, dt, be=be)", True),
    ("forward(x, Nf, Nt, dt, base=base)", True),
    ("forward(x, Nf, Nt, dt, window=MEYER)", True),
    ("Tr.forward(x, Nf, Nt, dt, window=wilson.MEYER)", True),
    ("forward(x, Nf, Nt, dt, window=Meyer(8))", True),
    ("forward(x, Nf, Nt, dt, window=windows.RampWindow(ramp, 'r'))", True),
    ("forward(x, Nf, Nt, dt, window=SampledWindow(fn, 's'))", True),
    ("forward(x, Nf, Nt, dt, window='meyer')", True),
    ("forward(x, Nf, Nt, dt, window=None)", True),
    ("inverse_many(e, Nf, Nt, dt, window=window if a == 0 else MEYER)", True),
    ("sample_window(MEYER, Nf, Nt, dt)", True),
    ("sample_window(window=wilson.MEYER, Nf=Nf, Nt=Nt, dt=dt)", True),
    ("window.sample(Nf, Nt, dt)", True),
    ("phi_centered(Nf, Nt, dt)", True),
])
def test_the_window_scan_flags_exactly_the_calls_that_lose_the_window(source, flagged):
    assert bool(_window_violations(source)) is flagged


_CASES = [c for c in wilson_cases(wilson, 8, 16, 1.0) if c[0].startswith(THREADED)]


@pytest.mark.parametrize("case", [c for _, c in _CASES], ids=[n for n, _ in _CASES])
def test_every_kernel_samples_the_window_it_is_given(case):
    w = CountingWindow(ALT_WINDOW)
    case({"window": w})
    assert w.calls > 0


@pytest.mark.parametrize("case", [c for _, c in _CASES], ids=[n for n, _ in _CASES])
def test_every_kernel_validates_the_window_it_is_given(case):
    with pytest.raises(WindowError, match="partition of unity"):
        case({"window": InadmissibleWindow()})


@pytest.mark.parametrize("bogus", ["meyer", None, Meyer], ids=["str", "None", "class"])
def test_atom_spectrum_checks_the_window_even_when_base_is_given(bogus):
    base = T.base_atom_spectra(8, 16, 1.0)
    with pytest.raises(TypeError, match="window must be"):
        T.atom_spectrum(1, 3, 8, 16, 1.0, base=base, window=bogus)


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


# ---------------------------------------------------------------------------
# Definition-level values: each kernel against another, independently derived kernel with the
# same window.  A kernel that samples the window but computes with MEYER fails these for every
# window but MEYER (the counting tests above only see that the window was sampled).
# ---------------------------------------------------------------------------


def _close(got, want, rtol: float, what: str) -> None:
    """``max |got - want| <= rtol * max |want|``."""
    err, scale = float(np.max(np.abs(got - want))), float(np.max(np.abs(want)))
    assert err <= rtol * scale, f"{what}: {err:.3e} > {rtol:.0e} * {scale:.3e}"


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_build_W_is_the_impulse_response(window, Nf, Nt, dt):
    """Column ``k`` of ``W`` is ``forward_time(e_k)`` (the definition; ``build_W`` probes
    ``2 (Nf+1)`` rows and rolls them by ``2 Nf`` samples instead), and the other matrix
    builders follow from it."""
    N = Nf * Nt
    W_def = np.stack([wilson.forward_time(e, Nf, Nt, dt, window=window).ravel() for e in np.eye(N)],
                     axis=1)
    mask = wilson.active_mask(Nf, Nt).ravel()
    C_def = W_def @ M.build_C_time(N, dt, psd) @ W_def.T
    _close(M.build_W(Nf, Nt, dt, window=window), W_def, RTOL, "build_W")
    _close(M.active_W(Nf, Nt, dt, window=window), W_def[mask], RTOL, "active_W")
    _close(M.build_C_wdm(Nf, Nt, dt, psd, window=window), C_def, RTOL, "build_C_wdm")
    _close(M.build_C_wdm_active(Nf, Nt, dt, psd, window=window), C_def[np.ix_(mask, mask)], RTOL,
           "build_C_wdm_active")


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_batched_inverse_is_the_per_row_inverse_and_round_trips(window, Nf, Nt, dt):
    X = np.random.default_rng(Nf * Nt + Nf).standard_normal((3, Nf * Nt))
    Wm = wilson.forward_time_many(X, Nf, Nt, dt, window=window)
    rows = np.stack([wilson.inverse(w, Nf, Nt, dt, window=window) for w in Wm])
    _close(wilson.inverse_many(Wm, Nf, Nt, dt, window=window), rows, RTOL, "inverse_many")
    _close(wilson.inverse_time_many(Wm, Nf, Nt, dt, window=window), X, RTOL, "inverse_time_many")


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_atom_spectra_are_the_rfft_of_the_rows_of_W(window, Nf, Nt, dt):
    W = M.build_W(Nf, Nt, dt, window=window)
    base = T.base_atom_spectra(Nf, Nt, dt, window=window)
    for (m, a), G in base.items():
        _close(G, np.fft.rfft(W[m * Nt + a]), RTOL, f"base_atom_spectra[{m}, {a}]")
    for m, n in zip(*np.nonzero(wilson.active_mask(Nf, Nt)), strict=True):
        _close(T.atom_spectrum(int(m), int(n), Nf, Nt, dt, window=window),
               np.fft.rfft(W[m * Nt + n]), RTOL_ATOM, f"atom_spectrum({m}, {n})")


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_forward_bands_matches_its_atoms_and_forward(window, Nf, Nt, dt):
    """``forward_bands`` (one IFFT per band) == ``Re(band_atoms @ z[window])`` (dense) ==
    ``forward(z)[bands] / sqrt(2 dt / N)`` (the unit-norm-atom scale)."""
    N = Nf * Nt
    bands = np.arange(1, Nf)
    z = np.fft.rfft(np.random.default_rng(N + Nf + 1).standard_normal(N))
    fb = T.forward_bands(z, 0, bands, Nf, Nt, dt, window=window)
    ell = (bands[:, None] - 1) * (Nt // 2) + np.arange(Nt)[None, :]
    atoms = T.band_atoms(bands, Nf, Nt, dt, window=window)
    _close(fb, np.real(np.einsum("bns,bs->bn", atoms, z[ell])), RTOL, "Re(band_atoms @ z)")
    _close(fb, wilson.forward(z, Nf, Nt, dt, window=window)[bands] / np.sqrt(2.0 * dt / N), RTOL,
           "forward(z)[bands] / sqrt(2 dt / N)")


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_inverse_bands_is_the_adjoint_of_forward_bands(window, Nf, Nt, dt):
    """``<forward_bands(z), w> == Re <z, inverse_bands(w)>``, to rounding of the
    Cauchy--Schwarz bound ``|forward_bands(z)| |w|``."""
    N = Nf * Nt
    n_bins = N // 2 + 1
    rng = np.random.default_rng(N + Nf + 2)
    bands = np.arange(1, Nf)
    z = rng.standard_normal(n_bins) + 1j * rng.standard_normal(n_bins)
    w = rng.standard_normal((bands.size, Nt))
    fz = T.forward_bands(z, 0, bands, Nf, Nt, dt, window=window)
    lhs = float(np.sum(fz * w))
    back = T.inverse_bands(w, 0, n_bins, bands, Nf, Nt, dt, window=window)
    rhs = float(np.real(np.vdot(z, back)))
    assert abs(lhs - rhs) <= RTOL * float(np.linalg.norm(fz) * np.linalg.norm(w))
