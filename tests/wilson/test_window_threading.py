"""Every window consumer samples, validates and computes with the window it is given, and
every admissible window gives an orthonormal Wilson basis with block-tridiagonal stationary
covariance."""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import domainia
from domainia import wilson
from domainia.testing import ALT_WINDOW, colored_noise, empirical_variance, smoothstep
from domainia.wilson import MEYER, Meyer, RampWindow, SampledWindow
from domainia.wilson.windows import WindowError

from .meyer_cases import THREADED, domainia_cases, psd, psd_ft, wilson_cases

#: Source files (relative to src/domainia) whose every consumer call must pass ``window=``:
#: every module of the package but UNSCANNED.  ``wilson/filter.py`` and ``wilson/windows.py``
#: must never be scanned: they are where the Meyer samples come from (``phi_centered``) and
#: where ``.sample`` is called, behind ``sample_window``'s validation.
SCANNED = ("wilson/transform.py", "wilson/matrix.py", "wilson/stationary.py", "wilson/noise.py",
           "wilson/evolutionary.py", "wilson/grid.py", "wilson/__init__.py", "wilson/_checks.py",
           "testing.py", "__init__.py", "_transform.py", "_covariance.py")
#: The two modules that legitimately sample the Meyer window directly.
UNSCANNED = frozenset({"wilson/filter.py", "wilson/windows.py"})
#: Meyer sampling helpers: only ``wilson/filter.py`` (and ``wilson/windows.py``) may call them.
MEYER_HELPERS = frozenset({"phi_centered"})

#: Every function that samples the window, directly or through another consumer.
CONSUMERS = frozenset({
    "_gather_bands", "forward", "forward_time", "inverse", "inverse_time", "_forward",
    "_forward_time", "_inverse", "_inverse_time", "base_atom_spectra", "atom_spectrum",
    "build_W", "build_C_wdm",
    "_slab_K", "build_lambda_blocks_full_active", "build_K_full_active", "psd_from_K",
    "apply_lambda_from_psd", "psd_from_lambda_blocks",
    "band_averaged_variance", "evolutionary_band_variance", "empirical_variance",
    "energy_fraction", "apply_time_mask", "wrap_fractions", "wrap_cut_columns", "exact_covariance",
    "eq45_plan",
    "dense_W", "wdm_congruence", "eq45_dense", "wdm_covariance",
})
#: Names that hard-code a window inside a kernel: the default instance and the constructors.
HARD_CODED = frozenset({"MEYER", "Meyer", "RampWindow", "SampledWindow"})
WINDOWS = [MEYER, Meyer(1), Meyer(8), RampWindow(smoothstep, "smoothstep"), ALT_WINDOW]
#: (16, 8, 0.25) is the only Nt = 8 geometry, and there the slab samples a Meyer rolloff at
#: x = 1/2 only, where every symmetric ramp agrees: MEYER, Meyer(1), Meyer(8) and smoothstep
#: coincide to 1 ulp, and only ALT_WINDOW tells windows apart.  (16, 16, 0.25) has the same
#: Nf and dt with no two test windows alike.
GRIDS = [(8, 16, 1.0), (16, 8, 0.25), (4, 32, 2.0), (16, 16, 0.25)]


def _osc0(u):
    """``cos(3 pi u / 2)`` on ``|u| < 1``, exactly 0 at ``|u| = 1/3``: admissible (power
    complementary), with a zero sample inside the slab where Meyer is near its peak."""
    u = np.asarray(u, float)
    v = np.where(np.abs(u) < 1.0, np.cos(1.5 * np.pi * u), 0.0)
    return np.where(np.abs(np.abs(u) - 1.0 / 3.0) < 1e-15, 0.0, v)


#: A window that vanishes on a sampled bin where Meyer does not.  A kernel that ranks or masks
#: candidate bins by Meyer's samples instead of this window's estimates 0 at that bin; every
#: window in WINDOWS is nonzero wherever Meyer is, which hides the mix-up.
ZEROS_WINDOW = SampledWindow(_osc0, "osc0")
#: The grids where ``|u| = 1/3`` is a sample (3 divides ``Nt / 2``).
ZEROS_GRIDS = [(4, 12, 1.0), (8, 24, 0.5)]
#: Relative tolerance (to the reference's peak) of the definition-level comparisons.
RTOL = 1e-13
#: atom_spectrum's phase ramp exp(-2 pi i k (n - a) / Nt) rounds its argument at every bin,
#: so its error grows with the bin index: worst seen on GRIDS 6.1e-14 relative (the others
#: stay below 1.3e-15).
RTOL_ATOM = 1e-12

T, M = wilson.transform, wilson.matrix
St, No, E = wilson.stationary, wilson.noise, wilson.evolutionary


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


def _window_name(node: ast.AST) -> str | None:
    """The HARD_CODED name ``node`` is (``MEYER``, ``wilson.MEYER``, ``Meyer``), else None."""
    name = (node.id if isinstance(node, ast.Name)
            else node.attr if isinstance(node, ast.Attribute) else None)
    return name if name in HARD_CODED else None


def _hard_coded(value: ast.expr) -> bool:
    """A literal, or an expression that names MEYER or a window constructor anywhere
    (``MEYER``, ``wilson.MEYER``, ``Meyer(8)``, ``window if a == 0 else MEYER``)."""
    return isinstance(value, ast.Constant) or any(_window_name(n) for n in ast.walk(value))


def _body_hard_codes(tree: ast.AST, reported: set[int]) -> list[tuple[int, str, str]]:
    """Every HARD_CODED name inside a function (or lambda) body: a rebinding
    ``window = MEYER`` on some path, a local alias ``w0 = MEYER``, a ``Meyer(8)`` built in place.

    Allowed: parameter defaults (``window: Window = MEYER``) and comparison operands
    (``window is MEYER``, ``window == Meyer(4)``).  Imports, module-level statements
    (``ALT_WINDOW = SampledWindow(...)``) and class-body statements (dataclass field defaults,
    ``window: Window = field(default=MEYER)``) are not in a function body, so they pass too.
    Names already in ``reported`` (flagged by a call rule) are not reported twice.
    """
    allowed, bodies = set(reported), []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            defaults = (*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None))
            for default in defaults:
                allowed.update(id(n) for n in ast.walk(default))
            bodies.extend(node.body if isinstance(node.body, list) else [node.body])
        elif isinstance(node, ast.Compare):
            for op in (node.left, *node.comparators):
                allowed.add(id(op.func if isinstance(op, ast.Call) else op))
    bad = []
    for stmt in bodies:
        for n in ast.walk(stmt):
            name = _window_name(n)
            if name and id(n) not in allowed:
                allowed.add(id(n))                      # a nested function's body is walked twice
                bad.append((n.lineno, name, "hard-codes a window in a function body"))
    return bad


def _window_violations(source: str) -> list[tuple[int, str, str]]:
    """``(line, name, reason)`` of every place in ``source`` that loses or bypasses the window.

    * a consumer call without ``window=`` (``base=`` excuses ``atom_spectrum`` only: the base
      carries the window it was built with), or with a hard-coded ``window=``;
    * ``sample_window`` of a hard-coded window;
    * any ``.sample(...)`` (it bypasses ``sample_window``'s validation) and any
      ``phi_centered`` (it samples Meyer directly);
    * a hard-coded window anywhere in a function body (:func:`_body_hard_codes`): the call rules
      see only keywords, so ``window = MEYER`` followed by ``window=window`` would pass them.

    Any other ``window=`` expression passes (``window=window``, ``window=self.dst.window``).
    """
    tree = ast.parse(source)
    bad, reported = [], set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callee(node)
        kw = {k.arg: k.value for k in node.keywords}
        if name in MEYER_HELPERS:
            bad.append((node.lineno, name, "samples Meyer directly"))
        elif name == "sample" and isinstance(node.func, ast.Attribute):
            bad.append((node.lineno, name, "bypasses sample_window's validation"))
        elif name == "sample_window":
            win = node.args[0] if node.args else kw.get("window")
            if win is None or _hard_coded(win):
                bad.append((node.lineno, name, "samples a hard-coded window"))
                if win is not None:
                    reported.update(id(n) for n in ast.walk(win))
        elif name in CONSUMERS:
            if "window" in kw:
                if _hard_coded(kw["window"]):
                    bad.append((node.lineno, name, "passes a hard-coded window"))
                    reported.update(id(n) for n in ast.walk(kw["window"]))
            elif not (name == "atom_spectrum" and "base" in kw):
                bad.append((node.lineno, name, "drops the window"))
    return sorted(bad + _body_hard_codes(tree, reported))


def _missing_window(path: Path) -> list[tuple[int, str, str]]:
    """:func:`_window_violations` of a source file."""
    return _window_violations(path.read_text())


@pytest.mark.parametrize("rel", SCANNED)
def test_no_kernel_call_drops_the_window(rel):
    assert _missing_window(Path(domainia.__file__).parent / rel) == []


def test_the_scan_covers_every_module_but_filter_and_windows():
    """A new module anywhere in the package is scanned unless it is added to UNSCANNED on
    purpose."""
    root = Path(domainia.__file__).parent
    modules = {p.relative_to(root).as_posix() for p in root.rglob("*.py")}
    assert UNSCANNED <= modules
    assert modules - UNSCANNED <= set(SCANNED)
    assert set(SCANNED) <= modules
    assert not UNSCANNED & set(SCANNED)


@pytest.mark.parametrize("source, flagged", [
    ("forward(x, Nf, Nt, dt, window=window)", False),
    ("forward(x, Nf, Nt, dt, window=self.dst.window)", False),
    ("sample_window(window, Nf, Nt, dt)", False),
    ("sample_window(self.window, Nf, Nt, dt)", False),
    ("atom_spectrum(m, n, Nf, Nt, dt, base=base)", False),
    ("np.fft.rfft(x)", False),
    ("forward(x, Nf, Nt, dt)", True),
    ("forward(x, Nf, Nt, dt, base=base)", True),
    ("forward(x, Nf, Nt, dt, window=MEYER)", True),
    ("Tr.forward(x, Nf, Nt, dt, window=wilson.MEYER)", True),
    ("forward(x, Nf, Nt, dt, window=Meyer(8))", True),
    ("forward(x, Nf, Nt, dt, window=windows.RampWindow(ramp, 'r'))", True),
    ("forward(x, Nf, Nt, dt, window=SampledWindow(fn, 's'))", True),
    ("forward(x, Nf, Nt, dt, window='meyer')", True),
    ("forward(x, Nf, Nt, dt, window=None)", True),
    ("inverse(e, Nf, Nt, dt, window=window if a == 0 else MEYER)", True),
    ("sample_window(MEYER, Nf, Nt, dt)", True),
    ("sample_window(window=wilson.MEYER, Nf=Nf, Nt=Nt, dt=dt)", True),
    ("window.sample(Nf, Nt, dt)", True),
    ("phi_centered(Nf, Nt, dt)", True),
    ("filter.phi_centered(Nf, Nt, dt, 8)", True),
])
def test_the_window_scan_flags_exactly_the_calls_that_lose_the_window(source, flagged):
    assert bool(_window_violations(source)) is flagged


_PASS_ON = "    return forward(x, Nf, Nt, dt, window=window)\n"


@pytest.mark.parametrize("source, flagged", [
    # Rebinding and aliases: every call passes window=window, the body hard-codes MEYER.
    ("def f(x, *, window=MEYER):\n    window = MEYER if x is None else window\n" + _PASS_ON, True),
    ("def f(x, *, bands=None, window=MEYER):\n    if bands is not None:\n        window = MEYER\n"
     + _PASS_ON, True),
    ("def f(x, *, window=MEYER):\n    window = window if x.ndim else wilson.MEYER\n" + _PASS_ON,
     True),
    ("def f(x, *, window=None):\n    window = Meyer(8) if window is None else window\n" + _PASS_ON,
     True),
    ("def f(x):\n    w0 = MEYER\n    return forward(x, Nf, Nt, dt, window=w0)\n", True),
    ("def f(x):\n    return g(x, SampledWindow(fn, 's'))\n", True),
    ("def f(x):\n    def inner(y):\n        return RampWindow(ramp, 'r')\n    return inner(x)\n", True),
    ("class A:\n    def f(self, x):\n        self.window = MEYER\n", True),
    ("h = lambda x: g(x, MEYER)\n", True),
    # Allowed: defaults, imports, module and class bodies, comparisons.
    ("def f(x, *, window: Window = MEYER):\n" + _PASS_ON, False),
    ("def f(x, window=wilson.MEYER, other=Meyer(8)):\n" + _PASS_ON, False),
    ("h = lambda x, window=MEYER: forward(x, Nf, Nt, dt, window=window)\n", False),
    ("from .windows import MEYER, Meyer, RampWindow, SampledWindow, Window\n", False),
    ("import domainia.wilson.windows as windows\nMEYER = windows.Meyer()\n", False),
    ("ALT_WINDOW = SampledWindow(wide_profile, 'wide-smoothstep')\n", False),
    ("@dataclass(frozen=True)\nclass D:\n    window: Window = MEYER\n", False),
    ("@dataclass(frozen=True)\nclass D:\n    window: Window = field(default=MEYER, kw_only=True)\n",
     False),
    ("def f(x, *, window=MEYER):\n    if window is MEYER:\n        return fast(x)\n" + _PASS_ON, False),
    ("def f(x, *, window=MEYER):\n    return window == wilson.MEYER or window != Meyer(4)\n", False),
    # The domainia layer's patterns: the window travels as data, checked but never rebuilt.
    ("class D:\n    def __init__(self, *, window: Window = MEYER):\n        self.window = window\n", False),
    ("class D:\n    @classmethod\n    def full(cls, grid, *, window=MEYER):\n"
     "        return cls(grid, window=window)\n", False),
    ("def f(self):\n    require_window(self.window)\n    if src.window != dst.window:\n"
     "        raise ValueError\n", False),
    ("@dataclass(frozen=True)\nclass D:\n    window: Window = field(default_factory=lambda: MEYER)\n",
     True),
    ("def f(window):\n    if isinstance(window, Meyer):\n        return fast(x)\n", True),
    ("def f(window):\n    return window in (MEYER,)\n", True),
])
def test_the_window_scan_flags_a_window_hard_coded_in_a_function_body(source, flagged):
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


#: The domainia-layer cases: ``kw`` goes to every window-taking call.
_DCASES = domainia_cases(wilson, 8, 16, 1.0)


@pytest.mark.parametrize("case", [c for _, c in _DCASES], ids=[n for n, _ in _DCASES])
def test_every_domainia_consumer_samples_the_window_it_is_given(case):
    w = CountingWindow(ALT_WINDOW)
    case({"window": w})
    assert w.calls > 0


@pytest.mark.parametrize("case", [c for _, c in _DCASES], ids=[n for n, _ in _DCASES])
def test_every_domainia_consumer_validates_the_window_it_is_given(case):
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
    A = wilson.matrix.build_W(Nf, Nt, dt, window=window)
    np.testing.assert_allclose(A @ A.T, dt * np.eye(A.shape[0]), rtol=0, atol=1e-12 * dt)


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_time_round_trip_and_energy(window, Nf, Nt, dt):
    x = np.random.default_rng(7).standard_normal(Nf * Nt)
    w = wilson.forward_time(x, Nf, Nt, dt, window=window)
    np.testing.assert_allclose(wilson.inverse_time(w, Nf, Nt, dt, window=window), x, rtol=0, atol=1e-12)
    assert float(np.sum(w ** 2)) == pytest.approx(dt * float(np.sum(x ** 2)), rel=1e-12)
    many = wilson.forward_time(np.stack([x, -x]), Nf, Nt, dt, window=window)
    np.testing.assert_allclose(many[0], w, rtol=0, atol=1e-13 * float(np.max(np.abs(w))))


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
def test_stationary_covariance_is_block_tridiagonal(window):
    Nf, Nt, dt = 8, 16, 1.0
    C = wilson.matrix.build_C_wdm(Nf, Nt, dt, psd, window=window)
    m = wilson.grid.cells(Nf, Nt)[0].ravel()               # the band of each folded slot
    far = float(np.max(np.abs(C[np.abs(m[:, None] - m[None, :]) > 1])))
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
    """Column ``k`` of ``W`` is ``forward_time(e_k)`` in folded flat order (the definition;
    ``build_W`` probes ``2 Nf`` rows and rolls them by ``2 Nf`` samples instead), and
    ``build_C_wdm`` follows from it."""
    N = Nf * Nt
    W_def = wilson.forward_time(np.eye(N), Nf, Nt, dt, window=window).reshape(N, N).T
    C_def = W_def @ M.build_C_time(N, dt, psd) @ W_def.T
    _close(M.build_W(Nf, Nt, dt, window=window), W_def, RTOL, "build_W")
    _close(M.build_C_wdm(Nf, Nt, dt, psd, window=window), C_def, RTOL, "build_C_wdm")


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_atom_spectra_are_the_rfft_of_the_rows_of_W(window, Nf, Nt, dt):
    """Every active cell from one ``base`` (``base=``), and three cells -- an interior one and
    one in each edge band -- built from ``window=`` alone (``atom_spectrum``'s own
    ``base_atom_spectra(window=)`` call)."""
    W = M.build_W(Nf, Nt, dt, window=window)
    row = wilson.grid.unfold(np.arange(Nf * Nt).reshape(Nf, Nt), Nf, Nt)    # cell (m, n)'s row
    base = T.base_atom_spectra(Nf, Nt, dt, window=window)
    for (m, a), G in base.items():
        _close(G, np.fft.rfft(W[row[m, a]]), RTOL, f"base_atom_spectra[{m}, {a}]")
    for m, n in zip(*np.nonzero(wilson.grid.active_mask(Nf, Nt)), strict=True):
        _close(T.atom_spectrum(int(m), int(n), Nf, Nt, dt, base=base, window=window),
               np.fft.rfft(W[row[m, n]]), RTOL_ATOM, f"atom_spectrum({m}, {n}, base=)")
    for m, n in ((1, 3), (0, 2), (Nf, 2)):
        _close(T.atom_spectrum(m, n, Nf, Nt, dt, window=window), np.fft.rfft(W[row[m, n]]),
               RTOL_ATOM, f"atom_spectrum({m}, {n})")


# ---------------------------------------------------------------------------
# The stationary, noise, evolutionary and inner-product kernels, each against a dense reference
# built from the value-verified matrix kernels above (build_W and build_C_wdm)
# with the same window.  A kernel that samples the window but computes with MEYER fails these
# for every window but MEYER.
# ---------------------------------------------------------------------------

#: Relative tolerance (to the reference's peak) of these kernels against the dense references.
#: Worst measured over WINDOWS x GRIDS: 7.8e-15 (eq45_blocks_gram), 6.9e-15 (eq45_direct),
#: 5.3e-15 (eq45_blocks_cheb), 4.2e-15 (exact_covariance), the others <= 2.2e-15.
RTOL_KERNEL = 1e-12


def _stationary(f, t):
    """``psd`` as an evolutionary ``Sn(f, t)`` that does not depend on ``t``."""
    return psd(f) + 0.0 * np.asarray(t, dtype=float)


def _read_only(a):
    a = np.array(a)
    a.setflags(write=False)
    return a


@lru_cache(maxsize=None)
def _dense(window, Nf, Nt, dt):
    """``(W, C, W_a, C_a)``: ``build_W`` and ``build_C_wdm`` of ``psd`` with ``window``, and the
    same with their rows (and columns) in band order, ``numpy.nonzero`` of the padded mask
    (read-only)."""
    W, C = M.build_W(Nf, Nt, dt, window=window), M.build_C_wdm(Nf, Nt, dt, psd, window=window)
    a = np.argsort(np.ravel_multi_index(wilson.grid.cells(Nf, Nt), (Nf + 1, Nt)), axis=None)
    return tuple(_read_only(x) for x in (W, C, W[a], C[np.ix_(a, a)]))


def _blocks_of(C, Nf, Nt):
    """The diagonal and upper band blocks of a dense folded ``N x N`` matrix (the
    ``build_lambda_blocks_full_active`` layout: band ``m``'s cells, time ascending)."""
    band = [np.flatnonzero(wilson.grid.cells(Nf, Nt)[0] == m) for m in range(Nf + 1)]
    return ([C[np.ix_(b, b)] for b in band],
            [C[np.ix_(b, c)] for b, c in zip(band[:-1], band[1:], strict=True)])


def _K_blocks(K_diag, K_off, n_active, Nf, Nt):
    """The blocks the Toeplitz vectors stand for, ``Lambda^{(m, m')}[n, n'] =
    Re(carrier_{m m'}[n, n'] K^{(m, m')}[(n - n') mod Nt]) / N`` on the active cells (the
    carrier is the window-free Wilson parity convention)."""
    n = np.arange(Nt)
    diff = (n[:, None] - n[None, :]) % Nt
    carrier = St._carriers(Nt)
    N = Nf * Nt
    diag = [np.real(carrier[m % 2, m % 2] * K_diag[m][diff])[np.ix_(n_active[m], n_active[m])] / N
            for m in range(Nf + 1)]
    offd = [np.real(carrier[m % 2, (m + 1) % 2] * K_off[m][diff])[np.ix_(n_active[m], n_active[m + 1])]
            / N for m in range(Nf)]
    return diag, offd


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_stationary_builders_are_the_blocks_of_the_dense_covariance(window, Nf, Nt, dt):
    """``build_lambda_blocks_full_active``, and the blocks ``build_K_full_active``'s ``K``
    stand for, assemble to ``build_C_wdm``."""
    C = _dense(window, Nf, Nt, dt)[1]
    d, o, na = St.build_lambda_blocks_full_active(Nf, Nt, dt, psd, window=window)
    _close(St.assemble_lambda_full_active(d, o, na, Nf, Nt), C, RTOL_KERNEL,
           "build_lambda_blocks_full_active")
    K_diag, K_off = St.build_K_full_active(Nf, Nt, dt, psd, window=window)
    _close(St.assemble_lambda_full_active(*_K_blocks(K_diag, K_off, na, Nf, Nt), na, Nf, Nt),
           C, RTOL_KERNEL, "build_K_full_active")


@pytest.mark.parametrize("Nf, Nt, dt, window", [
    *(pytest.param(*g, w, id=f"{g[0]}-{g[1]}-{g[2]}-{w.name}") for g in GRIDS for w in WINDOWS),
    *(pytest.param(*g, ZEROS_WINDOW, id=f"{g[0]}-{g[1]}-{g[2]}-{ZEROS_WINDOW.name}")
      for g in ZEROS_GRIDS)])
def test_stationary_backward_maps_recover_the_psd(window, Nf, Nt, dt):
    """``psd_from_K`` and ``psd_from_lambda_blocks`` invert the builders with the same window,
    and ``psd_from_lambda_blocks`` recovers the PSD from the blocks of ``build_C_wdm``; also for
    a window with a zero sample inside the slab (``ZEROS_WINDOW``)."""
    S = psd(np.fft.rfftfreq(Nf * Nt, dt))
    K_diag, _ = St.build_K_full_active(Nf, Nt, dt, psd, window=window)
    _close(St.psd_from_K(K_diag, Nf, Nt, dt, window=window), S, RTOL_KERNEL, "psd_from_K")
    d, o, na = St.build_lambda_blocks_full_active(Nf, Nt, dt, psd, window=window)
    _close(St.psd_from_lambda_blocks(d, o, na, Nf, Nt, dt, window=window), S, RTOL_KERNEL,
           "psd_from_lambda_blocks")
    dense = _blocks_of(_dense(window, Nf, Nt, dt)[1], Nf, Nt)
    _close(St.psd_from_lambda_blocks(*dense, na, Nf, Nt, dt, window=window), S, RTOL_KERNEL,
           "psd_from_lambda_blocks(blocks of build_C_wdm)")


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_apply_lambda_from_psd_is_the_dense_covariance(window, Nf, Nt, dt):
    """``apply_lambda_from_psd(x)`` is ``build_C_wdm @ x``."""
    N = Nf * Nt
    x = np.random.default_rng(N + Nf + 3).standard_normal((Nf, Nt))
    S = psd(np.fft.rfftfreq(N, dt))
    want = (_dense(window, Nf, Nt, dt)[1] @ x.ravel()).reshape(Nf, Nt)
    got = St.apply_lambda_from_psd(x, Nf, Nt, dt, S, window=window)
    _close(got, want, RTOL_KERNEL, "apply_lambda_from_psd")


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_band_variances_are_the_diagonal_of_the_dense_covariance(window, Nf, Nt, dt):
    """``band_averaged_variance`` and the stationary limit of ``evolutionary_band_variance``
    are ``diag(build_C_wdm)``."""
    want = np.diag(_dense(window, Nf, Nt, dt)[1]).reshape(Nf, Nt)
    _close(No.band_averaged_variance(Nf, Nt, dt, psd, window=window), want, RTOL_KERNEL,
           "band_averaged_variance")
    _close(No.evolutionary_band_variance(Nf, Nt, dt, _stationary, window=window), want,
           RTOL_KERNEL, "evolutionary_band_variance")


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_empirical_variance_is_the_mean_square_of_the_same_draws(window, Nf, Nt, dt):
    """``empirical_variance`` is the mean of ``(W x)^2`` over the same seeded draws."""
    N = Nf * Nt
    W = _dense(window, Nf, Nt, dt)[0]
    rng = np.random.default_rng(5)
    draws = [W @ np.fft.irfft(colored_noise(N, dt, psd, rng), n=N) for _ in range(3)]
    want = np.mean(np.square(draws), axis=0).reshape(Nf, Nt)
    m, n = wilson.grid.cells(Nf, Nt)
    _close(empirical_variance(Nf, Nt, dt, psd, 3, seed=5, window=window)[m, n], want, RTOL_KERNEL,
           "empirical_variance")


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_time_masks_are_the_dense_mask(window, Nf, Nt, dt):
    """``energy_fraction`` is ``diag(M)`` and ``apply_time_mask`` applies ``M`` (to a stack of
    rows and to one 1-D row), ``M = W_a diag(w2) W_a^T / dt``."""
    N = Nf * Nt
    rng = np.random.default_rng(N + Nf + 4)
    Wa = _dense(window, Nf, Nt, dt)[2]
    w2 = rng.uniform(0.0, 1.0, N)
    Mm = Wa @ (w2[:, None] * Wa.T) / dt
    Y = rng.standard_normal((2, N))
    _close(E.energy_fraction(w2, Nf, Nt, dt, window=window), np.diag(Mm), RTOL_KERNEL,
           "energy_fraction")
    _close(E.apply_time_mask(Y, w2, Nf, Nt, dt, window=window), Y @ Mm.T, RTOL_KERNEL,
           "apply_time_mask")
    _close(E.apply_time_mask(Y[0], w2, Nf, Nt, dt, window=window), (Mm @ Y[0])[None, :],
           RTOL_KERNEL, "apply_time_mask(1-D)")


def _wrap_reference(W_a, Nf, Nt):
    """The definition of ``wrap_fractions`` on the band-order rows ``W_a``: for the active cell
    ``(m, n)``, centred on sample ``c = n Nf``, the share of ``sum_t g_mn(t)^2`` at samples
    whose unwrapped position ``c + delta`` (``delta`` the signed offset from the centre, in
    ``[-N/2, N/2)``) lies outside ``[0, N)``; ``nan`` at inactive cells."""
    N = Nf * Nt
    m, n = np.nonzero(wilson.grid.active_mask(Nf, Nt))
    g2 = W_a ** 2
    c = n[:, None] * Nf
    pos = c + (np.arange(N)[None, :] - c + N // 2) % N - N // 2
    out = np.full((Nf + 1, Nt), np.nan)
    out[m, n] = np.sum(g2 * ((pos < 0) | (pos >= N)), axis=-1) / np.sum(g2, axis=-1)
    return out


def _cut_reference(frac, tol):
    """The cut rule of ``wrap_cut_columns`` on a fraction grid."""
    worst = np.nanmax(frac, axis=0)
    half = worst.size // 2
    bad = np.flatnonzero(worst > tol)
    lead = int(bad[bad < half].max()) + 1 if np.any(bad < half) else 0
    trail = worst.size - int(bad[bad >= half].min()) if np.any(bad >= half) else 0
    return lead, trail


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_wrap_rule_is_its_definition_on_the_rows_of_W(window, Nf, Nt, dt):
    """``wrap_fractions`` (all bands and a subset) is its definition on the rows of
    ``build_W``, and ``wrap_cut_columns`` is the cut rule on those fractions."""
    want = _wrap_reference(_dense(window, Nf, Nt, dt)[2], Nf, Nt)
    got = E.wrap_fractions(Nf, Nt, dt, window=window)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    _close(np.nan_to_num(got, nan=-1.0), np.nan_to_num(want, nan=-1.0), RTOL_KERNEL, "wrap_fractions")
    bands = np.array([0, Nf // 2, Nf])
    _close(np.nan_to_num(E.wrap_fractions(Nf, Nt, dt, bands=bands, window=window), nan=-1.0),
           np.nan_to_num(want[bands], nan=-1.0), RTOL_KERNEL, "wrap_fractions(bands)")
    for tol in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6):
        assert E.wrap_cut_columns(Nf, Nt, dt, tol, window=window) == _cut_reference(want, tol), tol


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_exact_covariance_is_the_dense_congruence(window, Nf, Nt, dt):
    """``exact_covariance`` is ``W_a Sigma W_a^T`` (``Sigma = time_domain_covariance``), and
    ``build_C_wdm`` in the stationary limit."""
    N = Nf * Nt
    Wa, Ca = _dense(window, Nf, Nt, dt)[2:]
    _close(E.exact_covariance(Nf, Nt, dt, _stationary, window=window), Ca, RTOL_KERNEL,
           "exact_covariance(stationary)")
    Sn = psd_ft(N * dt)
    _close(E.exact_covariance(Nf, Nt, dt, Sn, window=window),
           Wa @ E.time_domain_covariance(N, dt, Sn) @ Wa.T, RTOL_KERNEL, "exact_covariance")


def _assemble_cheb(plan, diag, up):
    """Dense kept-cell matrix from ``eq45_blocks_cheb``'s ragged blocks."""
    C = np.zeros((plan.cells.size, plan.cells.size))
    for m, c in enumerate(plan.band_cells):
        C[np.ix_(plan.pos[c], plan.pos[c])] = diag[m]
        if m < plan.Nf:
            c2 = plan.band_cells[m + 1]
            C[np.ix_(plan.pos[c], plan.pos[c2])] = up[m]
            C[np.ix_(plan.pos[c2], plan.pos[c])] = up[m].T
    return C


def _assemble_gram(gplan, diag, up, n):
    """Dense kept-cell matrix from ``eq45_blocks_gram``'s slotted blocks (dummy slots dropped)."""
    C = np.zeros((n, n))
    for m in range(gplan.nb):
        s = gplan.slot[m]
        i = np.flatnonzero(s >= 0)
        C[np.ix_(s[i], s[i])] = diag[m][np.ix_(i, i)]
        if m < gplan.nb - 1:
            s2 = gplan.slot[m + 1]
            j = np.flatnonzero(s2 >= 0)
            C[np.ix_(s[i], s2[j])] = up[m][np.ix_(i, j)]
            C[np.ix_(s2[j], s[i])] = up[m][np.ix_(i, j)].T
    return C


@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: w.name)
@pytest.mark.parametrize("Nf, Nt, dt", GRIDS)
def test_eq45_plan_carries_the_window_to_every_eq45_builder(window, Nf, Nt, dt):
    """``plan.base`` is ``base_atom_spectra(window=)``, and in the stationary limit the plan's
    ``eq45_direct``, ``eq45_blocks_cheb`` and ``eq45_blocks_gram`` are ``build_C_wdm``,
    on all cells and on a ``keep`` subset (the kept rows and columns)."""
    Ca = _dense(window, Nf, Nt, dt)[3]
    base = T.base_atom_spectra(Nf, Nt, dt, window=window)
    keep = ~np.isin(np.nonzero(wilson.grid.active_mask(Nf, Nt))[1], (0, 1, Nt - 1))
    for tag, plan, ref in (("", E.eq45_plan(Nf, Nt, dt, window=window), Ca),
                           ("(keep)", E.eq45_plan(Nf, Nt, dt, keep, window=window),
                            Ca[np.ix_(keep, keep)])):
        assert sorted(plan.base) == sorted(base)
        assert all(np.array_equal(plan.base[key], G) for key, G in base.items())
        _close(E.eq45_direct(plan, _stationary), ref, RTOL_KERNEL, f"eq45_direct{tag}")
        coeffs = E.eq45_coeffs_interp(_stationary, plan.f, plan.t_max, plan.P, k=plan.k_used)
        _close(_assemble_cheb(plan, *E.eq45_blocks_cheb(plan, coeffs)), ref, RTOL_KERNEL,
               f"eq45_blocks_cheb{tag}")
        gplan = E.eq45_gram_plan(plan)
        blocks = E.eq45_blocks_gram(gplan, np.ones((gplan.nb, gplan.B)),
                                    E.eq45_gram_gather(gplan, np.log(psd(plan.f))))
        _close(_assemble_gram(gplan, *blocks, ref.shape[0]), ref, RTOL_KERNEL,
               f"eq45_blocks_gram{tag}")
