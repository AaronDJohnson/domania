"""domainia.translate operators: a native covariance seen from another domain (spec section 3)."""

from __future__ import annotations

import dataclasses
import functools
import warnings

import numpy as np
import pytest

from domainia import wilson
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.testing import ALT_WINDOW, PointwiseOperator
from domainia.translate import (
    Operator,
    TranslatedOperator,
    log_coordinate_scale,
    transform,
    translate_operator,
)

NF, NT, DT = 8, 16, 0.25
N = NF * NT
TIME = TimeDomain(N, DT)
FOURIER = FourierDomain.full(N, DT)
WDM = WDMDomain.full(wilson.Grid(NF, NT, DT))
WDM_ALT = WDMDomain.full(wilson.Grid(16, 8, DT), window=ALT_WINDOW)
TARGETS = {"time": TIME, "fourier": FOURIER, "wdm": WDM, "wdm-alt": WDM_ALT}
PARTIAL = WDMDomain(CellSet.bands(WDM.cells.grid, (2, 3)))
_rng = np.random.default_rng(43)
X = {"A": _rng.standard_normal(N), "E": _rng.standard_normal(N)}


def _variances(shape):
    return {"A": 0.5 + _rng.random(shape), "E": 0.5 + _rng.random(shape)}


NATIVES = {
    "time": PointwiseOperator(TIME, _variances((N,))),
    "wdm": PointwiseOperator(WDM, _variances((NF + 1, NT))),
    "wdm-alt": PointwiseOperator(WDM_ALT, _variances((17, 8))),
}
PAIRS = [(a, b) for a in NATIVES for b in TARGETS]


def _y(domain):
    return transform(X, TIME, domain)


def _close(got, want, rel=1e-10):
    assert list(got) == list(want)
    for k in want:
        np.testing.assert_allclose(got[k], want[k], rtol=0, atol=rel * float(np.max(np.abs(want[k]))))


# -- independent references: numpy.fft.rfft and wilson.matrix.active_W only, never transform --
assert N % 2 == 0                                    # the Fourier maps below have a Nyquist bin
_H = N // 2 - 1                                      # interior rfft bins


def _cells(domain):
    g = domain.cells.grid
    return wilson.grid.packed_cells(g.Nf, g.Nt)


def _coordinates(domain):
    """``M``: time samples -> ``domain``'s real coordinates (time samples; Fourier ``Re X_0``,
    ``sqrt(2) Re/Im X_k``, ``Re X_{N/2}`` of ``X = dt rfft(x)``; WDM active cells)."""
    if isinstance(domain, TimeDomain):
        return np.eye(N)
    if isinstance(domain, FourierDomain):
        F = DT * np.fft.rfft(np.eye(N), axis=0)      # column j: dt * rfft(e_j)
        return np.vstack([F[:1].real, np.sqrt(2) * F[1:-1].real, np.sqrt(2) * F[1:-1].imag, F[-1:].real])
    g = domain.cells.grid
    return wilson.matrix.active_W(g.Nf, g.Nt, g.dt, window=domain.window)


@functools.cache
def _maps(domain):
    """``(M, M^-1)`` for ``domain``."""
    M = _coordinates(domain)
    return M, np.linalg.inv(M)


def _pack(domain, a):
    """Layout -> real coordinates."""
    a = np.asarray(a)
    if isinstance(domain, TimeDomain):
        return a.astype(float)
    if isinstance(domain, FourierDomain):
        return np.concatenate([a[:1].real, np.sqrt(2) * a[1:-1].real, np.sqrt(2) * a[1:-1].imag, a[-1:].real])
    m, n = _cells(domain)
    return a[m, n]


def _unpack(domain, r):
    """Real coordinates -> the canonical layout (inactive cells exactly 0, end bins exactly real)."""
    if isinstance(domain, TimeDomain):
        return np.array(r, dtype=float)
    if isinstance(domain, FourierDomain):
        out = np.empty(N // 2 + 1, dtype=complex)
        out[0], out[-1] = r[0], r[-1]
        out[1:-1] = (r[1:1 + _H] + 1j * r[1 + _H:-1]) / np.sqrt(2)
        return out
    g = domain.cells.grid
    out = np.zeros((g.Nf + 1, g.Nt))
    m, n = _cells(domain)
    out[m, n] = r
    return out


class _DenseOperator:
    """A duck-typed native (no PointwiseOperator): per channel a non-diagonal SPD covariance
    ``U Cr U^T`` on the span of the orthonormal columns ``U`` (``domain``'s real coordinates).
    ``apply``, ``solve`` and ``project`` are ``U Cr U^T``, ``U Cr^-1 U^T`` and ``U U^T``."""

    channel_names = ("A", "E")

    def __init__(self, domain, U, rng):
        self.domain, self.U = domain, U
        rank = U.shape[1]
        self.Cr = {}
        for k in self.channel_names:
            B = rng.standard_normal((rank, rank)) / np.sqrt(rank)
            self.Cr[k] = B @ B.T + 0.3 * np.eye(rank)
        self.degrees_of_freedom = len(self.channel_names) * rank

    def _z(self, values):
        return {k: self.U.T @ _pack(self.domain, values[k]) for k in self.channel_names}

    def apply(self, values):
        return {k: _unpack(self.domain, self.U @ (self.Cr[k] @ z)) for k, z in self._z(values).items()}

    def solve(self, values):
        return {k: _unpack(self.domain, self.U @ np.linalg.solve(self.Cr[k], z)) for k, z in self._z(values).items()}

    def project(self, values):
        return {k: _unpack(self.domain, self.U @ z) for k, z in self._z(values).items()}

    def quadratic_form(self, values):
        return float(sum(z @ np.linalg.solve(self.Cr[k], z) for k, z in self._z(values).items()))

    def log_determinant(self):
        return float(sum(np.linalg.slogdet(c)[1] for c in self.Cr.values()))


_drng = np.random.default_rng(44)
RANK = N - 23
#: The identity's columns without 23 consecutive time samples: a gap (a coordinate mask).
_GAP = np.eye(N)[:, np.r_[0:40, 40 + N - RANK:N]]


def _spread(rank):
    """An orthonormal basis of a random rank-``rank`` subspace: no coordinate structure."""
    return np.linalg.qr(_drng.standard_normal((N, rank)))[0]


DENSE = {
    "time-gap": _DenseOperator(TIME, _GAP, _drng),
    "fourier": _DenseOperator(FOURIER, _spread(RANK), _drng),
    "fourier-full": _DenseOperator(FOURIER, np.eye(N), _drng),
    "wdm": _DenseOperator(WDM, _spread(RANK), _drng),
    "wdm-alt": _DenseOperator(WDM_ALT, _spread(RANK), _drng),
}
EVERY_NATIVE = {**NATIVES, **{f"dense-{k}": v for k, v in DENSE.items()}}


def _native_covariance(op, k):
    """``(C, rank)``: ``op``'s channel-``k`` covariance in its domain's real coordinates, from
    its definition, and the rank it was built with."""
    if isinstance(op, PointwiseOperator):
        return np.diag(_pack(op.domain, op.variances[k])), N
    return op.U @ op.Cr[k] @ op.U.T, op.U.shape[1]


@pytest.mark.parametrize("name", TARGETS)
def test_the_reference_coordinate_maps(name):
    """The references are built without transform; check them against it once: ``M`` is
    ``exp(s)`` times an orthogonal matrix, transform is ``M`` in real coordinates, and
    ``_unpack`` gives a canonical layout that transform takes back."""
    d = TARGETS[name]
    M = _maps(d)[0]
    s2 = np.exp(2.0 * log_coordinate_scale(d))
    np.testing.assert_allclose(M @ M.T, s2 * np.eye(N), rtol=0, atol=1e-12 * s2)
    for k, x in X.items():
        r = M @ x
        np.testing.assert_allclose(_pack(d, _y(d)[k]), r, rtol=0, atol=1e-12 * np.max(np.abs(r)))
        v = _unpack(d, r)
        np.testing.assert_allclose(_pack(d, v), r, rtol=0, atol=1e-15 * np.max(np.abs(r)))
        _close(transform({k: v}, d, TIME), {k: x}, rel=1e-12)


def test_the_operators_satisfy_the_protocol():
    op = NATIVES["time"]
    t = translate_operator(op, WDM)
    assert isinstance(op, Operator) and isinstance(t, Operator) and isinstance(t, TranslatedOperator)
    assert t.domain == WDM and t.channel_names == ("A", "E") and t.degrees_of_freedom == 2 * N
    assert [f.name for f in dataclasses.fields(TranslatedOperator)] == ["native", "target"]
    assert all(isinstance(native, Operator) for native in EVERY_NATIVE.values())


@pytest.mark.parametrize("native, target", PAIRS)
def test_quadratic_forms_agree_across_domains(native, target):
    op = NATIVES[native]
    t = translate_operator(op, TARGETS[target])
    assert t.quadratic_form(_y(TARGETS[target])) == pytest.approx(op.quadratic_form(_y(op.domain)), rel=1e-10)


@pytest.mark.parametrize("native, target", PAIRS)
def test_the_log_determinant_carries_the_coordinate_jacobian(native, target):
    op, d = NATIVES[native], TARGETS[target]
    delta = log_coordinate_scale(d) - log_coordinate_scale(op.domain)
    want = op.log_determinant() + 2 * op.degrees_of_freedom * delta
    assert translate_operator(op, d).log_determinant() == pytest.approx(want, rel=1e-12, abs=1e-9)


@pytest.mark.parametrize("native, target", PAIRS)
def test_solve_inverts_apply_up_to_the_projection(native, target):
    t = translate_operator(NATIVES[native], TARGETS[target])
    y = _y(TARGETS[target])
    _close(t.solve(t.apply(y)), t.project(y))
    _close(t.apply(t.solve(y)), t.project(y))


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("native", EVERY_NATIVE)
def test_every_method_matches_the_dense_reference(native, target):
    """In the target's real coordinates the translated covariance is ``C_t = S C S^T`` with
    ``S = M_target M_native^-1`` (the reference maps).  ``solve`` is its pseudo-inverse,
    ``project`` the projector onto its range, ``quadratic_form`` ``r^T C_t^+ r``,
    ``log_determinant`` its log pseudo-determinant and ``degrees_of_freedom`` its rank."""
    op, d = EVERY_NATIVE[native], TARGETS[target]
    t = translate_operator(op, d)
    S = _maps(d)[0] @ _maps(op.domain)[1]
    y = _y(d)
    got = {name: getattr(t, name)(y) for name in ("apply", "solve", "project")}
    rank, logdet, qf = 0, 0.0, 0.0
    for k in op.channel_names:
        C, built = _native_covariance(op, k)
        C = S @ C @ S.T
        w, V = np.linalg.eigh(0.5 * (C + C.T))
        keep = w > 1e-9 * w.max()
        assert keep.sum() == built                                  # the reference itself
        w, V = w[keep], V[:, keep]
        r = _pack(d, y[k])
        want = {"apply": C @ r, "solve": V @ ((V.T @ r) / w), "project": V @ (V.T @ r)}
        for name, v in want.items():
            np.testing.assert_allclose(_pack(d, got[name][k]), v, rtol=0, atol=1e-10 * np.max(np.abs(v)),
                                       err_msg=f"{name}[{k}]")
        if built < N:
            assert not np.allclose(got["project"][k], y[k])        # the projection is not vacuous
        rank += built
        logdet += float(np.sum(np.log(w)))
        qf += float(r @ want["solve"])
    assert t.degrees_of_freedom == rank
    assert t.log_determinant() == pytest.approx(logdet, rel=1e-10, abs=1e-9)
    assert t.quadratic_form(y) == pytest.approx(qf, rel=1e-10)


def test_a_time_operator_seen_in_wdm_is_the_dense_congruence():
    op = NATIVES["time"]
    t = translate_operator(op, WDM)
    A = wilson.matrix.active_W(NF, NT, DT)
    m, n = wilson.grid.packed_cells(NF, NT)
    y = _y(WDM)
    np.testing.assert_allclose(A @ X["A"], y["A"][m, n], rtol=0, atol=1e-12)
    out = t.apply(y)
    for k in ("A", "E"):
        want = A @ (op.variances[k] * (A.T @ y[k][m, n]))
        np.testing.assert_allclose(out[k][m, n], want, rtol=0, atol=1e-10 * np.max(np.abs(want)))
    dense = sum(np.linalg.slogdet(A @ np.diag(op.variances[k]) @ A.T)[1] for k in ("A", "E"))
    assert t.log_determinant() == pytest.approx(dense, rel=1e-10)


def test_a_wdm_operator_seen_in_time_is_the_dense_congruence():
    op = NATIVES["wdm"]
    t = translate_operator(op, TIME)
    A = wilson.matrix.active_W(NF, NT, DT)
    m, n = wilson.grid.packed_cells(NF, NT)
    out = t.apply(X)
    for k in ("A", "E"):
        want = A.T @ (op.variances[k][m, n] * (A @ X[k])) / DT**2
        np.testing.assert_allclose(out[k], want, rtol=0, atol=1e-10 * np.max(np.abs(want)))
    dense = sum(np.linalg.slogdet(A.T @ np.diag(op.variances[k][m, n]) @ A / DT**2)[1] for k in ("A", "E"))
    assert t.log_determinant() == pytest.approx(dense, rel=1e-10)      # dof: the N active cells only


def test_chains_collapse_and_a_native_target_returns_the_native_operator():
    op = NATIVES["wdm"]
    assert translate_operator(op, WDM) is op
    t1 = translate_operator(op, FOURIER)
    assert translate_operator(t1, FOURIER) is t1
    t2 = translate_operator(t1, WDM_ALT)
    assert isinstance(t2, TranslatedOperator) and t2.native is op and t2.target == WDM_ALT
    assert translate_operator(t1, WDM) is op
    y = _y(WDM_ALT)
    _close(t2.apply(y), translate_operator(op, WDM_ALT).apply(y), rel=1e-14)


class _KernelRoundTrip(PointwiseOperator):
    """A native WDM operator whose ``apply`` output comes straight from the ``wilson`` kernels:
    its inactive cells carry ~1e-16 residue, which breaks the Operator output contract."""

    def apply(self, values):
        g, window = self.domain.cells.grid, self.domain.window
        return {k: wilson.forward_time(wilson.inverse_time(v, g.Nf, g.Nt, g.dt, window=window),
                                       g.Nf, g.Nt, g.dt, window=window)
                for k, v in super().apply(values).items()}


class _Canonicalised(_KernelRoundTrip):
    """The same operator with the documented remedy: the inactive cells zeroed."""

    def apply(self, values):
        g = self.domain.cells.grid
        out = super().apply(values)
        for v in out.values():
            v[..., ~wilson.active_mask(g.Nf, g.Nt)] = 0.0
        return out


def test_a_native_output_off_the_canonical_form_is_refused_not_repaired():
    """TranslatedOperator sends native outputs back through transform, which takes only exact
    canonical input: an un-zeroed kernel output is refused (blamed on the native method), and
    zeroing it is the fix."""
    variances = NATIVES["wdm"].variances
    raw = _KernelRoundTrip(WDM, variances)
    inactive = ~wilson.active_mask(NF, NT)
    assert any(np.any(v[inactive] != 0.0) for v in raw.apply(_y(WDM)).values())   # not vacuous
    with pytest.raises(ValueError, match=r"inactive WDM cell .* must be exactly 0") as info:
        translate_operator(raw, TIME).apply(X)
    assert str(info.value).startswith("_KernelRoundTrip.apply returned a dict off transform's canonical "
                                      "form (see domainia.translate.Operator): channel 'A': inactive WDM cell")
    assert isinstance(info.value.__cause__, ValueError)
    _close(translate_operator(_Canonicalised(WDM, variances), TIME).apply(X),
           translate_operator(NATIVES["wdm"], TIME).apply(X))


class _Stub:
    """A duck-typed native on ``domain`` whose ``apply``, ``solve`` and ``project`` return
    ``out(values)``."""

    channel_names = ("A", "E")
    degrees_of_freedom = 2 * N

    def __init__(self, domain, out=dict):
        self.domain, self._out = domain, out

    def apply(self, values):
        return self._out(values)

    solve = project = apply

    def quadratic_form(self, values):
        return 0.0

    def log_determinant(self):
        return 0.0


def test_a_native_output_with_other_channels_or_no_mapping_is_blamed_on_the_native():
    y = _y(WDM)
    assert translate_operator(_Stub(TIME), WDM).apply(y).keys() == y.keys()
    with pytest.raises(ValueError, match=r"^_Stub\.solve returned channels \('A',\); the operator has "
                                         r"channels \('A', 'E'\)$"):
        translate_operator(_Stub(TIME, lambda v: {"A": v["A"]}), WDM).solve(y)
    with pytest.raises(ValueError, match=r"^_Stub\.apply returned channels \('A', 'E', 'T'\)"):
        translate_operator(_Stub(TIME, lambda v: {**v, "T": v["A"]}), WDM).apply(y)
    with pytest.raises(TypeError, match=r"^_Stub\.project returned a list, not a \{channel: array\} mapping$"):
        translate_operator(_Stub(TIME, lambda v: list(v.values())), WDM).project(y)


def test_refusals():
    op = NATIVES["time"]
    t = translate_operator(op, WDM)
    with pytest.raises(TypeError, match="Operator"):
        translate_operator("not an operator", WDM)
    with pytest.raises(TypeError, match="^target must be a full-record domain"):
        translate_operator(op, "wdm")
    with pytest.raises(TypeError, match="native"):
        TranslatedOperator(t, FOURIER)
    with pytest.raises(TypeError, match="^target must be a full-record domain"):
        TranslatedOperator(op, "wdm")
    with pytest.raises(ValueError, match="^native and target time domains start at different t0"):
        TranslatedOperator(op, TimeDomain(N, DT, t0=2.0))
    with pytest.raises(ValueError, match="^native and target describe different records: N="):
        translate_operator(op, FourierDomain.full(N + 2, DT))
    with pytest.raises(ValueError, match="^source and target describe different records: N="):
        transform(X, TIME, FourierDomain.full(N + 2, DT))                # transform keeps its roles
    with pytest.raises(ValueError, match="^native and target time domains start at different t0"):
        translate_operator(op, TimeDomain(N, DT, t0=2.0))
    with pytest.raises(ValueError, match="^target: translate handles only WDMDomain.full"):
        translate_operator(op, PARTIAL)
    on_partial = _Stub(PARTIAL)                                  # PointwiseOperator itself refuses one
    assert isinstance(on_partial, Operator)
    with pytest.raises(ValueError, match="^native: translate handles only WDMDomain.full"):
        translate_operator(on_partial, PARTIAL)                  # before the same-domain shortcut
    with pytest.raises(ValueError, match="the operator has channels"):     # not the native's own check
        t.apply({"A": _y(WDM)["A"]})
    with pytest.raises(TypeError, match="mapping"):
        t.quadratic_form([_y(WDM)["A"]])


def test_the_pointwise_test_operator_refuses_bad_input():
    with pytest.raises(TypeError, match="TimeDomain or WDMDomain"):
        PointwiseOperator(FOURIER, {"A": np.ones(N // 2 + 1)})
    with pytest.raises(ValueError, match="WDMDomain.full"):
        PointwiseOperator(PARTIAL, {"A": np.ones((NF + 1, NT))})
    with pytest.raises(ValueError, match="positive"):
        PointwiseOperator(TIME, {"A": -np.ones(N)})
    with pytest.raises(ValueError, match="shape"):
        PointwiseOperator(TIME, {"A": np.ones(N + 1)})
    with pytest.raises(ValueError, match="channel"):
        PointwiseOperator(TIME, {})
    with pytest.raises(TypeError, match="mapping"):
        PointwiseOperator(TIME, [np.ones(N)])
    with pytest.raises(TypeError, match="strings"):
        PointwiseOperator(TIME, {1: np.ones(N)})


def test_the_pointwise_operator_outputs_meet_the_output_contract_for_any_input():
    """Inactive cells come out exactly 0 even when they go in non-zero."""
    op = NATIVES["wdm"]
    ones = {k: np.ones((NF + 1, NT)) for k in op.channel_names}
    inactive = ~wilson.active_mask(NF, NT)
    for method in (op.apply, op.solve, op.project):
        assert all(np.all(v[inactive] == 0.0) for v in method(ones).values()), method.__name__


def test_the_pointwise_operator_never_divides_by_its_inactive_variances():
    """Zeros at the inactive cells are not variances: they are ignored, without a warning."""
    op = PointwiseOperator(WDM, {"A": np.where(wilson.active_mask(NF, NT), 2.0, 0.0)})
    y = {"A": _y(WDM)["A"]}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solved, qf = op.solve(y), op.quadratic_form(y)
    _close(solved, {"A": y["A"] / 2.0}, rel=1e-15)
    assert qf == pytest.approx(float(np.sum(y["A"] ** 2)) / 2.0, rel=1e-14)
