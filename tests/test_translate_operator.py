"""domainia.translate operators: a native covariance seen from another domain (spec section 3)."""

from __future__ import annotations

import copy
import dataclasses

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


def test_the_operators_satisfy_the_protocol():
    op = NATIVES["time"]
    t = translate_operator(op, WDM)
    assert isinstance(op, Operator) and isinstance(t, Operator) and isinstance(t, TranslatedOperator)
    assert t.domain == WDM and t.channel_names == ("A", "E") and t.degrees_of_freedom == 2 * N
    assert [f.name for f in dataclasses.fields(TranslatedOperator)] == ["native", "target"]


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


def test_chains_collapse_and_a_native_target_returns_the_native_operator():
    op = NATIVES["wdm"]
    assert translate_operator(op, WDM) is op
    t1 = translate_operator(op, FOURIER)
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
    canonical input: an un-zeroed kernel output is refused, and zeroing it is the fix."""
    variances = NATIVES["wdm"].variances
    raw = _KernelRoundTrip(WDM, variances)
    inactive = ~wilson.active_mask(NF, NT)
    assert any(np.any(v[inactive] != 0.0) for v in raw.apply(_y(WDM)).values())   # not vacuous
    with pytest.raises(ValueError, match=r"inactive WDM cell .* must be exactly 0"):
        translate_operator(raw, TIME).apply(X)
    _close(translate_operator(_Canonicalised(WDM, variances), TIME).apply(X),
           translate_operator(NATIVES["wdm"], TIME).apply(X))


def test_refusals():
    op = NATIVES["time"]
    t = translate_operator(op, WDM)
    with pytest.raises(TypeError, match="Operator"):
        translate_operator("not an operator", WDM)
    with pytest.raises(TypeError, match="domain"):
        translate_operator(op, "wdm")
    with pytest.raises(TypeError, match="native"):
        TranslatedOperator(t, FOURIER)
    with pytest.raises(TypeError, match="domain"):
        TranslatedOperator(op, "wdm")
    with pytest.raises(ValueError, match="t0"):
        TranslatedOperator(op, TimeDomain(N, DT, t0=2.0))
    with pytest.raises(ValueError, match="N="):
        translate_operator(op, FourierDomain.full(N + 2, DT))
    with pytest.raises(ValueError, match="t0"):
        translate_operator(op, TimeDomain(N, DT, t0=2.0))
    with pytest.raises(ValueError, match="WDMDomain.full"):
        translate_operator(op, WDMDomain(CellSet.bands(WDM.cells.grid, (2, 3))))
    on_partial = copy.copy(NATIVES["wdm"])                   # PointwiseOperator itself refuses one
    object.__setattr__(on_partial, "domain", WDMDomain(CellSet.bands(WDM.cells.grid, (2, 3))))
    with pytest.raises(ValueError, match="WDMDomain.full"):  # before the same-domain shortcut
        translate_operator(on_partial, on_partial.domain)
    with pytest.raises(ValueError, match="the operator has channels"):     # not the native's own check
        t.apply({"A": _y(WDM)["A"]})
    with pytest.raises(TypeError, match="mapping"):
        t.quadratic_form([_y(WDM)["A"]])


def test_the_pointwise_test_operator_refuses_bad_input():
    with pytest.raises(TypeError, match="TimeDomain or WDMDomain"):
        PointwiseOperator(FOURIER, {"A": np.ones(N // 2 + 1)})
    with pytest.raises(ValueError, match="WDMDomain.full"):
        PointwiseOperator(WDMDomain(CellSet.bands(WDM.cells.grid, (2, 3))), {"A": np.ones((NF + 1, NT))})
    with pytest.raises(ValueError, match="positive"):
        PointwiseOperator(TIME, {"A": -np.ones(N)})
    with pytest.raises(ValueError, match="shape"):
        PointwiseOperator(TIME, {"A": np.ones(N + 1)})
    with pytest.raises(ValueError, match="channel"):
        PointwiseOperator(TIME, {})
