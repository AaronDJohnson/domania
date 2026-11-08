"""domainia.operators.CovarianceOperator: a domainia covariance fixed at ``nu`` on a WDM
support, as an Operator that translate_operator moves to time and Fourier."""

from __future__ import annotations

import numpy as np
import pytest

from domainia import wilson
from domainia.covariance import (
    BlockTridiagonalCovariance,
    ChannelStack,
    DenseCovariance,
    DiagonalCovariance,
    EvolutionaryCovariance,
)
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.operators import CovarianceOperator, operator_matrix
from domainia.spectra import GenericPSD
from domainia.testing import ALT_WINDOW
from domainia.translate import Operator, to_coordinates, transform, translate_operator

NF, NT, DT = 4, 8, 0.5
N = NF * NT
GRID = wilson.Grid(NF, NT, DT)
ALL = CellSet.all_active(GRID)
TIME = TimeDomain(N, DT)
FOURIER = FourierDomain.full(N, DT)
WDM = WDMDomain.full(GRID)
NU = np.array([-1.3, 0.4])
_rng = np.random.default_rng(61)


def _drifting(f, t, nu):
    alpha = nu[0] + nu[1] * (np.asarray(t, float) / (N * DT) - 0.5)
    return np.exp(alpha * np.log(np.maximum(np.asarray(f, float), 0.5)))


PSD = GenericPSD(_drifting)


def _spd(n):
    B = _rng.standard_normal((n, n))
    return B @ B.T / n + 0.5 * np.eye(n)


def _blocktri(cells):
    """Random SPD band blocks on ``cells`` (``nu`` scales them)."""
    sizes = tuple(int(np.sum(cells.m_array == m)) for m in range(NF + 1))
    S = _spd(cells.size)
    off = np.concatenate([[0], np.cumsum(sizes)])
    diag = [S[off[m]:off[m + 1], off[m]:off[m + 1]] for m in range(len(sizes))]
    up = [S[off[m]:off[m + 1], off[m + 1]:off[m + 2]] for m in range(len(sizes) - 1)]

    def blocks(nu, be):
        s = be.xp.exp(nu[0])
        return [s * be.asarray(d) for d in diag], [s * be.asarray(u) for u in up]

    return BlockTridiagonalCovariance(blocks, sizes)


BANDS = CellSet.bands(GRID, (1, 2))
CUT = ALL.drop_columns(2, 1)
_V = 0.5 + _rng.random(2 * ALL.size)


def _diagonal(v):
    """A DiagonalCovariance of the variances ``v`` scaled by ``exp(nu[0])``."""
    return DiagonalCovariance(lambda nu, be: be.asarray(v) * be.xp.exp(nu[0]), v.size)


#: (covariance, support, channel_names): every kind of covariance, both kinds of partial support,
#: and a support in a non-Meyer window.
CASES = {
    "diagonal": lambda: (_diagonal(_V[:ALL.size]), WDM, ("A",)),
    "blocktri": lambda: (_blocktri(ALL), WDM, ("A",)),
    "evolutionary-blocktri": lambda: (EvolutionaryCovariance(PSD, ALL), WDM, ("A",)),
    "evolutionary-dense": lambda: (EvolutionaryCovariance(PSD, ALL, assembly="dense"), WDM, ("A",)),
    "dense-two-channels": lambda: (DenseCovariance(_spd(2 * ALL.size), 2 * ALL.size, chol_impl="numpy"),
                                   WDM, ("A", "E")),
    "stack": lambda: (ChannelStack([EvolutionaryCovariance(PSD, ALL), _diagonal(_V[ALL.size:])], 2), WDM,
                      ("A", "E")),
    "band-subset": lambda: (EvolutionaryCovariance(PSD, BANDS), WDMDomain(BANDS), ("A",)),
    "drop-columns": lambda: (EvolutionaryCovariance(PSD, CUT), WDMDomain(CUT), ("A",)),
    "drop-columns-stack": lambda: (ChannelStack([_blocktri(CUT), EvolutionaryCovariance(PSD, CUT)], 2),
                                   WDMDomain(CUT), ("A", "E")),
    "alt-window": lambda: (EvolutionaryCovariance(PSD, ALL, window=ALT_WINDOW), WDMDomain(ALL, ALT_WINDOW),
                           ("A",)),
}


def _nu(cov):
    return np.concatenate([NU, NU[::-1]]) if isinstance(cov, ChannelStack) else NU


def _P(support, K):
    """``(K s, K N)``: row ``c s + i`` selects packed cell ``support.cells.packed_positions()[i]``
    of channel ``c``, so ``C = P^T Sigma P`` on the stacked coordinates (channel-major)."""
    return np.kron(np.eye(K), np.eye(N)[support.cells.packed_positions()])


def _fourier_coordinates():
    """``(N, N)``: time samples -> Fourier real coordinates, from ``numpy.fft.rfft`` alone."""
    F = DT * np.fft.rfft(np.eye(N), axis=0)
    return np.vstack([F[:1].real, np.sqrt(2) * F[1:-1].real, np.sqrt(2) * F[1:-1].imag, F[-1:].real])


def _references(cov, support, K):
    """The stacked matrices on WDM, time and Fourier from ``cov.dense(nu)`` alone."""
    C_wdm = _P(support, K).T @ cov.dense(_nu(cov)) @ _P(support, K)
    W = np.kron(np.eye(K), wilson.matrix.active_W(NF, NT, DT, window=support.window))
    C_time = W.T @ C_wdm @ W / DT**2
    M = np.kron(np.eye(K), _fourier_coordinates())
    return {"wdm": C_wdm, "time": C_time, "fourier": M @ C_time @ M.T}


def _close(got, want, rel):
    assert np.max(np.abs(got - want)) <= rel * np.max(np.abs(want))


@pytest.mark.parametrize("case", CASES)
def test_operator_matrix_is_the_selected_covariance_on_every_domain(case):
    """Native: ``P^T Sigma P``.  Seen from time: ``W^T P^T Sigma P W / dt^2``; from Fourier: the
    same mapped by the Fourier coordinates."""
    cov, support, names = CASES[case]()
    op = CovarianceOperator(cov, _nu(cov), support, names)
    assert isinstance(op, Operator) and op.channel_names == names
    assert op.domain == WDMDomain.full(GRID, support.window)
    ref = _references(cov, support, len(names))
    _close(operator_matrix(op), ref["wdm"], 1e-12)
    for target, d in (("time", TIME), ("fourier", FOURIER)):
        _close(operator_matrix(translate_operator(op, d)), ref[target], 1e-10)


@pytest.mark.parametrize("case", CASES)
def test_every_method_matches_the_dense_moore_penrose_reference(case):
    """On WDM, time and Fourier: ``solve`` is the pseudo-inverse, ``project`` the projector onto
    the range, ``quadratic_form`` ``r^T C^+ r``, ``log_determinant`` the log pseudo-determinant
    and ``degrees_of_freedom`` the rank ``K s``.  The translated pseudo-determinant is
    ``log det Sigma + 2 K s Delta`` because ``P W W^T P^T = dt I_s``: the change of basis
    scales every non-zero eigenvalue by the same ``exp(2 Delta)``."""
    cov, support, names = CASES[case]()
    K = len(names)
    op = CovarianceOperator(cov, _nu(cov), support, names)
    assert op.degrees_of_freedom == K * support.size
    assert op.log_determinant() == pytest.approx(np.linalg.slogdet(cov.dense(_nu(cov)))[1], rel=1e-10)
    for target, C in _references(cov, support, K).items():
        d = {"wdm": op.domain, "time": TIME, "fourier": FOURIER}[target]
        t = translate_operator(op, d)
        w, V = np.linalg.eigh(0.5 * (C + C.T))
        keep = w > 1e-9 * w.max()
        assert keep.sum() == K * support.size
        w, V = w[keep], V[:, keep]
        y = transform({k: _rng.standard_normal(N) for k in names}, TIME, d)
        r = to_coordinates(y, d, names).reshape(-1)
        want = {"apply": C @ r, "solve": V @ ((V.T @ r) / w), "project": V @ (V.T @ r)}
        for method, v in want.items():
            out = getattr(t, method)(y)
            assert list(out) == list(names)
            _close(to_coordinates(out, d, names).reshape(-1), v, 1e-10)
        assert t.degrees_of_freedom == keep.sum()
        assert t.quadratic_form(y) == pytest.approx(float(r @ want["solve"]), rel=1e-10)
        assert t.log_determinant() == pytest.approx(float(np.sum(np.log(w))), rel=1e-10, abs=1e-9)


@pytest.mark.parametrize("case", ["dense-two-channels", "stack"])
@pytest.mark.parametrize("target", ["wdm", "time"])
def test_a_reversed_channel_dict_meets_the_same_blocks(case, target):
    cov, support, names = CASES[case]()
    op = CovarianceOperator(cov, _nu(cov), support, names)
    o = op if target == "wdm" else translate_operator(op, TIME)
    y = transform({k: _rng.standard_normal(N) for k in names}, TIME, o.domain)
    flipped = {"E": y["E"], "A": y["A"]}
    for method in ("apply", "solve", "project"):
        a, b = getattr(o, method)(y), getattr(o, method)(flipped)
        assert list(a) == list(b) == ["A", "E"]
        for k in names:
            if method == "solve":   # triangular solves are not bit-reproducible on every BLAS
                _close(b[k], a[k], 1e-12)
            else:
                np.testing.assert_array_equal(a[k], b[k])
    assert o.quadratic_form(flipped) == pytest.approx(o.quadratic_form(y), rel=1e-12)


def test_projection_is_zero_off_the_support_and_outputs_are_canonical():
    cov, support, names = CASES["band-subset"]()
    op = CovarianceOperator(cov, NU, support, names)
    y = transform({"A": _rng.standard_normal(N)}, TIME, WDM)
    off = ~np.isin(np.arange(N), support.cells.packed_positions())
    for method in ("apply", "solve", "project"):
        out = getattr(op, method)(y)
        assert np.all(to_coordinates(out, WDM)[0, off] == 0.0)
        assert np.all(out["A"][~wilson.active_mask(NF, NT)] == 0.0)
    kept = to_coordinates(op.project(y), WDM)[0, ~off]
    np.testing.assert_array_equal(kept, to_coordinates(y, WDM)[0, ~off])


def test_the_push_refusal_names_the_route_and_the_route_works():
    """``PushedCovariance`` cannot push a cell covariance to time; the route its error names,
    a CovarianceOperator seen from time, gives ``W^T Lambda W / dt^2``."""
    from domainia.maps import WDMToTime

    cov = DiagonalCovariance(2.0 * np.ones(N), N)
    with pytest.raises(TypeError, match=r"domainia\.operators\.CovarianceOperator.*translate_operator"):
        WDMToTime(WDM, TIME).push(cov).rows([0])
    got = operator_matrix(translate_operator(CovarianceOperator(cov, np.zeros(0), WDM, ("A",)), TIME))
    _close(got, (2.0 / DT) * np.eye(N), 1e-12)                  # W^T W = dt I


class _WithNuisance(DiagonalCovariance):
    n_u = 1


def test_the_arguments_are_validated():
    ev = EvolutionaryCovariance(PSD, ALL)
    with pytest.raises(TypeError, match="cov must be a domainia.covariance.Covariance"):
        CovarianceOperator(np.eye(N), NU, WDM, ("A",))
    with pytest.raises(TypeError, match="support must be a WDMDomain"):
        CovarianceOperator(ev, NU, ALL, ("A",))
    with pytest.raises(TypeError, match="channel_names must be a tuple"):
        CovarianceOperator(ev, NU, WDM, "A")
    with pytest.raises(ValueError, match="unique"):
        CovarianceOperator(ChannelStack([ev, ev], 2), np.r_[NU, NU], WDM, ("A", "A"))
    with pytest.raises(TypeError, match="nu must be a real numeric array"):
        CovarianceOperator(ev, "nu", WDM, ("A",))
    with pytest.raises(ValueError, match=r"^cov\.n = 32, but 2 channel\(s\) on a support of 32 cells need "
                                         r"cov\.n = 64$"):
        CovarianceOperator(ev, NU, WDM, ("A", "E"))
    with pytest.raises(ValueError, match="the ChannelStack has 2 channels; channel_names has 1"):
        CovarianceOperator(ChannelStack([ev, ev], 2), np.r_[NU, NU], WDM, ("A",))
    with pytest.raises(ValueError, match=r"every ChannelStack channel needs n = support.size = 32"):
        CovarianceOperator(ChannelStack([ev, EvolutionaryCovariance(PSD, CUT)], 2), np.r_[NU, NU], WDM,
                           ("A", "E"))
    with pytest.raises(ValueError, match="window wide-smoothstep but the support uses meyer"):
        CovarianceOperator(EvolutionaryCovariance(PSD, ALL, window=ALT_WINDOW), NU, WDM, ("A",))
    with pytest.raises(ValueError, match="window wide-smoothstep but the support uses meyer"):
        CovarianceOperator(ChannelStack([ev, EvolutionaryCovariance(PSD, ALL, window=ALT_WINDOW)], 2),
                           np.r_[NU, NU], WDM, ("A", "E"))       # each child is checked
    left, right = ALL.drop_columns(2, 0), ALL.drop_columns(0, 2)
    assert left.size == right.size and left != right
    with pytest.raises(ValueError, match="are not support.cells"):
        CovarianceOperator(EvolutionaryCovariance(PSD, left), NU, WDMDomain(right), ("A",))
    stack = ChannelStack([EvolutionaryCovariance(PSD, right), EvolutionaryCovariance(PSD, left)], 2)
    with pytest.raises(ValueError, match="are not support.cells"):
        CovarianceOperator(stack, np.r_[NU, NU], WDMDomain(right), ("A", "E"))   # each child is checked
    with pytest.raises(ValueError, match=r"cov\.n_u = 1"):
        CovarianceOperator(_WithNuisance(np.ones(N), N), NU, WDM, ("A",))
    with pytest.raises(ValueError, match="^DenseCovariance is not positive definite at nu$"):
        CovarianceOperator(DenseCovariance(-np.eye(N), N, chol_impl="numpy"), NU, WDM, ("A",))
    op = CovarianceOperator(ev, NU, WDM, ("A",))
    assert not op.nu.flags.writeable
    with pytest.raises(ValueError, match="channel_names are"):
        op.apply({"E": np.zeros((NF + 1, NT))})
