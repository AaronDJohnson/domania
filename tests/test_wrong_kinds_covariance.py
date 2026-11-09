"""``domainia.covariance`` refuses a wrong kind with ``TypeError`` and a wrong value with
``ValueError`` (spec 2026-11-08 §2.3, §2.5, §2.6, §2.9)."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from domainia import wilson
from domainia.covariance import (BlockTridiagonalCovariance, ChannelStack, DenseCovariance,
                                 DiagonalCovariance, EvolutionaryCovariance, dense_factor, quad_form)
from domainia.domains import CellSet
from domainia.spectra import GenericPSD
from domainia.variance import BandAveragedVariance
from domainia.wilson.backend import NUMPY

NF, NT, DT = 8, 16, 1 / 16
T_REC = NF * NT * DT
CELLS = CellSet.all_active(wilson.Grid(NF, NT, DT))
NU = np.array([-1.7, 0.9])
PSD = GenericPSD(lambda f, t, nu: 1.0 + 0.0 * f * t)
V = np.array([1.0, 2.0, 4.0, 8.0])
S = np.diag(V)
Y = np.ones(4)
DIRECT = {"builder": "direct", "assembly": "dense"}


def _blocks(nu, be):
    """Two bands of two cells, ragged ``(diag, up)``."""
    return [be.asarray(np.eye(2)), be.asarray(2.0 * np.eye(2))], [be.asarray(np.zeros((2, 2)))]


def _ev(**kw):
    return EvolutionaryCovariance(PSD, CELLS, **kw)


def _cases(slots, wrong):
    """``(make, name, value, kind)`` for every slot and wrong value.  A slot whose third entry
    is True has ``None`` as its documented default, so ``None`` is not a wrong kind there."""
    return [pytest.param(make, name, value, kind, id=f"{sid}-{value!r}")
            for sid, (make, name, none_is_default) in slots.items()
            for value, kind in wrong if not (value is None and none_is_default)]


# --- integers (spec §2.3, §2.9) -------------------------------------------------------------

INT_SLOTS = {
    "DiagonalCovariance.n": (lambda v: DiagonalCovariance(V, v), "n", False),
    "DiagonalCovariance.channel": (
        lambda v: DiagonalCovariance(BandAveragedVariance(PSD, CELLS), CELLS.size, channel=v), "channel", True),
    "DenseCovariance.n": (lambda v: DenseCovariance(S, v, chol_impl="numpy"), "n", False),
    "ChannelStack.params_per_channel": (
        lambda v: ChannelStack([DiagonalCovariance(V, 4)], v), "params_per_channel", False),
    "BlockTridiagonalCovariance.sizes": (lambda v: BlockTridiagonalCovariance(_blocks, (2, v)), r"sizes\[1\]", False),
    "EvolutionaryCovariance.P": (lambda v: _ev(P=v), "P", False),
    "EvolutionaryCovariance.P-direct": (lambda v: _ev(P=v, **DIRECT), "P", False),
}
WRONG_INTS = [(2.0, "float"), (2.7, "float"), (True, "bool"), (np.True_, "bool"), ("2", "str"),
              (None, "NoneType"), (np.array(2), "ndarray"), (np.float64(2), "float64")]


@pytest.mark.parametrize(("make", "name", "value", "kind"), _cases(INT_SLOTS, WRONG_INTS))
def test_counts_of_the_wrong_kind_raise_type_error(make, name, value, kind):
    """``int()`` used to truncate ``2.7`` to 2 and take ``True``, ``"2"`` and a 0-d array;
    ``P`` was read only by ``builder="interp"``, at the first factorisation."""
    with pytest.raises(TypeError, match=f"^{name} must be an integer, got {kind}$"):
        make(value)


def test_numpy_integer_counts_pass_and_are_stored_as_int():
    assert type(DiagonalCovariance(V, np.int64(4)).n) is int
    assert type(DenseCovariance(S, np.int64(4), chol_impl="numpy").n) is int
    assert type(ChannelStack([DiagonalCovariance(V, 4)], np.int64(2)).k) is int
    d = DiagonalCovariance(BandAveragedVariance(PSD, CELLS), CELLS.size, channel=np.int64(1))
    assert d.variance.kwargs == (("channel", 1),) and type(d.variance.kwargs[0][1]) is int
    b = BlockTridiagonalCovariance(_blocks, np.array([2, 2]))
    assert b.sizes == (2, 2) and all(type(s) is int for s in b.sizes) and b.n == 4
    ev = _ev(P=np.int64(4))
    assert ev.block_evaluator.P == 4 and type(ev.block_evaluator.P) is int
    assert len(ev.blocks(NU)[0]) == NF + 1


@pytest.mark.parametrize(("sizes", "kind"), [((2.0, 2.0), "float"), (np.array([2.0, 2.0]), "float64"),
                                             ((True, 2), "bool"), ("22", "str")])
def test_sizes_are_checked_element_by_element(sizes, kind):
    with pytest.raises(TypeError, match=rf"^sizes\[0\] must be an integer, got {kind}$"):
        BlockTridiagonalCovariance(_blocks, sizes)


def test_count_ranges_stay_value_errors():
    """``P < 1`` is now refused at construction, for both builders, with the plan's message."""
    with pytest.raises(ValueError, match="params_per_channel must be non-negative; got -1"):
        ChannelStack([DiagonalCovariance(V, 4)], -1)
    for kw in ({}, DIRECT):
        for P in (0, -1):
            with pytest.raises(ValueError, match=f"^P must be >= 1, got {P}$"):
                _ev(P=P, **kw)


# --- flags (spec §2.5) ----------------------------------------------------------------------

FLAG_SLOTS = {
    "DenseCovariance.symmetrize": (
        lambda v: DenseCovariance(S, 4, chol_impl="numpy", symmetrize=v), "symmetrize", False),
    "dense_factor.symmetrize": (
        lambda v: dense_factor(S, chol_impl="numpy", symmetrize=v, be=NUMPY), "symmetrize", False),
    "EvolutionaryCovariance.symmetrize-blocktri": (lambda v: _ev(symmetrize=v), "symmetrize", False),
    "EvolutionaryCovariance.symmetrize-dense": (lambda v: _ev(assembly="dense", symmetrize=v), "symmetrize", False),
    "Covariance.factor.jit": (lambda v: DiagonalCovariance(V, 4).factor(NU, jit=v), "jit", True),
}
WRONG_FLAGS = [("no", "str"), (1, "int"), (0, "int"), (0.5, "float"), (None, "NoneType"),
               (np.array(True), "ndarray")]


@pytest.mark.parametrize(("make", "name", "value", "kind"), _cases(FLAG_SLOTS, WRONG_FLAGS))
def test_flags_of_the_wrong_kind_raise_type_error(make, name, value, kind):
    """``bool()`` used to read ``"no"`` as True; ``EvolutionaryCovariance`` read ``symmetrize``
    only for ``assembly="dense"``, and ``factor`` read ``jit`` only on JAX."""
    with pytest.raises(TypeError, match=f"^{name} must be a bool, got {kind}$"):
        make(value)


def test_numpy_bool_flags_pass_and_are_stored_as_bool():
    assert DenseCovariance(S, 4, chol_impl="numpy", symmetrize=np.False_).symmetrize is False
    assert _ev(assembly="dense", symmetrize=np.False_).impl.symmetrize is False
    F = dense_factor(S, chol_impl="numpy", symmetrize=np.True_, be=NUMPY)
    assert float(F.logdet) == pytest.approx(np.log(64.0), rel=1e-15)
    for jit in (None, np.False_, True):
        F = DiagonalCovariance(V, 4).factor(NU, jit=jit)
        assert float(F.logdet) == pytest.approx(np.log(64.0), rel=1e-15)


# --- string options (spec §2.5) -------------------------------------------------------------

DIAG_F = DiagonalCovariance(V, 4).factor(NU)
CHOL_F = DenseCovariance(S, 4, chol_impl="numpy").factor(NU)
BLOCK_F = BlockTridiagonalCovariance(_blocks, (2, 2)).factor(NU)
STACK_F = ChannelStack([DiagonalCovariance(V, 4), DiagonalCovariance(V, 4)], 0).factor(np.zeros(0))

STR_SLOTS = {
    "DenseCovariance.chol_impl": (lambda v: DenseCovariance(S, 4, chol_impl=v), "chol_impl", False),
    "dense_factor.chol_impl": (
        lambda v: dense_factor(S, chol_impl=v, symmetrize=True, be=NUMPY), "chol_impl", False),
    "EvolutionaryCovariance.chol_impl": (lambda v: _ev(chol_impl=v), "chol_impl", False),
    "BlockTridiagonalCovariance.layout": (
        lambda v: BlockTridiagonalCovariance(_blocks, (2, 2), layout=v), "layout", False),
    "EvolutionaryCovariance.layout": (lambda v: _ev(layout=v), "layout", True),
    "EvolutionaryCovariance.builder": (lambda v: _ev(builder=v), "builder", False),
    "EvolutionaryCovariance.assembly": (lambda v: _ev(assembly=v), "assembly", False),
    "DiagFactor.quad.form": (lambda v: DIAG_F.quad(Y, form=v), "form", False),
    "CholFactor.quad.form": (lambda v: CHOL_F.quad(Y, form=v), "form", False),
    "BlockTriCovFactor.quad.form": (lambda v: BLOCK_F.quad(Y, form=v), "form", False),
    "StackFactor.quad.form": (lambda v: STACK_F.quad(np.ones((2, 4)), form=v), "form", False),
    "quad_form.form": (lambda v: quad_form(DIAG_F, Y, v, NUMPY), "form", False),
}
WRONG_STRS = [(3, "int"), (0, "int"), (False, "bool"), (None, "NoneType"), (b"solve", "bytes"),
              (["solve"], "list"), (np.array("elementwise"), "ndarray")]


@pytest.mark.parametrize(("make", "name", "value", "kind"), _cases(STR_SLOTS, WRONG_STRS))
def test_string_options_of_the_wrong_kind_raise_type_error(make, name, value, kind):
    """A wrong kind used to reach the membership ``ValueError``, or pass:
    ``EvolutionaryCovariance(layout=0)`` was ``"ragged"``, and ``DiagFactor.quad`` took
    ``np.array("elementwise")`` as ``"elementwise"``."""
    with pytest.raises(TypeError, match=f"^{name} must be a string, got {kind}$"):
        make(value)


def test_numpy_str_options_pass():
    assert DenseCovariance(S, 4, chol_impl=np.str_("scipy")).chol_impl == "scipy"
    assert BlockTridiagonalCovariance(_blocks, (2, 2), layout=np.str_("padded")).jittable
    assert _ev(builder=np.str_("direct"), assembly=np.str_("dense")).n == CELLS.size
    assert float(DIAG_F.quad(Y, form=np.str_("elementwise"))) == pytest.approx(1.875, rel=1e-15)
    assert float(CHOL_F.quad(Y, form=np.str_("whiten"))) == pytest.approx(1.875, rel=1e-15)


@pytest.mark.parametrize(("make", "match"), [
    (lambda: DenseCovariance(S, 4, chol_impl="lapack"), "chol_impl must be 'numpy' or 'scipy'; got 'lapack'"),
    (lambda: BlockTridiagonalCovariance(_blocks, (2, 2), layout="dense"),
     "layout must be 'ragged' or 'padded'; got 'dense'"),
    (lambda: _ev(layout=""), "layout must be 'ragged' or 'padded'; got ''"),
    (lambda: _ev(builder="bessel"), "unknown builder 'bessel'"),
    (lambda: _ev(assembly="banded"), "assembly must be 'blocktri' or 'dense'"),
    (lambda: DIAG_F.quad(Y, form="bogus"), "DiagFactor has no quadratic form 'bogus'"),
    (lambda: CHOL_F.quad(Y, form="elementwise"), "CholFactor has no quadratic form 'elementwise'"),
], ids=["chol_impl", "layout", "layout-empty", "builder", "assembly", "form-diag", "form-chol"])
def test_unknown_strings_stay_value_errors(make, match):
    """``layout=""`` used to be read as ``"ragged"`` by ``EvolutionaryCovariance``."""
    with pytest.raises(ValueError, match=match):
        make()


# --- covariances and callables (spec §2.6) --------------------------------------------------

@pytest.mark.parametrize(("covs", "index", "kind"), [
    ([1], 0, "int"), ("ab", 0, "str"), ([None], 0, "NoneType"), ([S], 0, "ndarray"),
    ([DiagonalCovariance(V, 4), object()], 1, "object"),
])
def test_channel_stack_channels_must_be_covariances(covs, index, kind):
    """Each used to fail with ``AttributeError: ... has no attribute 'n'``."""
    with pytest.raises(TypeError, match=rf"^covs\[{index}\] must be a Covariance, got {kind}$"):
        ChannelStack(covs, 1)


def test_channel_stack_takes_any_covariances_and_still_needs_one():
    st = ChannelStack((DiagonalCovariance(V, 4), DenseCovariance(S, 4, chol_impl="numpy")), 1)
    assert st.n == 8
    with pytest.raises(ValueError, match="ChannelStack needs at least one channel"):
        ChannelStack([], 1)


def test_the_psd_must_be_callable():
    with pytest.raises(TypeError, match="^psd must be callable, got object$"):
        EvolutionaryCovariance(object(), CELLS)


# --- t_max (spec §2.4, §2.9) ----------------------------------------------------------------

@pytest.mark.parametrize("kw", [{}, DIRECT], ids=["interp", "direct"])
@pytest.mark.parametrize(("t_max", "kind"), [
    ("300", "str"), (True, "bool"), (np.True_, "bool"), (1 + 0j, "complex"), (np.complex128(8.0), "complex128"),
    (Decimal("8"), "Decimal"), (np.array(8.0), "ndarray"), ([8.0], "list"),
])
def test_t_max_of_the_wrong_kind_raises_type_error_at_construction(t_max, kind, kw):
    """``t_max="300"`` used to be accepted, and parsed by ``float()`` at the first factorisation."""
    with pytest.raises(TypeError, match=f"^t_max must be a real number, got {kind}$"):
        _ev(t_max=t_max, **kw)


def test_real_t_max_passes_as_a_float_and_its_range_stays_lazy():
    ref = _ev().blocks(NU)
    for t_max in (np.float64(T_REC), np.int64(8)):
        ev = _ev(t_max=t_max)
        assert type(ev.block_evaluator.t_max) is float
        got = ev.blocks(NU)
        assert all(np.array_equal(a, b) for a, b in zip(got[0], ref[0], strict=True))
    late = _ev(t_max=0.5 * T_REC)                 # constructs: the range is the plan's check
    with pytest.raises(ValueError, match="pair mid-times"):
        late.blocks(NU)
