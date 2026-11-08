"""domainia.translate.to_coordinates / from_coordinates: the real coordinates of the Operator
contract, one row per channel, in channel_names order."""

from __future__ import annotations

import numpy as np
import pytest
from test_translate import REFUSALS

from domainia import wilson
from domainia.domains import CellSet, FourierDomain, TimeDomain, WDMDomain
from domainia.testing import ALT_WINDOW
from domainia.translate import from_coordinates, log_coordinate_scale, to_coordinates, transform

NF, NT, DT = 8, 16, 0.25
N = NF * NT
GRID = wilson.Grid(NF, NT, DT)
TIME = TimeDomain(N, DT)
ODD_TIME = TimeDomain(15, 0.5)
DOMAINS = {
    "time": (TIME, TIME),
    "fourier": (TIME, FourierDomain.full(N, DT)),
    "fourier-odd": (ODD_TIME, FourierDomain.full(15, 0.5)),
    "wdm": (TIME, WDMDomain.full(GRID)),
    "wdm-alt": (TIME, WDMDomain.full(wilson.Grid(16, 8, DT), window=ALT_WINDOW)),
}
_rng = np.random.default_rng(47)


def _record(time):
    return {"A": _rng.standard_normal(time.N), "E": _rng.standard_normal(time.N)}


def _reference(values, domain):
    """The coordinates by their definition (numpy only, no to_coordinates), one row per channel."""
    rows = []
    for v in values.values():
        if isinstance(domain, TimeDomain):
            rows.append(v)
        elif isinstance(domain, FourierDomain):
            h = (domain.N - 1) // 2
            r = [v[:1].real, np.sqrt(2) * v[1:h + 1].real, np.sqrt(2) * v[1:h + 1].imag]
            rows.append(np.concatenate(r + ([v[-1:].real] if domain.N % 2 == 0 else [])))
        else:
            g = domain.cells.grid
            m, n = wilson.grid.packed_cells(g.Nf, g.Nt)
            rows.append(v[m, n])
    return np.stack(rows)


@pytest.mark.parametrize("name", DOMAINS)
def test_the_coordinates_are_the_documented_layout(name):
    time, d = DOMAINS[name]
    y = transform(_record(time), time, d)
    R = to_coordinates(y, d)
    assert R.dtype == np.float64 and R.shape == (2, time.N)
    np.testing.assert_array_equal(R, _reference(y, d))
    if isinstance(d, FourierDomain):
        X = d.dt * np.fft.rfft(transform(y, d, time)["A"])      # dt * rfft(x), independently
        h = (d.N - 1) // 2
        np.testing.assert_allclose(R[0, 1:h + 1], np.sqrt(2) * X[1:h + 1].real, rtol=0,
                                   atol=1e-13 * np.max(np.abs(X)))
        np.testing.assert_allclose(R[0, h + 1:2 * h + 1], np.sqrt(2) * X[1:h + 1].imag, rtol=0,
                                   atol=1e-13 * np.max(np.abs(X)))


def test_wdm_coordinates_follow_the_all_active_cell_set():
    d = DOMAINS["wdm"][1]
    y = transform(_record(TIME), TIME, d)
    cells = CellSet.all_active(GRID)
    m, n = wilson.grid.packed_cells(NF, NT)
    assert cells.m == tuple(m.tolist()) and cells.n == tuple(n.tolist())
    np.testing.assert_array_equal(to_coordinates(y, d)[1], y["E"][cells.m_array, cells.n_array])


@pytest.mark.parametrize("name", DOMAINS)
def test_from_coordinates_inverts_to_the_canonical_layout(name):
    time, d = DOMAINS[name]
    y = transform(_record(time), time, d)
    back = from_coordinates(to_coordinates(y, d), d, ("A", "E"))
    assert list(back) == ["A", "E"]
    for k in y:
        assert back[k].dtype == y[k].dtype and back[k].shape == y[k].shape
        if isinstance(d, FourierDomain):                          # sqrt(2) out and back: 1 ulp
            np.testing.assert_allclose(back[k], y[k], rtol=0, atol=1e-15 * np.max(np.abs(y[k])))
        else:
            np.testing.assert_array_equal(back[k], y[k])
    R = _rng.standard_normal((2, time.N))
    out = from_coordinates(R, d, ("A", "E"))
    if isinstance(d, FourierDomain):
        assert all(v[0].imag == 0.0 for v in out.values())
        if d.N % 2 == 0:
            assert all(v[-1].imag == 0.0 for v in out.values())
    if isinstance(d, WDMDomain):
        g = d.cells.grid
        assert all(np.all(v[~wilson.active_mask(g.Nf, g.Nt)] == 0.0) for v in out.values())
    np.testing.assert_allclose(to_coordinates(transform(out, d, d), d), R, rtol=0,
                               atol=1e-15 * np.max(np.abs(R)))


@pytest.mark.parametrize("name", DOMAINS)
def test_the_coordinates_are_an_orthogonal_map_of_the_time_samples_times_the_log_scale(name):
    """Column ``j`` of ``Q`` is the coordinates of the unit sample ``e_j``, scaled by
    ``exp(-log_coordinate_scale)``: ``Q^T Q = I``."""
    time, d = DOMAINS[name]
    Q = np.stack([to_coordinates(transform({"A": e}, time, d), d)[0] for e in np.eye(time.N)], axis=1)
    Q *= np.exp(-log_coordinate_scale(d))
    np.testing.assert_allclose(Q.T @ Q, np.eye(time.N), rtol=0, atol=1e-12)


def test_channel_names_pin_the_row_order():
    d = DOMAINS["fourier"][1]
    y = transform(_record(TIME), TIME, d)
    flipped = {"E": y["E"], "A": y["A"]}
    np.testing.assert_array_equal(to_coordinates(flipped, d, ("A", "E")), to_coordinates(y, d))
    np.testing.assert_array_equal(to_coordinates(flipped, d)[0], to_coordinates(y, d)[1])  # dict order
    out = from_coordinates(to_coordinates(y, d), d, ("E", "A"))
    assert list(out) == ["E", "A"]
    np.testing.assert_allclose(out["E"], y["A"], rtol=0, atol=1e-15 * np.max(np.abs(y["A"])))
    with pytest.raises(ValueError, match=r"^channel_data has channels \('A', 'E'\); channel_names are "
                                         r"\('A', 'T'\)$"):
        to_coordinates(y, d, ("A", "T"))
    with pytest.raises(ValueError, match="channel_names are"):
        to_coordinates({"A": y["A"]}, d, ("A", "E"))


def _refusal(fn, *args):
    """``(type, message)`` of the refusal ``fn(*args)`` raises, or ``None``; the role word that
    names a bad domain (``source`` in transform, ``domain`` here) is read as ``domain``."""
    try:
        fn(*args)
    except (TypeError, ValueError) as err:
        msg = str(err)
        return type(err), "domain" + msg[len("source"):] if msg.startswith("source") else msg
    return None


@pytest.mark.parametrize("case", REFUSALS)
def test_input_refusals_are_transforms(case):
    """Every refusal case of tests/test_translate.py, on each of its two domains in turn:
    ``to_coordinates(data, d)`` refuses exactly when ``transform(data, d, d)`` does, with the
    same type and message.  ``t0`` is a rule of the pair alone, so on that case both domains
    accept the data."""
    data, src, dst, _, _ = REFUSALS[case]
    seen = []
    for d in (src, dst):
        want = _refusal(transform, data, d, d)
        assert _refusal(to_coordinates, data, d) == want
        seen.append(want)
    assert any(seen) or case == "t0"


def test_argument_refusals():
    d = DOMAINS["wdm"][1]
    y = transform(_record(TIME), TIME, d)
    R = to_coordinates(y, d)
    with pytest.raises(TypeError, match="channel_names must be a tuple"):
        to_coordinates(y, d, "AE")
    with pytest.raises(TypeError, match="channel names must be non-empty strings"):
        from_coordinates(R, d, ("A", 1))
    with pytest.raises(ValueError, match="unique"):
        from_coordinates(R, d, ("A", "A"))
    with pytest.raises(ValueError, match="channel_names is empty"):
        from_coordinates(R[:0], d, ())
    with pytest.raises(ValueError, match=r"^R has shape \(1, 128\); 2 channel\(s\) on this WDMDomain need "
                                         r"shape \(2, 128\)$"):
        from_coordinates(R[:1], d, ("A", "E"))
    with pytest.raises(ValueError, match="complex"):
        from_coordinates(R + 0j, d, ("A", "E"))
    with pytest.raises(ValueError, match="not finite"):
        from_coordinates(np.where(np.arange(N) == 3, np.nan, R), d, ("A", "E"))
    with pytest.raises(TypeError, match="real coordinates"):
        from_coordinates(R.astype(object), d, ("A", "E"))
    with pytest.raises(TypeError, match="masked"):
        from_coordinates(np.ma.MaskedArray(R), d, ("A", "E"))
    with pytest.raises(ValueError, match="^domain: translate handles only WDMDomain.full"):
        from_coordinates(R, WDMDomain(CellSet.bands(GRID, (2, 3))), ("A", "E"))
    with pytest.raises(TypeError, match="^domain must be a full-record domain"):
        from_coordinates(R, GRID, ("A", "E"))
