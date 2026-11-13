"""The grid, the Hz window and the atoms."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from domainia import wilson
from domainia.wilson.filter import phi_centered
from domainia.wilson.grid import Grid, active_mask, endpoint_weights
from domainia.wilson.matrix import active_W
from domainia.wilson.transform import atom_spectrum, base_atom_spectra, forward_time

from .conftest import _oracle_paths

#: The galactic-binary toy model's tree under $DOMAINIA_ORACLE_ROOT.
GB_SUBDIR = "toy_models/galactic_binary"
#: (Nf, Nt, dt): small grids plus the three galactic-binary analysis grids (N // NT, NT, 5 s).
ATOM_GRIDS = [(4, 8, 1.0), (16, 32, 0.5), (6570, 240, 5.0), (7200, 438, 5.0), (7008, 900, 5.0)]


def test_grid_is_frozen_positional_and_unpacks():
    g = Grid(8, 16, 0.5)
    assert tuple(g) == (8, 16, 0.5) and g.N == 128
    with pytest.raises(AttributeError):
        g.Nf = 4  # type: ignore[misc]


def test_endpoint_weights():
    c = endpoint_weights(9)
    assert c[0] == c[-1] == 0.5 and np.all(c[1:-1] == 1.0)


@pytest.mark.parametrize("grid", [(8, 16, 1.0), (16, 32, 0.5), (8, 32, 1.7), (64, 256, 2.0**-10)])
def test_hz_identity(grid):
    """The Hz window (Paper I eq:Meyer-def, flat value 1/sqrt(Delta F)) with kappa = sqrt(2)/Nf
    gives the product of the omega-normalised window of releases <= 0.11 (1/sqrt(2 pi Delta F),
    kappa = 2 sqrt(pi)/Nf, the frozen oracle) to 1e-15."""
    import vendored_2f7136c.filter as ofilter

    Nf, Nt, dt = grid
    h = (np.sqrt(2.0) / Nf) * phi_centered(Nf, Nt, dt)
    w = (2.0 * np.sqrt(np.pi) / Nf) * ofilter.phi_centered(Nf, Nt, dt)
    nz = w != 0
    assert np.array_equal(nz, h != 0)
    assert np.max(np.abs(h[nz] / w[nz] - 1.0)) <= 1e-15
    assert np.max(phi_centered(Nf, Nt, dt)) == 1.0 / np.sqrt(wilson.grid.delta_F(Nf, dt))


#: grids whose window arithmetic is not "nice": dt with long mantissas, odd Nf, the
#: galactic binary's 5 s.
WINDOW_GRIDS = [(8, 16, 1.0), (16, 32, 0.5), (8, 32, 1.7), (64, 256, 2.0**-10), (6, 10, 0.3),
                (32, 64, 1.0 / 3.0), (4, 8, 7.123456789e-5), (12, 24, 123.456), (6570, 240, 5.0)]


def _phi_hz_june(Nf, Nt, dt, d=4):
    """June ``wdm/filter.py`` arithmetic, verbatim (the pre-``grid.delta_F`` phi_hz body)."""
    from domainia.wilson.filter import nu_d

    T = Nf * Nt * dt
    f = (np.arange(Nt) - Nt // 2) / T
    delta_F = 1.0 / (2.0 * dt * Nf)
    A = delta_F / 4.0
    B = delta_F / 2.0
    amp = 1.0 / np.sqrt(delta_F)
    f_abs = np.abs(f)
    out = np.where(f_abs < A, amp, 0.0)
    rolloff = (f_abs >= A) & (f_abs < A + B)
    return np.where(rolloff, amp * np.cos(np.pi / 2.0 * nu_d((f_abs - A) / B, d)), out)


@pytest.mark.parametrize("grid", WINDOW_GRIDS)
def test_windows_are_bitwise_the_oracle(grid):
    """phi_centered == the June Hz arithmetic bit for bit (now taking Delta F from
    grid.delta_F), and sqrt(2 pi) times the frozen omega oracle to 2 ulp."""
    import vendored_2f7136c.filter as ofilter

    Nf, Nt, dt = grid
    phi = phi_centered(Nf, Nt, dt)
    assert np.array_equal(phi, _phi_hz_june(Nf, Nt, dt))
    assert np.max(np.abs(phi - np.sqrt(2.0 * np.pi) * ofilter.phi_centered(Nf, Nt, dt))) \
        <= 4.5e-16 * np.max(phi)


@pytest.mark.parametrize("dt", [5e-324, 1e-320, 1e308])
def test_unrepresentable_delta_F_raises_instead_of_returning_zeros(dt):
    """dt is legal on its own but 1/(2 dt Nf) is 0 or inf: the transform used to return
    a finite all-zero grid (lisa/wdm b5973b7 fixed it in filter.py via grid.delta_F)."""
    Nf, Nt = 8, 16
    calls = {
        "phi_centered": lambda: phi_centered(Nf, Nt, dt),
        "forward_time": lambda: forward_time(np.ones(Nf * Nt), Nf, Nt, dt),
        "inverse_time": lambda: wilson.inverse_time(np.ones((Nf + 1, Nt)), Nf, Nt, dt),
    }
    for name, call in calls.items():
        with pytest.raises(ValueError, match="delta_F is not representable"):
            call()
            pytest.fail(f"{name} returned for dt={dt!r}")


def test_delta_F_rejects_unrepresentable_widths():
    from domainia.wilson.grid import delta_F

    assert delta_F(8, 1.0) == 0.0625
    with pytest.raises(ValueError, match="not representable"):
        delta_F(8, 5e-324)
    with pytest.raises(ValueError, match="Nf must be even"):
        delta_F(7, 1.0)


def test_rfft_packing_rejects_odd_N():
    from domainia.wilson.matrix import real_to_rfft, rfft_to_real

    x = np.random.default_rng(0).standard_normal(32)
    assert np.allclose(real_to_rfft(rfft_to_real(np.fft.rfft(x), 32), 32), np.fft.rfft(x))
    with pytest.raises(ValueError, match="N must be even"):
        rfft_to_real(np.fft.rfft(x[:5]), 5)
    with pytest.raises(ValueError, match="N must be even"):
        real_to_rfft(np.zeros(5), 5)


def test_atoms_are_the_rows_of_the_transform():
    Nf, Nt, dt = 8, 16, 0.5
    Wa = active_W(Nf, Nt, dt)
    base = base_atom_spectra(Nf, Nt, dt)
    assert sorted(base) == [(m, a) for m in range(Nf + 1) for a in (0, 1)
                            if active_mask(Nf, Nt)[m, a]]
    m, n = np.nonzero(active_mask(Nf, Nt))
    for p in range(m.size):
        atom = np.fft.irfft(atom_spectrum(int(m[p]), int(n[p]), Nf, Nt, dt, base=base), n=Nf * Nt)
        assert np.allclose(atom, Wa[p], rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="not an active cell"):
        atom_spectrum(0, 1, Nf, Nt, dt, base=base)


def _gb_reference(tmp_path: Path, grids) -> dict:
    """phi computed by the frozen galactic-binary code, from a temporary copy."""
    (gb_dir,) = _oracle_paths(GB_SUBDIR)
    copy = tmp_path / "gb"
    copy.mkdir()
    for name in ("config.py", "wdm_atoms.py", "geometry.py"):
        shutil.copyfile(gb_dir / name, copy / name)
    out = tmp_path / "gb_ref.npz"
    code = (
        "import sys, numpy as np\n"
        "from wdm_atoms import phi_centered\n"
        f"grids = {grids!r}\n"
        "res = {}\n"
        "for i, (Nf, Nt, dt) in enumerate(grids):\n"
        "    N = Nt * Nf\n"
        "    phi = phi_centered(Nt, Nf, dt, N * dt)\n"
        "    res[f'phi{i}'] = phi\n"
        f"np.savez({str(out)!r}, **res)\n"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(copy))
    subprocess.run([sys.executable, "-P", "-c", code], env=env, cwd=str(copy), check=True)
    return dict(np.load(out))


def test_window_is_bitwise_the_gb_copy(tmp_path):
    ref = _gb_reference(tmp_path, ATOM_GRIDS)
    for i, (Nf, Nt, dt) in enumerate(ATOM_GRIDS):
        assert np.array_equal(phi_centered(Nf, Nt, dt), ref[f"phi{i}"])


def test_atom_spectrum_rejects_out_of_range_n_and_foreign_base():
    Nf, Nt, dt = 8, 16, 1.0
    base = base_atom_spectra(Nf, Nt, dt)
    ref = atom_spectrum(3, 5, Nf, Nt, dt)               # base built internally
    assert np.array_equal(atom_spectrum(3, 5, Nf, Nt, dt, base=base), ref)
    # n outside 0..Nt-1 used to wrap silently onto another cell's atom
    for n in (Nt, Nt + 1, -1, 100):
        with pytest.raises(ValueError, match=r"n must lie in 0\.\.15"):
            atom_spectrum(3, n, Nf, Nt, dt, base=base)
    for m in (-1, Nf + 1):
        with pytest.raises(ValueError, match="not an active cell"):
            atom_spectrum(m, 4, Nf, Nt, dt, base=base)
    # a base built for another grid: same N with another Nf, another N, another dt
    for other, match in (((16, 8, dt), "base does not hold"),
                         ((4, 8, dt), "base does not hold"),
                         ((8, 32, dt), "base does not hold"),
                         ((Nf, Nt, 2.0), "not built for dt=1.0"),
                         ((Nf, Nt, 0.5), "not built for dt=1.0")):
        foreign = base_atom_spectra(*other)
        with pytest.raises(ValueError, match=match):
            atom_spectrum(3, 5, Nf, Nt, dt, base=foreign)


def test_time_pixel_centers_doctest_value():
    assert wilson.grid.time_pixel_centers(8, 4, 1.0).tolist() == [0.0, 8.0, 16.0, 24.0]
