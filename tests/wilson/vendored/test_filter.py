"""Tests for wdm.filter."""

from __future__ import annotations

import numpy as np
import pytest

from domania.wilson.filter import nu_d, phi_centered


def test_nu_d_endpoints():
    """nu_d(0) = 0, nu_d(1) = 1 (boundary conditions for the rolloff)."""
    assert nu_d(np.array([0.0])).item() == 0.0
    assert nu_d(np.array([1.0])).item() == 1.0


def test_nu_d_clips_outside_unit_interval():
    """nu_d clips x to [0, 1] internally; outside-range values give boundary values."""
    assert nu_d(np.array([-0.5])).item() == 0.0
    assert nu_d(np.array([1.5])).item() == 1.0


def test_nu_d_monotonic():
    """nu_d is monotonically nondecreasing on [0, 1]."""
    x = np.linspace(0.0, 1.0, 200)
    y = nu_d(x)
    assert np.all(np.diff(y) >= -1e-15)


def test_phi_centered_peak_at_middle(simple_grid):
    """phi_centered attains its maximum at index Nt/2.

    The filter has a flat top, so several indices share the max value;
    we just check that index Nt/2 is one of them.
    """
    g = simple_grid
    phi = phi_centered(g.Nf, g.Nt, g.dt)
    assert phi[g.Nt // 2] == phi.max()


def test_phi_compact_support(medium_grid):
    """phi is zero outside |f| < 3*dF/4."""
    g = medium_grid
    phi = phi_centered(g.Nf, g.Nt, g.dt)
    f = (np.arange(g.Nt) - g.Nt // 2) / g.T
    cutoff = 3.0 * g.dF / 4.0
    outside = np.abs(f) >= cutoff
    assert np.allclose(phi[outside], 0.0, atol=1e-15)


def test_phi_amplitude_is_one_over_sqrt_dF(simple_grid):
    """In the flat region, phi == 1/sqrt(Delta F) (Paper I eq:Meyer-def, Hz)."""
    g = simple_grid
    phi = phi_centered(g.Nf, g.Nt, g.dt)
    # The center value (f = 0) is in the flat region.
    expected = 1.0 / np.sqrt(g.dF)
    assert np.isclose(phi[g.Nt // 2], expected, rtol=1e-12)


@pytest.mark.parametrize("d", [1, 2, 4, 8])
def test_smoothness_order_d_changes_the_window(simple_grid, d):
    """``d`` is a public parameter on two functions and must be plumbed.

    Dropping ``d`` in the call chain used to pass the whole suite, because
    nothing ever varied it away from the default of 4.
    """
    g = simple_grid
    phi = phi_centered(g.Nf, g.Nt, g.dt, d)
    default = phi_centered(g.Nf, g.Nt, g.dt)

    assert phi.shape == (g.Nt,)
    assert phi[g.Nt // 2] == phi.max()
    # Support is d-independent; the rolloff shape is not.
    f = (np.arange(g.Nt) - g.Nt // 2) / g.T
    assert np.allclose(phi[np.abs(f) >= 0.75 * g.dF], 0.0, atol=1e-15)
    if d != 4:
        assert not np.allclose(phi, default)


@pytest.mark.parametrize("bad_d", [0, -1])
def test_invalid_smoothness_order_raises(bad_d):
    """``betainc(0, 0, x)`` is NaN, so a bad ``d`` must not slip through."""
    with pytest.raises(ValueError, match="d must be >= 1"):
        nu_d(np.array([0.5]), d=bad_d)


def test_non_integer_smoothness_order_raises_type_error():
    with pytest.raises(TypeError, match="d must be an integer, got float"):
        nu_d(np.array([0.5]), d=2.5)
