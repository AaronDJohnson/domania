r"""The sinusoid's exact time covariance through its Vandermonde square root (WDM II Eqs. 52,
95-97; spec section 11.1): :class:`VandermondeExactTime` and the Fourier-Cauchy factor
:class:`FCFactor`.

For the drifting power law :math:`S = A_n e^{\alpha(t) L(f)}`, :math:`\alpha(t) = \alpha_c +
\alpha_d (t/T - 1/2)`, :math:`T = N\,\delta t`, Eq. 95 is :math:`\tilde C = C/A_n = B_c B_c^{\mathsf H}`
with the square, column-scaled complex Vandermonde matrix

.. math::  B_c = V\,\mathrm{diag}(\sigma),\quad V_{ik} = x_k^i,\quad x_k = \rho_k\omega^k,\quad
           \omega = e^{2\pi i/N},\quad \rho_k = e^{u_k},\quad u_k = \alpha_d L_k / (2N),

:math:`\sigma_k^2 = (\delta f/2)\,e^{(\alpha_c - \alpha_d/2) L_k}`, :math:`\delta f = 1/(N\delta t)`,
:math:`L_k = L(|f_k|)` on the two-sided ``fftfreq`` index ``k = 0..N-1``.  The DC node is
:math:`\rho_0`, the Nyquist node :math:`-\rho_{N/2}` (:math:`\omega^{N/2} = -1` exactly in the
tables), and :math:`x_{N-k} = \bar x_k`: the endpoint weights :math:`\eta_k` of Eq. 51 come out of
the two-sided sum by themselves.  The real voice factor ``ExactTimeCovariance.voice_factor`` is
:math:`G = B_c P` with ``P`` block-unitary, so :meth:`FCFactor.whiten` (packed by
:func:`pack_real`) is :math:`G^{-1} b`.

* **Log-determinant** (closed form, no fit): :math:`\ln|\tilde C| = \alpha_c S_L - \alpha_d S_L/(2N)
  - N\ln(2\,\delta t) + R(\alpha_d)` -- the Weyl term of ``ExactTimeCovariance.weyl_logdet`` plus
  :math:`R = \sum_{j<k}\mathrm{log1p}\bigl(\sinh^2\tfrac{u_k - u_j}{2} / \sin^2\tfrac{\pi(k-j)}{N}\bigr)
  \ge 0` (even in :math:`\alpha_d`, :math:`R(0) = 0`; the exact quantity the historical
  :math:`c_N\alpha_d^2` fitted).  Time domain: no :math:`\ln\delta t` of the WDM Jacobian.
* **Solve** (Fourier-Cauchy): with :math:`X = \mathrm{fft}(x)`, :math:`e = \mathrm{expm1}(u)`,
  :math:`g_d = 1/(1 - \omega^d) = 1/2 + (i/2)\cot(\pi d/N)` (:math:`g_0 = 0`),

  .. math:: (V^{-1}x)_k = \Bigl[\hat b_k X_k - \sum_{q\ne k}\frac{e_q\hat b_q X_q}{\omega^{q-k} - \rho_k}\Bigr]
            \Big/ (N\hat b'_k),\quad \hat b_q = \prod_{j\ne q}(1 + e_j g_{q-j}),\quad
            \hat b'_k = \rho_k^{N-1}\prod_{j\ne k}\bigl(1 + \mathrm{expm1}(u_j - u_k)\,g_{k-j}\bigr).

  Every factor has :math:`\mathrm{Re}(1 + e g) = 1 + e/2 > 0`; at :math:`\alpha_d = 0`
  :math:`V^{-1}x = X/N` exactly (no 0/0 anywhere).  :math:`O(N^2)` per noise state (the two log-sums
  and ``R``, ``lax.map`` over row blocks, ``group`` factors per complex log) and per right-hand side
  (one Cauchy sum), :math:`O(N)` storage.
* **Fourier modes** (the flat marginal's discarded coordinates): a vector whose DFT has one or two
  nonzero bins is a combination of closed-form columns of the inverse, :math:`O(N)` each
  (:meth:`FCState.solve_fourier_modes`; :func:`fourier_mode_table` finds the bins).
* **Geometric right-hand sides** :math:`\tau^i`, :math:`\tau = e^{2\pi i\kappa/N}` (the sinusoid
  template, :math:`\kappa = f_0 N\delta t`): :math:`V^{-1}\tau^{(\cdot)} = \ell(\tau)`, the Lagrange
  basis of the nodes, in :math:`O(N)` (:meth:`FCState.solve_geometric`), normalised by the
  :math:`\alpha_d = 0` (Dirichlet) product with the nearest root of unity handled exactly.

**Two layers** (the sampler's cache is keyed on :math:`\alpha_d`): :meth:`VandermondeExactTime.state`
(:math:`\alpha_d` only, :math:`O(N^2)`) gives an :class:`FCState`; :meth:`VandermondeExactTime.complete`
(:math:`O(N)`) adds :math:`\alpha_c` through :math:`\sigma^2` and the log-determinant;
``_factor = complete(state)``.  :meth:`FCFactor.at_alpha_c` re-completes a cached state.

Scope: ``ln S(t, f_k)`` affine in ``t`` for every ``k``, uniform sampling, the record's own clock
(``psd.T == N dt``, no ``time_shifted`` PSD): the sinusoid's ``LogLinearPSD``.  Not the GB
foreground, not the pads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from domainia.wilson.backend import Backend, get_backend, pytree_dataclass

from .base import Covariance, QuadForm, attach_constants, factor_constants, quad_form

__all__ = ["BLOCK", "GROUP", "FCFactor", "FCState", "VandermondeExactTime", "fourier_mode_table", "pack_real",
           "pair_sum_R"]

#: Rows per ``map`` block of the O(N^2) kernels, and factors multiplied before one complex log
#: (exact algebra: only ``exp`` of the log-sums is used).  Both are reduced to divisors of ``N``.
BLOCK = 32
GROUP = 16
#: :func:`fourier_mode_table`: a bin is in a column's support above ``SUPPORT_RTOL`` of the
#: column's largest ``|fft|``; everything else must be round-off (``<= ROUNDOFF_RTOL``).
SUPPORT_RTOL = 1e-8
ROUNDOFF_RTOL = 1e-12


def _divisor(N: int, want: int) -> int:
    """The largest divisor of ``N`` that is at most ``want``."""
    return max(d for d in range(1, max(1, min(int(want), N)) + 1) if N % d == 0)


def _tables(N: int) -> dict[str, np.ndarray]:
    r"""Exactly (anti)symmetric tables over ``d = 0..N-1``: ``g`` (:math:`1/(1-\omega^d)`, ``g_0 = 0``),
    ``cm1`` (:math:`\cos(2\pi d/N) - 1 = -2\sin^2(\pi d/N)`), ``sn`` (:math:`\sin(2\pi d/N)`, ``0`` at
    ``N/2``) and ``s2`` (:math:`\sin^2(\pi d/N)`, ``1`` at ``d = 0``)."""
    h = N // 2
    d = np.arange(h + 1)
    half = np.sin(np.pi * d / N)
    s2h = half * half
    snh = np.sin(2.0 * np.pi * d / N)
    snh[h] = 0.0
    coth = np.zeros(h + 1)
    coth[1:] = np.cos(np.pi * d[1:] / N) / half[1:]
    coth[h] = 0.0
    s2 = np.concatenate([s2h, s2h[1:h][::-1]])
    sn = np.concatenate([snh, -snh[1:h][::-1]])
    cot = np.concatenate([coth, -coth[1:h][::-1]])
    g = np.where(np.arange(N) == 0, 0.0, 0.5 + 0.5j * cot)
    return {"g": g, "cm1": -2.0 * s2, "sn": sn, "s2": np.where(np.arange(N) == 0, 1.0, s2)}


def pack_real(z: Any, be: Backend | None = None) -> Any:
    r"""``(N,)`` or ``(N, m)`` complex :math:`z` with :math:`z_{N-k} = \bar z_k` (``V^{-1}`` of a real
    vector) as the real vector of ``ExactTimeCovariance.voice_factor``'s column order:
    ``[Re z_0, sqrt2 Re z_1..z_{N/2-1}, Re z_{N/2}, -sqrt2 Im z_1..z_{N/2-1}]``.  Inner products are kept:
    ``pack_real(a) . pack_real(b) = Re(a^H b)``; only ``k <= N/2`` is read."""
    be = be or get_backend(z)
    xp = be.xp
    h = z.shape[0] // 2
    re, im = xp.real(z), xp.imag(z)
    r2 = float(np.sqrt(2.0))
    return xp.concatenate([re[:1], r2 * re[1:h], re[h:h + 1], -r2 * im[1:h]], axis=0)


def pair_sum_R(L: Any, alpha_d: float, *, chunk: int = 256) -> float:
    r"""Host NumPy :math:`R(\alpha_d) = \sum_{j<k}\mathrm{log1p}(\sinh^2((u_k-u_j)/2)/\sin^2(\pi(k-j)/N))`,
    ``u = alpha_d L / (2N)``, ``L`` two-sided (``(N,)``): every term non-negative (no cancellation);
    O(N^2) time, O(chunk N) memory.  The independent closed form the factor's in-kernel ``R`` is
    tested against."""
    L = np.asarray(L, dtype=float)
    N = L.size
    u = float(alpha_d) * L / (2.0 * N)
    k = np.arange(N)
    tot = 0.0
    for s in range(0, N, chunk):
        j = k[s:s + chunk]
        d = k[None, :] - j[:, None]
        up = d > 0
        sn = np.sin(np.pi * np.where(up, d, 1) / N)
        r = np.sinh(0.5 * (u[None, :] - u[j][:, None])) ** 2 / (sn * sn)
        tot += float(np.sum(np.log1p(np.where(up, r, 0.0))))
    return tot


def fourier_mode_table(U: np.ndarray, *, support_rtol: float = SUPPORT_RTOL,
                       roundoff_rtol: float = ROUNDOFF_RTOL) -> tuple[np.ndarray, np.ndarray] | None:
    """``(bins, vals)`` (``(n, 2)`` int64, ``(n, 2)`` complex) with ``fft(U[:, c])`` equal to ``vals[c]``
    at ``bins[c]`` and zero elsewhere up to round-off -- the Fourier-band nuisance basis (one bin
    for DC / Nyquist, stored twice with the second value 0; two conjugate bins otherwise) -- or
    ``None`` when some column has more than two bins or off-support entries above round-off."""
    U = np.asarray(U, dtype=float)
    if U.ndim != 2 or U.shape[1] == 0:
        return None
    X = np.fft.fft(U, axis=0)
    a = np.abs(X)
    n = U.shape[1]
    bins = np.zeros((n, 2), dtype=np.int64)
    vals = np.zeros((n, 2), dtype=complex)
    for c in range(n):
        mx = float(a[:, c].max())
        if not mx > 0.0:
            return None
        sup = np.flatnonzero(a[:, c] > support_rtol * mx)
        rest = np.delete(a[:, c], sup)
        if sup.size > 2 or (rest.size and float(rest.max()) > roundoff_rtol * mx):
            return None
        bins[c, :sup.size], vals[c, :sup.size] = sup, X[sup, c]
        if sup.size == 1:
            bins[c, 1] = sup[0]
    return bins, vals


def _cauchy(C: Any, e: Any, rows: Any, cols: Any, N: int, xp: Any) -> Any:
    r"""``K[r, c] = 1/(omega^{q-k} - rho_k)`` with ``q - k = cols - rows`` (``rows`` the ``k`` of
    ``rho_k`` when ``e`` is indexed by them) and ``0`` where ``q = k``."""
    d = (cols - rows) % N
    den = (C["cm1"][d] - e) + 1j * C["sn"][d]
    return xp.where(d == 0, 0.0, 1.0 / xp.where(d == 0, 1.0, den))


@pytree_dataclass(data=("u", "e", "bh", "bp", "R"), meta=("N", "block"))
@dataclass(frozen=True, eq=False)
class FCState:
    r"""The :math:`\alpha_d`-only Fourier-Cauchy state: ``u``, ``e = expm1(u)``, ``bh`` (:math:`\hat b_q`),
    ``bp`` (:math:`\hat b'_k`) and the pair sum ``R``; ``O(N)`` arrays.  Its methods read the
    tables ``C["cm1"]``, ``C["sn"]`` (``VandermondeExactTime.constants(be)``)."""

    u: Any
    e: Any
    bh: Any
    bp: Any
    R: Any
    N: int
    block: int

    def vsolve(self, x: Any, *, C: Any, be: Backend | None = None) -> Any:
        """``V^{-1} x`` (complex), ``x`` real or complex ``(N,)`` / ``(N, m)``: one FFT and one
        O(N^2) Cauchy sum by row blocks."""
        be = be or get_backend(x, self.u)
        xp, N, B = be.xp, self.N, self.block
        X = be.fft.fft(x, axis=0)
        one = X.ndim == 1
        X = X[:, None] if one else X
        y = (self.e * self.bh)[:, None] * X
        q = xp.arange(N)

        def rows(k0: Any) -> Any:
            k = k0 + xp.arange(B)
            return _cauchy(C, self.e[k][:, None], k[:, None], q[None, :], N, xp) @ y

        s = be.map(rows, xp.arange(0, N, B)).reshape(N, X.shape[1])
        w = (self.bh[:, None] * X - s) / (N * self.bp)[:, None]
        return w[:, 0] if one else w

    def vsolve_T(self, v: Any, *, C: Any, be: Backend | None = None) -> Any:
        """``V^{-T} v`` (transpose, no conjugate): ``fft(K'^T (v / (N bp)))`` with the same Cauchy
        kernel summed over the other index."""
        be = be or get_backend(v, self.u)
        xp, N, B = be.xp, self.N, self.block
        one = v.ndim == 1
        v1 = (v[:, None] if one else v) / (N * self.bp)[:, None]
        k = xp.arange(N)

        def rows(q0: Any) -> Any:
            q = q0 + xp.arange(B)
            return _cauchy(C, self.e[None, :], k[None, :], q[:, None], N, xp) @ v1

        t = be.map(rows, xp.arange(0, N, B)).reshape(N, v1.shape[1])
        out = be.fft.fft(self.bh[:, None] * v1 - (self.e * self.bh)[:, None] * t, axis=0)
        return out[:, 0] if one else out

    def solve_fourier_modes(self, bins: Any, vals: Any, *, C: Any, be: Backend | None = None) -> Any:
        """``(N, n)``: ``V^{-1} u_c`` for the columns ``u_c`` whose DFT is ``vals[c]`` at ``bins[c]``
        (``(n, S)``; :func:`fourier_mode_table`): closed-form columns of the inverse, O(N) each."""
        be = be or get_backend(vals, self.u)
        xp, N = be.xp, self.N
        k = xp.arange(N)[:, None, None]
        K = _cauchy(C, self.e[:, None, None], k, bins[None, :, :], N, xp)      # (N, n, S): q - k
        bq = self.bh[bins] * vals
        d0 = (bins[None, :, :] - k) == 0
        term = xp.where(d0, bq[None], -(self.e[bins] * bq)[None] * K)
        return xp.sum(term, axis=2) / (N * self.bp)[:, None]

    def solve_geometric(self, kappa: Any, *, be: Backend | None = None) -> Any:
        r"""``V^{-1} (tau^i)_i`` for :math:`\tau = e^{2\pi i\kappa/N}` (``kappa`` any real): the Lagrange
        basis :math:`\ell_k(\tau) = \prod_{j\ne k}(\tau - x_j)/P'(x_k)` in O(N), normalised by the
        :math:`\alpha_d = 0` product: with :math:`\delta_j = 2\pi(\kappa - j)/N` reduced to
        :math:`[-\pi, \pi)`, :math:`D_j = (e^{i\delta_j} - 1) - e_j` and :math:`j^*` the nearest root
        of unity, :math:`\ell_k = R_* e^Q D_{j^*}/(N D_k \hat b'_k)` (:math:`k \ne j^*`),
        :math:`\ell_{j^*} = R_* e^Q/(N\hat b'_{j^*})`, :math:`Q = \sum_{j\ne j^*}\mathrm{log1p}(e_j G_j)`,
        :math:`G_j = 1/2 + (i/2)\cot(\delta_j/2)`, :math:`R_* = e^{i(N-1)\delta_*/2}\sin(N\delta_*/2)/
        \sin(\delta_*/2)` (``N`` at :math:`\delta_* = 0`: ``tau`` on a root, e.g. an on-bin ``f_0``)."""
        be = be or get_backend(kappa, self.u)
        xp, N = be.xp, self.N
        j = xp.arange(N)
        r = xp.mod(kappa - j + 0.5 * N, N) - 0.5 * N
        dl = 2.0 * np.pi * r / N
        sh = xp.sin(0.5 * dl)
        diff = (-2.0 * sh * sh - self.e) + 1j * xp.sin(dl)
        js = xp.argmin(xp.abs(r))
        ds = dl[js]
        at0 = ds == 0.0
        sd = xp.where(at0, 1.0, xp.sin(0.5 * ds))
        Rs = xp.where(at0, float(N), xp.exp(0.5j * (N - 1) * ds) * xp.sin(0.5 * N * ds) / sd)
        m = j != js
        tq = xp.tan(0.5 * xp.where(m, dl, 1.0))
        G = 0.5 + 0.5j / tq
        Q = xp.sum(xp.where(m, xp.log1p(self.e * G), 0.0))
        c = Rs * xp.exp(Q) / N
        return xp.where(m, c * diff[js] / (xp.where(m, diff, 1.0) * self.bp), c / self.bp)

    def solve_cosine(self, amp: Any, kappa: Any, phase: Any, *, be: Backend | None = None) -> Any:
        r"""``V^{-1}`` of ``amp cos(2 pi kappa i / N + phase)`` (the sinusoid template with
        :math:`\kappa = f_0 N\delta t`): :math:`(A/2)[e^{i\phi}\ell(\kappa) + e^{-i\phi}\ell(-\kappa)]`, O(N)."""
        be = be or get_backend(kappa, self.u)
        xp = be.xp
        return 0.5 * amp * (xp.exp(1j * phase) * self.solve_geometric(kappa, be=be)
                            + xp.exp(-1j * phase) * self.solve_geometric(-kappa, be=be))


def _complete(C: Any, state: FCState, nu: Any, N: int, be: Backend) -> FCFactor:
    """``state`` at ``nu = (alpha_c, alpha_d)``: ``log sigma^2`` and ``ln det C~ = Weyl + R`` (O(N))."""
    xp = be.xp
    log_sig2 = C["log_half_df"] + (nu[0] - 0.5 * nu[1]) * C["L"]
    weyl = nu[0] * C["S_L"] - nu[1] * C["S_L"] / (2.0 * N) - N * C["log_2dt"]
    logdet = weyl + state.R
    return FCFactor(state, nu, log_sig2, logdet, xp.isfinite(logdet))


@pytree_dataclass(data=("state", "nu", "log_sig2", "logdet", "ok"), meta=())
@dataclass(frozen=True, eq=False)
class FCFactor:
    r"""``C~ = B_c B_c^H`` at ``nu``: the :class:`FCState` plus ``log_sig2`` (:math:`\ln\sigma_k^2`),
    ``logdet`` (closed form) and ``ok`` (``isfinite(logdet)``: True for every finite ``nu``, no
    positive-definiteness branch).  ``solve`` is :math:`\tilde C^{-1} b` (real), ``whiten`` is
    :math:`G^{-1} b` (real, :func:`pack_real` of :math:`B_c^{-1} b`), ``quad`` defaults to
    ``form="whiten"`` (one Cauchy sum; ``"solve"`` costs two)."""

    state: FCState
    nu: Any
    log_sig2: Any
    logdet: Any
    ok: Any

    n_u = 0

    def _be(self, b: Any, be: Backend | None) -> Backend:
        return be or get_backend(b, self.log_sig2)

    def whiten(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        be = self._be(b, be)
        C = factor_constants(self, C, "whiten")
        w = self.state.vsolve(b, C=C, be=be)
        s = be.xp.exp(0.5 * self.log_sig2)
        return pack_real(w / (s if b.ndim == 1 else s[:, None]), be)

    def solve(self, b: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        r""":math:`\tilde C^{-1} b = \mathrm{Re}\,V^{-T}(\bar w/\sigma^2)`, :math:`w = V^{-1} b` (two Cauchy sums)."""
        be = self._be(b, be)
        xp = be.xp
        C = factor_constants(self, C, "solve")
        w = self.state.vsolve(b, C=C, be=be)
        s2 = xp.exp(self.log_sig2)
        return xp.real(self.state.vsolve_T(xp.conj(w) / (s2 if b.ndim == 1 else s2[:, None]), C=C, be=be))

    def quad(self, b: Any, *, C: Any = None, form: QuadForm = "whiten", be: Backend | None = None) -> Any:
        be = self._be(b, be)
        return quad_form(self, b, form, be, factor_constants(self, C, "quad"))

    def whiten_fourier_modes(self, bins: Any, vals: Any, *, C: Any = None, be: Backend | None = None) -> Any:
        """``(N, n)`` real: ``whiten`` of the Fourier-mode columns (:func:`fourier_mode_table`), O(N) each."""
        be = self._be(vals, be)
        C = factor_constants(self, C, "whiten_fourier_modes")
        W = self.state.solve_fourier_modes(bins, vals, C=C, be=be)
        return pack_real(W / be.xp.exp(0.5 * self.log_sig2)[:, None], be)

    def at_alpha_c(self, alpha_c: Any, *, C: Any = None, be: Backend | None = None) -> FCFactor:
        """The same :math:`\\alpha_d` state completed at another ``alpha_c`` (O(N)); the constants
        stay attached."""
        be = self._be(self.nu, be)
        C = factor_constants(self, C, "at_alpha_c")
        out = _complete(C, self.state, be.xp.stack([be.asarray(alpha_c, dtype=float), self.nu[1]]),
                        self.state.N, be)
        return attach_constants(out, C)


class VandermondeExactTime(Covariance):
    """The exact time-domain covariance of a ``LogLinearPSD`` on ``N`` samples of ``dt`` (Eq. 95
    / 97, ``A_n`` outside) by its Vandermonde square root (module docstring).  ``nu = (alpha_c,
    alpha_d)``; jittable, no static key, ``marginal_form = "whiten"`` (``FlatMarginal``'s default
    order over this covariance; with the Fourier-band nuisance it takes the O(N) mode route)."""

    marginal_form = "whiten"

    def __init__(self, psd: Any, N: int, dt: float, *, block: int = BLOCK, group: int = GROUP) -> None:
        if "operator" not in getattr(psd, "capabilities", frozenset()):
            raise ValueError(f"VandermondeExactTime needs a log-linear PSD on the record's clock; got "
                             f"{type(psd).__name__} (a time_shifted PSD withdraws the capability)")
        if float(getattr(psd, "t0", 0.0)) != 0.0:
            raise ValueError("VandermondeExactTime: the PSD is time-shifted (a padded grid's clock)")
        if int(N) < 2 or int(N) % 2:
            raise ValueError(f"N must be even and positive; got {N}")
        if not float(dt) > 0.0:
            raise ValueError(f"dt must be positive; got {dt}")
        T, NT = float(getattr(psd, "T", np.nan)), int(N) * float(dt)
        if not np.isclose(T, NT, rtol=1e-12, atol=0.0):
            raise ValueError(f"VandermondeExactTime: alpha(t) uses the record's T = N dt = {NT!r}; this PSD has "
                             f"T = {T!r}")
        if int(block) < 1 or int(group) < 1:
            raise ValueError(f"block and group must be positive; got {block}, {group}")
        self.psd, self.N, self.dt = psd, int(N), float(dt)
        self.n = self.N
        self.block, self.group = _divisor(self.N, block), _divisor(self.N, group)
        self.L = np.asarray(psd.L(np.abs(np.fft.fftfreq(self.N, d=self.dt))), float)
        self.S_L = float(np.sum(self.L))
        self.jittable = True

    def host_constants(self) -> dict[str, Any]:
        return {"L": self.L, **_tables(self.N), "S_L": np.float64(self.S_L),
                "log_2dt": np.float64(np.log(2.0 * self.dt)),
                "log_half_df": np.float64(np.log(0.5 / (self.N * self.dt)))}

    def weyl_logdet(self, nu: Any) -> Any:
        r""":math:`\alpha_c S_L - \alpha_d S_L/(2N) - N\ln(2\,\delta t)` (``ExactTimeCovariance.weyl_logdet``)."""
        S_L, N = self.S_L, self.N
        return nu[0] * S_L - nu[1] * S_L / (2.0 * N) - N * np.log(2.0 * self.dt)

    def nodes(self, alpha_d: float) -> np.ndarray:
        """Host ``(N,)`` complex nodes ``x_k = rho_k omega^k`` (DC ``rho_0``, Nyquist ``-rho_{N/2}``)."""
        T = _tables(self.N)
        rho = np.exp(float(alpha_d) * self.L / (2.0 * self.N))
        return rho * ((1.0 + T["cm1"]) + 1j * T["sn"])

    def fourier_mode_table(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """:func:`fourier_mode_table` of ``U`` (``(N, n_u)``): the ``FlatMarginal`` fast path's bins."""
        U = np.asarray(U, dtype=float)
        if U.ndim != 2 or U.shape[0] != self.N:
            return None
        return fourier_mode_table(U)

    def state(self, C: Any, alpha_d: Any, be: Backend) -> FCState:
        """The :math:`\\alpha_d`-only part (O(N^2), ``map`` over ``block`` rows): ``bh``, ``bp`` and ``R``."""
        xp, N, B, G = be.xp, self.N, self.block, self.group
        u = alpha_d * C["L"] / (2.0 * N)
        e = xp.expm1(u)
        g, s2 = C["g"], C["s2"]
        j = xp.arange(N)

        def rows(q0: Any) -> Any:
            q = q0 + xp.arange(B)
            d = (q[:, None] - j[None, :]) % N
            gg = g[d]
            ee = xp.expm1(u[None, :] - u[q][:, None])
            f1 = 1.0 + e[None, :] * gg
            f2 = 1.0 + ee * gg
            r = xp.log1p(ee * ee / (4.0 * (1.0 + ee) * s2[d]))            # 0 at d = 0 (ee = 0)
            if G > 1:
                f1 = xp.prod(f1.reshape(B, N // G, G), axis=2)
                f2 = xp.prod(f2.reshape(B, N // G, G), axis=2)
            return xp.sum(xp.log(f1), axis=1), xp.sum(xp.log(f2), axis=1), xp.sum(r, axis=1)

        lbh, lbp, rr = be.map(rows, xp.arange(0, N, B))
        lbp = lbp.reshape(N) + (N - 1) * u
        return FCState(u, e, xp.exp(lbh.reshape(N)), xp.exp(lbp), 0.5 * xp.sum(rr), N, B)

    def complete(self, C: Any, state: FCState, nu: Any, be: Backend) -> FCFactor:
        """``state`` (at ``nu[1]``) completed at ``nu`` (O(N))."""
        return _complete(C, state, nu, self.N, be)

    def _factor(self, C: Any, nu: Any, be: Backend) -> FCFactor:
        return self.complete(C, self.state(C, nu[1], be), nu, be)

    def _matrix(self, C: Any, nu: Any, be: Backend) -> Any:
        """The dense ``B_c B_c^H`` (host tests; N <= 4096)."""
        nu = np.asarray(nu, dtype=float)
        V = self.vandermonde(nu[1])
        sig2 = np.exp(np.asarray(C["log_half_df"]) + (nu[0] - 0.5 * nu[1]) * np.asarray(C["L"]))
        return be.asarray(((V * sig2[None, :]) @ V.conj().T).real)

    def vandermonde(self, alpha_d: float) -> np.ndarray:
        """Host dense ``V`` (``V_ik = rho_k^i omega^{ik}``, exact phases ``ik mod N``; tests, N <= 4096)."""
        T = _tables(self.N)
        i = np.arange(self.N)
        ph = np.outer(i, i) % self.N
        u = float(alpha_d) * self.L / (2.0 * self.N)
        return np.exp(np.outer(i, u)) * ((1.0 + T["cm1"][ph]) + 1j * T["sn"][ph])
