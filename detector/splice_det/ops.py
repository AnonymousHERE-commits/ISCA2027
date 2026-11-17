"""SPLICE neck operations (SPLICE-V section 2), backend-neutral.

Every function takes the backend `B` first (backend.NumpyBackend or TorchBackend) so the
same code is exercised by the numpy tests here and runs on GPUs during training.  These
are the float forms of the integer operations in reference/splicev_ref; where the integer
reference fixes a convention (neighbour order, band telescoping, unassigned handling,
null option, clip) the float form follows it so that a trained model exports cleanly.

Cell tensors are (N, C, H, W); fragment-level operations work per image on flattened
cells x (n, C) with fragment ids (n,) in [0, F) and -1 for unassigned cells.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

# eight-neighbour order of the integer walk (reference/splicev_ref/walk.py NEIGH)
NEIGH: Tuple[Tuple[int, int], ...] = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
OPP = tuple(7 - k for k in range(8))            # NEIGH[OPP[k]] == -NEIGH[k]
UNASSIGNED = -1


# --------------------------------------------------------------------------- preserve

def edge_weights(B, q, pi):
    """w_ij = q_i q_j pi_ij on the eight-neighbour lattice, symmetric, zero across the border.

    q: (N, 1, H, W) purity in [0, 1];  pi: (N, 8, H, W) directed edge probabilities.
    The undirected probability is the mean of the two directions, so W is symmetric."""
    ws = []
    for k, (dy, dx) in enumerate(NEIGH):
        back = B.shift2d(pi[:, OPP[k]:OPP[k] + 1], dy, dx)
        qj = B.shift2d(q, dy, dx)
        ws.append(q * qj * 0.5 * (pi[:, k:k + 1] + back))
    return B.concat(ws, axis=1)


def walk_step(B, x, w, wsum):
    """Anchored lazy walk P = 1/2 I + 1/2 S^-1 (I + W), S = I + diag(W 1); x (N, C, H, W)."""
    acc = x
    for k, (dy, dx) in enumerate(NEIGH):
        acc = acc + w[:, k:k + 1] * B.shift2d(x, dy, dx)
    return 0.5 * x + 0.5 * acc / (1.0 + wsum)


def walk_powers(B, x, w, steps: int) -> List:
    """[P^0 x, P^1 x, ..., P^steps x] (the only walks computed, SPLICE-V section 2)."""
    wsum = B.sum(w, axis=1, keepdims=True)
    out = [x]
    for _ in range(steps):
        out.append(walk_step(B, out[-1], w, wsum))
    return out


def beta_chain(B, u):
    """Monotone per-cell chain 0 < beta_1 < ... < beta_4 < 1 from four unconstrained maps.

    u: (N, 4, H, W).  beta_k = c_k / (1 + c_4) with c = cumsum(softplus(u))."""
    c = B.cumsum(B.softplus(u) + 1e-4, axis=1)
    return c / (1.0 + c[:, 3:4])


def bank(B, powers: Sequence, beta):
    """A(beta) x = sum_{t<s} (1 - beta) beta^t P^t x + beta^s P^s x; beta (N, 1, H, W)."""
    s = len(powers) - 1
    out = (1.0 - beta) * powers[0]
    bt = beta
    for t in range(1, s):
        out = out + (1.0 - beta) * bt * powers[t]
        bt = bt * beta
    return out + bt * powers[s]


def bands(B, powers: Sequence, betas) -> Tuple[List, object]:
    """Signed bands Z_j = (A(beta_j) - A(beta_{j+1})) x, j = 0..3 with beta_0 = 0, and the
    low-pass A(beta_4) x.  sum_j Z_j + low = A(0) x = x (exact telescoping in float).

    betas: (N, 4, H, W) increasing in (0, 1)."""
    A = [powers[0]]
    for j in range(4):
        A.append(bank(B, powers, betas[:, j:j + 1]))
    return [A[j] - A[j + 1] for j in range(4)], A[4]


def band_codes(B, z, lsb_x: float, exponent: int):
    """Integer-code view of a band (DD-04): code = Z(x / lsb_x) * 2^e, before rounding."""
    return z * (2.0 ** exponent / lsb_x)


def modulated_taps(B, x, w):
    """Nine shifted copies of x with the eight lateral copies scaled by the edge affinity
    they cross (typed modulation of the fusion convolutions); a 1x1 convolution over the
    result equals a 3x3 convolution whose lateral taps are modulated.  Returns
    (N, 9, C, H, W) in the 3x3 raster order with the centre at index 4."""
    out = []
    for k9 in range(9):
        dy, dx = k9 // 3 - 1, k9 % 3 - 1
        if dy == 0 and dx == 0:
            out.append(x)
            continue
        k = NEIGH.index((dy, dx))
        out.append(w[:, k:k + 1] * B.shift2d(x, dy, dx))
    return B.stack(out, axis=1)


# --------------------------------------------------------------------------- coarsen

def _ids_with_dump(B, ids, F: int):
    """Map unassigned ids (< 0 or >= F) to the dump row F."""
    return B.where((ids >= 0) & (ids < F), ids, ids * 0 + F)


def restrict(B, x, ids, weights, F: int, eps: float = 1e-12):
    """H = R X: per-fragment weighted means of the rows of x (n, C); rows of unassigned
    cells are ignored; a fragment with zero mass gets a zero row.  Returns (H, mass)."""
    idx = _ids_with_dump(B, ids, F)
    w = weights[:, None]
    num = B.scatter_add(F + 1, idx, x * w)[:F]
    den = B.scatter_add(F + 1, idx, w)[:F]
    return num / B.maximum(den, eps), den[:, 0]


def level_weights(B, e, ids, F: int):
    """DD-08: eligibility weights; a fragment with zero mass at the level uses uniform weights."""
    idx = _ids_with_dump(B, ids, F)
    mass = B.scatter_add(F + 1, idx, e[:, None])[:, 0]
    zero = B.take(mass, idx) <= 0
    return B.where(zero, e * 0 + 1.0, e)


def fill(B, H, ids):
    """(M H)_i = H[id_i]; zero rows for unassigned cells."""
    F = H.shape[0]
    ext = B.concat([H, B.zeros((1, H.shape[1]), like=H)], axis=0)
    return B.take(ext, _ids_with_dump(B, ids, F), axis=0)


def detail(B, x, H, ids):
    """D = X - M H (within-fragment detail)."""
    return x - fill(B, H, ids)


def descriptor_nu(B, band_planes: Sequence, ids, F: int, n_transport: int = 48, shift: int = 7, cap: float = 127.0):
    """Per-fragment band energies of one level (DD-05): nu_j = min(cap, mean_cells mean_c z_j^2 / 2^shift)
    with z in band-code units.  band_planes: four (n, C) arrays; returns (F, 4)."""
    cols = []
    ones = B.ones((ids.shape[0],), like=band_planes[0])
    for z in band_planes:
        e = B.mean(z[:, :n_transport] ** 2, axis=1)
        m, _ = restrict(B, e[:, None], ids, ones, F)
        cols.append(B.minimum(m[:, 0] / float(2 ** shift), cap))
    return B.stack(cols, axis=1)


# --------------------------------------------------------------------------- exchange inputs

def fuse_hypotheses(B, mu, prec, ids, F: int, eps: float = 1e-12):
    """Precision-weighted fusion of anchor box hypotheses per fragment.
    mu, prec: (n, 4).  Returns mu_f (F, 4), var_f (F, 4)."""
    idx = _ids_with_dump(B, ids, F)
    ps = B.scatter_add(F + 1, idx, prec)[:F]
    ms = B.scatter_add(F + 1, idx, prec * mu)[:F]
    ps = B.maximum(ps, eps)
    return ms / ps, 1.0 / ps


def mahalanobis2(B, mu_f, var_f):
    """d^2_fg = sum_d (mu_f - mu_g)^2 / (var_f + var_g); (F, F)."""
    diff = mu_f[:, None, :] - mu_f[None, :, :]
    return B.sum(diff * diff / (var_f[:, None, :] + var_f[None, :, :]), axis=2)


def gate_stratum(n_frag: int, bounds: Sequence[int]) -> int:
    s = 0
    for b in bounds:
        if n_frag >= b:
            s += 1
    return s


def admissible(B, d2, tau: float):
    """C_f: pairs whose distance passes the conformal threshold, no self pairs."""
    F = d2.shape[0]
    eye = B.eye(F, like=d2) > 0.5
    return (d2 <= tau) & (~eye)


# --------------------------------------------------------------------------- exposure budget

def band_tau(B, betas, steps: int):
    """Diffusion-time weight of each band per cell (Lemma 3(iii)):
    tau_j = sum_t t |a_t(beta_j) - a_t(beta_{j+1})|, j = 0..3 with beta_0 = 0, where
    a_t(beta) = (1 - beta) beta^t for t < s and a_s = beta^s.  betas (N, 4, H, W)."""
    out = []
    for j in range(4):
        hi = betas[:, j:j + 1]
        lo = betas[:, j - 1:j] if j > 0 else None
        acc = hi * 0
        for t in range(1, steps + 1):
            a_hi = (1.0 - hi) * hi ** t if t < steps else hi ** t
            if lo is None:
                acc = acc + t * a_hi                                  # a_t(0) = 0 for t >= 1
            else:
                a_lo = (1.0 - lo) * lo ** t if t < steps else lo ** t
                acc = acc + t * B.abs(a_lo - a_hi)
        out.append(acc)
    return B.concat(out, axis=1)


def lowpass_time(B, betas, steps: int):
    """Diffusion time T(beta_j) = sum_t t a_t(beta_j) of the four low-pass responses."""
    out = []
    for j in range(4):
        b = betas[:, j:j + 1]
        acc = b * 0
        for t in range(1, steps + 1):
            acc = acc + t * ((1.0 - b) * b ** t if t < steps else b ** t)
        out.append(acc)
    return B.concat(out, axis=1)


def leave_prob(B, w, code):
    """One-step probability of leaving one's class under the lazy walk, per cell.

    w (N, 8, H, W) edge weights; code (N, 1, H, W) class code of every cell (a positive
    number per fragment, one more value shared by the unassigned cells).  The walk is
    P = 1/2 I + 1/2 S^-1 (I + W), so the step to neighbour k has probability
    w_k / (2 (1 + sum_k w_k)); weights across the border are zero."""
    wsum = B.sum(w, axis=1, keepdims=True)
    acc = wsum * 0
    for k, (dy, dx) in enumerate(NEIGH):
        other = B.shift2d(code, dy, dx) != code
        acc = acc + B.where(other, w[:, k:k + 1], w[:, k:k + 1] * 0)
    return 0.5 * acc / (1.0 + wsum)


def segment_amax(B, x, ids, F: int):
    """Per-fragment maximum of the nonnegative rows of x (n, C); zero for empty fragments."""
    return B.scatter_amax(F + 1, _ids_with_dump(B, ids, F), x)[:F]


def exposure_discount(B, kappa, shares, tau, eps: float = 2.0 ** -4):
    """Exposure budget of a source fragment (section 2, 'What the bands do'):
    xi_g = min(1 - eps, 2 kappa_g chi_g),  chi_g = sum_j share_gj tau_gj.

    kappa (F,) largest one-step leaving probability of the fragment's cells; shares (F, J)
    band-energy shares summing to one; tau (F, J) diffusion-time weights.  Returns
    (xi (F,), log(1 - xi) (F,)), the additive discount of the forecast logit."""
    chi = B.sum(shares * tau, axis=1)
    xi = B.minimum(2.0 * kappa * chi, 1.0 - eps)
    return xi, B.log(1.0 - xi)


# --------------------------------------------------------------------------- route

def forecast(B, K, Uv, VO, Hs, l0, alpha_logit, adm, d: int = 64, clip_b: float = 64.0, eps: float = 2.0 ** -4,
             logit_clamp=(-8.0, 127.0 / 16.0), disc=None):
    """Confined forecast, Eq. (3).

    l_fg = <K_f, K_g>/sqrt(d) + <Psi nu_f, Psi nu_g> + log(1 - xi_g)  (g in C_f),  l_f0 = psi_0^T nu_f
    y_f. = softmax over C_f union {0};  alpha_f = (1 - eps) sigmoid(psi_alpha^T nu_f)
    dHhat_f = alpha_f sum_{g in C_f} y_fg clip_Theta(VO_g - Hs_f)
    Masked donors contribute zero, the null enters only the denominator, an empty C_f gives
    zero.  `disc` (F,) is the exposure discount log(1 - xi_g) of each source, or None.
    Logits are clamped to the manifest's table range (DD-16, [-8, 8)) unless logit_clamp is
    None.  Returns (dHhat (F, Ct), y (F, F), y0 (F,), alpha (F,))."""
    L = (K @ K.T) / math.sqrt(d) + Uv @ Uv.T
    if disc is not None:
        L = L + disc[None, :]
    if logit_clamp is not None:
        L = B.clip(L, logit_clamp[0], logit_clamp[1])
        l0 = B.clip(l0, logit_clamp[0], logit_clamp[1])
    neg = -1e30
    Lm = B.where(adm, L, L * 0 + neg)
    m = B.maximum(B.amax(Lm, axis=1), l0)
    m = B.detach(m)
    e = B.where(adm, B.exp(B.minimum(Lm - m[:, None], 0.0)), L * 0)
    e0 = B.exp(l0 - m)
    den = e0 + B.sum(e, axis=1)
    a = e / den[:, None]
    a0 = e0 / den
    innov = B.clip(VO[None, :, :] - Hs[:, None, :], -clip_b, clip_b)
    alpha = (1.0 - eps) * B.sigmoid(alpha_logit)
    dh = alpha[:, None] * B.einsum("fg,fgc->fc", a, innov)
    return dh, a, a0, alpha


def forecast_free(B, K, VO, Hs, d: int = 64):
    """Arm (d): the free forecast softmax(K K^T / sqrt(d)) V O_F over all other fragments,
    without the admissible set, the null option, the shrink or the clip:
    dHhat_f = sum_{g != f} y_fg (VO_g - Hs_f).  Returns (dHhat, y)."""
    F = K.shape[0]
    L = (K @ K.T) / math.sqrt(d)
    off = B.eye(F, like=L) < 0.5
    Lm = B.where(off, L, L * 0 - 1e30)
    m = B.detach(B.amax(Lm, axis=1))
    e = B.where(off, B.exp(B.minimum(Lm - m[:, None], 0.0)), L * 0)
    den = B.maximum(B.sum(e, axis=1), 1e-30)
    y = e / den[:, None]
    has = B.sum(e, axis=1) > 0
    dh = y @ VO - B.where(has, Hs[:, 0] * 0 + 1.0, Hs[:, 0] * 0)[:, None] * Hs
    return dh, y


def synthesize(B, xs, ids, dh_hat, lo: float = None, hi: float = None):
    """X's = X^s + M dHhat (Eq. (4) without the detail correction); optional saturation."""
    out = xs + fill(B, dh_hat, ids)
    if lo is not None:
        out = B.clip(out, lo, hi)
    return out


def class_band(B, N, ids, weights, F: int, rz=None):
    """Pi N + M (r Z): fragment-constant class-band input; unassigned rows are zero."""
    H, _ = restrict(B, N, ids, weights, F)
    if rz is not None:
        H = H + rz
    return fill(B, H, ids)


def box_band(B, N, ids, weights, F: int):
    """Q N = N - Pi N: within-object band read by regression (unassigned rows keep N)."""
    H, _ = restrict(B, N, ids, weights, F)
    return N - fill(B, H, ids)


def class_score(B, logits, gate):
    """p = sigma(l) g (Prop. 3: the gate multiplies the class score)."""
    return B.sigmoid(logits) * gate
