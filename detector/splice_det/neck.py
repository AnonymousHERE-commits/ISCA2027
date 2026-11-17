"""SPLICE-B neck in PyTorch: the SPLICE core in place of YOLO26's native neck (SPLICE-V section 2).

Per frame:
  preserve  latent X_l (64 = 48 shared + 16 protected) per backbone tap; supervised cell
            heads (edge pi, purity q, eligibility omega, thin t, gate gamma, beta chain);
            lazy walks P_l^t X_l (t <= 4, 8, 12), signed bands Z_j, typed mixing R_w
  coarsen   reserve anchors by farthest-point sampling, certified rounds of the guarded
            average linkage on the window pair scores (integer reference, CPU),
            memberships M_l, restriction H = R X
  exchange  per-anchor box hypotheses fused per fragment, conformal admissibility test ->
            C_f; evidence, need zeta in [0, 1], reliability, error moments; Eq. (1) marginals
            and the trust solve of Eq. (2) -> dH
  route     the readout arm (config.ARMS).  Deployed arm (e): confined forecast dHhat
            (Eq. (3), with the exposure discount log(1 - xi_g)) through the zero-initialised
            injection map, X'^s = X^s + M dHhat + R_w^perp, bottom-up path, readout
            cls_in = Pi X'' + M r O_l (r = dH - dHhat), box_in = Q X''

The fragment-level steps run per image (ragged F); the cell-level steps are batched.
Memberships, partner lists, active sets and admissibility decisions are made outside
autograd (integer reference and exchange_float, CPU pool); gradients reach the exchange
through Eq. (1) and the optimality system of Eq. (2) (exchange_torch).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import exchange_float, exchange_torch, linkage_bridge as lb, ops
from . import targets as tg
from .backend import TorchBackend
from .config import SpliceConfig

LINK_FWD = tg.LINK_FWD
NOMINAL_BETAS = (0.2, 0.4, 0.6, 0.8)          # start of the learned mixing (descriptor 'learned_mix')


def conv_bn_act(cin: int, cout: int, k: int = 1, s: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout), nn.SiLU())


class CellHeads(nn.Module):
    """Supervised per-cell heads of one level: pi (8), q, e, t, g (logits) and the beta chain (4)."""

    def __init__(self, c: int):
        super().__init__()
        self.body = conv_bn_act(c, c, 3)
        self.out = nn.Conv2d(c, 16, 1)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.zero_()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        o = self.out(self.body(x))
        return {"pi_logit": o[:, 0:8], "q_logit": o[:, 8:9], "e_logit": o[:, 9:10], "t_logit": o[:, 10:11],
                "g_logit": o[:, 11:12], "beta_u": o[:, 12:16],
                "pi": torch.sigmoid(o[:, 0:8]), "q": torch.sigmoid(o[:, 8:9]), "e": torch.sigmoid(o[:, 9:10]),
                "t": torch.sigmoid(o[:, 10:11]), "g": torch.sigmoid(o[:, 11:12])}


class ModFuse(nn.Module):
    """Fusion block: 1x1 reduce of the concatenated inputs, then a 3x3 mixing whose eight
    lateral taps are scaled by the edge affinity they cross (typed modulation)."""

    def __init__(self, cin: int, c: int):
        super().__init__()
        self.reduce = conv_bn_act(cin, c, 1)
        self.mix = nn.Conv2d(9 * c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        y = self.reduce(x)
        taps = ops.modulated_taps(TorchBackend(), y, w)
        n, _, c, h, ww = taps.shape
        return y + self.act(self.bn(self.mix(taps.reshape(n, 9 * c, h, ww))))


def _zeros_linear(cin: int, cout: int, bias: bool = False) -> nn.Linear:
    m = nn.Linear(cin, cout, bias=bias)
    nn.init.zeros_(m.weight)
    if bias:
        nn.init.zeros_(m.bias)
    return m


def _band_coefficients(steps: int, betas: Sequence[float] = NOMINAL_BETAS) -> np.ndarray:
    """Coefficient of P^t in band j at a fixed chain: a_t(beta_j) - a_t(beta_{j+1}), (4, steps + 1)."""
    def a(beta: float) -> np.ndarray:
        v = np.array([(1.0 - beta) * beta ** t for t in range(steps + 1)])
        v[steps] = beta ** steps
        return v
    chain = [0.0] + list(betas)
    return np.stack([a(chain[j]) - a(chain[j + 1]) for j in range(4)])


class SpliceNeck(nn.Module):
    def __init__(self, in_channels: Sequence[int], out_channels: Sequence[int], cfg: Optional[SpliceConfig] = None,
                 workers: int = 0, strides: Sequence[int] = (8, 16, 32)):
        super().__init__()
        self.cfg = cfg = cfg or SpliceConfig()
        L, Ct = cfg.latent, cfg.transport
        self.strides = tuple(strides)
        self.proj = nn.ModuleList([conv_bn_act(c, L, 1) for c in in_channels])
        self.heads = nn.ModuleList([CellHeads(L) for _ in range(3)])
        self.mix_gain = nn.Parameter(torch.zeros(3, 4))                 # R_w = sum_j gain_lj Z_j (MIX unit)
        self.score_emb4 = nn.Conv2d(L, cfg.score_dim, 1)
        self.score_emb3 = nn.Conv2d(L, cfg.score_dim, 1)
        self.score_bias = nn.Parameter(torch.zeros(1))
        self.hyp = nn.Sequential(nn.Linear(L, L), nn.SiLU(), nn.Linear(L, 8))
        self.W_K = nn.Linear(L, cfg.key_dim, bias=False)                  # keys K_f
        self.W_U = nn.Linear(cfg.nu_dim, cfg.u_dim, bias=False)           # Psi
        self.u0 = nn.Linear(cfg.nu_dim, 1)                                # psi_0 (null option)
        self.u_alpha = nn.Linear(cfg.nu_dim, 1)                           # psi_alpha (shrink)
        self.O_F = nn.Linear(L, Ct, bias=False)                           # forecast values V O_F
        self.O_T = nn.Linear(L, Ct, bias=False)                           # translated values V O_T
        self.W_S = nn.Linear(L, cfg.evidence_dim, bias=False)             # symmetric evidence scorer
        self.trust = nn.Linear(L, 6)                                      # need, reliability, 4 error moments
        self.E_F = _zeros_linear(Ct, Ct)                                  # the zero-initialised injection map
        self.Z = nn.ModuleList([_zeros_linear(Ct, L) for _ in range(3)])  # residual maps O_l (class head only)
        if cfg.arm_spec["offset"]:
            self.score_off = nn.ModuleList([_zeros_linear(Ct, 1) for _ in range(3)])      # arm (c)
        if cfg.descriptor == "learned_mix":
            init = np.zeros((3, 4, max(cfg.steps) + 1), dtype=np.float32)     # coefficient of P^t, zero beyond s_l
            for l, s in enumerate(cfg.steps):
                init[l, :, :s + 1] = _band_coefficients(s)
            self.mix_coef = nn.Parameter(torch.tensor(init))
        if cfg.descriptor == "local":
            init = np.zeros((3, 4, 9), dtype=np.float32)
            init[:, :, 4] = 1.0                                           # centre minus a ring of growing weight
            for j in range(4):
                init[:, j, [0, 1, 2, 3, 5, 6, 7, 8]] = -(j + 1) / 32.0
            self.local_coef = nn.Parameter(torch.tensor(init))
        self.td4, self.td3 = ModFuse(2 * L, L), ModFuse(2 * L, L)
        self.bu4, self.bu5 = ModFuse(2 * L, L), ModFuse(2 * L, L)
        self.down3, self.down4 = conv_bn_act(L, L, 3, 2), conv_bn_act(L, L, 3, 2)
        self.cls_out = nn.ModuleList([conv_bn_act(L, o, 1) for o in out_channels])
        self.box_out = nn.ModuleList([conv_bn_act(L, o, 1) for o in out_channels])
        self.register_buffer("gate_tau", torch.tensor(cfg.gate_tau, dtype=torch.float32))
        self.register_buffer("prior", torch.tensor(cfg.prior, dtype=torch.float32))
        self.register_buffer("band_exp", torch.tensor([[1, 2, 2, 3]] * 3, dtype=torch.float32))
        with torch.no_grad():
            self.u_alpha.bias.fill_(-2.0)
            self.trust.bias.copy_(torch.tensor([0.0, 0.5413, 0.0, 0.0, 0.0, 0.0]))   # zeta = 1/2, rho = softplus(0.5413) = 1
        self.workers = workers
        self._pool = None
        self.quant = None          # set by qat.attach(): fake quantization at the manifest sites
        self.diag_exchange = False  # diagnostics: also run the exchange where inference drops it (arm f)

    # ------------------------------------------------------------------ helpers
    def __getstate__(self):
        """The process pool is not copied (EMA deep copies, pickling)."""
        state = super().__getstate__() if hasattr(super(), "__getstate__") else self.__dict__.copy()
        state = dict(state)
        state["_pool"] = None
        return state

    @property
    def pool(self) -> lb.Pool:
        if self._pool is None:
            self._pool = lb.Pool(self.workers)
        return self._pool

    def _q(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return self.quant(name, x) if self.quant is not None else x

    def window_logits(self, phi4: torch.Tensor) -> torch.Tensor:
        """Pair logits of the P4 forward half window: (N, 12, H4, W4)."""
        B = TorchBackend()
        d = math.sqrt(phi4.shape[1])
        out = [(phi4 * B.shift2d(phi4, dy, dx)).sum(1) / d for (dy, dx) in LINK_FWD]
        return torch.stack(out, 1) + self.score_bias

    def reserve_logits(self, phi3_n: torch.Tensor, phi4_n: torch.Tensor, reserve: np.ndarray, rr_p: np.ndarray,
                       w3: int):
        """Reserve-window logits (R, 25) and reserve-pair logits (R, 8) of one image."""
        c, h4, w4 = phi4_n.shape
        d = math.sqrt(c)
        r = torch.as_tensor(reserve, device=phi3_n.device, dtype=torch.long)
        if r.numel() == 0:
            z = phi3_n.new_zeros((0, 25)), phi3_n.new_zeros((0, 8))
            return z
        f3 = phi3_n.flatten(1)[:, r].T                                         # (R, c)
        py, px = (r // w3) >> 1, (r % w3) >> 1
        dy = torch.arange(25, device=r.device) // 5 - 2
        dx = torch.arange(25, device=r.device) % 5 - 2
        yy, xx = py[:, None] + dy[None], px[:, None] + dx[None]
        inside = (yy >= 0) & (yy < h4) & (xx >= 0) & (xx < w4)
        idx = (yy.clamp(0, h4 - 1) * w4 + xx.clamp(0, w4 - 1))
        f4 = phi4_n.flatten(1).T[idx]                                         # (R, 25, c)
        rs = (f3[:, None, :] * f4).sum(-1) / d + self.score_bias
        rs = torch.where(inside, rs, torch.full_like(rs, -30.0))
        rp = torch.as_tensor(rr_p, device=r.device, dtype=torch.long)
        valid = rp != 0xFFFF
        rr = (f3[:, None, :] * f3[rp.clamp(max=r.numel() - 1)]).sum(-1) / d + self.score_bias
        rr = torch.where(valid, rr, torch.full_like(rr, -30.0))
        return rs, rr

    def G(self, X: List[torch.Tensor], W: List[torch.Tensor]) -> List[torch.Tensor]:
        """The neck after the synthesis (4): top-down and bottom-up paths."""
        x3, x4, x5 = X
        t5 = x5
        t4 = self.td4(torch.cat([x4, F.interpolate(t5, size=x4.shape[-2:], mode="nearest")], 1), W[1])
        t3 = self.td3(torch.cat([x3, F.interpolate(t4, size=x3.shape[-2:], mode="nearest")], 1), W[0])
        n3 = t3
        n4 = self.bu4(torch.cat([t4, self.down3(n3)], 1), W[1])
        n5 = self.bu5(torch.cat([t5, self.down4(n4)], 1), W[2])
        return [n3, n4, n5]

    # ------------------------------------------------------------------ descriptor
    def _descriptor(self, l: int, X: torch.Tensor, W: torch.Tensor, powers: List[torch.Tensor], betas: torch.Tensor,
                    Z: List[torch.Tensor]):
        """Planes read by the forecast descriptor (four per level), their diffusion-time
        weights tau (N, 4, H, W) and the walk whose leaving probability sets the exposure."""
        cfg, B = self.cfg, TorchBackend()
        s = cfg.steps[l]
        kind = cfg.descriptor
        if kind in ("bands", "no_discount"):
            return Z, ops.band_tau(B, betas, s), W
        if kind == "learned_mix":
            c = self.mix_coef[l][:, :s + 1]
            planes = [sum(c[j, t] * powers[t] for t in range(s + 1)) for j in range(4)]
            t_idx = torch.arange(s + 1, device=X.device, dtype=c.dtype)
            tau = (c.detach().abs() * t_idx[None]).sum(1)                       # (4,)
            return planes, tau[None, :, None, None] + betas * 0, W
        if kind == "lowpass":
            planes = [self._q(f"band{l}{j}", ops.bank(B, powers, betas[:, j:j + 1])) for j in range(4)]
            return planes, ops.lowpass_time(B, betas, s), W
        if kind == "blind":
            one = torch.ones_like(W[:, :1])
            Wb = torch.cat([B.shift2d(one, dy, dx) for (dy, dx) in ops.NEIGH], 1)   # 1 inside the lattice
            Zb, _ = ops.bands(B, ops.walk_powers(B, X, Wb, s), betas)
            return [self._q(f"band{l}{j}", z) for j, z in enumerate(Zb)], ops.band_tau(B, betas, s), Wb
        if kind == "local":
            c = self.local_coef[l]
            shifts = [X if (k9 == 4) else B.shift2d(X, k9 // 3 - 1, k9 % 3 - 1) for k9 in range(9)]
            planes = [sum(c[j, k] * shifts[k] for k in range(9)) for j in range(4)]
            return planes, betas * 0 + 1.0, W
        if kind == "single_step":
            b = torch.cat([betas[:, :1] * 0, betas], 1)
            step = powers[0] - powers[1]                                        # X - P X
            planes = [(b[:, j + 1:j + 2] - b[:, j:j + 1]) * step for j in range(4)]
            return planes, b[:, 1:] - b[:, :-1], W
        raise ValueError(kind)

    # ------------------------------------------------------------------ forward
    def forward(self, feats: Sequence[torch.Tensor], boxes: Optional[List[np.ndarray]] = None,
                probe: bool = False, stats: bool = False) -> Dict:
        """feats: three backbone taps (strides 8, 16, 32).  boxes: per-image GT boxes (xyxy
        pixels), carried for the callers (losses, calibration).  probe: add the sensitivity
        term.  stats: keep the pair-level forecast quantities for the diagnostics.
        Returns head inputs and auxiliaries."""
        cfg = self.cfg
        arm = cfg.arm_spec
        B = TorchBackend()
        X = [self._q("latent", p(f)) for p, f in zip(self.proj, feats)]
        hd = [h(x) for h, x in zip(self.heads, X)]
        W = [ops.edge_weights(B, h["q"], h["pi"]) for h in hd]
        Rw, planes, taus, Wd = [], [], [], []
        for l in range(3):
            betas = ops.beta_chain(B, hd[l]["beta_u"])
            powers = ops.walk_powers(B, X[l], W[l], cfg.steps[l])
            Z, _low = ops.bands(B, powers, betas)
            Z = [self._q(f"band{l}{j}", z) for j, z in enumerate(Z)]
            Rw.append(sum(self.mix_gain[l, j] * Z[j] for j in range(4)))
            pl, tau, wd = self._descriptor(l, X[l], W[l], powers, betas, Z)
            planes.append(pl)
            taus.append(tau.detach())
            Wd.append(wd.detach())
        n_img = X[0].shape[0]
        (h3, w3), (h4, w4), (h5, w5) = [x.shape[-2:] for x in X]
        e_codes = [lb.to_u8(h["e"][:, 0].detach().float().cpu().numpy()) for h in hd]
        t3_codes = lb.to_u8(hd[0]["t"][:, 0].detach().float().cpu().numpy())
        reserves = self.pool.map(lb.reserve_job, [(e_codes[0][n].ravel(), t3_codes[n].ravel(), cfg.n_reserve, w3)
                                                  for n in range(n_img)])
        phi4, phi3 = self.score_emb4(X[1]), self.score_emb3(X[0])
        win = self.window_logits(phi4)                                          # (N, 12, H4, W4)
        rr_ps = [lb.rr_pairs(r, w3) for r in reserves]
        res_logits = [self.reserve_logits(phi3[n], phi4[n], reserves[n], rr_ps[n], w3) for n in range(n_img)]
        win_codes = lb.to_u8(torch.sigmoid(win).detach().float().cpu().numpy())
        jobs = []
        for n in range(n_img):
            sc = np.transpose(win_codes[n], (1, 2, 0)).reshape(h4 * w4, 12)
            rs = lb.to_u8(torch.sigmoid(res_logits[n][0]).detach().float().cpu().numpy())
            rr = lb.to_u8(torch.sigmoid(res_logits[n][1]).detach().float().cpu().numpy())
            jobs.append((sc, reserves[n], rs, rr_ps[n], rr, e_codes[1][n], e_codes[0][n], h3, w3, h4, w4, h5, w5,
                         cfg.link_budget, cfg.f_cap))
        links = self.pool.map(lb.linkage_job, jobs)

        use_exchange = cfg.needs_exchange(self.training) or (self.diag_exchange and arm["exchange"] != "never")
        frag, ex_jobs = [], []
        dev = X[0].device
        zero_ct = lambda: X[0].new_zeros((0, cfg.transport))
        for n in range(n_img):
            lk = links[n]
            Fn = lk["F"]
            ids = {k: torch.as_tensor(v, device=dev) for k, v in lk["ids"].items()}
            res = torch.as_tensor(reserves[n], device=dev, dtype=torch.long)
            x_anchor = torch.cat([X[1][n].flatten(1).T, X[0][n].flatten(1).T[res]], 0)
            ids_a = torch.as_tensor(lk["ids_anchor"], device=dev)
            e_a = torch.as_tensor(lk["e_anchor"], device=dev, dtype=x_anchor.dtype)
            omega = torch.where(ids_a >= 0, e_a, torch.zeros_like(e_a))
            hyp = self.hyp(x_anchor)
            mu, prec, st = self._hypotheses(hyp, res, w3, h4, w4)
            rec = {"F": Fn, "ids": ids, "ids_anchor": ids_a, "reserve": reserves[n], "rr_p": rr_ps[n],
                   "res_logits": res_logits[n], "mu": mu, "logprec": hyp[:, 4:8], "stride_a": st, "link": lk,
                   "dh_hat": None, "dH": None, "r": None}
            frag.append(rec)
            if Fn == 0:
                rec.update(dh_hat=zero_ct() if arm["forecast"] else None, dH=zero_ct() if use_exchange else None)
                continue
            H, _ = ops.restrict(B, x_anchor, ids_a, omega, Fn)
            H = self._q("H", H)
            Hs = H[:, :cfg.transport]
            nu = torch.cat([ops.descriptor_nu(B, [self._band_code(z[n], l, j) for j, z in enumerate(planes[l])],
                                              ids[f"P{l + 3}"], Fn, cfg.transport) for l in range(3)], 1)
            mu_f, var_f = ops.fuse_hypotheses(B, mu.detach(), prec.detach(), ids_a, Fn)
            s = ops.gate_stratum(Fn, cfg.gate_strata)
            adm = ops.admissible(B, ops.mahalanobis2(B, mu_f, var_f), float(self.gate_tau[s]))
            rec.update(H=H, nu=nu, adm=adm, mu_f=mu_f, var_f=var_f, stratum=s)
            # ---- forecast (Eq. (3)) per arm
            if arm["forecast"] == "confined":
                K = self._q("K", self.W_K(H))
                Uv = self._q("Uv", self.W_U(nu))
                l0 = self.u0(nu)[:, 0]
                al = self.u_alpha(nu)[:, 0]
                VO = self._q("VO", self.O_F(H))
                disc = None
                if cfg.discount:
                    xi, disc, kappa = self._exposure(n, ids, Fn, planes, taus, Wd)
                    rec.update(xi=xi, kappa=kappa)
                dh_hat, y, y0, alpha = ops.forecast(B, K, Uv, VO, Hs, l0, al, adm, cfg.key_dim, cfg.clip_b,
                                                    cfg.eps_shrink, disc=disc)
                dh_hat = self._q("dHhat", dh_hat)
                rec.update(K=K, Uv=Uv, l0=l0, al=al, VO=VO, dh_hat=dh_hat, a0=y0, alpha=alpha)
                if stats:
                    mag = torch.clamp(VO[None] - Hs[:, None], -cfg.clip_b, cfg.clip_b).abs().sum(-1)
                    clipped = ((VO[None] - Hs[:, None]).abs() >= cfg.clip_b).float().mean(-1)
                    rec.update(transport_fg=(alpha[:, None] * y * mag).detach(), y=y.detach(),
                               clip_fg=(y * clipped).detach())
            elif arm["forecast"] == "free":
                K = self._q("K", self.W_K(H))
                VO = self._q("VO", self.O_F(H))
                dh_hat, y = ops.forecast_free(B, K, VO, Hs, cfg.key_dim)
                dh_hat = self._q("dHhat", dh_hat.clamp(-cfg.transport_lsb * 127, cfg.transport_lsb * 127))
                rec.update(K=K, VO=VO, dh_hat=dh_hat)
                if stats:
                    rec.update(transport_fg=(y * (VO[None] - Hs[:, None]).abs().sum(-1)).detach(), y=y.detach())
            # ---- exchange inputs (Eq. (1), (2))
            if use_exchange:
                Sx = self.W_S(H)
                ev = (Sx @ Sx.T / math.sqrt(Sx.shape[1])).clamp(-127.9, 127.9)
                tr = self.trust(H)
                need = torch.sigmoid(tr[:, 0])                                  # zeta in [0, 1]
                rho = F.softplus(tr[:, 1]) + 1e-3
                phi = 0.25 * torch.tanh(tr[:, 2:6])
                vo_t = self._q("VO_T", self.O_T(H))
                np64 = lambda t: t.detach().double().cpu().numpy()
                ex_jobs.append((n, (adm.cpu().numpy(), np64(ev), np64(need), np64(rho), np64(phi), np64(vo_t), np64(Hs),
                                    list(self.prior.cpu().numpy()), cfg.a_r, cfg.b_r, cfg.c_r, cfg.ms, cfg.n_cap, cfg.kmax)))
                rec.update(need=need, rho=rho, phi=phi, ev=ev, vo_t=vo_t)
        # ---- exchange: decisions on the CPU pool, values inside autograd when training
        if ex_jobs:
            exs = self.pool.map(exchange_float.exchange_job, [j for _, j in ex_jobs])
            lim = cfg.transport_lsb * 127
            if self.training:
                items = []
                for (n, job), ex in zip(ex_jobs, exs):
                    rec = frag[n]
                    items.append({"ev": rec["ev"], "need": rec["need"], "rho": rec["rho"], "phi": rec["phi"],
                                  "vo_t": rec["vo_t"], "hs": rec["H"][:, :cfg.transport],
                                  "dec": exchange_float.decisions(job[0], ex["lam"], cfg.n_cap, cfg.ms)})
                outs = exchange_torch.exchange_batch(items, list(self.prior.cpu().numpy()), cfg.a_r, cfg.b_r, cfg.c_r,
                                                     cfg.kmax, cfg.ms)
                for (n, _), ex, o in zip(ex_jobs, exs, outs):
                    frag[n].update(dH=self._q("dH", o["dV"].clamp(-lim, lim)), lam=ex["lam"], mutual=ex["mutual"],
                                   screened=ex["screened"])
            else:
                for (n, _), ex in zip(ex_jobs, exs):
                    dH = torch.as_tensor(ex["dV"], device=dev, dtype=X[0].dtype).clamp(-lim, lim)
                    frag[n].update(dH=self._q("dH", dH), lam=ex["lam"], mutual=ex["mutual"], screened=ex["screened"])
        # ---- per image: forecast loss, injection, residual, score offset
        fc_loss = X[0].new_zeros(())
        n_rows = 0
        inj_maps = [[None] * n_img for _ in range(3)]            # per level, per image: (Ct, H, W) or None
        off_maps = [[None] * n_img for _ in range(3)] if arm["offset"] else None
        for n, rec in enumerate(frag):
            if rec["F"] == 0:
                continue
            dh_hat, dH = rec["dh_hat"], rec["dH"]
            if dh_hat is not None and dH is not None:
                fc_loss = fc_loss + ((dh_hat - dH.detach()) ** 2).sum()
                n_rows += dh_hat.numel()
            moved = dH if arm["inject"] == "exact" else dh_hat if arm["inject"] == "forecast" else None
            if moved is not None:
                inj = self.E_F(moved)
                for l in range(3):
                    inj_maps[l][n] = ops.fill(B, inj, rec["ids"][f"P{l + 3}"]).T.reshape(cfg.transport, *X[l].shape[-2:])
            if arm["residual"] and dH is not None:
                rec["r"] = dH - dh_hat if dh_hat is not None else dH
            if arm["offset"] and dH is not None:
                for l in range(3):
                    off_maps[l][n] = ops.fill(B, self.score_off[l](dH), rec["ids"][f"P{l + 3}"]).T.reshape(1, *X[l].shape[-2:])
        # synthesis, Eq. (4): X'^s = X^s + M dHhat + R_w^perp,s ; X'^u = X^u + R_w^perp,u, where the
        # detail correction R_w^perp = Q R_w changes no fragment state
        Xp = []
        for l in range(3):
            shp = X[l].shape[-2:]
            per = []
            for n in range(n_img):
                Fn = frag[n]["F"]
                idl = frag[n]["ids"][f"P{l + 3}"]
                rw = Rw[l][n].flatten(1).T
                if Fn:
                    wl = ops.level_weights(B, self._level_e(hd[l]["e"][n]), idl, Fn)
                    rw = rw - ops.fill(B, ops.restrict(B, rw, idl, wl, Fn)[0], idl)
                add = rw.T.reshape(-1, *shp)
                if inj_maps[l][n] is not None:
                    add = add + torch.cat([inj_maps[l][n], add.new_zeros((add.shape[0] - cfg.transport,) + tuple(shp))], 0)
                per.append(add)
            Xp.append(self._q("Xsyn", X[l] + torch.stack(per, 0)))
        Nl = self.G(Xp, W)
        cls_feats, box_feats = [], []
        for l in range(3):
            cbs, bbs = [], []
            for n in range(n_img):
                Fn = frag[n]["F"]
                idl = frag[n]["ids"][f"P{l + 3}"]
                if Fn == 0:                                  # every cell unassigned: shared part 0, detail part X''
                    cbs.append(torch.zeros_like(Nl[l][n]))
                    bbs.append(Nl[l][n])
                    continue
                rows = Nl[l][n].flatten(1).T
                wl = ops.level_weights(B, self._level_e(hd[l]["e"][n]), idl, Fn)
                rz = self.Z[l](frag[n]["r"]) if frag[n]["r"] is not None else None
                cbs.append(ops.class_band(B, rows, idl, wl, Fn, rz).T.reshape(Nl[l].shape[1:]))
                bbs.append(ops.box_band(B, rows, idl, wl, Fn).T.reshape(Nl[l].shape[1:]))
            cls_feats.append(self.cls_out[l](self._q(f"cls_in{l}", torch.stack(cbs, 0))))
            box_feats.append(self.box_out[l](self._q(f"box_in{l}", torch.stack(bbs, 0))))
        offsets = None
        if off_maps is not None:
            offsets = [torch.stack([m if m is not None else X[l].new_zeros((1,) + tuple(X[l].shape[-2:]))
                                    for m in off_maps[l]], 0) for l in range(3)]
        out = {"cls_feats": cls_feats, "box_feats": box_feats, "gates": [h["g"] for h in hd], "heads": hd,
               "window_logits": win, "frag": frag, "N": Nl, "fc_loss": fc_loss / max(n_rows, 1),
               "fc_rows": n_rows, "W": W, "Wd": Wd, "taus": taus, "score_offset": offsets, "arm": cfg.arm,
               "Xp": Xp if stats else None}
        if probe:
            out["sens_loss"] = self._sensitivity(Xp, W, frag, hd, Nl)
        return out

    # ------------------------------------------------------------------ pieces
    def _band_code(self, z: torch.Tensor, level: int, band: int) -> torch.Tensor:
        """Band plane of one image (C, H, W) -> rows (HW, C) in band-code units (DD-04)."""
        e = float(self.band_exp[level, band])
        return (z * (2.0 ** e / self.cfg.transport_lsb)).flatten(1).T

    def _level_e(self, e_map: torch.Tensor) -> torch.Tensor:
        """Eligibility codes (u8 scale, detached) of one image's level, flattened."""
        return torch.round(e_map.detach()[0].flatten() * 255.0)

    def _exposure(self, n: int, ids: Dict[str, torch.Tensor], Fn: int, planes, taus, Wd):
        """Exposure budget xi_g and its logit discount log(1 - xi_g) for the fragments of image n
        (constants of the graph).  Per level: band-energy shares of the fragment, the largest
        diffusion-time weight tau of its cells and its largest one-step leaving probability."""
        cfg, B = self.cfg, TorchBackend()
        with torch.no_grad():
            en, tf, kp = [], [], []
            for l in range(3):
                idl = ids[f"P{l + 3}"]
                ones = torch.ones_like(idl, dtype=planes[l][0].dtype)
                for j in range(4):
                    e = (planes[l][j][n][:cfg.transport].detach() ** 2).mean(0).flatten()
                    en.append(ops.restrict(B, e[:, None], idl, ones, Fn)[0][:, 0])
                tf.append(ops.segment_amax(B, taus[l][n].flatten(1).T, idl, Fn))               # (F, 4)
                code = (idl + 2).to(Wd[l].dtype).reshape(1, 1, *Wd[l].shape[-2:])
                kp.append(ops.segment_amax(B, ops.leave_prob(B, Wd[l][n:n + 1], code).flatten()[:, None], idl, Fn)[:, 0])
            E = torch.stack(en, 1)                                                              # (F, 12)
            shares = E / E.sum(1, keepdim=True).clamp(min=1e-20)
            kappa = torch.stack(kp, 1).amax(1)
            xi, disc = ops.exposure_discount(B, kappa, shares, torch.cat(tf, 1), cfg.eps_shrink)
        return xi, disc, kappa

    def _hypotheses(self, hyp: torch.Tensor, reserve: torch.Tensor, w3: int, h4: int, w4: int):
        """Box hypothesis per anchor (P4 cells, then the P3 reserves): mu (x0, y0, x1, y1) in
        pixels from softplus distances to the anchor centre, the precision in 1/px^2 and the
        anchor's stride.  The precision is exp(logprec) per stride^2, so the logits stay in
        the same range at every level (the fusion and the test use pixels)."""
        dev = hyp.device
        n4 = h4 * w4
        ys4, xs4 = torch.div(torch.arange(n4, device=dev), w4, rounding_mode="floor"), torch.arange(n4, device=dev) % w4
        s4, s3 = float(self.strides[1]), float(self.strides[0])
        cx = torch.cat([(xs4.float() + 0.5) * s4, ((reserve % w3).float() + 0.5) * s3])
        cy = torch.cat([(ys4.float() + 0.5) * s4, (torch.div(reserve, w3, rounding_mode="floor").float() + 0.5) * s3])
        st = torch.cat([torch.full((n4,), s4, device=dev), torch.full((reserve.numel(),), s3, device=dev)])
        d = F.softplus(hyp[:, 0:4]) * st[:, None] * 2.0
        mu = torch.stack([cx - d[:, 0], cy - d[:, 1], cx + d[:, 2], cy + d[:, 3]], 1)
        prec = torch.exp(hyp[:, 4:8].clamp(-12.0, 2.0)) / (st[:, None] ** 2)
        return mu, prec, st

    def probe_response(self, Xp, W, frag, hd, Nl):
        """Response of the neck after (4) to a random fragment-constant probe v ~ N(0, (probe
        sigma)^2) on the shared channels: returns (sum ||Q dN||^2, sum ||dN||^2, sum ||M v||^2,
        number of entries) over the cells of the batch."""
        cfg = self.cfg
        B = TorchBackend()
        deltas = [torch.randn(rec["F"], cfg.transport, device=Xp[0].device) * cfg.probe_sigma_lsb * cfg.transport_lsb
                  if rec["F"] else None for rec in frag]
        Xd = []
        probe_sq = Xp[0].new_zeros(())
        for l in range(3):
            shp = Xp[l].shape[-2:]
            per = []
            for n, rec in enumerate(frag):
                z = Xp[l].new_zeros(Xp[l].shape[1:])
                if deltas[n] is not None:
                    m = ops.fill(B, deltas[n], rec["ids"][f"P{l + 3}"]).T.reshape(cfg.transport, *shp)
                    probe_sq = probe_sq + (m ** 2).sum()
                    z = torch.cat([m, z[cfg.transport:]], 0)
                per.append(z)
            Xd.append(Xp[l] + torch.stack(per, 0))
        Nd = self.G(Xd, W)
        tot = Xp[0].new_zeros(())
        full = Xp[0].new_zeros(())
        cnt = 0
        for l in range(3):
            for n, rec in enumerate(frag):
                if rec["F"] == 0:
                    continue
                idl = rec["ids"][f"P{l + 3}"]
                diff = (Nd[l][n] - Nl[l][n]).flatten(1).T
                wl = ops.level_weights(B, self._level_e(hd[l]["e"][n]), idl, rec["F"])
                q = ops.box_band(B, diff, idl, wl, rec["F"])          # Q = I on unassigned cells
                tot = tot + (q ** 2).sum()
                full = full + (diff ** 2).sum()
                cnt += q.numel()
        return tot, full, probe_sq, cnt

    def _sensitivity(self, Xp, W, frag, hd, Nl) -> torch.Tensor:
        """mu_s term: || Q [G(X' + M v) - G(X')] ||^2 (mean over cells and channels)."""
        tot, _full, _probe, cnt = self.probe_response(Xp, W, frag, hd, Nl)
        return tot / max(cnt, 1)
