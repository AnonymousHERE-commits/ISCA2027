"""Configuration of the SPLICE detector; numeric defaults come from the integer manifest
(generated/py/splicev_gen.py) so the float model and the device share one contract.

Readout arms (SPLICE-V section 2, Table 7).  Every arm keeps preserve, coarsen, the
supervised heads, the bottom-up path and the Pi / Q readout split; the arms differ in
where the transport enters:

  arm  inject (before the bottom-up path)   class head reads            exchange at inference
  a    exact dH                             Pi N                        yes
  b    nothing                              Pi N + M dH O               yes
  c    nothing                              Pi N, logits + M (dH o)     yes
  d    free forecast dHhat                  Pi N + M (dH - dHhat) O     yes
  e    confined forecast dHhat (deployed)   Pi N + M (dH - dHhat) O     yes
  f    confined forecast dHhat              Pi N                        no (training target only)
  g    nothing                              Pi N                        no
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

from . import REPO  # noqa: F401  (path setup)
import splicev_gen as gen

ARMS: Dict[str, Dict] = {
    "a": {"inject": "exact", "forecast": None, "residual": False, "offset": False, "exchange": "always",
          "label": "transport injected before the bottom-up path"},
    "b": {"inject": None, "forecast": None, "residual": True, "offset": False, "exchange": "always",
          "label": "injection at the readout, no forecast"},
    "c": {"inject": None, "forecast": None, "residual": False, "offset": True, "exchange": "always",
          "label": "per-fragment score offset"},
    "d": {"inject": "forecast", "forecast": "free", "residual": True, "offset": False, "exchange": "always",
          "label": "free forecast + residual"},
    "e": {"inject": "forecast", "forecast": "confined", "residual": True, "offset": False, "exchange": "always",
          "label": "confined forecast + residual (deployed)"},
    "f": {"inject": "forecast", "forecast": "confined", "residual": False, "offset": False, "exchange": "train",
          "label": "confined forecast alone"},
    "g": {"inject": None, "forecast": None, "residual": False, "offset": False, "exchange": "never",
          "label": "no transport"},
}

# Forecast descriptor read by arm (e) (Table 7, lower block)
DESCRIPTORS: Dict[str, str] = {
    "bands": "signed, ownership-typed bands with the exposure discount (deployed)",
    "no_discount": "typed bands, no exposure discount",
    "learned_mix": "learned mixing of the same walk powers",
    "lowpass": "low-pass stack A(beta_j) X",
    "blind": "ownership-blind bands (pi = q = 1)",
    "local": "local descriptors of equal cost (channel-shared 3x3 stencils)",
    "single_step": "single-step stencil only",
}


@dataclass
class SpliceConfig:
    # geometry (section 2, Table 5)
    latent: int = gen.C_LATENT if hasattr(gen, "C_LATENT") else 64
    transport: int = gen.C_TRANSPORT
    steps: Tuple[int, int, int] = (gen.LEVELS["P3"]["steps"], gen.LEVELS["P4"]["steps"], gen.LEVELS["P5"]["steps"])
    key_dim: int = gen.KEY_D
    u_dim: int = gen.FC_UDIM
    nu_dim: int = gen.NU_DIM
    score_dim: int = 16                 # pair-scorer embedding (DD: linkage.score_format)
    evidence_dim: int = 32              # symmetric evidence scorer width
    n_reserve: int = gen.N_RESERVE
    f_cap: int = gen.F_CAP
    n_cap: int = gen.N_CAP
    link_budget: int = gen.LINK_BUDGET
    kmax: int = gen.KMAX
    ms: int = gen.MS
    # forecast, Eq. (3)
    clip_b_lsb: int = gen.FC_B          # clip Theta in transport LSB
    eps_shrink: float = 2.0 ** -4
    transport_lsb: float = 1.0 / 16.0   # float value of one transport LSB (QAT scale of X^s, H^s, dH)
    # exchange, Eq. (1)-(2)
    a_r: float = 0.25
    b_r: float = 1.0
    c_r: float = 0.5
    prior: Tuple[float, ...] = (0.5, 0.3, 0.2)
    gate_strata: Tuple[int, ...] = (32, 96, 192)
    gate_tau: Tuple[float, ...] = (13.3, 13.3, 13.3, 13.3)     # replaced by calibration (delta = 0.01)
    gate_delta: float = 0.01
    # training (section 2)
    mu_forecast: float = 0.1
    lambda_sens: float = 0.05           # mu_s of the paper
    probe_sigma_lsb: float = 4.0        # DD: fragment-constant probe v ~ N(0, (4 LSB)^2) per channel
    aux_weight: float = 1.0             # DD: weight of the supervised neck heads
    hyp_weight: float = 0.05            # DD: weight of the hypothesis NLL
    # readout arm and forecast descriptor (Table 7)
    variant: str = "SPLICE-B"
    arm: str = "e"
    descriptor: str = "bands"
    theta_det: float = 0.25             # deployment score threshold of the skip rule

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; choose from {sorted(ARMS)}")
        if self.descriptor not in DESCRIPTORS:
            raise ValueError(f"unknown descriptor {self.descriptor!r}; choose from {sorted(DESCRIPTORS)}")

    @property
    def clip_b(self) -> float:
        return self.clip_b_lsb * self.transport_lsb

    @property
    def arm_spec(self) -> Dict:
        return ARMS[self.arm]

    def needs_exchange(self, training: bool) -> bool:
        mode = self.arm_spec["exchange"]
        return mode == "always" or (mode == "train" and training)

    @property
    def discount(self) -> bool:
        """The exposure discount log(1 - xi_g) of the confined forecast."""
        return self.arm_spec["forecast"] == "confined" and self.descriptor != "no_discount"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SpliceConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        for k in ("steps", "prior", "gate_strata", "gate_tau"):
            if k in known:
                known[k] = tuple(known[k])
        return cls(**known)
