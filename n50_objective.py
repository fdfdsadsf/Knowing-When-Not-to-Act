# N=50 portfolio objective experiment from the ICLR 2027 submission.
# Direct policy optimization, feedback-BPTT continuation ratios,
# Yosida action realization, and fixed-time control refinement.
# Core numerical/training hyperparameters are unchanged from the supplied code.

import os, math, time, random, re, json, hashlib, platform, sys
from dataclasses import asdict
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


@dataclass
class Config:
    N: int = 2
    T: float = 1.0
    n_steps: int = 12
    r: float = 0.02

    mu_base: Tuple[float, ...] = (0.08, 0.10, 0.09, 0.11, 0.095, 0.105, 0.115, 0.085)
    sigma_base: Tuple[float, ...] = (0.20, 0.25, 0.22, 0.18, 0.21, 0.24, 0.23, 0.19)
    corr_mode: str = "random_spd"       # "equicorr" or "random_spd"
    rho: float = 0.2                    # only for equicorr
    corr_bound: float = 0.2
    corr_seed_offset: int = 12345

    # Correlated sector rotation: gross risk 80%, cash 20%, 32pp sector swap.
    mu_from_merton_target: bool = True
    sector_merton_A: float = 0.56
    sector_merton_B: float = 0.24
    sector_initial_A: float = 0.24
    sector_initial_B: float = 0.56
    # Global feasible simplex sampler including exact cash/holding zero faces.
    train_gross_risky_max: float = 1.00
    # Emergency terminal guard only; valid long-only GBM paths remain positive.
    exit_liq_floor: float = 0.05

    gamma: float = 3.0
    alpha: float = 0.005
    eta: float = 0.01
    u_max: float = 10.0

    # u_theta training
    hidden: int = 96
    depth: int = 2
    batch_size: int = 128
    n_train_steps: int = 600
    lr: float = 1e-3
    print_every: int = 200

    # PINN for eta-regularized HJB
    pinn_hidden: int = 128
    pinn_depth: int = 3
    pinn_batch_size: int = 128
    pinn_train_steps: int = 1000
    pinn_lr: float = 5e-4
    pinn_print_every: int = 100
    pinn_hutchinson_samples: int = 1
    pinn_monotonicity_weight: float = 10.0
    pinn_vx_floor: float = 1e-8
    pinn_grad_clip: float = 10.0
    pinn_residual_scale: float = 2.0
    pinn_regime_tol: float = 1e-4
    pinn_use_float64: bool = True

    # Evaluation / Rhat_BPTT
    outer_eval_paths: int = 128
    inner_mc_paths: int = 48
    eval_chunk_size: int = 64
    marginal_floor: float = 1e-10

    # Trajectory distillation: teacher rollouts are DISJOINT from held-out J0 paths.
    distill_teacher_paths: int = 64
    distill_teacher_seed_extra: int = 600_000  # held-out evaluation uses 300_000
    distill_hidden: int = 256
    distill_depth: int = 3
    distill_batch_size: int = 512
    distill_steps: int = 1000
    distill_lr: float = 5e-4
    distill_eval_every: int = 25
    distill_early_stop_patience: int = 12

    # Same-time P_eta ratio-map depths. [star] uses p_eta_star_max_iterations.
    p_eta_repeat_depths: Tuple[int, ...] = (1, 2, 4, 8)
    p_eta_star_max_iterations: int = 500
    same_time_position_tol: float = 1e-10
    no_trade_rate_tol: float = 1e-8
    boundary_plot_resolution: int = 21

    # Hard action-magnitude thresholds for the single-step u_theta,0 ablation.
    # Units are the normalized policy-rate units returned by _u_theta_action.
    # IMPORTANT for paper experiments: choose/freeze these on validation data;
    # do not select the best tau on the held-out test objective.
    hard_action_thresholds: Tuple[float, ...] = (1e-3, 1e-2, 5e-2)

    x0_eval: float = 0.20
    logw_low: float = math.log(0.8)
    logw_high: float = math.log(1.2)

    use_common_random_numbers: bool = True
    use_antithetic: bool = True
    seed: int = 10
    device: str = "auto"
    outdir: str = "near_zero_to_zero"
    param_extension_mode: str = "tile"

    @property
    def dt(self) -> float:
        return self.T / self.n_steps

    def _expand_param(self, base: Tuple[float, ...], name: str) -> np.ndarray:
        arr = np.asarray(base, dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"{name}_base must be a nonempty 1D tuple/list")
        if self.N <= arr.size:
            return arr[:self.N].copy()
        if self.param_extension_mode == "tile":
            reps = int(np.ceil(self.N / arr.size))
            return np.tile(arr, reps)[:self.N].astype(np.float64)
        if self.param_extension_mode == "linear":
            if arr.size == 1:
                return np.full(self.N, float(arr[0]), dtype=np.float64)
            return np.linspace(float(arr[0]), float(arr[-1]), self.N, dtype=np.float64)
        raise ValueError(f"Unknown param_extension_mode={self.param_extension_mode}")

    @property
    def merton_target_pi(self) -> np.ndarray:
        if self.N < 2:
            raise ValueError("Two-sector correlation design needs N >= 2")
        nA = self.N // 2
        nB = self.N - nA
        return np.r_[np.full(nA, self.sector_merton_A/nA),
                     np.full(nB, self.sector_merton_B/nB)]

    @property
    def mu(self) -> np.ndarray:
        if self.mu_from_merton_target:
            # Invert pi^M = (gamma Sigma)^(-1) (mu - r * 1) for each N.
            # Do not tile a 50-dimensional mu vector at other dimensions.
            return self.r + self.gamma * (self.cov @ self.merton_target_pi)
        return self._expand_param(self.mu_base, "mu")

    @property
    def sigma(self) -> np.ndarray:
        return self._expand_param(self.sigma_base, "sigma")

    @property
    def y0_eval(self) -> np.ndarray:
        if self.N < 2:
            raise ValueError("Two-sector correlation design needs N >= 2")
        nA = self.N // 2
        nB = self.N - nA
        return np.r_[np.full(nA, self.sector_initial_A/nA),
                     np.full(nB, self.sector_initial_B/nB)]

    @property
    def corr(self) -> np.ndarray:
        if self.corr_mode == "equicorr":
            C = np.full((self.N, self.N), float(self.rho), dtype=np.float64)
            np.fill_diagonal(C, 1.0)
            return C
        if self.corr_mode == "random_spd":
            return _make_random_corr_matrix(
                N=self.N, bound=self.corr_bound,
                seed=self.seed + self.corr_seed_offset,
            )
        raise ValueError(f"Unknown corr_mode={self.corr_mode}")

    @property
    def cov(self) -> np.ndarray:
        D = np.diag(self.sigma)
        return D @ self.corr @ D

    @property
    def chol(self) -> np.ndarray:
        return np.linalg.cholesky(self.cov)

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _choose_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)

def _build_cache(cfg: Config, device: torch.device) -> Dict[str, torch.Tensor]:
    # IMPORTANT:
    # - Market simulation first creates correlated standard-normal shocks using
    #   chol_corr and then multiplies assetwise by sigma.
    # - The PINN diffusion trace uses chol_cov, satisfying
    #       chol_cov @ chol_cov.T = cov.
    chol_corr = np.linalg.cholesky(cfg.corr)
    return {
        "mu": torch.tensor(cfg.mu, dtype=torch.float32, device=device),
        "sigma": torch.tensor(cfg.sigma, dtype=torch.float32, device=device),
        "chol": torch.tensor(chol_corr, dtype=torch.float32, device=device),
        "chol_cov": torch.tensor(cfg.chol, dtype=torch.float64, device=device),
        "y0_eval": torch.tensor(cfg.y0_eval, dtype=torch.float32, device=device),
    }

def _make_random_corr_matrix(N: int, bound: float, seed: int) -> np.ndarray:
    """Deterministic B, C_beta=(1-beta)I+beta BB', signed sector exposures.
    beta changes correlation strength without resampling factor loadings.
    """
    beta = float(bound)
    if not (0.0 <= beta < 1.0):
        raise ValueError("factor strength beta must be in [0,1)")
    if N < 2:
        raise ValueError("Two sectors require N >= 2")
    rng = np.random.default_rng(seed)
    B = np.zeros((N, 6), dtype=np.float64)
    B[:, 0] = 0.45
    B[:N//2, 1] = 0.85
    B[N//2:, 1] = -0.85
    B[:, 2:] = 0.18 * rng.standard_normal((N, 4))
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    C = (1.0-beta) * np.eye(N) + beta * (B @ B.T)
    C = 0.5*(C+C.T)
    np.fill_diagonal(C, 1.0)
    if np.linalg.eigvalsh(C)[0] <= 0:
        raise ArithmeticError("Covariance design lost positive definiteness")
    return C

# Audit records are isolated by method; training / label creation does not enter evaluation logs.
_AUDIT = None
_CODE_SHA256 = "notebook_cell_source_unavailable; use delivered file SHA256"

def _audit_inc(key, value=1):
    if _AUDIT is not None:
        _AUDIT["counters"][key] = _AUDIT["counters"].get(key, 0) + int(value)

def _audit_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def _decision_begin(device):
    if _AUDIT is None:
        return None
    _audit_sync(device)
    return time.perf_counter()

def _decision_end(start, batch_size, device, kind="policy_to_net_order", query_count=None):
    if start is None:
        return
    _audit_sync(device)
    elapsed = time.perf_counter()-start
    _AUDIT["decisions"].append({"decision_seconds_per_batch":elapsed,
                                 "batch_size":int(batch_size),
                                 "decision_seconds_amortized":elapsed/max(1,int(batch_size)),
                                 "scope":kind,
                                 "ratio_queries_in_batch":None if query_count is None else int(query_count)})

def _decision_eval(fn, bsz, device, kind="policy_to_net_order"):
    t0=_decision_begin(device)
    out=fn()
    _decision_end(t0,bsz,device,kind)
    return out

def _terminal_exit(x, y, cfg, *, context="market"):
    """Emergency absorbing exit, not triggered by x=0 or an individual y_i=0."""
    liq=_liq_wealth(x,y,cfg.alpha)
    invalid=(~torch.isfinite(liq)) | (liq<=cfg.exit_liq_floor)
    if context=="market" and _AUDIT is not None:
        was_absorbed=(y.abs().sum(-1)==0)&(x<=cfg.exit_liq_floor*(1+1e-5))
        _audit_inc("market_new_exit_events",(invalid & ~was_absorbed).sum().item())
        _audit_inc("market_exit_events",invalid.sum().item())
        _audit_inc("market_path_steps",invalid.numel())
    x=torch.where(invalid,torch.full_like(x,cfg.exit_liq_floor),x)
    y=torch.where(invalid.unsqueeze(-1),torch.zeros_like(y),y)
    return x,y

def _utility_t(x: torch.Tensor, gamma: float) -> torch.Tensor:
    x = torch.clamp(x, min=1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return torch.log(x)
    return (x.pow(1.0 - gamma) - 1.0) / (1.0 - gamma)

def _liq_wealth(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    """Long-only liquidation, including correct one-sided derivative at y_i=0."""
    return x + (1.0-float(alpha))*y.sum(dim=-1)

def _gross_wealth(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y.sum(dim=-1)

def _correlated_dW(z: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
    return z @ chol.transpose(-1, -2)

def _merton_pi(cfg: Config) -> np.ndarray:
    excess = cfg.mu - cfg.r
    try:
        pi = np.linalg.solve(cfg.gamma * cfg.cov, excess)
    except np.linalg.LinAlgError:
        pi = np.linalg.pinv(cfg.gamma * cfg.cov) @ excess
    return pi


def _validate_and_report_portfolio_setup(cfg: Config, outdir_n: str) -> None:
    init, target = cfg.y0_eval, cfg.merton_target_pi
    merton = _merton_pi(cfg)
    x0 = float(cfg.x0_eval)
    if not np.isclose(x0 + init.sum(), 1.0, atol=1e-10, rtol=0):
        raise ValueError("Initial gross wealth must equal one")
    if min(x0, init.min(), target.min(), 1-target.sum()) < -1e-12:
        raise ValueError("Initial and target must belong to long-only/no-borrowing simplex")
    if not np.allclose(merton, target, rtol=1e-8, atol=1e-8):
        raise ValueError("Merton inversion mismatch")
    if cfg.exit_liq_floor <= 0 or cfg.alpha < 0 or cfg.alpha >= 1:
        raise ValueError("Need 0 < emergency floor and alpha in [0,1)")
    if not (0 < cfg.train_gross_risky_max <= 1.0):
        raise ValueError("Long-only training gross risk maximum must lie in (0,1]")
    if cfg.eta < 0 or cfg.u_max <= 0 or cfg.n_steps < 1:
        raise ValueError("Invalid regularization, control bound or time steps")
    eig = np.linalg.eigvalsh(cfg.corr)
    if eig[0] <= 0: raise ValueError("Correlation matrix not SPD")
    delta = target - init
    x_after = x0 - np.maximum(delta,0).sum() + (1-cfg.alpha)*np.maximum(-delta,0).sum()
    print(f"[setup N={cfg.N}] x0={x0:.4f} min_y0={init.min():.5f} "
          f"min_merton={merton.min():.5f} target_cash_after_fee={x_after:.5f} "
          f"corr_condition={eig[-1]/eig[0]:.2f}")
    pd.DataFrame({"asset":np.arange(1,cfg.N+1),"mu":cfg.mu,"sigma":cfg.sigma,
                  "initial_risky_weight":init,"merton_risky_weight":merton,
                  "initial_cash_weight":x0,"merton_cash_weight":1.0-merton.sum()
                  }).to_csv(os.path.join(outdir_n,"portfolio_setup.csv"),index=False)
    np.savetxt(os.path.join(outdir_n,"correlation_matrix.csv"),cfg.corr,delimiter=",")
    np.savetxt(os.path.join(outdir_n,"covariance_matrix.csv"),cfg.cov,delimiter=",")
    manifest={"model":"long-only/no-borrowing closed simplex, buys allowed with sale proceeds",
              "code_sha256":_CODE_SHA256,"python":sys.version,"torch":torch.__version__,
              "device":str(_choose_device(cfg.device)),"config":asdict(cfg),"dt":cfg.dt,
              "state_space":"x>=0; each y_i>=0; W>0; boundary included",
              "terminal_payoff":"U(x+(1-alpha)*sum(y)), emergency absorbing exit only at L<=floor",
              "control":"Euclidean feasible-rate projection with implicit cash-budget derivative",
              "virtual":"fee-free gross-wealth preserving constrained virtual updates; one net fee on execution",
              "PINN":"feasibility-projected Hamiltonian surrogate; NOT certified state-constraint HJB",
              "no_trade":"projected feasible action is zero; interior wedge and blocked boundary tracked separately",
              "limitations":"finite-step cash constraint; nonsmooth active-set changes; no theorem for BPTT derivative at switches"}
    with open(os.path.join(outdir_n,"implementation_manifest.json"),"w",encoding="utf8") as f:
        json.dump(manifest,f,indent=2,ensure_ascii=False,default=str)

class PolicyNet(nn.Module):
    def __init__(self, N: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        # Inputs: (t/T, log W, pi_1, ..., pi_N)
        in_dim = 2 + N
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers.append(nn.Linear(d, N))
        self.net = nn.Sequential(*layers)
        self.N = N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def _features(t_frac: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Non-refinement feature: (t/T, log W, pi_1, ..., pi_N)."""
    W = torch.clamp(_gross_wealth(x, y), min=1e-12)
    logW = torch.log(W)
    pi = y / W.unsqueeze(-1)
    return torch.cat([t_frac.unsqueeze(-1), logW.unsqueeze(-1), pi], dim=-1)

def _sample_init(cfg: Config, bsz: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample the CLOSED nonnegative simplex, including cash=0 / risky_i=0.

    Boundary states are sampled during training rather than just plotted later.
    No dependence on evaluation initial weights or the Merton target.
    """
    W=torch.exp(torch.empty(bsz,device=device).uniform_(cfg.logw_low,cfg.logw_high))
    raw=torch.rand((bsz,cfg.N),device=device).clamp_min(1e-7)
    zero_mask=torch.rand((bsz,cfg.N),device=device)<0.12
    raw=torch.where(zero_mask,torch.zeros_like(raw),raw)
    # If all risky coordinates are zero, the state is the pure-cash corner.
    denom=raw.sum(-1,keepdim=True)
    shares=raw/denom.clamp_min(1e-12)
    risk=torch.rand(bsz,device=device)*float(cfg.train_gross_risky_max)
    face=torch.rand(bsz,device=device)
    risk=torch.where(face<0.18,torch.zeros_like(risk),risk)
    risk=torch.where((face>=0.18)&(face<0.40),torch.ones_like(risk),risk)
    risk=torch.where(denom[:,0]==0,torch.zeros_like(risk),risk)
    y=W.unsqueeze(-1)*risk.unsqueeze(-1)*shares
    x=W-y.sum(-1)
    # Cash boundary is EXACT, never replace individual zero holdings by eps.
    x=torch.where(risk>=1.0,torch.zeros_like(x),x.clamp_min(0.0))
    if not bool((_liq_wealth(x,y,cfg.alpha)>cfg.exit_liq_floor).all()):
        raise RuntimeError("Simplex sampler violated positive liquidation")
    return x,y


def _cash_after_rate(x: torch.Tensor,y: torch.Tensor,u: torch.Tensor,cfg:Config):
    """Cash after the finite-duration rate control, including eta cash leakage."""
    L=_liq_wealth(x,y,cfg.alpha)
    A=L*cfg.dt
    return x-A*(torch.relu(u).sum(-1)
                -(1-cfg.alpha)*torch.relu(-u).sum(-1)
                +0.5*cfg.eta*u.square().sum(-1))


def _is_feasible(x: torch.Tensor,y: torch.Tensor,u: torch.Tensor,cfg:Config)->torch.Tensor:
    L=_liq_wealth(x,y,cfg.alpha)
    x2=_cash_after_rate(x,y,u,cfg)
    y2=y+(L*cfg.dt).unsqueeze(-1)*u
    tol=1e-7*(1+_gross_wealth(x,y).abs())
    return (torch.isfinite(x2)&torch.isfinite(y2).all(-1)
            &(x2>=-tol)&(y2>=-tol.unsqueeze(-1)).all(-1)
            &(_liq_wealth(x2,y2,cfg.alpha)>cfg.exit_liq_floor))


def _project_u(u_raw: torch.Tensor,x: torch.Tensor,y: torch.Tensor,cfg:Config)->torch.Tensor:
    """Project normalized rate to long-only AND cash-nonnegative feasible set.

    The feasible set is convex: -y_i/(L dt)<=u_i<=u_max and
    x-Ldt[sum(u_+)-(1-alpha)sum(u_-)+eta/2*||u||^2]>=0.
    Exact Euclidean projection is solved via one nonnegative scalar multiplier.
    It shifts ALL rates, hence cash=0 still permits sell-to-buy rotation even
    if the raw policy requested positive rates only.

    Bisection finds lambda numerically without a gradient through Boolean tests.
    Its continuous active-set Jacobian is supplied by implicit differentiation
    of the scalar cash constraint (except at genuine nonsmooth face switches).
    """
    finite=torch.isfinite(u_raw).all(-1)
    if _AUDIT is not None:_audit_inc("raw_action_fallback_paths",(~finite).sum().item())
    raw=torch.where(finite.unsqueeze(-1),u_raw,torch.zeros_like(u_raw))
    W=_gross_wealth(x,y)
    L=_liq_wealth(x,y,cfg.alpha)
    A=(L*cfg.dt).clamp_min(1e-12)
    lower=-torch.minimum(torch.full_like(y,float(cfg.u_max)),y.clamp_min(0)/A.unsqueeze(-1))
    upper=torch.full_like(y,float(cfg.u_max))
    dead=(y.abs().sum(-1)==0)&(x<=cfg.exit_liq_floor*(1+1e-5))

    def prox(lam):
        lam=lam.unsqueeze(-1)
        denom=1.0+lam*A.unsqueeze(-1)*cfg.eta
        buy=(raw-lam*A.unsqueeze(-1))/denom
        sell=(raw-lam*A.unsqueeze(-1)*(1-cfg.alpha))/denom
        buy=torch.minimum(torch.relu(buy),upper)
        sell=torch.maximum(torch.minimum(sell,torch.zeros_like(sell)),lower)
        return torch.where(raw>lam*A.unsqueeze(-1),buy,
               torch.where(raw<lam*A.unsqueeze(-1)*(1-cfg.alpha),sell,torch.zeros_like(raw)))

    def budget(v):
        return x-A*(torch.relu(v).sum(-1)
              -(1-cfg.alpha)*torch.relu(-v).sum(-1)
              +0.5*cfg.eta*v.square().sum(-1))

    u0=prox(torch.zeros_like(x))
    cash0=budget(u0)
    tol=1e-7*(1+W.abs())
    binding=(cash0 < -tol)&(~dead)&finite
    if not bool(binding.any()):
        u=torch.where(dead.unsqueeze(-1),torch.zeros_like(u0),u0)
        if _AUDIT is not None:
            _audit_inc("long_only_lower_bound_binding",(u<=lower+1e-8).sum().item())
            _audit_inc("feasibility_scaled_paths",(u0!=raw).any(-1).sum().item())
        return u
    with torch.no_grad():
        lo=torch.zeros_like(x)
        hi=torch.ones_like(x)
        # Bracket lambda, only needed for cash-binding rows.
        for _ in range(44):
            not_bracketed=binding&(budget(prox(hi)) < 0)
            hi=torch.where(not_bracketed,hi*2,hi)
            if not bool(not_bracketed.any()):break
        if bool((binding&(budget(prox(hi)) < -tol)).any()):
            raise RuntimeError("Cash-budget dual projection failed to bracket its root")
        for _ in range(42):
            mid=(lo+hi)*0.5
            below=(budget(prox(mid)) < 0)&binding
            lo=torch.where(below,mid,lo)
            hi=torch.where(binding&(~below),mid,hi)
        lam=torch.where(binding,hi,torch.zeros_like(x)).detach()

    # Implicit derivative of cash(lambda, raw, x,y)=0 for binding rows.
    u_at=prox(lam)
    g=budget(u_at)
    active=u_at.abs()>1e-9
    free=(active&(u_at>lower+1e-8)&(u_at<upper-1e-8))
    c=torch.where(u_at>0,torch.ones_like(u_at),torch.full_like(u_at,1-cfg.alpha))
    slope=c+cfg.eta*u_at
    denom=1+lam.unsqueeze(-1)*A.unsqueeze(-1)*cfg.eta
    g_lambda=(A.square().unsqueeze(-1)*slope.square()/denom*free).sum(-1)
    differentiable=binding&(g_lambda>1e-12)
    lam_implicit=torch.where(differentiable,
        lam-(g-g.detach())/g_lambda.detach().clamp_min(1e-12),lam)
    u=prox(lam_implicit)
    u=torch.where(dead.unsqueeze(-1),torch.zeros_like(u),u)
    cash=budget(u)
    if bool(((cash<-5e-5*(1+W.abs()))|(~torch.isfinite(cash))).any()):
        raise RuntimeError("Projected rate violates cash budget")
    if bool((y+A.unsqueeze(-1)*u < -5e-5*(1+W.abs()).unsqueeze(-1)).any()):
        raise RuntimeError("Projected rate creates a short position")
    if _AUDIT is not None:
        _audit_inc("cash_budget_binding",binding.sum().item())
        _audit_inc("long_only_lower_bound_binding",(u <= lower+1e-8).sum().item())
        _audit_inc("feasibility_scaled_paths",(binding | (u0!=raw).any(-1)).sum().item())
    return u

def _u_theta_action(policy: PolicyNet,
                   t_frac: torch.Tensor,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   cfg: Config) -> torch.Tensor:
    inp = _features(t_frac, x, y)
    raw = policy(inp.reshape(-1, inp.shape[-1])).reshape(*inp.shape[:-1], cfg.N)
    u = cfg.u_max * torch.tanh(raw)
    if _AUDIT is not None: _audit_inc("raw_policy_query_states",x.numel())
    return _project_u(u, x, y, cfg)

def _step(x: torch.Tensor,y: torch.Tensor,u: torch.Tensor,dW: torch.Tensor,
          cfg:Config,cache:Dict[str,torch.Tensor])->Tuple[torch.Tensor,torch.Tensor]:
    """Fee/regularization paid once; no x/y 0-boundary absorption."""
    bankrupt_before=(y.abs().sum(-1)==0)&(x<=cfg.exit_liq_floor*(1+1e-5))
    L=_liq_wealth(x,y,cfg.alpha)
    u=_project_u(u,x,y,cfg)
    x2=_cash_after_rate(x,y,u,cfg)
    y2=y+(L*cfg.dt).unsqueeze(-1)*u
    tol=5e-5*(1+_gross_wealth(x,y).abs())
    if bool(((x2 < -tol)|(y2 < -tol.unsqueeze(-1)).any(-1)).any()):
        raise RuntimeError("Step escaped long-only/cash constraints")
    x2=x2.clamp_min(0.0)  # only roundoff, not a wealth/default floor
    y2=y2.clamp_min(0.0)
    x2=x2*math.exp(cfg.r*cfg.dt)
    mu=cache["mu"].to(dtype=y.dtype)
    sig=cache["sigma"].to(dtype=y.dtype)
    expo=(mu-.5*sig.square())*cfg.dt+sig*math.sqrt(cfg.dt)*dW
    y2=y2*torch.exp(expo)
    x2,y2=_terminal_exit(x2,y2,cfg,context="market")
    return (torch.where(bankrupt_before,torch.full_like(x2,cfg.exit_liq_floor),x2),
            torch.where(bankrupt_before.unsqueeze(-1),torch.zeros_like(y2),y2))

def _build_fwd_bank(start_step: int, cfg: Config, device: torch.device, bsz: int,
                    cache: Dict[str, torch.Tensor], seed_extra: int = 0) -> Optional[torch.Tensor]:
    if not cfg.use_common_random_numbers:
        return None
    remaining = cfg.n_steps - start_step
    if remaining <= 0:
        return None

    gen = torch.Generator(device="cpu")
    gen.manual_seed(cfg.seed + 2000 * (start_step + 1) + 17 + seed_extra)
    z = torch.randn((remaining, bsz, cfg.N), generator=gen, device="cpu").to(device)
    return _correlated_dW(z, cache["chol"].to(dtype=z.dtype))

def _build_inner_bank(start_step: int, cfg: Config, device: torch.device,
                      m: int, cache: Dict[str, torch.Tensor],
                      seed_extra: int = 0) -> Optional[torch.Tensor]:
    if not cfg.use_common_random_numbers:
        return None
    remaining = cfg.n_steps - start_step
    if remaining <= 0:
        return None

    use_anti = cfg.use_antithetic
    half = int(math.ceil(m / 2)) if use_anti else m
    total = 2 * half if use_anti else m

    gen = torch.Generator(device="cpu")
    gen.manual_seed(cfg.seed + 1000 * (start_step + 1) + int(seed_extra))
    z = torch.randn((remaining, half, cfg.N), generator=gen, device="cpu").to(device)
    base = _correlated_dW(z, cache["chol"].to(dtype=z.dtype))
    if use_anti:
        noise = torch.cat([base, -base], dim=1)[:, :total, :]
    else:
        noise = base
    return noise

def _rollout(policy: PolicyNet, x0: torch.Tensor, y0: torch.Tensor,
             start_step: int, noise_bank: Optional[torch.Tensor],
             cfg: Config, cache: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = x0, y0
    device = x.device

    for j in range(cfg.n_steps - start_step):
        step = start_step + j
        t_frac = torch.full(x.shape, float(step) / cfg.n_steps, device=device, dtype=x.dtype)
        u = _u_theta_action(policy, t_frac, x, y, cfg)

        if noise_bank is None:
            z = torch.randn((*x.shape, cfg.N), device=device, dtype=y.dtype)
            dW = _correlated_dW(z, cache["chol"].to(dtype=z.dtype))
        else:
            dW = noise_bank[j]

        x, y = _step(x, y, u, dW, cfg, cache)

    return x, y

def train_u_theta(cfg:Config,device:torch.device,cache:Dict[str,torch.Tensor])->PolicyNet:
    policy=PolicyNet(N=cfg.N,hidden=cfg.hidden,depth=cfg.depth).to(device)
    opt=optim.Adam(policy.parameters(),lr=cfg.lr)
    history=[]
    for step in range(cfg.n_train_steps+1):
        x0,y0=_sample_init(cfg,cfg.batch_size,device)
        xT,yT=_rollout(policy,x0,y0,0,None,cfg,cache)
        obj=_utility_t(_liq_wealth(xT,yT,cfg.alpha),cfg.gamma).mean()
        if not bool(torch.isfinite(obj)):
            raise RuntimeError(f"Nonfinite training objective at step {step}")
        opt.zero_grad(set_to_none=True)
        (-obj).backward()
        grad_norm=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in policy.parameters()
                                if p.grad is not None))
        opt.step()
        if step%cfg.print_every==0 or step==cfg.n_train_steps:
            with torch.no_grad():
                at0=torch.zeros_like(x0)
                u0=_u_theta_action(policy,at0,x0,y0,cfg)
                record={"step":step,"objective":obj.item(),"grad_norm":grad_norm,
                        "initial_cash_boundary_ratio":(x0==0).float().mean().item(),
                        "initial_risky_face_ratio":(y0==0).float().mean().item(),
                        "active_rate_ratio":(u0.abs()>cfg.no_trade_rate_tol).float().mean().item(),
                        "mean_abs_rate":u0.abs().mean().item(),
                        "terminal_min_liq":_liq_wealth(xT,yT,cfg.alpha).min().item()}
                history.append(record)
                print(f"  [u_theta | N={cfg.N}] step={step:5d} E[U]={obj.item():.6f} "
                      f"grad={grad_norm:.3g} active={record['active_rate_ratio']:.3f} "
                      f"cash_face={record['initial_cash_boundary_ratio']:.3f}")
    policy.training_history=pd.DataFrame(history)
    return policy

class PINNValueNet(nn.Module):
    """
    Homogeneity-enforced CRRA value network on the closed long-only simplex.

    Let W = x + sum_i y_i and pi_i = y_i / W.  For gamma != 1,

        V(t,x,y)
          = [ W^(1-gamma) q_theta(t,pi) - 1 ] / (1-gamma),

    where q_theta is positive and exactly satisfies the terminal condition

        q(T,pi) = [1 - alpha * sum_i pi_i]^(1-gamma), pi_i>=0.

    The multiplicative exponential residual guarantees q_theta > 0 and the
    factor (1-t/T) imposes the terminal condition exactly.
    """
    def __init__(
        self,
        N: int,
        gamma: float,
        alpha: float,
        hidden: int = 128,
        depth: int = 3,
        residual_scale: float = 2.0,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        # Residual network inputs: (t/T, log W, pi_1, ..., pi_N)
        d = N + 2
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.N = N
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.residual_scale = float(residual_scale)

        # Start from the exact terminal utility extended backward in time.
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        t_frac: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        W = torch.clamp(_gross_wealth(x, y), min=1e-10)
        logW = torch.log(W)
        pi = y / W.unsqueeze(-1)
        inp = torch.cat([t_frac.unsqueeze(-1), logW.unsqueeze(-1), pi], dim=-1)
        raw = self.net(inp).squeeze(-1)

        liq_factor = torch.clamp(
            1.0 - self.alpha * pi.sum(dim=-1),
            min=1e-8,
        )
        time_to_go = torch.clamp(1.0 - t_frac, min=0.0, max=1.0)
        log_multiplier = (
            time_to_go
            * self.residual_scale
            * torch.tanh(raw)
        )

        if abs(self.gamma - 1.0) < 1e-12:
            # Log utility:
            # V(T)=log(W*liq_factor), with an additive homogeneous correction.
            return (
                torch.log(W)
                + torch.log(liq_factor)
                + log_multiplier
            )

        terminal_q = liq_factor.pow(1.0 - self.gamma)
        q = terminal_q * torch.exp(log_multiplier)
        return (
            W.pow(1.0 - self.gamma) * q - 1.0
        ) / (1.0 - self.gamma)

def _pinn_dtype(cfg: Config) -> torch.dtype:
    return torch.float64 if cfg.pinn_use_float64 else torch.float32

def _sample_pinn_states(
    cfg: Config,
    bsz: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample interior collocation points from the same wealth/fraction band used
    for neural policy initialization.
    """
    x, y = _sample_init(cfg, bsz, device)
    t_frac = torch.rand(bsz, device=device, dtype=dtype)

    x = x.to(dtype=dtype).detach().requires_grad_(True)
    y = y.to(dtype=dtype).detach().requires_grad_(True)
    t_frac = t_frac.detach().requires_grad_(True)
    return t_frac, x, y

def _hutchinson_diffusion_trace(
    V_y: torch.Tensor,
    y: torch.Tensor,
    chol_cov: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """
    Estimate

        Tr[ diag(y) Cov diag(y) Hess_yy V ]

    using Rademacher Hessian-vector products.

    If A A^T = Cov and eta has identity covariance, then
        v = diag(y) A eta
    satisfies E[v v^T] = diag(y) Cov diag(y), and
        E[v^T Hess(V) v]
    is the required trace.
    """
    estimates = []
    chol_cov = chol_cov.to(device=y.device, dtype=y.dtype)

    for _ in range(max(1, int(n_samples))):
        eta = (
            torch.empty_like(y)
            .bernoulli_(0.5)
            .mul_(2.0)
            .sub_(1.0)
        )
        correlated = eta @ chol_cov.transpose(-1, -2)
        # Detach v: the Hessian-vector product treats the direction as fixed.
        v = (y * correlated).detach()

        directional_first = (V_y * v).sum()
        Hv = torch.autograd.grad(
            directional_first,
            y,
            create_graph=True,
            retain_graph=True,
        )[0]
        estimates.append((Hv * v).sum(dim=-1))

    return torch.stack(estimates, dim=0).mean(dim=0)

def _pinn_hjb_terms(
    value_net: PINNValueNet,
    t_frac: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
    cache: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Compute the regularized HJB residual

        V_t + r x V_x + sum_i mu_i y_i V_{y_i}
        + 1/2 Tr[D_y Cov D_y Hess_yy V]
        + L/(2 eta V_x) sum_i [
              (V_{y_i}-V_x)_+^2
            + ((1-alpha)V_x-V_{y_i})_+^2
          ] = 0.
    """
    if cfg.eta <= 0.0:
        raise ValueError("PINN HJB training requires eta > 0.")

    V = value_net(t_frac, x, y)
    V_tfrac, V_x, V_y = torch.autograd.grad(
        V.sum(),
        [t_frac, x, y],
        create_graph=True,
        retain_graph=True,
    )
    V_t = V_tfrac / cfg.T

    diffusion_trace = _hutchinson_diffusion_trace(
        V_y=V_y,
        y=y,
        chol_cov=cache["chol_cov"],
        n_samples=cfg.pinn_hutchinson_samples,
    )

    mu_t = cache["mu"].to(device=y.device, dtype=y.dtype)
    drift_term = (
        cfg.r * x * V_x
        + (mu_t.unsqueeze(0) * y * V_y).sum(dim=-1)
    )

    L = _liq_wealth(x, y, cfg.alpha)
    Vx_col = V_x.unsqueeze(-1)
    buy_gap = V_y - Vx_col
    sell_gap = (1.0 - cfg.alpha) * Vx_col - V_y

    # Discrete-time feasibility-projected Hamiltonian surrogate. It agrees
    # with the interior HJB for unconstrained-optimal admissible rates, but is
    # not claimed to solve the exact state-constraint viscosity problem.
    ratio = V_y / torch.clamp(V_x.unsqueeze(-1),min=cfg.pinn_vx_floor)
    candidate = _P_eta_raw(ratio,cfg.eta,cfg.alpha)
    u_feasible = _project_u(candidate,x,y,cfg)
    regularized_gain = L*(
        (torch.relu(u_feasible)*buy_gap).sum(-1)
        +(torch.relu(-u_feasible)*sell_gap).sum(-1)
        -0.5*cfg.eta*V_x*u_feasible.square().sum(-1)
    )

    residual = (
        V_t
        + drift_term
        + 0.5 * diffusion_trace
        + regularized_gain
    )

    return {
        "V": V,
        "V_t": V_t,
        "V_x": V_x,
        "V_y": V_y,
        "ratio": ratio,
        "buy_gap": buy_gap,
        "sell_gap": sell_gap,
        "diffusion_trace": diffusion_trace,
        "drift_term": drift_term,
        "regularized_gain": regularized_gain,
        "residual": residual,
    }

def train_pinn(
    cfg: Config,
    device: torch.device,
    cache: Dict[str, torch.Tensor],
) -> Tuple[PINNValueNet, pd.DataFrame]:
    """
    Train a scalable PINN for the regularized N-asset HJB.

    The terminal condition is hard-wired into the architecture.  The objective
    combines a normalized HJB residual with a penalty enforcing V_x > 0.
    """
    dtype = _pinn_dtype(cfg)
    value_net = PINNValueNet(
        N=cfg.N,
        gamma=cfg.gamma,
        alpha=cfg.alpha,
        hidden=cfg.pinn_hidden,
        depth=cfg.pinn_depth,
        residual_scale=cfg.pinn_residual_scale,
    ).to(device=device, dtype=dtype)

    opt = optim.Adam(value_net.parameters(), lr=cfg.pinn_lr)
    rows = []

    for step in range(cfg.pinn_train_steps + 1):
        t_frac, x, y = _sample_pinn_states(
            cfg=cfg,
            bsz=cfg.pinn_batch_size,
            device=device,
            dtype=dtype,
        )

        terms = _pinn_hjb_terms(
            value_net=value_net,
            t_frac=t_frac,
            x=x,
            y=y,
            cfg=cfg,
            cache=cache,
        )

        # Normalize pathwise to avoid a few large-utility states dominating.
        residual_scale = (
            1.0
            + terms["V"].detach().abs()
            + terms["drift_term"].detach().abs()
            + 0.5 * terms["diffusion_trace"].detach().abs()
            + terms["regularized_gain"].detach().abs()
        )
        normalized_residual = terms["residual"] / residual_scale

        loss_pde = normalized_residual.square().mean()
        loss_mono = torch.relu(
            cfg.pinn_vx_floor - terms["V_x"]
        ).square().mean()
        loss = (
            loss_pde
            + cfg.pinn_monotonicity_weight * loss_mono
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.pinn_grad_clip is not None and cfg.pinn_grad_clip > 0:
            nn.utils.clip_grad_norm_(
                value_net.parameters(),
                max_norm=cfg.pinn_grad_clip,
            )
        opt.step()

        if (
            step % cfg.pinn_print_every == 0
            or step == cfg.pinn_train_steps
        ):
            with torch.no_grad():
                row = {
                    "N": cfg.N,
                    "step": step,
                    "loss": float(loss.item()),
                    "loss_pde": float(loss_pde.item()),
                    "loss_monotonicity": float(loss_mono.item()),
                    "mean_abs_raw_residual": float(
                        terms["residual"].detach().abs().mean().item()
                    ),
                    "max_abs_raw_residual": float(
                        terms["residual"].detach().abs().max().item()
                    ),
                    "mean_Vx": float(
                        terms["V_x"].detach().mean().item()
                    ),
                    "min_Vx": float(
                        terms["V_x"].detach().min().item()
                    ),
                    "Vx_nonpositive_ratio": float(
                        (terms["V_x"].detach() <= 0.0)
                        .double()
                        .mean()
                        .item()
                    ),
                }
                rows.append(row)
                print(
                    f"  [PINN | N={cfg.N}] step={step:5d}  "
                    f"loss={row['loss']:.4e}  "
                    f"|HJB|={row['mean_abs_raw_residual']:.4e}  "
                    f"min(Vx)={row['min_Vx']:.4e}"
                )

    return value_net, pd.DataFrame(rows)

def _summary_stats(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().double().cpu()
    n = int(values.numel())

    mean = float(values.mean().item())
    std = float(values.std(unbiased=True).item()) if n > 1 else 0.0
    se = std / math.sqrt(max(n, 1))

    return {
        "mean": mean,
        "std": std,
        "se": se,
        "n_paths": n,
    }

def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def _timed_call(fn, device: torch.device):
    _sync_device(device)
    t0 = time.perf_counter()
    out = fn()
    _sync_device(device)
    return out, time.perf_counter() - t0

def _count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))

def _build_pathwise_value_df(
    method: str,
    N: int,
    liquidation: torch.Tensor,
    utility: torch.Tensor,
) -> pd.DataFrame:
    """
    One row per outer Monte-Carlo path.

    path_value is the realized terminal CRRA utility U(L_T) on that path.
    The reported objective_mean is exactly path_value.mean().
    """
    liquidation = liquidation.detach().cpu().reshape(-1).double()
    utility = utility.detach().cpu().reshape(-1).double()
    if liquidation.numel() != utility.numel():
        raise RuntimeError(
            f"Pathwise size mismatch: liquidation={liquidation.numel()} "
            f"utility={utility.numel()}"
        )

    n = int(utility.numel())
    return pd.DataFrame({
        "N": np.full(n, int(N), dtype=np.int64),
        "method": np.full(n, str(method), dtype=object),
        "path_id": np.arange(n, dtype=np.int64),
        "terminal_liq": liquidation.numpy(),
        "terminal_utility": utility.numpy(),
        "path_value": utility.numpy(),
    })


# Rhat acquisition and structure-preserving P_eta map
# Rhat_BPTT is differentiated directly through the frozen
# Stage-1 u_{theta,eta} feedback policy.

def _P_eta_raw(R_hat: torch.Tensor, eta: float, alpha: float) -> torch.Tensor:
    """Coordinatewise Yosida/dead-zone map P_eta(R_hat)."""
    if eta <= 0.0:
        raise ValueError("eta must be strictly positive in P_eta.")
    return (
        torch.relu(R_hat - 1.0) / float(eta)
        - torch.relu((1.0 - float(alpha)) - R_hat) / float(eta)
    )


def _project_P_eta_for_outer_execution(
    R_hat: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    """P_eta(Rhat), feasibility-projected for eta=0 outer execution."""
    u_raw = _P_eta_raw(R_hat, cfg.eta, cfg.alpha)
    return _project_u(u_raw, x, y, replace(cfg, eta=0.0))


def _Rhat_BPTT(
    policy: PolicyNet,
    x: torch.Tensor,
    y: torch.Tensor,
    t_value: float,
    cfg: Config,
    device: torch.device,
    cache: Dict[str, torch.Tensor],
    inner_bank: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimate local continuation marginal values by feedback BPTT and return

        Rhat_BPTT = V_y_hat / V_x_hat.

    The policy parameters are fixed during this query, but the full state-to-action
    feedback Jacobian remains in the differentiated rollout.
    """
    start_step = int(round(t_value / cfg.dt))
    start_step = max(0, min(cfg.n_steps - 1, start_step))
    bsz = x.shape[0]

    mc = int(cfg.inner_mc_paths)
    use_anti = bool(cfg.use_antithetic)
    half = int(math.ceil(mc / 2)) if use_anti else mc
    total = 2 * half if use_anti else mc

    x_rep = x.unsqueeze(0).expand(total, bsz).clone().detach().requires_grad_(True)
    y_rep = y.unsqueeze(0).expand(total, bsz, cfg.N).clone().detach().requires_grad_(True)
    remaining = cfg.n_steps - start_step

    if inner_bank is None:
        if remaining > 0:
            z = torch.randn((remaining, half, cfg.N), device=device, dtype=y.dtype)
            base = _correlated_dW(z, cache["chol"].to(dtype=z.dtype))
            noise = torch.cat([base, -base], dim=1)[:, :total, :] if use_anti else base
            noise = noise[:, :, None, :].expand(remaining, total, bsz, cfg.N)
        else:
            noise = None
    else:
        noise = (
            inner_bank[:, :, None, :].expand(remaining, total, bsz, cfg.N)
            if remaining > 0 else None
        )

    xT, yT = _rollout(
        policy, x_rep, y_rep, start_step, noise,
        cfg, cache,
    )
    payoff = _utility_t(_liq_wealth(xT, yT, cfg.alpha), cfg.gamma)
    gx, gy = torch.autograd.grad(payoff.sum(), [x_rep, y_rep], create_graph=False)

    Vx_hat = gx.mean(dim=0)
    Vy_hat = gy.mean(dim=0)
    good=(torch.isfinite(Vx_hat)&(Vx_hat>cfg.marginal_floor)&torch.isfinite(Vy_hat).all(-1))
    Vx_safe=torch.where(good,Vx_hat,torch.ones_like(Vx_hat))
    raw=Vy_hat/Vx_safe.unsqueeze(-1)
    good=good&torch.isfinite(raw).all(-1)
    R_hat=torch.where(good.unsqueeze(-1),raw,torch.full_like(raw,1-cfg.alpha/2))
    _audit_inc("BPTT_ratio_query_states",bsz)
    _audit_inc("BPTT_ratio_batch_calls",1)
    if _AUDIT is not None: _audit_inc("BPTT_ratio_query_failed_states",(~good).sum().item())
    return R_hat.detach(),Vx_hat.detach(),Vy_hat.detach()


def _P_eta_Rhat_BPTT_action(
    policy: PolicyNet,
    x: torch.Tensor,
    y: torch.Tensor,
    t_value: float,
    cfg: Config,
    device: torch.device,
    cache: Dict[str, torch.Tensor],
    inner_bank: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    R_hat, Vx_hat, Vy_hat = _Rhat_BPTT(
        policy, x, y, t_value, cfg, device, cache,
        inner_bank=inner_bank,
    )
    valid=torch.isfinite(Vx_hat)&(Vx_hat>cfg.marginal_floor)&torch.isfinite(Vy_hat).all(-1)
    u=_project_P_eta_for_outer_execution(R_hat,x,y,cfg)
    valid=valid&torch.isfinite(Vy_hat/Vx_hat.unsqueeze(-1)).all(-1)
    u=torch.where(valid.unsqueeze(-1),u,torch.zeros_like(u))
    return u,{"R_hat":R_hat,"Vx_hat":Vx_hat,"Vy_hat":Vy_hat,"ratio_valid":valid}


def _Rhat_PINN(
    value_net: PINNValueNet,
    t_frac: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = next(value_net.parameters()).dtype
    with torch.enable_grad():
        x_req = x.detach().to(dtype=dtype).requires_grad_(True)
        y_req = y.detach().to(dtype=dtype).requires_grad_(True)
        t_req = t_frac.detach().to(dtype=dtype)
        V = value_net(t_req, x_req, y_req)
        Vx, Vy = torch.autograd.grad(
            V.sum(), [x_req, y_req], create_graph=False, retain_graph=False
        )
        good=torch.isfinite(Vx)&(Vx>cfg.pinn_vx_floor)&torch.isfinite(Vy).all(-1)
        raw=Vy/torch.where(good,Vx,torch.ones_like(Vx)).unsqueeze(-1)
        good=good&torch.isfinite(raw).all(-1)
        R_hat=torch.where(good.unsqueeze(-1),raw,torch.full_like(raw,1-cfg.alpha/2))
        _audit_inc("PINN_ratio_query_states",x.numel())
        _audit_inc("PINN_ratio_batch_calls",1)
        if _AUDIT is not None: _audit_inc("PINN_ratio_query_failed_states",(~good).sum().item())
    return (
        R_hat.detach().to(device=x.device, dtype=y.dtype),
        Vx.detach().to(device=x.device, dtype=x.dtype),
        Vy.detach().to(device=x.device, dtype=y.dtype),
    )


def _P_eta_Rhat_PINN_action(
    value_net: PINNValueNet,
    t_frac: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    R_hat, Vx_hat, Vy_hat = _Rhat_PINN(value_net, t_frac, x, y, cfg)
    valid=torch.isfinite(Vx_hat)&(Vx_hat>cfg.pinn_vx_floor)&torch.isfinite(Vy_hat).all(-1)
    u=_project_P_eta_for_outer_execution(R_hat,x,y,cfg)
    valid=valid&torch.isfinite(Vy_hat/Vx_hat.unsqueeze(-1)).all(-1)
    u=torch.where(valid.unsqueeze(-1),u,torch.zeros_like(u))
    return u,{"R_hat":R_hat,"Vx_hat":Vx_hat,"Vy_hat":Vy_hat,"ratio_valid":valid}


# Trajectory-only BPTT ratio distillation (separate teacher and test paths)

class RatioNet(nn.Module):
    """Supervise marginal-value ratios, NOT projected controls or P_eta actions."""
    def __init__(self, N: int, eta: float, alpha: float, hidden: int, depth: int):
        super().__init__()
        layers = []
        dim = N + 2  # (t/T, log gross wealth, risky weights)
        for _ in range(depth):
            layers.extend([nn.Linear(dim, hidden), nn.Tanh()])
            dim = hidden
        layers.append(nn.Linear(dim, N))
        self.net = nn.Sequential(*layers)
        self.eta = float(eta)
        self.midpoint = 1.0 - float(alpha) / 2.0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Scale the output and the regression loss in policy-relevant eta units.
        return self.midpoint + self.eta * self.net(features)


class BPTTRatioCollector:
    """Archive every teacher query; training uses only active, valid observations."""
    def __init__(self, cfg: Config, method: str, seed_extra: int):
        self.cfg = cfg
        self.method = method
        self.seed_extra = int(seed_extra)
        self.blocks = []
        self.step = None
        self.chunk_start = None

    def set_context(self, step: int, chunk_start: int):
        self.step = int(step)
        self.chunk_start = int(chunk_start)

    def observe(self, x, y, aux, iteration: int, is_endpoint: bool, active):
        if self.step is None or self.chunk_start is None:
            raise RuntimeError("BPTT collector requires step and chunk context")
        size = int(x.shape[0])
        def cpu(t): return t.detach().cpu().numpy().copy()
        self.blocks.append({
            "x": cpu(x), "y": cpu(y), "R_hat": cpu(aux["R_hat"]),
            "Vx_hat": cpu(aux["Vx_hat"]), "Vy_hat": cpu(aux["Vy_hat"]),
            "ratio_valid": cpu(aux["ratio_valid"]).astype(np.bool_),
            "active": cpu(active).astype(np.bool_),
            "path_id": np.arange(self.chunk_start, self.chunk_start + size, dtype=np.int64),
            "step": np.full(size, self.step, dtype=np.int32),
            "map_iteration": np.full(size, int(iteration), dtype=np.int32),
            "is_endpoint": np.full(size, bool(is_endpoint), dtype=np.bool_),
        })

    def arrays(self):
        if not self.blocks:
            raise RuntimeError(f"No BPTT labels were collected for {self.method}")
        names = self.blocks[0].keys()
        return {key: np.concatenate([b[key] for b in self.blocks], axis=0)
                for key in names}

    def save(self, filename: str, arrays: dict):
        # NPZ holds ALL calls, valid or invalid; records x/y and Vx/Vy as well as R.
        np.savez_compressed(filename, **arrays)
        meta = {
            "teacher_method": self.method,
            "source": "frozen_u_theta_eta_feedback_BPTT",
            "split": "teacher_rollout_only; not held_out_objective",
            "teacher_seed_extra": self.seed_extra,
            "teacher_inner_seed_extra": self.seed_extra,
            "heldout_seed_extra": 300_000,
            "heldout_inner_seed_extra": 0,
            "n_teacher_outer_paths": int(self.cfg.distill_teacher_paths),
            "n_ratio_queries_all": int(len(arrays["step"])),
            "n_ratio_queries_valid": int(arrays["ratio_valid"].sum()),
            "n_ratio_queries_active_valid": int((arrays["ratio_valid"] & arrays["active"]).sum()),
            "endpoint_queries": int(arrays["is_endpoint"].sum()),
            "fields": list(arrays),
            "note": "All BPTT calls archived including inactive and failed queries; invalid targets excluded from SGD.",
        }
        with open(filename.replace(".npz", "_manifest.json"), "w", encoding="utf8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
        return meta


def train_distilled_ratio(cfg: Config, device: torch.device, arrays: dict, seed: int):
    """Train/validate split by outer path ID; no held-out objective or labels enter SGD."""
    if cfg.distill_teacher_paths < 5:
        raise ValueError("distill_teacher_paths must be >=5 for path-disjoint validation")
    valid = arrays["ratio_valid"] & np.isfinite(arrays["R_hat"]).all(axis=1)
    # Intermediate queries on already converged/stalled rows were still archived,
    # but duplicates must not dominate the learning distribution.
    selected = valid & (arrays["active"] | arrays["is_endpoint"])
    paths = arrays["path_id"]
    # Identical initial portfolios appear on every path. Exclude t=0 from
    # validation when later, path-specific states exist; otherwise validation
    # RMSE would be spuriously optimistic due to identical states across splits.
    validation = selected & (paths % 5 == 0)
    if cfg.n_steps > 1:
        later = validation & (arrays["step"] > 0)
        if later.any():
            validation = later
    training = selected & (paths % 5 != 0)
    if training.sum() == 0 or validation.sum() == 0:
        raise RuntimeError("Insufficient disjoint valid teacher labels for distillation")

    _set_seed(seed)
    net = RatioNet(cfg.N, cfg.eta, cfg.alpha, cfg.distill_hidden, cfg.distill_depth).to(device)
    opt = optim.Adam(net.parameters(), lr=cfg.distill_lr)
    # Materialize only SELECTED labels on the accelerator. Archival includes
    # inactive/failed observations but does not occupy additional GPU memory.
    def selected_features(mask):
        x_selected = torch.from_numpy(arrays["x"][mask]).float()
        y_selected = torch.from_numpy(arrays["y"][mask]).float()
        time_selected = torch.from_numpy(arrays["step"][mask].astype(np.float32)) / cfg.n_steps
        return _features(time_selected, x_selected, y_selected).detach().to(device)

    train_features = selected_features(training)
    val_features = selected_features(validation)
    midpoint = 1.0 - cfg.alpha / 2.0
    train_target_scaled = (torch.from_numpy(arrays["R_hat"][training]).float().to(device)
                           - midpoint) / cfg.eta
    val_target_scaled = (torch.from_numpy(arrays["R_hat"][validation]).float().to(device)
                         - midpoint) / cfg.eta
    n_train = int(training.sum())
    n_val = int(validation.sum())

    history = []
    best_loss = float("inf")
    best_state = None
    best_step = -1
    stale = 0
    eval_every = max(1, int(cfg.distill_eval_every))
    for step in range(int(cfg.distill_steps) + 1):
        net.train()
        indices = torch.randint(n_train, (cfg.distill_batch_size,), device=device)
        predicted_scaled = (net(train_features[indices]) - midpoint) / cfg.eta
        loss = (predicted_scaled - train_target_scaled[indices]).square().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite ratio-distillation loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        if step % eval_every == 0 or step == int(cfg.distill_steps):
            net.eval()
            with torch.no_grad():
                pred_val = net(val_features)
                true_val = midpoint + cfg.eta * val_target_scaled
                residual = pred_val - true_val
                val_loss = float(((residual / cfg.eta) ** 2).mean().item())
                val_mae = float(residual.abs().mean().item())
                val_rmse = float(residual.square().mean().sqrt().item())
                same_wedge = (((pred_val >= 1 - cfg.alpha) & (pred_val <= 1)) ==
                              ((true_val >= 1 - cfg.alpha) & (true_val <= 1)))
                wedge_accuracy = float(same_wedge.float().mean().item())
            history.append({"step": step, "train_loss_eta_scaled": float(loss.item()),
                            "val_loss_eta_scaled": val_loss, "val_R_mae": val_mae,
                            "val_R_rmse": val_rmse,
                            "val_wedge_coordinate_agreement": wedge_accuracy,
                            "n_train_labels": n_train,
                            "n_val_labels": n_val})
            if val_loss < best_loss:
                best_loss = val_loss
                best_step = step
                best_state = {name: value.detach().cpu().clone()
                              for name, value in net.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= int(cfg.distill_early_stop_patience):
                break
    if best_state is None:
        raise RuntimeError("No valid distillation checkpoint")
    net.load_state_dict(best_state)
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    hist = pd.DataFrame(history)
    best_record = hist.loc[hist.step == best_step].iloc[-1].to_dict()
    diagnostics = {"N": cfg.N, "best_step": best_step, "best_val_loss_eta_scaled": best_loss,
                   "total_archived_queries": int(len(arrays["step"])),
                   "valid_selected_labels": int(selected.sum()),
                   "n_train_labels": n_train,
                   "n_validation_labels": n_val,
                   "teacher_paths_disjoint_from_test": True, **best_record}
    return net, hist, diagnostics


def _P_eta_Rtheta_action(ratio_net: RatioNet, t_frac, x, y, cfg: Config):
    """Use a learned ratio at inference: strictly no BPTT or teacher lookup."""
    with torch.no_grad():
        r = ratio_net(_features(t_frac, x, y)).detach()
        valid = torch.isfinite(r).all(dim=-1)
        neutral = torch.full_like(r, 1.0 - cfg.alpha / 2.0)
        r = torch.where(valid.unsqueeze(-1), r, neutral)
        u = _project_P_eta_for_outer_execution(r, x, y, cfg)
        u = torch.where(valid.unsqueeze(-1), u, torch.zeros_like(u))
    _audit_inc("Rtheta_ratio_query_states", x.numel())
    _audit_inc("Rtheta_ratio_batch_calls", 1)
    if _AUDIT is not None:
        _audit_inc("Rtheta_ratio_query_failed_states", (~valid).sum().item())
    return u, {"R_hat": r, "ratio_valid": valid}

# Same-time application of P_eta^[m] / P_eta^[star]

def _virtual_P_eta_step(x:torch.Tensor,y:torch.Tensor,u:torch.Tensor,cfg:Config):
    """Same-calendar-time NO fee; keep x>=0,y>=0; allow boundary trading."""
    cfg0=replace(cfg,eta=0.0)
    rate=_project_u(u,x,y,cfg0)
    W=_gross_wealth(x,y)
    L=_liq_wealth(x,y,cfg.alpha)
    yy=(y+(L*cfg.dt).unsqueeze(-1)*rate).clamp_min(0)
    xx=W-yy.sum(-1)
    if bool((xx < -5e-5*(1+W.abs())).any()):
        raise RuntimeError("Virtual step exceeded available gross wealth")
    xx=xx.clamp_min(0)
    return xx,yy,(yy-y).abs().sum(-1)

def _execute_target_once_unregularized(x0,y0,y_target,cfg):
    """One physical NET order: sell existing holdings, use sale proceeds to BUY.

    Budget is tested after the sale fee. There is no ban on buying at x=0:
    concurrent sales fund purchases. Never charge fees during virtual steps.
    """
    finite=torch.isfinite(y_target).all(-1)
    requested=torch.where(finite.unsqueeze(-1),y_target,y0)
    desired=requested-y0
    sale=torch.minimum(torch.relu(-desired),y0.clamp_min(0.0))
    cash_available=x0+(1-cfg.alpha)*sale.sum(-1)
    buy_requested=torch.relu(desired)
    buy_total=buy_requested.sum(-1)
    buy_scale=torch.minimum(torch.ones_like(buy_total),
                 cash_available.clamp_min(0)/buy_total.clamp_min(1e-12))
    buy=buy_requested*buy_scale.unsqueeze(-1)
    y1=(y0-sale+buy).clamp_min(0)
    x1=(cash_available-buy.sum(-1)).clamp_min(0)
    delta=y1-y0
    cost=cfg.alpha*sale.sum(-1)
    if bool(((y1<-1e-6).any(-1)|(x1<-1e-6)).any()):
        raise RuntimeError("Single net execution violated no-short/no-borrow")
    if _AUDIT is not None:
        _audit_inc("net_order_scaled_paths",(buy_scale<1-1e-8).sum().item())
        _audit_inc("target_fallback_paths",(~finite).sum().item())
    return x1,y1,delta.abs().sum(-1),cost,buy_scale

def _market_step_after_target(x,y,dW,cfg,cache):
    absorbed=(y.abs().sum(-1)==0)&(x<=cfg.exit_liq_floor*(1+1e-5))
    x2=x*math.exp(cfg.r*cfg.dt)
    mu=cache["mu"].to(dtype=y.dtype)
    sig=cache["sigma"].to(dtype=y.dtype)
    expo=(mu-.5*sig.square())*cfg.dt+sig*math.sqrt(cfg.dt)*dW
    y2=y*torch.exp(expo)
    x2,y2=_terminal_exit(x2,y2,cfg,context="market")
    return (torch.where(absorbed,torch.full_like(x2,cfg.exit_liq_floor),x2),
            torch.where(absorbed.unsqueeze(-1),torch.zeros_like(y2),y2))

def _apply_P_eta_depth(
    x: torch.Tensor,
    y: torch.Tensor,
    action_builder,
    no_trade_builder,
    cfg: Config,
    repeat_depth: Optional[int],
    ratio_observer=None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Implements P_eta^[m] when repeat_depth=m and P_eta^[star] when repeat_depth=None.

    For finite m, no-trade is absorbing: reaching it early is equivalent to all
    remaining P_eta applications being exactly zero.
    """
    if repeat_depth is not None and int(repeat_depth) < 1:
        raise ValueError("repeat_depth must be a positive integer or None for [star].")

    max_maps = int(cfg.p_eta_star_max_iterations) if repeat_depth is None else int(repeat_depth)
    x_initial, y_initial = x.detach().clone(), y.detach().clone()
    x_virtual, y_virtual = x_initial.clone(), y_initial.clone()
    bsz = x.shape[0]

    converged = torch.zeros(bsz, dtype=torch.bool, device=x.device)
    stalled = torch.zeros(bsz, dtype=torch.bool, device=x.device)
    query_failed=torch.zeros(bsz,dtype=torch.bool,device=x.device)
    n_queries=torch.zeros(bsz,dtype=torch.long,device=x.device)
    n_maps = torch.zeros(bsz, dtype=torch.long, device=x.device)
    virtual_path_length = torch.zeros(bsz, dtype=x.dtype, device=x.device)

    for map_iteration in range(max_maps):
        u_instruction, aux = action_builder(x_virtual, y_virtual)
        n_queries += 1  # counts ALL states actually submitted to the batch query
        if ratio_observer is not None:
            ratio_observer(x_virtual, y_virtual, aux, map_iteration, False,
                           ~(converged | stalled))
        valid=aux.get("ratio_valid",torch.ones(bsz,dtype=torch.bool,device=x.device))
        newly_failed=(~valid)&(~converged)&(~stalled)
        query_failed |= newly_failed
        stalled |= newly_failed
        nt_now = no_trade_builder(u_instruction, aux) & valid
        if nt_now.ndim != 1 or nt_now.shape[0] != bsz:
            raise ValueError("no_trade_builder must return shape (batch,).")

        live = ~(converged | stalled)
        converged = converged | (live & nt_now)
        to_move = live & (~nt_now)
        if not bool(to_move.any()):
            break

        x_candidate, y_candidate, path_inc = _virtual_P_eta_step(
            x_virtual, y_virtual, u_instruction, cfg
        )
        dx = (x_candidate - x_virtual).abs()
        dy = (y_candidate - y_virtual).abs().amax(dim=-1)
        immobile = to_move & (torch.maximum(dx, dy) <= cfg.same_time_position_tol)
        stalled = stalled | immobile
        move_mask = to_move & (~immobile)

        x_virtual = torch.where(move_mask, x_candidate, x_virtual)
        y_virtual = torch.where(move_mask.unsqueeze(-1), y_candidate, y_virtual)
        n_maps = n_maps + move_mask.long()
        virtual_path_length = virtual_path_length + torch.where(
            move_mask, path_inc, torch.zeros_like(path_inc)
        )

    final_u, final_aux = action_builder(x_virtual, y_virtual)
    n_queries += 1  # final endpoint audit query, including finite-depth / failed paths
    if ratio_observer is not None:
        ratio_observer(x_virtual, y_virtual, final_aux, map_iteration + 1, True,
                       ~(converged | stalled))
    final_valid=final_aux.get("ratio_valid",torch.ones(bsz,dtype=torch.bool,device=x.device))
    query_failed |= ~final_valid
    final_nt = no_trade_builder(final_u, final_aux) & final_valid
    raw_wedge = ((final_aux["R_hat"]>=1-cfg.alpha)&
                 (final_aux["R_hat"]<=1.0)).all(-1)&final_valid
    constraint_hold = final_nt & (~raw_wedge)
    reached_no_trade = final_nt & (~query_failed)
    hit_depth_cap = (~reached_no_trade) & (~stalled) & (~query_failed)

    x_exec, y_exec, turnover, transaction_cost, execution_scale = (
        _execute_target_once_unregularized(x_initial, y_initial, y_virtual, cfg)
    )

    initial_W = _gross_wealth(x_initial, y_initial)
    virtual_W = _gross_wealth(x_virtual, y_virtual)
    executed_W = _gross_wealth(x_exec, y_exec)

    if _AUDIT is not None:
        _audit_inc("replay_physical_decisions",bsz)
        _audit_inc("replay_end_no_trade",(reached_no_trade&~query_failed).sum().item())
        _audit_inc("replay_end_constraint_hold",(constraint_hold&~query_failed).sum().item())
        _audit_inc("replay_end_stall",(stalled&~query_failed).sum().item())
        _audit_inc("replay_end_cap",hit_depth_cap.sum().item())
        _audit_inc("replay_end_query_failure",query_failed.sum().item())
        _audit_inc("fallback_physical_decisions",query_failed.sum().item())
    diagnostics = {
        "n_P_eta_maps": n_maps,
        "n_ratio_queries":n_queries,
        "query_failed":query_failed,
        "termination_code":torch.where(query_failed,torch.full_like(n_maps,3),
             torch.where(reached_no_trade,torch.zeros_like(n_maps),
             torch.where(stalled,torch.ones_like(n_maps),torch.full_like(n_maps,2)))),
        "reached_no_trade": reached_no_trade,
        "raw_wedge_endpoint":raw_wedge,
        "constraint_hold":constraint_hold,
        "stalled": stalled,
        "hit_depth_cap": hit_depth_cap,
        "repeat_depth": torch.full(
            (bsz,), -1 if repeat_depth is None else int(repeat_depth),
            device=x.device, dtype=torch.long,
        ),
        "turnover": turnover,
        "transaction_cost": transaction_cost,
        "execution_scale": execution_scale,
        "execution_scaled": execution_scale < (1.0 - 1e-8),
        "virtual_path_length": virtual_path_length,
        "virtual_gross_wealth_error": (virtual_W - initial_W).abs(),
        "initial_pi": y_initial / torch.clamp(initial_W.unsqueeze(-1), min=1e-12),
        "virtual_target_pi": y_virtual / torch.clamp(virtual_W.unsqueeze(-1), min=1e-12),
        "target_pi": y_exec / torch.clamp(executed_W.unsqueeze(-1), min=1e-12),
        "final_instruction": final_u.detach(),
    }
    return x_exec, y_exec, diagnostics, {k: v.detach() for k, v in final_aux.items()}


def _method_name_P_eta(source: str, repeat_depth: Optional[int]) -> str:
    idx = "star" if repeat_depth is None else str(int(repeat_depth))
    if source == "Rtheta":
        return f"P_eta^[{idx}](R_theta)"
    return f"P_eta^[{idx}](Rhat_{source})"


def _threshold_label(tau: float) -> str:
    """Compact, deterministic label used in method names / CSV keys."""
    return f"{float(tau):.6g}"


def _method_name_threshold_u0(tau: float) -> str:
    return f"Threshold[tau={_threshold_label(tau)}](u_theta,0)"


def _method_name_hold_gated_u0() -> str:
    return "Hold-gated(u_theta,0|Rhat_BPTT)"


def _plot_label(method: str, eta_label: str) -> str:
    if method == "No-Action":
        return "No-Action"
    if method == "u_theta,0":
        return r"$u_{\theta,0}$"
    if method == "u_theta,eta":
        return r"$u_{\theta,\eta}$"
    if method == _method_name_hold_gated_u0():
        return r"Hold-gated $u_{\theta,0}$"
    m = re.match(r"Threshold\[tau=(.+?)\]\(u_theta,0\)", method)
    if m:
        tau = m.group(1)
        return rf"Threshold $u_{{\theta,0}}$, $\tau={tau}$"
    m_student = re.match(r"P_eta\^\[(.+?)\]\(R_theta\)", method)
    if m_student:
        idx_tex = r"\star" if m_student.group(1) == "star" else m_student.group(1)
        return rf"$P_\eta^{{[{idx_tex}]}}(R_\theta)$"
    m = re.match(r"P_eta\^\[(.+?)\]\(Rhat_(BPTT|PINN)\)", method)
    if m:
        idx, source = m.group(1), m.group(2)
        idx_tex = r"\star" if idx == "star" else idx
        return rf"$P_\eta^{{[{idx_tex}]}}(\widehat{{R}}_{{\rm {source}}})$"
    return method


def _safe_method_key(method: str) -> str:
    key = (
        method.replace("P_eta^[", "Peta_m")
        .replace("](Rhat_", "_")
        .replace("](R_theta)", "_Rtheta")
        .replace("u_theta,0", "u_theta_0")
        .replace("u_theta,eta", "u_theta_eta")
        .replace("No-Action", "NoAction")
        .replace("star", "star")
    )
    # Make every newly added method safe as a CSV column / filename.
    key = re.sub(r"[^A-Za-z0-9_]+", "_", key).strip("_")
    key = re.sub(r"_+", "_", key)
    return key


# Objective evaluation helpers

@torch.no_grad()
def _trade_only_unregularized(
    x: torch.Tensor, y: torch.Tensor, u: torch.Tensor, cfg: Config
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg0 = replace(cfg, eta=0.0)
    u_exec = _project_u(u, x, y, cfg0)
    L = _liq_wealth(x, y, cfg.alpha)
    buy, sell = torch.relu(u_exec), torch.relu(-u_exec)
    x2 = x - L * buy.sum(dim=-1) * cfg.dt + (1.0 - cfg.alpha) * L * sell.sum(dim=-1) * cfg.dt
    y2 = y + L.unsqueeze(-1) * u_exec * cfg.dt
    turnover = L * u_exec.abs().sum(dim=-1) * cfg.dt
    return x2, y2, u_exec, turnover


@torch.no_grad()
def _eval_no_action_unregularized(
    cfg: Config, device: torch.device, cache: Dict[str, torch.Tensor],
    n_paths: Optional[int] = None, seed_extra: int = 0,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    method = "No-Action"
    n_paths = cfg.outer_eval_paths if n_paths is None else int(n_paths)
    chunk = max(1, cfg.eval_chunk_size)
    utility_all, liq_all, rows = [], [], []
    for s0 in range(0, n_paths, chunk):
        bsz = min(chunk, n_paths - s0)
        x = torch.full((bsz,), cfg.x0_eval, device=device, dtype=torch.float32)
        y = cache["y0_eval"].view(1, -1).repeat(bsz, 1)
        outer_bank = _build_fwd_bank(0, cfg, device, bsz, cache, seed_extra=seed_extra+s0)
        for step in range(cfg.n_steps):
            _decision_eval(lambda: torch.zeros_like(y),bsz,device,"zero_net_order")
            dW = outer_bank[step, :bsz, :] if outer_bank is not None else _correlated_dW(
                torch.randn((bsz, cfg.N), device=device, dtype=y.dtype), cache["chol"].to(dtype=y.dtype)
            )
            x, y = _market_step_after_target(x, y, dW, cfg, cache)
            rows.append({"N":cfg.N,"chunk_start":s0,"step":step,"method":method,
                         "execution_rule":"u_identically_zero","mean_turnover":0.0})
        liqT = _liq_wealth(x, y, cfg.alpha)
        utility_all.append(_utility_t(liqT, cfg.gamma).cpu()); liq_all.append(liqT.cpu())
    utility, liquidation = torch.cat(utility_all), torch.cat(liq_all)
    us, ls = _summary_stats(utility), _summary_stats(liquidation)
    return ({"objective_mean":us["mean"],"objective_std":us["std"],"objective_se":us["se"],
             "terminal_liq_mean":ls["mean"],"terminal_liq_std":ls["std"],"n_paths":us["n_paths"],
             "eta_outer_evaluation":0.0,"execution_rule":"u_identically_zero"},
            pd.DataFrame(rows), _build_pathwise_value_df(method,cfg.N,liquidation,utility))


@torch.no_grad()
def _eval_u_theta_unregularized(
    policy: PolicyNet, cfg_policy: Config, cfg_outer: Config,
    device: torch.device, cache: Dict[str, torch.Tensor], method_name: str,
    n_paths: Optional[int] = None, seed_extra: int = 0,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    n_paths = cfg_outer.outer_eval_paths if n_paths is None else int(n_paths)
    chunk = max(1, cfg_outer.eval_chunk_size)
    utility_all, liq_all, rows = [], [], []
    policy.eval()
    for s0 in range(0, n_paths, chunk):
        bsz = min(chunk, n_paths-s0)
        x = torch.full((bsz,), cfg_outer.x0_eval, device=device, dtype=torch.float32)
        y = cache["y0_eval"].view(1,-1).repeat(bsz,1)
        outer_bank = _build_fwd_bank(0,cfg_outer,device,bsz,cache,seed_extra=seed_extra+s0)
        for step in range(cfg_outer.n_steps):
            t_frac = torch.full((bsz,), step/cfg_outer.n_steps, device=device, dtype=x.dtype)
            xt,yt,u_exec,turnover=_decision_eval(
                lambda:_trade_only_unregularized(x,y,
                    _u_theta_action(policy,t_frac,x,y,cfg_policy),cfg_outer),
                bsz,device)
            dW = outer_bank[step,:bsz,:] if outer_bank is not None else _correlated_dW(
                torch.randn((bsz,cfg_outer.N),device=device,dtype=y.dtype),cache["chol"].to(dtype=y.dtype)
            )
            x,y = _market_step_after_target(xt,yt,dW,cfg_outer,cache)
            rows.append({"N":cfg_outer.N,"chunk_start":s0,"step":step,"method":method_name,
                         "eta_policy_training":float(cfg_policy.eta),"eta_outer_evaluation":0.0,
                         "execution_rule":"one_u_theta_rate_per_calendar_step",
                         "mean_abs_rate":float(u_exec.abs().mean().item()),
                         "mean_turnover":float(turnover.mean().item())})
        liqT=_liq_wealth(x,y,cfg_outer.alpha)
        utility_all.append(_utility_t(liqT,cfg_outer.gamma).cpu()); liq_all.append(liqT.cpu())
    utility, liquidation=torch.cat(utility_all),torch.cat(liq_all)
    us,ls=_summary_stats(utility),_summary_stats(liquidation)
    return ({"objective_mean":us["mean"],"objective_std":us["std"],"objective_se":us["se"],
             "terminal_liq_mean":ls["mean"],"terminal_liq_std":ls["std"],"n_paths":us["n_paths"],
             "eta_outer_evaluation":0.0,"execution_rule":"one_u_theta_rate_per_calendar_step"},
            pd.DataFrame(rows), _build_pathwise_value_df(method_name,cfg_outer.N,liquidation,utility))


@torch.no_grad()
def _eval_threshold_u_theta0_unregularized(
    policy: PolicyNet,
    tau: float,
    cfg0: Config,
    device: torch.device,
    cache: Dict[str, torch.Tensor],
    n_paths: Optional[int] = None,
    seed_extra: int = 0,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Hard-threshold baseline:

        u_i^thr(t,z) = u_{theta,0,i}(t,z) * 1{|u_{theta,0,i}(t,z)| > tau}.

    The threshold is coordinatewise.  No BPTT ratio, Yosida map, refinement,
    or same-time replay is used.  This isolates the benefit of simply deleting
    small raw neural actions.
    """
    tau = float(tau)
    if tau < 0.0:
        raise ValueError("tau must be nonnegative.")

    method = _method_name_threshold_u0(tau)
    n_paths = cfg0.outer_eval_paths if n_paths is None else int(n_paths)
    chunk = max(1, cfg0.eval_chunk_size)
    utility_all, liq_all, rows = [], [], []
    policy.eval()

    for s0 in range(0, n_paths, chunk):
        bsz = min(chunk, n_paths - s0)
        x = torch.full(
            (bsz,), cfg0.x0_eval, device=device, dtype=torch.float32
        )
        y = cache["y0_eval"].view(1, -1).repeat(bsz, 1)
        outer_bank = _build_fwd_bank(
            0, cfg0, device, bsz, cache, seed_extra=seed_extra + s0
        )

        for step in range(cfg0.n_steps):
            t_frac = torch.full(
                (bsz,),
                step / cfg0.n_steps,
                device=device,
                dtype=x.dtype,
            )
            decision_start=_decision_begin(device)
            u_raw = _u_theta_action(
                policy, t_frac, x, y, cfg0
            )
            keep = u_raw.abs() > tau
            u_thr = torch.where(keep, u_raw, torch.zeros_like(u_raw))

            xt, yt, u_exec, turnover = _trade_only_unregularized(
                x, y, u_thr, cfg0
            )
            _decision_end(decision_start,bsz,device)
            dW = (
                outer_bank[step, :bsz, :]
                if outer_bank is not None
                else _correlated_dW(
                    torch.randn(
                        (bsz, cfg0.N),
                        device=device,
                        dtype=y.dtype,
                    ),
                    cache["chol"].to(dtype=y.dtype),
                )
            )
            x, y = _market_step_after_target(
                xt, yt, dW, cfg0, cache
            )

            rows.append({
                "N": cfg0.N,
                "chunk_start": s0,
                "step": step,
                "method": method,
                "tau": tau,
                "eta_policy_training": 0.0,
                "eta_outer_evaluation": 0.0,
                "execution_rule": "coordinatewise_hard_threshold_u_theta_0",
                "mean_abs_raw_rate": float(
                    u_raw.abs().mean().item()
                ),
                "active_coordinate_ratio": float(
                    keep.float().mean().item()
                ),
                "exact_zero_coordinate_ratio": float(
                    (u_exec == 0.0).float().mean().item()
                ),
                "all_zero_state_ratio": float(
                    (u_exec.abs().amax(dim=-1) == 0.0)
                    .float()
                    .mean()
                    .item()
                ),
                "mean_turnover": float(turnover.mean().item()),
            })

        liqT = _liq_wealth(x, y, cfg0.alpha)
        utility_all.append(
            _utility_t(liqT, cfg0.gamma).detach().cpu()
        )
        liq_all.append(liqT.detach().cpu())

    utility = torch.cat(utility_all)
    liquidation = torch.cat(liq_all)
    us, ls = _summary_stats(utility), _summary_stats(liquidation)
    return (
        {
            "objective_mean": us["mean"],
            "objective_std": us["std"],
            "objective_se": us["se"],
            "terminal_liq_mean": ls["mean"],
            "terminal_liq_std": ls["std"],
            "n_paths": us["n_paths"],
            "tau": tau,
            "eta_outer_evaluation": 0.0,
            "execution_rule": "coordinatewise_hard_threshold_u_theta_0",
        },
        pd.DataFrame(rows),
        _build_pathwise_value_df(
            method, cfg0.N, liquidation, utility
        ),
    )


def _eval_hold_gated_u_theta0_unregularized(
    policy_u0: PolicyNet,
    detector_policy_ueta: PolicyNet,
    cfg: Config,
    device: torch.device,
    cache: Dict[str, torch.Tensor],
    n_paths: Optional[int] = None,
    seed_extra: int = 0,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Hold-gated baseline:

      1) obtain the existing BPTT detector Rhat_BPTT from the frozen
         u_{theta,eta} continuation;
      2) for each coordinate i independently, if
             1-alpha <= Rhat_i <= 1,
         force u_{theta,0,i}=0;
      3) outside the detected hold coordinate, preserve the ORIGINAL
         u_{theta,0,i} unchanged.

    Therefore this baseline adds only detector-located exact zeros.  It does NOT
    replace the active-region sign or magnitude with P_eta.
    """
    method = _method_name_hold_gated_u0()
    cfg0 = replace(cfg, eta=0.0)
    n_paths = cfg.outer_eval_paths if n_paths is None else int(n_paths)
    chunk = max(1, cfg.eval_chunk_size)
    utility_all, liq_all, rows = [], [], []

    policy_u0.eval()
    detector_policy_ueta.eval()

    for s0 in range(0, n_paths, chunk):
        bsz = min(chunk, n_paths - s0)
        x = torch.full(
            (bsz,), cfg.x0_eval, device=device, dtype=torch.float32
        )
        y = cache["y0_eval"].view(1, -1).repeat(bsz, 1)
        outer_bank = _build_fwd_bank(
            0, cfg0, device, bsz, cache, seed_extra=seed_extra + s0
        )

        for step in range(cfg.n_steps):
            t_frac = torch.full(
                (bsz,),
                step / cfg.n_steps,
                device=device,
                dtype=x.dtype,
            )

            decision_start=_decision_begin(device)
            # Raw eta=0 action to be preserved outside detected hold.
            with torch.no_grad():
                u0 = _u_theta_action(
                    policy_u0, t_frac, x, y, cfg0
                )

            # BPTT detector source: the frozen Stage-1 u_{theta,eta}
            # continuation policy directly.
            inner_bank = _build_inner_bank(
                step, cfg, device, cfg.inner_mc_paths, cache
            )
            R_hat, Vx_hat, Vy_hat = _Rhat_BPTT(
                detector_policy_ueta,
                x,
                y,
                step * cfg.dt,
                cfg,
                device,
                cache,
                inner_bank=inner_bank,
            )

            hold_coord = (
                (R_hat >= (1.0 - cfg.alpha))
                & (R_hat <= 1.0)
            )
            u_gated = torch.where(
                hold_coord, torch.zeros_like(u0), u0
            )

            with torch.no_grad():
                xt, yt, u_exec, turnover = _trade_only_unregularized(
                    x, y, u_gated, cfg0
                )
                _decision_end(decision_start,bsz,device)
                dW = (
                    outer_bank[step, :bsz, :]
                    if outer_bank is not None
                    else _correlated_dW(
                        torch.randn(
                            (bsz, cfg.N),
                            device=device,
                            dtype=y.dtype,
                        ),
                        cache["chol"].to(dtype=y.dtype),
                    )
                )
                x, y = _market_step_after_target(
                    xt, yt, dW, cfg0, cache
                )

            rows.append({
                "N": cfg.N,
                "chunk_start": s0,
                "step": step,
                "method": method,
                "ratio_source": "BPTT",
                "eta_detector_continuation": float(cfg.eta),
                "eta_outer_evaluation": 0.0,
                "execution_rule": (
                    "zero_u_theta_0_coordinates_detected_as_hold;"
                    "preserve_u_theta_0_elsewhere"
                ),
                "detected_hold_coordinate_ratio": float(
                    hold_coord.float().mean().item()
                ),
                "detected_joint_hold_state_ratio": float(
                    hold_coord.all(dim=-1).float().mean().item()
                ),
                "exact_zero_coordinate_ratio": float(
                    (u_exec == 0.0).float().mean().item()
                ),
                "mean_abs_raw_u0_rate": float(
                    u0.abs().mean().item()
                ),
                "mean_turnover": float(turnover.mean().item()),
                "mean_Vx_hat": float(Vx_hat.mean().item()),
                "mean_abs_Rhat_minus_midwedge": float(
                    (
                        R_hat - (1.0 - cfg.alpha / 2.0)
                    )
                    .abs()
                    .mean()
                    .item()
                ),
            })

        liqT = _liq_wealth(x, y, cfg.alpha)
        utility_all.append(
            _utility_t(liqT, cfg.gamma).detach().cpu()
        )
        liq_all.append(liqT.detach().cpu())

    utility = torch.cat(utility_all)
    liquidation = torch.cat(liq_all)
    us, ls = _summary_stats(utility), _summary_stats(liquidation)
    return (
        {
            "objective_mean": us["mean"],
            "objective_std": us["std"],
            "objective_se": us["se"],
            "terminal_liq_mean": ls["mean"],
            "terminal_liq_std": ls["std"],
            "n_paths": us["n_paths"],
            "eta_detector_continuation": float(cfg.eta),
            "eta_outer_evaluation": 0.0,
            "ratio_source": "BPTT",
            "execution_rule": (
                "hold_gate_u_theta_0_with_BPTT_detector"
            ),
        },
        pd.DataFrame(rows),
        _build_pathwise_value_df(
            method, cfg.N, liquidation, utility
        ),
    )


def _eval_P_eta_unregularized(
    source: str,
    repeat_depth: Optional[int],
    cfg: Config,
    device: torch.device,
    cache: Dict[str, torch.Tensor],
    policy: Optional[PolicyNet] = None,
    pinn_net: Optional[PINNValueNet] = None,
    ratio_net: Optional[RatioNet] = None,
    n_paths: Optional[int] = None,
    seed_extra: int = 0,
    ratio_collector: Optional[BPTTRatioCollector] = None,
    inner_seed_extra: int = 0,
) -> Tuple[Dict[str,float],pd.DataFrame,pd.DataFrame]:
    """Shared evaluator: BPTT teacher, PINN and learned-ratio student use identical execution."""
    source = "Rtheta" if source.lower() == "rtheta" else source.upper()
    if source not in {"BPTT", "PINN", "Rtheta"}:
        raise ValueError("source must be BPTT, PINN, or Rtheta")
    if source == "BPTT" and policy is None:
        raise ValueError("BPTT source requires policy")
    if source == "PINN" and pinn_net is None:
        raise ValueError("PINN source requires pinn_net")
    if source == "Rtheta" and ratio_net is None:
        raise ValueError("Rtheta source requires a frozen trained RatioNet")
    if ratio_collector is not None and source != "BPTT":
        raise ValueError("Only the BPTT teacher rollout may populate ratio labels")

    method=_method_name_P_eta(source,repeat_depth)
    cfg0=replace(cfg,eta=0.0)
    n_paths=cfg.outer_eval_paths if n_paths is None else int(n_paths)
    chunk=max(1,cfg.eval_chunk_size)
    utility_all,liq_all,rows=[],[],[]
    if policy is not None: policy.eval()
    if pinn_net is not None: pinn_net.eval()
    if ratio_net is not None: ratio_net.eval()

    for s0 in range(0,n_paths,chunk):
        bsz=min(chunk,n_paths-s0)
        x=torch.full((bsz,),cfg.x0_eval,device=device,dtype=torch.float32)
        y=cache["y0_eval"].view(1,-1).repeat(bsz,1)
        outer_bank=_build_fwd_bank(0,cfg0,device,bsz,cache,seed_extra=seed_extra+s0)

        for step in range(cfg.n_steps):
            t_frac=torch.full((bsz,),step/cfg.n_steps,device=device,dtype=x.dtype)
            decision_start=_decision_begin(device)
            if source=="BPTT":
                inner_bank=_build_inner_bank(
                    step,cfg,device,cfg.inner_mc_paths,cache,
                    seed_extra=inner_seed_extra)
                def action_builder(x_now,y_now):
                    return _P_eta_Rhat_BPTT_action(
                        policy,x_now,y_now,step*cfg.dt,cfg,device,cache,
                        inner_bank=inner_bank,
                    )
                def no_trade_builder(u_now,aux_now):
                    return u_now.abs().amax(dim=-1) <= cfg.no_trade_rate_tol
            elif source == "PINN":
                def action_builder(x_now,y_now):
                    return _P_eta_Rhat_PINN_action(pinn_net,t_frac,x_now,y_now,cfg)
                def no_trade_builder(u_now,aux_now):
                    return u_now.abs().amax(dim=-1) <= cfg.no_trade_rate_tol
            else:
                def action_builder(x_now,y_now):
                    return _P_eta_Rtheta_action(ratio_net,t_frac,x_now,y_now,cfg)
                def no_trade_builder(u_now,aux_now):
                    return u_now.abs().amax(dim=-1) <= cfg.no_trade_rate_tol

            if ratio_collector is not None:
                ratio_collector.set_context(step, s0)
            xt,yt,diag,final_aux=_apply_P_eta_depth(
                x,y,action_builder,no_trade_builder,cfg,repeat_depth,
                ratio_observer=ratio_collector.observe if ratio_collector is not None else None
            )
            _decision_end(decision_start,bsz,device,"BPTT_or_PINN_replay_to_net_order",
                          query_count=diag["n_ratio_queries"].sum().item())
            dW=outer_bank[step,:bsz,:] if outer_bank is not None else _correlated_dW(
                torch.randn((bsz,cfg.N),device=device,dtype=y.dtype),cache["chol"].to(dtype=y.dtype)
            )
            x,y=_market_step_after_target(xt,yt,dW,cfg0,cache)
            Rf=final_aux["R_hat"]
            rows.append({
                "N":cfg.N,"chunk_start":s0,"step":step,"method":method,"ratio_source":source,
                "repeat_depth":"star" if repeat_depth is None else int(repeat_depth),
                "eta_map":float(cfg.eta),"eta_outer_evaluation":0.0,
                "execution_rule":"same_time_P_eta_requery_then_one_net_trade",
                "mean_P_eta_maps":float(diag["n_P_eta_maps"].float().mean().item()),
                "max_P_eta_maps":int(diag["n_P_eta_maps"].max().item()),
                "mean_actual_ratio_queries":float(diag["n_ratio_queries"].float().mean().item()),
                "query_failure_ratio":float(diag["query_failed"].float().mean().item()),
                "end_no_trade_ratio":float((diag["termination_code"]==0).float().mean().item()),
                "end_stall_ratio":float((diag["termination_code"]==1).float().mean().item()),
                "end_cap_ratio":float((diag["termination_code"]==2).float().mean().item()),
                "end_query_failure_ratio":float((diag["termination_code"]==3).float().mean().item()),
                "no_trade_endpoint_ratio":float(diag["reached_no_trade"].float().mean().item()),
                "constraint_hold_ratio":float(diag["constraint_hold"].float().mean().item()),
                "depth_cap_ratio":float(diag["hit_depth_cap"].float().mean().item()),
                "stalled_ratio":float(diag["stalled"].float().mean().item()),
                "mean_turnover":float(diag["turnover"].mean().item()),
                "mean_transaction_cost":float(diag["transaction_cost"].mean().item()),
                "mean_virtual_path_length":float(diag["virtual_path_length"].mean().item()),
                "mean_abs_Rhat_minus_midwedge":float((Rf-(1.0-cfg.alpha/2.0)).abs().mean().item()),
            })

        liqT=_liq_wealth(x,y,cfg.alpha)
        utility_all.append(_utility_t(liqT,cfg.gamma).detach().cpu()); liq_all.append(liqT.detach().cpu())
    utility,liquidation=torch.cat(utility_all),torch.cat(liq_all)
    us,ls=_summary_stats(utility),_summary_stats(liquidation)
    return ({"objective_mean":us["mean"],"objective_std":us["std"],"objective_se":us["se"],
             "terminal_liq_mean":ls["mean"],"terminal_liq_std":ls["std"],"n_paths":us["n_paths"],
             "eta_map":float(cfg.eta),"eta_outer_evaluation":0.0,
             "ratio_source":source,"repeat_depth":"star" if repeat_depth is None else int(repeat_depth)},
            pd.DataFrame(rows),_build_pathwise_value_df(method,cfg.N,liquidation,utility))


def _eta_label(value: float) -> str:
    return f"{float(value):.12g}"


def _eta_tag(value: float) -> str:
    return _eta_label(value).replace("-","m").replace("+","p").replace(".","p")



# Per-dimension runner

def _plot_boundary_policy_map(cfg:Config,device:torch.device,cache:Dict[str,torch.Tensor],
                              policy0:PolicyNet,policy_eta:PolicyNet,outdir_n:str):
    """Visualize feasible simplex INCLUDING ALL THREE ZERO FACES for any N>=2.

    A/B represent sector totals, internally divided uniformly among assets.
    Color = actual feasible normalized rate magnitude; quiver = target delta.
    Display real data even when every rate is zero (scatter, never empty contour).
    """
    m=max(5,int(cfg.boundary_plot_resolution))
    A,B=[],[]
    for i in range(m):
        for j in range(m-i):
            A.append(i/(m-1));B.append(j/(m-1))
    a=np.asarray(A); b=np.asarray(B); c=1-a-b
    nA=cfg.N//2;nB=cfg.N-nA
    assert nA>=1 and nB>=1
    holdings=np.concatenate((np.repeat((a/nA)[:,None],nA,axis=1),
                              np.repeat((b/nB)[:,None],nB,axis=1)),axis=-1)
    x=torch.tensor(c,dtype=torch.float32,device=device).clamp_min(0)
    y=torch.tensor(holdings,dtype=torch.float32,device=device)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,(name,pol,conf) in zip(axes,(("u_theta,0",policy0,replace(cfg,eta=0.0)),
                                        ("u_theta,eta",policy_eta,cfg))):
        with torch.no_grad():
            t=torch.zeros_like(x)
            u=_u_theta_action(pol,t,x,y,conf)
            L=_liq_wealth(x,y,cfg.alpha)
            dy=L.unsqueeze(-1)*u*cfg.dt
            dA=dy[:,:nA].sum(-1).cpu().numpy()
            dB=dy[:,nA:].sum(-1).cpu().numpy()
            strength=u.abs().mean(-1).cpu().numpy()
            u_np=u.cpu().numpy()
        im=ax.scatter(a,b,c=strength,cmap="viridis",s=19,vmin=0,
                      vmax=max(float(np.max(strength)),1e-12),zorder=2)
        stride=max(1,len(a)//130)
        ax.quiver(a[::stride],b[::stride],dA[::stride],dB[::stride],
                  angles="xy",scale_units="xy",scale=1.0,alpha=.65,width=.0025)
        ax.plot([0,1,0,0],[0,0,1,0],color="black",lw=1.4,zorder=3)
        ax.scatter([cfg.sector_initial_A,cfg.sector_merton_A],
                   [cfg.sector_initial_B,cfg.sector_merton_B],
                   marker="x",s=70,color=["tab:red","tab:orange"],zorder=5)
        ax.set(xlim=(-.025,1.025),ylim=(-.025,1.025),aspect="equal",
               xlabel="Sector A weight",ylabel="Sector B weight",
               title=f"{name}: x=0 & y=0 included; active={np.mean(np.max(np.abs(u_np),axis=1)>cfg.no_trade_rate_tol):.2%}")
        fig.colorbar(im,ax=ax,label="mean absolute feasible rate",shrink=.83)
    path=os.path.join(outdir_n,"simplex_boundary_policy_map.png")
    fig.savefig(path,dpi=175,bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame({"sector_A":a,"sector_B":b,"cash_weight":c}).to_csv(
        os.path.join(outdir_n,"simplex_boundary_grid.csv"),index=False)
    print(f"  [boundary plot] {path}; includes cash=0 & holding=0 edges")


def run_all_methods_for_N(N: int, base_cfg: Config) -> Dict:
    cfg=replace(base_cfg,N=N)
    cfg0=replace(cfg,eta=0.0)
    eta_label=_eta_label(cfg.eta)
    eta_tag=_eta_tag(cfg.eta)
    depths=tuple(sorted(set(int(m) for m in cfg.p_eta_repeat_depths if int(m)>=1)))
    thresholds=tuple(sorted(set(float(t) for t in cfg.hard_action_thresholds if float(t)>=0.0)))
    if not thresholds:
        raise ValueError("hard_action_thresholds must contain at least one nonnegative tau.")

    _set_seed(cfg.seed)
    device=_choose_device(cfg.device)
    cache=_build_cache(cfg,device)
    outdir_n=os.path.join(cfg.outdir,f"N{N}")
    os.makedirs(outdir_n,exist_ok=True)
    _validate_and_report_portfolio_setup(cfg, outdir_n)

    threshold_u0_methods=[
        _method_name_threshold_u0(tau) for tau in thresholds
    ]
    hold_gated_methods=[_method_name_hold_gated_u0()]
    bptt_methods=[_method_name_P_eta("BPTT",m) for m in depths]+[_method_name_P_eta("BPTT",None)]
    pinn_methods=[_method_name_P_eta("PINN",m) for m in depths]+[_method_name_P_eta("PINN",None)]
    distilled_methods=[_method_name_P_eta("Rtheta",m) for m in depths]+[_method_name_P_eta("Rtheta",None)]
    method_order=(
        ["No-Action","u_theta,0","u_theta,eta"]
        + threshold_u0_methods
        + hold_gated_methods
        + pinn_methods
        + bptt_methods
        + distilled_methods
    )

    print("\n"+"="*120)
    print(f"N={N} | eta={eta_label}")
    print("Methods: "+" / ".join(method_order))
    print("All reported objectives use eta=0 outer execution semantics.")
    print("="*120)

    # u_{theta,0}
    print(f"\n[N={N}] Training u_theta,0")
    _set_seed(cfg.seed+1)
    u_theta_0, t_u0=_timed_call(lambda:train_u_theta(cfg0,device,cache),device)
    torch.save(u_theta_0.state_dict(),os.path.join(outdir_n,"u_theta_0.pt"))
    u_theta_0.training_history.to_csv(os.path.join(outdir_n,"u_theta_0_training_diagnostics.csv"),index=False)

    # u_{theta,eta}
    print(f"\n[N={N}] Training u_theta,eta with eta={eta_label}")
    _set_seed(cfg.seed+2)
    u_theta_eta,t_ueta=_timed_call(lambda:train_u_theta(cfg,device,cache),device)
    torch.save(u_theta_eta.state_dict(),os.path.join(outdir_n,f"u_theta_eta_{eta_tag}.pt"))
    u_theta_eta.training_history.to_csv(os.path.join(outdir_n,f"u_theta_eta_{eta_tag}_training_diagnostics.csv"),index=False)
    # Stage-1 continuation policy is frozen for all subsequent BPTT queries.
    u_theta_eta.eval()
    for p in u_theta_eta.parameters():
        p.requires_grad_(False)
    _plot_boundary_policy_map(cfg,device,cache,u_theta_0,u_theta_eta,outdir_n)

    # PINN ratio source
    print(f"\n[N={N}] Training PINN for eta={eta_label}")
    _set_seed(cfg.seed+7)
    (pinn_net,pinn_history),t_pinn=_timed_call(lambda:train_pinn(cfg,device,cache),device)
    pinn_history.to_csv(os.path.join(outdir_n,f"PINN_eta_{eta_tag}_history.csv"),index=False)
    torch.save(pinn_net.state_dict(),os.path.join(outdir_n,f"PINN_eta_{eta_tag}.pt"))

    # Separate teacher rollouts: preserve every intermediate BPTT ratio from the
    # exact corresponding P_eta^[m] / P_eta^[star] objective procedure.
    # NEVER collect from common_outer_seed (held-out). Train and validate the
    # student only on these independent paths; the teacher test remains untouched.
    common_outer_seed=300_000
    if cfg.distill_teacher_seed_extra == common_outer_seed:
        raise ValueError("Distillation teacher and held-out outer seeds must differ")
    if not cfg.use_common_random_numbers:
        raise ValueError("Distillation's paired held-out objective comparison requires CRN")
    distilled_nets, distill_build_seconds, distill_diagnostics = {}, {}, []
    distill_summaries = []
    for method_idx, depth in enumerate((*depths, None)):
        teacher_name = _method_name_P_eta("BPTT", depth)
        student_name = _method_name_P_eta("Rtheta", depth)
        collector = BPTTRatioCollector(cfg, teacher_name, cfg.distill_teacher_seed_extra)
        print(f"\n[N={N}] Collecting independent teacher ratios: {teacher_name}")
        (teacher_stats, _, _), teacher_seconds = _timed_call(
            lambda depth=depth, collector=collector: _eval_P_eta_unregularized(
                "BPTT", depth, cfg, device, cache, policy=u_theta_eta,
                n_paths=cfg.distill_teacher_paths,
                seed_extra=cfg.distill_teacher_seed_extra,
                ratio_collector=collector,
                inner_seed_extra=cfg.distill_teacher_seed_extra), device)
        arrays = collector.arrays()
        key = _safe_method_key(student_name)
        archive = os.path.join(outdir_n, f"{key}_teacher_ratio_queries.npz")
        manifest = collector.save(archive, arrays)
        print(f"  archived {manifest['n_ratio_queries_all']} BPTT queries, "
              f"valid {manifest['n_ratio_queries_valid']} -> {archive}")
        (ratio_net, hist, distill_diag), fit_seconds = _timed_call(
            lambda arrays=arrays, method_idx=method_idx: train_distilled_ratio(
                cfg, device, arrays, seed=cfg.seed + 20_000 + method_idx), device)
        distilled_nets[student_name] = ratio_net
        distill_build_seconds[student_name] = teacher_seconds + fit_seconds
        torch.save({"state_dict": ratio_net.state_dict(), "N": cfg.N,
                    "eta": cfg.eta, "alpha": cfg.alpha,
                    "hidden": cfg.distill_hidden, "depth": cfg.distill_depth,
                    "teacher": teacher_name, "teacher_seed_extra": cfg.distill_teacher_seed_extra},
                   os.path.join(outdir_n, f"{key}.pt"))
        hist.to_csv(os.path.join(outdir_n, f"{key}_supervised_history.csv"), index=False)
        distill_diag.update({"method": student_name, "teacher_method": teacher_name,
                             "teacher_collection_seconds": float(teacher_seconds),
                             "supervised_fit_seconds": float(fit_seconds),
                             "ratio_archive": os.path.basename(archive),
                             "teacher_training_rollout_J0_mean": float(teacher_stats["objective_mean"])})
        distill_diagnostics.append(distill_diag)
        distill_summaries.append(manifest)
        del arrays, collector
        print(f"  student best-step={distill_diag['best_step']} "
              f"val RMSE(R)={distill_diag['val_R_rmse']:.5g} "
              f"wedge agreement={distill_diag['val_wedge_coordinate_agreement']:.3%}")
    pd.DataFrame(distill_diagnostics).to_csv(
        os.path.join(outdir_n, "Rtheta_supervised_learning_summary.csv"), index=False)

    stats:Dict[str,Dict[str,float]]={}
    profiles:Dict[str,pd.DataFrame]={}
    timing_rows=[]
    diagnostics_rows=[]
    path_values:Dict[str,pd.DataFrame]={}
    eval_seconds:Dict[str,float]={}

    def timed_eval(name,fn):
        global _AUDIT
        print(f"\n[N={N}] Objective evaluation: {name}")
        if _AUDIT is not None:
            raise RuntimeError("Nested evaluation audit is not supported")
        _AUDIT={"method":name,"counters":{},"decisions":[]}
        if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
        try:
            (st,pr,pv),sec=_timed_call(fn,device)
            audit=_AUDIT
            peak=(torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0)
        finally:
            _AUDIT=None
        stats[name],profiles[name],path_values[name],eval_seconds[name]=st,pr,pv,float(sec)
        safe=_safe_method_key(name)
        pr.to_csv(os.path.join(outdir_n,f"{safe}_profile.csv"),index=False)
        pv.to_csv(os.path.join(outdir_n,f"{safe}_path_values.csv"),index=False)
        d=pd.DataFrame(audit["decisions"])
        d.insert(0,"method",name)
        d.insert(0,"N",N)
        d.to_csv(os.path.join(outdir_n,f"{safe}_decision_latencies.csv"),index=False)
        if len(d):
            # Batch latency and amortized throughput are different metrics.
            batch_p50=float(d.decision_seconds_per_batch.median())
            batch_p95=float(d.decision_seconds_per_batch.quantile(.95))
            amortized=float(d.decision_seconds_amortized.mean())
            ndec=int(d.batch_size.sum())
            actual_queries=sum(int(q) for q in d.ratio_queries_in_batch if pd.notna(q))
            for row in d.to_dict("records"):
                timing_rows.append(row)
        else:
            batch_p50=batch_p95=amortized=float("nan")
            ndec=actual_queries=0
        counters=audit["counters"]
        actual_ratio_state_queries=(counters.get("BPTT_ratio_query_states",0)
                                    +counters.get("PINN_ratio_query_states",0)
                                    +counters.get("Rtheta_ratio_query_states",0))
        diag={"N":N,"method":name,"outer_eval_seconds":float(sec),
              "n_physical_decisions":ndec,"n_policy_batches":len(d),
              "mean_amortized_decision_sec":amortized,
              "median_batch_decision_sec":batch_p50,
              "p95_batch_decision_sec":batch_p95,
              "total_measured_decision_seconds":float(d.decision_seconds_per_batch.sum()) if len(d) else 0.0,
              "actual_ratio_queries_timed":actual_ratio_state_queries,
              "peak_gpu_memory_bytes":int(peak),"device":str(device),
              "dtype_policy":"float32","inner_MC":cfg.inner_mc_paths,
              "outer_paths":cfg.outer_eval_paths,"n_steps":cfg.n_steps,
              "exit_floor":cfg.exit_liq_floor,
              "ratio_failure_denominator":actual_ratio_state_queries,
              "fallback_decision_denominator":ndec,
              "termination_denominator":counters.get("replay_physical_decisions",0),
              **counters}
        diagnostics_rows.append(diag)
        print(f"  [audit] batches={len(d)} decisions={ndec} batch_p50={batch_p50:.5g}s "
              f"batch_p95={batch_p95:.5g}s ratio_query_states={actual_ratio_state_queries} "
              f"ratio_failures={counters.get('BPTT_ratio_query_failed_states',0)+counters.get('PINN_ratio_query_failed_states',0)+counters.get('Rtheta_ratio_query_failed_states',0)} "
              f"market_exits={counters.get('market_exit_events',0)} "
              f"new_exits={counters.get('market_new_exit_events',0)} "
              f"budget_binding={counters.get('cash_budget_binding',0)}")


    timed_eval("No-Action",lambda:_eval_no_action_unregularized(
        cfg0,device,cache,n_paths=cfg.outer_eval_paths,seed_extra=common_outer_seed))
    timed_eval("u_theta,0",lambda:_eval_u_theta_unregularized(
        u_theta_0,cfg0,cfg0,device,cache,"u_theta,0",n_paths=cfg.outer_eval_paths,seed_extra=common_outer_seed))
    timed_eval("u_theta,eta",lambda:_eval_u_theta_unregularized(
        u_theta_eta,cfg,cfg0,device,cache,"u_theta,eta",n_paths=cfg.outer_eval_paths,seed_extra=common_outer_seed))

    # New ablation A: hard threshold on u_{theta,0}.
    for tau in thresholds:
        name=_method_name_threshold_u0(tau)
        timed_eval(name,lambda tau=tau:_eval_threshold_u_theta0_unregularized(
            u_theta_0,tau,cfg0,device,cache,
            n_paths=cfg.outer_eval_paths,seed_extra=common_outer_seed))

    # New ablation B: correct-location exact-zero only.
    # The BPTT detector decides hold coordinates; outside hold we preserve
    # u_{theta,0} exactly rather than replacing sign/magnitude by P_eta.
    name=_method_name_hold_gated_u0()
    timed_eval(name,lambda:_eval_hold_gated_u_theta0_unregularized(
        u_theta_0,u_theta_eta,cfg,device,cache,
        n_paths=cfg.outer_eval_paths,seed_extra=common_outer_seed))

    for m in depths:
        name=_method_name_P_eta("PINN",m)
        timed_eval(name,lambda m=m:_eval_P_eta_unregularized(
            "PINN",m,cfg,device,cache,pinn_net=pinn_net,
            n_paths=cfg.outer_eval_paths,seed_extra=common_outer_seed))
    name=_method_name_P_eta("PINN",None)
    timed_eval(name,lambda:_eval_P_eta_unregularized(
        "PINN",None,cfg,device,cache,pinn_net=pinn_net,
        n_paths=cfg.outer_eval_paths,seed_extra=common_outer_seed))

    for m in depths:
        name=_method_name_P_eta("BPTT",m)
        timed_eval(name,lambda m=m:_eval_P_eta_unregularized(
            "BPTT",m,cfg,device,cache,policy=u_theta_eta,
            n_paths=cfg.outer_eval_paths,seed_extra=common_outer_seed))
    name=_method_name_P_eta("BPTT",None)
    timed_eval(name,lambda:_eval_P_eta_unregularized(
        "BPTT",None,cfg,device,cache,policy=u_theta_eta,
        n_paths=cfg.outer_eval_paths,seed_extra=common_outer_seed))

    # Held-out Rtheta objective is evaluated only after its network was frozen.
    # CRN and path IDs match every corresponding BPTT teacher evaluation.
    for depth in (*depths, None):
        name=_method_name_P_eta("Rtheta",depth)
        timed_eval(name,lambda depth=depth,name=name:_eval_P_eta_unregularized(
            "Rtheta",depth,cfg,device,cache,ratio_net=distilled_nets[name],
            n_paths=cfg.outer_eval_paths,seed_extra=common_outer_seed))

    # Paired BPTT-vs-distilled gap on the genuinely held-out common paths.
    teacher_student_rows=[]
    for depth in (*depths, None):
        teacher=_method_name_P_eta("BPTT",depth)
        student=_method_name_P_eta("Rtheta",depth)
        a=path_values[teacher].sort_values("path_id")
        b=path_values[student].sort_values("path_id")
        if not np.array_equal(a.path_id.to_numpy(),b.path_id.to_numpy()):
            raise RuntimeError("Teacher/student held-out paths are not aligned")
        diff=b.path_value.to_numpy()-a.path_value.to_numpy()
        teacher_student_rows.append({"N":N,"depth":"star" if depth is None else depth,
              "teacher":teacher,"student":student,"n_outer_paths":len(diff),
              "student_minus_teacher_J0":float(diff.mean()),
              "paired_SE":float(diff.std(ddof=1)/np.sqrt(len(diff))) if len(diff)>1 else 0.0,
              "student_J0":stats[student]["objective_mean"],
              "teacher_J0":stats[teacher]["objective_mean"]})
    pd.DataFrame(teacher_student_rows).to_csv(
        os.path.join(outdir_n,"Rtheta_vs_BPTT_paired_heldout_J0.csv"),index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    paired = pd.DataFrame(teacher_student_rows)
    depth_labels = paired["depth"].astype(str).to_list()
    xx = np.arange(len(depth_labels))
    ax.errorbar(xx, paired["student_minus_teacher_J0"],
                yerr=paired["paired_SE"], fmt="o-", capsize=4,
                label=r"$J_0(P_\eta(R_\theta))-J_0(P_\eta(\widehat R_{\rm BPTT}))$")
    ax.axhline(0, linewidth=1, linestyle="--", color="0.35")
    ax.set_xticks(xx, depth_labels)
    ax.set(xlabel="Same-time mapping depth", ylabel="Paired held-out objective difference",
           title=f"N={N}: supervised ratio versus BPTT teacher")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir_n, "Rtheta_vs_BPTT_paired_heldout_J0.png"), dpi=180)
    plt.close(fig)

    path_values_df=pd.concat([path_values[m] for m in method_order],ignore_index=True)
    path_values_df.to_csv(os.path.join(outdir_n,"path_values_all_methods.csv"),index=False)
    paired_rows=[]
    for baseline in ("u_theta,0","u_theta,eta"):
        base=path_values[baseline].sort_values("path_id")
        for method in method_order:
            current=path_values[method].sort_values("path_id")
            if not np.array_equal(current.path_id.to_numpy(),base.path_id.to_numpy()):
                raise RuntimeError("Paired objective requires matching path IDs")
            gap=current.path_value.to_numpy()-base.path_value.to_numpy()
            paired_rows.append({"N":N,"method":method,"baseline":baseline,
                "n_outer_paths":len(gap),"paired_mean_gap":float(gap.mean()),
                "paired_SE":float(gap.std(ddof=1)/np.sqrt(len(gap))) if len(gap)>1 else 0.0,
                "paired_ci_lower":float(gap.mean()-1.96*gap.std(ddof=1)/np.sqrt(len(gap))) if len(gap)>1 else float(gap.mean()),
                "paired_ci_upper":float(gap.mean()+1.96*gap.std(ddof=1)/np.sqrt(len(gap))) if len(gap)>1 else float(gap.mean()),
                "scope":"outer_MC_only; single training seed; asymptotic normal CI"})
    pd.DataFrame(paired_rows).to_csv(os.path.join(outdir_n,"paired_objective_gaps.csv"),index=False)

    # Construction accounting: repeated-depth variants share the same learned source.
    construction={"No-Action":0.0,"u_theta,0":t_u0,"u_theta,eta":t_ueta}
    params={"No-Action":0,"u_theta,0":_count_parameters(u_theta_0),
            "u_theta,eta":_count_parameters(u_theta_eta)}

    # Threshold(u_theta,0) reuses the already-trained eta=0 policy.
    for method in threshold_u0_methods:
        construction[method]=t_u0
        params[method]=_count_parameters(u_theta_0)

    # Hold-gated uses u_theta,0 for execution and BPTT ratios from the
    # frozen Stage-1 u_theta,eta continuation policy.
    for method in hold_gated_methods:
        construction[method]=t_u0+t_ueta
        params[method]=(
            _count_parameters(u_theta_0)
            +_count_parameters(u_theta_eta)
        )

    for method in pinn_methods:
        construction[method]=t_pinn; params[method]=_count_parameters(pinn_net)
    for method in bptt_methods:
        construction[method]=t_ueta
        params[method]=_count_parameters(u_theta_eta)
    # Student construction includes frozen-teacher policy, distinct teacher
    # trajectory collection and supervised fitting; on-line query is student only.
    for method in distilled_methods:
        construction[method]=t_ueta+distill_build_seconds[method]
        params[method]=_count_parameters(distilled_nets[method])

    cost_rows=[]
    for method in method_order:
        if method.startswith("P_eta^["):
            eta_map=float(cfg.eta)
        else:
            # No Yosida map is used by direct, threshold, or hold-gated baselines.
            eta_map=np.nan
        cost_rows.append({"N":N,"method":method,"eta_map":eta_map,
                          "eta_outer_evaluation":0.0,"end_to_end_construction_seconds":float(construction[method]),
                          "continuation_eval_seconds":float(eval_seconds[method]),
                          "total_end_to_end_plus_eval_seconds":float(construction[method]+eval_seconds[method]),
                          "trainable_parameters":int(params[method])})
    cost_df=pd.DataFrame(cost_rows)
    diag_df=pd.DataFrame(diagnostics_rows)
    timing_df=pd.DataFrame(timing_rows)
    diag_df.to_csv(os.path.join(outdir_n,"execution_failure_and_latency.csv"),index=False)
    timing_df.to_csv(os.path.join(outdir_n,"decision_latency_all_batches.csv"),index=False)
    cost_df=cost_df.merge(diag_df.drop(columns=["N"]),on="method",how="left")
    cost_df.to_csv(os.path.join(outdir_n,"computation_costs.csv"),index=False)

    row={"N":N,"eta_regularized":float(cfg.eta),"eta_outer_evaluation":0.0,"n_paths":int(cfg.outer_eval_paths)}
    for method in method_order:
        p=_safe_method_key(method); st=stats[method]
        for k in ["objective_mean","objective_std","objective_se","terminal_liq_mean","terminal_liq_std"]:
            row[f"{p}_{k}"]=st[k]
        row[f"{p}_eval_seconds"]=eval_seconds[method]
    if not pinn_history.empty:
        last=pinn_history.iloc[-1]
        row["PINN_final_mean_abs_HJB_residual"]=last.get("mean_abs_raw_residual",np.nan)
        row["PINN_final_Vx_nonpositive_ratio"]=last.get("Vx_nonpositive_ratio",np.nan)
    pd.DataFrame([row]).to_csv(os.path.join(outdir_n,"objective_summary.csv"),index=False)

    print("\nController objective summary")
    for method in method_order:
        st=stats[method]
        print(f"  {method:32s}: objective={st['objective_mean']:.8f} ± {st['objective_se']:.2e} (SE), eval={eval_seconds[method]:.3f}s")

    return {"N":N,"summary_row":row,"cost_df":cost_df,"stats":stats,"profiles":profiles,
            "path_values":path_values,"path_values_df":path_values_df,"method_order":method_order,
            "u_theta_0":u_theta_0,"u_theta_eta":u_theta_eta,"PINN":pinn_net,
            "Rtheta_nets":distilled_nets,
            "Rtheta_supervised_diagnostics":pd.DataFrame(distill_diagnostics),
            "Rtheta_teacher_student_paired":pd.DataFrame(teacher_student_rows),
            "execution_audit":diag_df,"decision_latency":timing_df,
            "cfg_eta0":cfg0,"cfg_eta":cfg,
            "eta_label":eta_label,"eta_tag":eta_tag,"cache":cache,"device":device}


# Run block: edit only here for experiments

OUTPUT_DIR="long_only_no_borrowing_controller_comparison"
os.makedirs(OUTPUT_DIR,exist_ok=True)

base_cfg=Config(
    T=1.0,n_steps=20,r=0.045,
    # Recompute mu = r*1 + gamma*Sigma*pi_target independently for each N.
    mu_from_merton_target=True,
    mu_base=(0.06,),  # ignored when mu_from_merton_target=True
    sigma_base=(0.20,),
    corr_mode="random_spd",rho=0.20,corr_bound=0.70,
    sector_merton_A=0.70,
    sector_merton_B=0.10,

    sector_initial_A=0.08,
    sector_initial_B=0.72,
    train_gross_risky_max=1.00, exit_liq_floor=1e-8,
    gamma=3.0,alpha=0.005,eta=0.01,u_max=10.0,

    hidden=256,depth=3,batch_size=256,
    n_train_steps=3000,lr=5e-4,print_every=100,

    pinn_hidden=128,pinn_depth=3,pinn_batch_size=256,
    pinn_train_steps=1000,pinn_lr=5e-4,pinn_print_every=1000,
    pinn_hutchinson_samples=1,pinn_monotonicity_weight=1.0,
    pinn_vx_floor=1e-8,pinn_grad_clip=10.0,
    pinn_residual_scale=2.0,pinn_regime_tol=1e-4,pinn_use_float64=True,

    outer_eval_paths=256,
    inner_mc_paths=2048*2,
    eval_chunk_size=16,
    distill_teacher_paths=64,  # INDEPENDENT teacher objective rollouts per depth
    distill_teacher_seed_extra=600_000,
    distill_hidden=256,distill_depth=3,
    distill_batch_size=512,distill_steps=1000,distill_lr=5e-4,

    # Main ablation requested: finite [m] depths plus adaptive [star].
    p_eta_repeat_depths=(1,2,3),
    p_eta_star_max_iterations=500,
    same_time_position_tol=1e-4,

    # Hard thresholds are normalized policy-rate units.
    # Freeze these using validation data before using the final held-out test.
    hard_action_thresholds=(1e-3,1e-2,1e-1),

    x0_eval=0.20,  # W0=1: cash=20%, risky=80%
    use_common_random_numbers=True,
    use_antithetic=True,
    seed=800,
    device="cuda:7",  # original GPU choice; change to "auto" on other machines
    outdir=OUTPUT_DIR,
    param_extension_mode="tile",
)

N_LIST=[5,10,20,50]  # sector correlation robustness; edit here if needed
RUN_CORRELATION_SWEEP=False  # True => extra beta-by-dimension retraining (VERY expensive)
CORRELATION_BETA_LIST=(0.0,0.2,0.5,0.7)
all_results={}
for N in N_LIST:
    all_results[N]=run_all_methods_for_N(N,base_cfg)

# Optional external beta grid: retrain ALL baselines at each covariance.
# No current-test selection of favorable beta, seed, depth, or threshold.
if RUN_CORRELATION_SWEEP:
    beta_results=[]
    for beta in CORRELATION_BETA_LIST:
        for N in N_LIST:
            # Reuse the main run only at its exact beta to avoid duplicated training.
            if abs(beta-base_cfg.corr_bound)<1e-12:
                result=all_results[N]
            else:
                bcfg=replace(base_cfg,corr_bound=float(beta),
                             outdir=os.path.join(OUTPUT_DIR,f"beta_{beta:.2f}"))
                result=run_all_methods_for_N(N,bcfg)
            for method,st in result["stats"].items():
                beta_results.append({"beta":beta,"N":N,"method":method,
                    "objective_mean":st["objective_mean"],"objective_se":st["objective_se"]})
    bd=pd.DataFrame(beta_results)
    bd.to_csv(os.path.join(OUTPUT_DIR,"correlation_beta_sweep_objective.csv"),index=False)
    for N in N_LIST:
        fig,ax=plt.subplots(figsize=(9,5))
        for method in ["No-Action","u_theta,0","u_theta,eta",_method_name_P_eta("BPTT",1),_method_name_P_eta("BPTT",None),_method_name_P_eta("Rtheta",1),_method_name_P_eta("Rtheta",None)]:
            subset=bd[(bd.N==N)&(bd.method==method)].sort_values("beta")
            ax.errorbar(subset.beta,subset.objective_mean,yerr=subset.objective_se,
                        marker="o",capsize=3,label=method)
        ax.set(xlabel="Factor correlation strength beta",ylabel="Expected utility J0",
               title=f"N={N}: correlation-structure robustness")
        ax.legend(fontsize=8);ax.grid(alpha=.3);fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR,f"N{N}_correlation_beta_sweep.png"),dpi=170)
        plt.close(fig)

df_summary=pd.DataFrame([all_results[N]["summary_row"] for N in N_LIST]).sort_values("N")
df_cost=pd.concat([all_results[N]["cost_df"] for N in N_LIST],ignore_index=True).sort_values(["N","method"])
df_path_values=pd.concat([all_results[N]["path_values_df"] for N in N_LIST],ignore_index=True).sort_values(["N","method","path_id"])

df_summary.to_csv(os.path.join(OUTPUT_DIR,"objective_by_dimension.csv"),index=False)
df_cost.to_csv(os.path.join(OUTPUT_DIR,"computation_cost_by_dimension.csv"),index=False)
df_path_values.to_csv(os.path.join(OUTPUT_DIR,"path_values_by_dimension.csv"),index=False)

eta_label=_eta_label(base_cfg.eta)
depths=tuple(sorted(set(int(m) for m in base_cfg.p_eta_repeat_depths if int(m)>=1)))
thresholds=tuple(sorted(set(float(t) for t in base_cfg.hard_action_thresholds if float(t)>=0.0)))
method_order=all_results[N_LIST[0]]["method_order"]

print("\n"+"="*140)
print("Objective comparison")
print("="*140)
for _,r in df_summary.iterrows():
    N=int(r["N"])
    print(f"N={N}")
    for method in method_order:
        p=_safe_method_key(method)
        print(f"  {method:32s}: {r[f'{p}_objective_mean']:.8f} ± {r[f'{p}_objective_se']:.2e}")

# Figure 1: objective versus dimension for every controller.
plt.figure(figsize=(13,7))
markers=["o","s","^","v","D","P","X","<",">","h","d","*","p","8","H","+"]
linestyles=["-","--","-.",":"]
for j,method in enumerate(method_order):
    p=_safe_method_key(method)
    plt.errorbar(df_summary["N"],df_summary[f"{p}_objective_mean"],
                 yerr=df_summary[f"{p}_objective_se"],marker=markers[j%len(markers)],
                 linestyle=linestyles[j%len(linestyles)],capsize=3,
                 label=_plot_label(method,eta_label))
plt.xlabel("Asset dimension N")
plt.ylabel("Expected terminal utility")
plt.title(r"Long-only, no-borrowing: $P_\eta^{[m]}$ depth ablation")
plt.xticks(N_LIST)
plt.grid(alpha=0.3)
plt.legend(ncol=3,fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"objective_by_dimension_P_eta_depths.png"),dpi=180)
plt.close()

# Figure 2: for each N, categorical P_eta repeat-depth ablation [m,...,star].
for N in N_LIST:
    stats=all_results[N]["stats"]
    x_labels=[str(m) for m in depths]+[r"$\star$"]
    x=np.arange(len(x_labels))
    plt.figure(figsize=(8,5))
    for source,marker in [("BPTT","o"),("PINN","s"),("Rtheta","^")]:
        methods=[_method_name_P_eta(source,m) for m in depths]+[_method_name_P_eta(source,None)]
        means=[stats[m]["objective_mean"] for m in methods]
        ses=[stats[m]["objective_se"] for m in methods]
        ratio_label = r"$R_\theta$" if source == "Rtheta" else rf"$\widehat{{R}}_{{\rm {source}}}$"
        plt.errorbar(x,means,yerr=ses,marker=marker,capsize=3,label=ratio_label)
    plt.xticks(x,x_labels)
    plt.xlabel(r"Same-time depth $m$ in $P_\eta^{[m]}$ ($\star$: first-hit)")
    plt.ylabel("Expected terminal utility")
    plt.title(f"N={N}: objective versus regime re-detection depth")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,f"N{N}_objective_vs_P_eta_depth.png"),dpi=180)
    plt.close()

# Figure 3: terminal liquidation wealth versus dimension.
plt.figure(figsize=(13,7))
for j,method in enumerate(method_order):
    p=_safe_method_key(method)
    plt.plot(df_summary["N"],df_summary[f"{p}_terminal_liq_mean"],
             marker=markers[j%len(markers)],linestyle=linestyles[j%len(linestyles)],
             label=_plot_label(method,eta_label))
plt.xlabel("Asset dimension N")
plt.ylabel("Mean terminal liquidation wealth")
plt.title("Terminal liquidation wealth")
plt.xticks(N_LIST)
plt.grid(alpha=0.3)
plt.legend(ncol=3,fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"terminal_liquidation_by_dimension.png"),dpi=180)
plt.close()

# Figure 4: evaluation cost; [m] variants expose the computational price of re-detection.
plt.figure(figsize=(13,7))
for method,g in df_cost.groupby("method"):
    g=g.sort_values("N")
    plt.plot(g["N"],g["continuation_eval_seconds"],marker="o",label=_plot_label(method,eta_label))
plt.xlabel("Asset dimension N")
plt.ylabel("Evaluation seconds")
plt.title(r"Evaluation cost of $P_\eta^{[m]}$ regime re-detection")
plt.xticks(N_LIST)
plt.yscale("log")
plt.grid(alpha=0.3)
plt.legend(ncol=3,fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"evaluation_cost_by_dimension.png"),dpi=180)
plt.close()

# Main budget performance-time: fixed trained policies, full measured decision cost.
for N in N_LIST:
    d=all_results[N]["cost_df"]
    d=d[d["mean_amortized_decision_sec"].notna()].copy()
    if len(d):
        fig,ax=plt.subplots(figsize=(9,6))
        for source in ("BPTT","PINN","Rtheta"):
            token = "(R_theta)" if source == "Rtheta" else f"Rhat_{source}"
            subset=d[d["method"].str.contains(token,regex=False)].copy()
            if len(subset):
                subset=subset.sort_values("mean_amortized_decision_sec")
                ax.plot(subset["mean_amortized_decision_sec"],
                        [all_results[N]["stats"][m]["objective_mean"] for m in subset["method"]],
                        "-o",label=f"{source} ratio map")
                for _,rr in subset.iterrows():
                    ax.annotate(rr["method"].split("[")[-1].split("]")[0],
                         (rr["mean_amortized_decision_sec"],
                          all_results[N]["stats"][rr["method"]]["objective_mean"]),
                          xytext=(3,3),textcoords="offset points",fontsize=7)
        for baseline in ("u_theta,0","u_theta,eta"):
            match=d[d["method"]==baseline]
            if len(match):
                rr=match.iloc[0]
                ax.scatter([rr["mean_amortized_decision_sec"]],
                           [all_results[N]["stats"][baseline]["objective_mean"]],label=baseline)
        ax.set_xlabel("Complete decision seconds / state (amortized batch throughput)")
        ax.set_ylabel("Held-out objective J0");ax.set_xscale("log")
        ax.set_title(f"N={N}: performance vs real full-decision cost")
        ax.grid(alpha=.3);ax.legend(fontsize=8);fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR,f"N{N}_performance_time_main.png"),dpi=170)
        plt.close(fig)

print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
print("Done.")
