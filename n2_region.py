import os

import torch

import math

import json

import time

import random

from dataclasses import dataclass, asdict, field

from pathlib import Path

from typing import Dict, List, Optional, Tuple, Any, Union

import dataclasses

import numpy as np

import pandas as pd

import scipy.sparse as sp

import scipy.sparse.linalg as spla

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.tri as mtri

from matplotlib.colors import BoundaryNorm, ListedColormap

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from numpy.polynomial.hermite import hermgauss

from matplotlib.lines import Line2D

from matplotlib.colors import LinearSegmentedColormap

import torch.nn as nn

import torch.optim as optim

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

class Config2D:
    # Horizon / market
    T: float = 5.0
    n_steps: int = 100
    r: float = 0.02
    mu: Tuple[float, float] = (0.08, 0.10)
    sigma: Tuple[float, float] = (0.20, 0.25)
    rho: float = 0.0
    gamma: float = 3.0      # CRRA: u(W) = W^(1-gamma)/(1-gamma), gamma>0, gamma!=1

    # Proportional transaction cost (symmetric: lambda=mu=alpha)
    alpha: float = 0.03

    # Quadratic regularization on trading rate vector u=(u1,u2)
    eps_quad: float = 0.01
    u_max: float = 10.0

    # Training
    hidden: int = 128
    depth: int = 3
    batch_size: int = 256
    n_train_steps: int = 200
    lr: float = 1e-3
    print_every: int = 200

    # Initial-state sampling
    init_pi_sum_high: float = 1.6
    logw_low: float = math.log(0.8)
    logw_high: float = math.log(1.2)

    # Evaluation / projector
    outer_eval_paths: int = 256
    inner_mc_paths: int = 1024
    eval_chunk_size: int = 64
    lambda_floor: float = 1e-10

    # Default evaluation initial condition
    x0_eval: float = 0.30
    y10_eval: float = 0.35
    y20_eval: float = 0.35

    # Variance reduction
    use_common_random_numbers: bool = True
    use_antithetic: bool = True

    # Plane in WEALTH-FRACTION coordinates (y1, y2) = (X1/W, X2/W)
    # where W = X0+X1+X2 (gross wealth)
    plane_time: float = 0.0
    plane_points: int = 101          # 논문 스타일 플레인 해상도
    plane_chunk_size: int = 128

    # Plot bounds in (pi1, pi2) = wealth fraction coords
    # solvency region: pi1>=0, pi2>=0, pi1+pi2<=1
    pi1_min: float = 0.0
    pi1_max: float = 0.85
    pi2_min: float = 0.0
    pi2_max: float = 0.85
    plane_ref_W: float = 1.0         # reference gross wealth for plane
    plane_min_liquidation: float = 1e-8

    # finite-horizon DP/QVI benchmark benchmark parameters (Section 4 of paper)
    # Solves the 2D variational inequality in (y1,y2) = wealth-fraction coords
    dz_Ny: int = 60                  # grid points per axis (논문 수준)
    dz_K_factor: float = 1000.0     # K = dz_K_factor / dt  (K*dt = const)
    dz_newton_tol: float = 1e-8
    dz_newton_maxiter: int = 50
    dz_n_snapshots: int = 4         # legacy; not used when benchmark_kind="finite_horizon_dp_qvi"

    # Finite-horizon DP/QVI benchmark replacement
    benchmark_kind: str = "finite_horizon_dp_qvi"
    dp_n_grid: int = 61
    dp_p1_max: float = 1.0
    dp_p2_max: float = 1.0
    dp_p_sum_max: float = 2.0
    dp_n_steps: int = 24
    dp_gh_order: int = 5
    dp_buy_cost: Tuple[float, float] = (0.0, 0.0)
    dp_sell_cost: Optional[Tuple[float, float]] = None  # None -> (alpha, alpha)
    dp_hold_value_tol: float = 1e-10
    dp_trade_dollar_tol: float = 1e-10
    dp_snapshot_points: int = 151
    dp_run_convergence_check: bool = False
    dp_convergence_n_grids: Tuple[int, ...] = (41, 51, 61)
    dp_convergence_n_steps: Tuple[int, ...] = (16, 24)

    # Recovered-region classification
    region_tol: float = 0.001

    # Misc
    seed: int = 10
    device: str = "auto"
    outdir: str = "paper_2asset_direct_policy_dp_qvi"

    @property
    def dt(self) -> float:
        return self.T / self.n_steps

    # DP/QVI paper uses gamma < 1 notation (u(W)=W^gamma/gamma).
    # Our cfg.gamma is the CRRA coefficient (gamma>0).
    # Paper's gamma_paper = 1 - cfg.gamma  (so paper's gamma < 0 when cfg.gamma > 1)
    @property
    def dz_gamma(self) -> float:
        """Paper's gamma: u(W) = W^{gamma_paper}/gamma_paper. For CRRA with cfg.gamma>0,
        gamma_paper = 1 - cfg.gamma."""
        return 1.0 - self.gamma

class PolicyNet2D(nn.Module):
    def __init__(self, hidden: int = 96, depth: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = 4  # t/T, logW, y1, y2  (wealth-fraction coords)
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.Tanh())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def choose_device(device_arg: str) -> torch.device:
    """Robust device selector for notebooks.

    - "auto": use cuda:0 if CUDA is available, otherwise CPU.
    - "cuda" or "cuda:k": use the requested CUDA device if available.
      If the requested index is invalid, fall back to cuda:0 and print a warning.
    - Any CPU-like string returns torch.device("cpu").
    """
    device_arg = str(device_arg).strip().lower()

    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")

    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            print(f"[device] Requested {device_arg}, but CUDA is unavailable. Falling back to CPU.")
            return torch.device("cpu")

        if device_arg == "cuda":
            return torch.device("cuda:0")

        try:
            idx = int(device_arg.split(":", 1)[1])
        except Exception:
            print(f"[device] Could not parse device '{device_arg}'. Falling back to cuda:0.")
            return torch.device("cuda:0")

        if idx < 0 or idx >= torch.cuda.device_count():
            print(f"[device] Requested cuda:{idx}, but only {torch.cuda.device_count()} CUDA device(s) are visible. Falling back to cuda:0.")
            return torch.device("cuda:0")
        return torch.device(f"cuda:{idx}")

    return torch.device("cpu")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj: Dict, path: str) -> None:
    def _to_serializable(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    obj2 = {k: _to_serializable(v) for k, v in obj.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj2, f, indent=2)

def save_npz(path: str, **arrays) -> None:
    np.savez_compressed(path, **arrays)

def utility(x: torch.Tensor, gamma: float) -> torch.Tensor:
    x = torch.clamp(x, min=1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return torch.log(x)
    return (x.pow(1.0 - gamma) - 1.0) / (1.0 - gamma)

def liquidation_wealth_2d(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    """Liquidation wealth L = x + (1-alpha)*(y1+y2)"""
    return x + (1.0 - alpha) * y.sum(dim=-1)

def liquidation_wealth_2d_np(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, alpha: float) -> np.ndarray:
    return x + (1.0 - alpha) * (y1 + y2)

def gross_wealth_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gross wealth W = x + y1 + y2"""
    return x + y.sum(dim=-1)

def corr_matrix_2d(cfg: "Config2D") -> np.ndarray:
    rho = float(cfg.rho)
    rho = max(-0.999999, min(0.999999, rho))
    return np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)

def chol_corr_torch(cfg: "Config2D", dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    C = torch.tensor(corr_matrix_2d(cfg), dtype=dtype, device=device)
    return torch.linalg.cholesky(C)

def make_correlated_dW(z: torch.Tensor, cfg: "Config2D") -> torch.Tensor:
    if z.shape[-1] != 2:
        raise ValueError(f"Expected last dimension 2, got {z.shape[-1]}")
    chol = chol_corr_torch(cfg, dtype=z.dtype, device=z.device)
    return z @ chol.T

def state_features_2d(t_frac: torch.Tensor, x: torch.Tensor, y: torch.Tensor, cfg: "Config2D") -> torch.Tensor:
    """
    Features: (t/T, log(W), pi1, pi2)
    where W = x+y1+y2 is gross wealth, pi_i = y_i / W.
    """
    W = gross_wealth_2d(x, y)
    logW = torch.log(torch.clamp(W, min=1e-12))
    pi = y / torch.clamp(W.unsqueeze(-1), min=1e-12)
    return torch.cat([t_frac.unsqueeze(-1), logW.unsqueeze(-1), pi], dim=-1)

def sample_initial_states_2d(cfg: "Config2D", batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    logw = torch.empty(batch_size, device=device).uniform_(cfg.logw_low, cfg.logw_high)
    W = torch.exp(logw)

    u = torch.rand(batch_size, 2, device=device)
    row_sum = torch.clamp(u.sum(dim=-1, keepdim=True), min=1e-12)
    direction = u / row_sum
    scale = torch.empty(batch_size, 1, device=device).uniform_(0.0, cfg.init_pi_sum_high)
    pi = direction * scale

    y = pi * W.unsqueeze(-1)
    x = W - y.sum(dim=-1)
    return x, y

def is_feasible_post_trade(x: torch.Tensor, y: torch.Tensor, u: torch.Tensor, cfg: "Config2D") -> torch.Tensor:
    L = liquidation_wealth_2d(x, y, cfg.alpha)
    dt = cfg.dt
    buy = torch.relu(u)
    sell = torch.relu(-u)

    x_trade = (
        x
        - L * buy.sum(dim=-1) * dt
        + (1.0 - cfg.alpha) * L * sell.sum(dim=-1) * dt
        - 0.5 * cfg.eps_quad * L * (u ** 2).sum(dim=-1) * dt
    )
    y_trade = y + L.unsqueeze(-1) * u * dt
    L_trade = x_trade + (1.0 - cfg.alpha) * y_trade.sum(dim=-1)
    cond = (y_trade >= -1e-12).all(dim=-1) & (L_trade > cfg.plane_min_liquidation)
    return cond

def project_rate_vector(u_raw: torch.Tensor, x: torch.Tensor, y: torch.Tensor, cfg: "Config2D") -> torch.Tensor:
    u = torch.clamp(u_raw, min=-cfg.u_max, max=cfg.u_max)
    L = liquidation_wealth_2d(x, y, cfg.alpha)
    lower = -y / torch.clamp(L.unsqueeze(-1) * cfg.dt, min=1e-12)
    u = torch.maximum(u, lower)
    u = torch.clamp(u, min=-cfg.u_max, max=cfg.u_max)

    feasible = is_feasible_post_trade(x, y, u, cfg)
    if feasible.all():
        return u

    out = u.clone()
    idx_bad = torch.where(~feasible)[0]
    if idx_bad.numel() == 0:
        return out

    u_bad = u[idx_bad]
    x_bad = x[idx_bad]
    y_bad = y[idx_bad]

    lo = torch.zeros(idx_bad.numel(), device=u.device)
    hi = torch.ones(idx_bad.numel(), device=u.device)

    for _ in range(28):
        mid = 0.5 * (lo + hi)
        u_mid = u_bad * mid.unsqueeze(-1)
        ok = is_feasible_post_trade(x_bad, y_bad, u_mid, cfg)
        lo = torch.where(ok, mid, lo)
        hi = torch.where(ok, hi, mid)

    out[idx_bad] = u_bad * lo.unsqueeze(-1)
    return out

def policy_action_2d(policy: PolicyNet2D, t_frac: torch.Tensor, x: torch.Tensor, y: torch.Tensor, cfg: "Config2D") -> torch.Tensor:
    inp = state_features_2d(t_frac, x, y, cfg)
    raw = policy(inp)
    u = cfg.u_max * torch.tanh(raw)
    return project_rate_vector(u, x, y, cfg)

def step_dynamics_from_rate_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    u: torch.Tensor,
    dW: torch.Tensor,
    cfg: "Config2D"
) -> Tuple[torch.Tensor, torch.Tensor]:
    L = liquidation_wealth_2d(x, y, cfg.alpha)
    u = project_rate_vector(u, x, y, cfg)
    buy = torch.relu(u)
    sell = torch.relu(-u)
    dt = cfg.dt

    x_trade = (
        x
        - L * buy.sum(dim=-1) * dt
        + (1.0 - cfg.alpha) * L * sell.sum(dim=-1) * dt
        - 0.5 * cfg.eps_quad * L * (u ** 2).sum(dim=-1) * dt
    )
    y_trade = y + L.unsqueeze(-1) * u * dt
    y_trade = torch.clamp(y_trade, min=0.0)

    x_next = x_trade * math.exp(cfg.r * dt)

    mu_t = torch.tensor(cfg.mu, dtype=y.dtype, device=y.device).view(1, 2)
    sigma_t = torch.tensor(cfg.sigma, dtype=y.dtype, device=y.device).view(1, 2)
    expo = (mu_t - 0.5 * sigma_t ** 2) * dt + sigma_t * math.sqrt(dt) * dW
    y_next = y_trade * torch.exp(expo)
    return x_next, y_next

def rollout_policy_2d(
    policy: PolicyNet2D,
    x0: torch.Tensor,
    y0: torch.Tensor,
    start_step: int,
    noise_bank: Optional[torch.Tensor],
    cfg: "Config2D"
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x0
    y = y0
    batch = x.shape[0]
    device = x.device
    remaining = cfg.n_steps - int(start_step)

    for j in range(remaining):
        step = int(start_step) + j
        t_frac = torch.full((batch,), float(step) / cfg.n_steps, device=device)
        u = policy_action_2d(policy, t_frac, x, y, cfg)

        if noise_bank is None:
            dW_raw = torch.randn(batch, 2, device=device)
            dW = make_correlated_dW(dW_raw, cfg)
        else:
            if noise_bank.shape[1] != batch:
                raise ValueError(f"noise_bank width {noise_bank.shape[1]} != batch {batch}")
            dW = noise_bank[j]

        x, y = step_dynamics_from_rate_2d(x, y, u, dW, cfg)

    return x, y

def _base_noise_count(m: int, antithetic: bool) -> Tuple[int, int]:
    if not antithetic:
        return int(m), int(m)
    half = int(math.ceil(m / 2))
    total = 2 * half
    return half, total

def build_shared_noise_bank(start_step: int, cfg: "Config2D", device: torch.device, m: Optional[int] = None) -> Optional[torch.Tensor]:
    if not cfg.use_common_random_numbers:
        return None
    remaining = cfg.n_steps - int(start_step)
    if remaining <= 0:
        return None
    m_eff = cfg.inner_mc_paths if m is None else int(m)
    base_count, total_count = _base_noise_count(m_eff, cfg.use_antithetic)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(cfg.seed + 1000 * (start_step + 1))
    z = torch.randn((remaining, base_count, 2), generator=gen, device="cpu").to(device)
    base = make_correlated_dW(z, cfg)
    if cfg.use_antithetic:
        noise = torch.cat([base, -base], dim=1)[:, :total_count, :]
    else:
        noise = base
    return noise

def train_direct_policy_2d(policy: PolicyNet2D, cfg: "Config2D", device: torch.device) -> Dict:
    opt = optim.Adam(policy.parameters(), lr=cfg.lr)
    curve_steps, curve_vals = [], []

    for step in range(cfg.n_train_steps + 1):
        x0, y0 = sample_initial_states_2d(cfg, cfg.batch_size, device)
        xT, yT = rollout_policy_2d(policy, x0, y0, 0, None, cfg)
        terminal_liq = liquidation_wealth_2d(xT, yT, cfg.alpha)
        obj = utility(terminal_liq, cfg.gamma).mean()
        loss = -obj

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.print_every == 0 or step == cfg.n_train_steps:
            curve_steps.append(int(step))
            curve_vals.append(float(obj.item()))
            print(f"[train] step={step:5d}  E[U(L_T)]={obj.item(): .6f}")

    return {"steps": curve_steps, "values": curve_vals}

def _expand_shared_noise_2d(noise_bank: torch.Tensor, batch_states: int) -> torch.Tensor:
    return noise_bank.repeat_interleave(batch_states, dim=1)

def recovered_action_batch_2d(
    policy: PolicyNet2D,
    x: torch.Tensor,
    y: torch.Tensor,
    t_value: float,
    cfg: "Config2D",
    device: torch.device,
    noise_bank: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    start_step = int(round(float(t_value) / cfg.dt))
    start_step = max(0, min(cfg.n_steps - 1, start_step))

    batch_states = x.numel()
    m_eff = int(cfg.inner_mc_paths)

    if cfg.use_antithetic:
        _, total_count = _base_noise_count(m_eff, True)
        m_paths = total_count
    else:
        m_paths = m_eff

    x_rep = x.view(1, batch_states).repeat(m_paths, 1).reshape(-1).clone().detach().requires_grad_(True)
    y_rep = y.view(1, batch_states, 2).repeat(m_paths, 1, 1).reshape(-1, 2).clone().detach().requires_grad_(True)

    if noise_bank is None:
        remaining = cfg.n_steps - start_step
        if remaining > 0:
            base_count, total_count = _base_noise_count(m_eff, cfg.use_antithetic)
            z = torch.randn((remaining, base_count, batch_states, 2), device=device)
            base = make_correlated_dW(z, cfg)
            if cfg.use_antithetic:
                noise = torch.cat([base, -base], dim=1)[:, :total_count, :, :]
            else:
                noise = base
            noise = noise.permute(0, 2, 1, 3).reshape(remaining, batch_states * m_paths, 2)
        else:
            noise = None
    else:
        noise = _expand_shared_noise_2d(noise_bank[:, :m_paths, :], batch_states)

    xT, yT = rollout_policy_2d(policy, x_rep, y_rep, start_step, noise, cfg)
    payoff = utility(liquidation_wealth_2d(xT, yT, cfg.alpha), cfg.gamma)
    grad_x, grad_y = torch.autograd.grad(payoff.sum(), [x_rep, y_rep], create_graph=False)

    lam_x = grad_x.view(m_paths, batch_states).mean(dim=0)
    lam_y = grad_y.view(m_paths, batch_states, 2).mean(dim=0)

    lam_x_safe = torch.clamp(lam_x, min=cfg.lambda_floor).unsqueeze(-1)
    buy_gap = lam_y - lam_x.unsqueeze(-1)
    sell_gap = (1.0 - cfg.alpha) * lam_x.unsqueeze(-1) - lam_y

    denom = torch.clamp(cfg.eps_quad * lam_x_safe, min=cfg.lambda_floor)
    u_buy = torch.relu(buy_gap / denom)
    u_sell = torch.relu(sell_gap / denom)
    u_recovered = u_buy - u_sell
    u_recovered = project_rate_vector(u_recovered, x.detach(), y.detach(), cfg)

    return u_recovered.detach(), lam_x.detach(), lam_y.detach(), buy_gap.detach(), sell_gap.detach()

def _valid_pi_plane_mask(PI1: np.ndarray, PI2: np.ndarray, cfg: "Config2D") -> np.ndarray:
    """Valid wealth-fraction states: pi1>=0, pi2>=0, pi1+pi2 < 1, cash > 0 after liquidation."""
    x = cfg.plane_ref_W * (1.0 - PI1 - PI2)
    y1 = cfg.plane_ref_W * PI1
    y2 = cfg.plane_ref_W * PI2
    L = liquidation_wealth_2d_np(x, y1, y2, cfg.alpha)
    return (PI1 >= 0.0) & (PI2 >= 0.0) & (L > cfg.plane_min_liquidation)

def pi_plane_to_xy(PI1: np.ndarray, PI2: np.ndarray, cfg: "Config2D") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    W = cfg.plane_ref_W
    y1 = W * PI1
    y2 = W * PI2
    x = W * (1.0 - PI1 - PI2)
    return x, y1, y2

def classify_recovered_region(u1: np.ndarray, u2: np.ndarray, tol: float) -> np.ndarray:
    a1 = np.zeros_like(u1, dtype=np.int64)
    a2 = np.zeros_like(u2, dtype=np.int64)
    a1[u1 > tol] = 1
    a1[u1 < -tol] = -1
    a2[u2 > tol] = 1
    a2[u2 < -tol] = -1

    out = np.zeros_like(a1, dtype=np.int64)
    out[(a1 == 0) & (a2 == 0)] = 0
    out[(a1 == 1) & (a2 == 0)] = 1
    out[(a1 == -1) & (a2 == 0)] = 2
    out[(a1 == 0) & (a2 == 1)] = 3
    out[(a1 == 0) & (a2 == -1)] = 4
    out[(a1 == 1) & (a2 == 1)] = 5
    out[(a1 == -1) & (a2 == -1)] = 6
    out[(a1 == 1) & (a2 == -1)] = 7
    out[(a1 == -1) & (a2 == 1)] = 8
    return out

def compute_recovered_region_plane_pi_at_time(
    policy: PolicyNet2D,
    cfg: "Config2D",
    device: torch.device,
    time_value: float
) -> Dict:
    """Compute Recovered plane in (pi1, pi2) = wealth-fraction coordinates."""
    pi1s = np.linspace(cfg.pi1_min, cfg.pi1_max, cfg.plane_points, dtype=np.float64)
    pi2s = np.linspace(cfg.pi2_min, cfg.pi2_max, cfg.plane_points, dtype=np.float64)
    PI1, PI2 = np.meshgrid(pi1s, pi2s)

    valid = _valid_pi_plane_mask(PI1, PI2, cfg)
    X, Y1, Y2 = pi_plane_to_xy(PI1, PI2, cfg)

    baseline_u1 = np.full_like(PI1, np.nan, dtype=np.float64)
    baseline_u2 = np.full_like(PI1, np.nan, dtype=np.float64)
    recovered_u1 = np.full_like(PI1, np.nan, dtype=np.float64)
    recovered_u2 = np.full_like(PI1, np.nan, dtype=np.float64)
    lam_x = np.full_like(PI1, np.nan, dtype=np.float64)
    lam_y1 = np.full_like(PI1, np.nan, dtype=np.float64)
    lam_y2 = np.full_like(PI1, np.nan, dtype=np.float64)
    buy_gap1 = np.full_like(PI1, np.nan, dtype=np.float64)
    sell_gap1 = np.full_like(PI1, np.nan, dtype=np.float64)
    buy_gap2 = np.full_like(PI1, np.nan, dtype=np.float64)
    sell_gap2 = np.full_like(PI1, np.nan, dtype=np.float64)
    recovered_region = np.full(PI1.shape, -1, dtype=np.int64)

    coords = np.argwhere(valid)
    if coords.size == 0:
        raise RuntimeError("No valid points in (pi1, pi2) plotting plane.")

    step = int(round(float(time_value) / cfg.dt))
    step = max(0, min(cfg.n_steps - 1, step))
    shared_noise = build_shared_noise_bank(step, cfg, device=device) if cfg.use_common_random_numbers else None

    for start in range(0, coords.shape[0], cfg.plane_chunk_size):
        chunk = coords[start:start + cfg.plane_chunk_size]
        j = chunk[:, 0]
        i = chunk[:, 1]

        x_t = torch.tensor(X[j, i], dtype=torch.float32, device=device)
        y_t = torch.tensor(np.stack([Y1[j, i], Y2[j, i]], axis=-1), dtype=torch.float32, device=device)
        t_frac = torch.full((x_t.numel(),), float(step) / cfg.n_steps, device=device)

        with torch.no_grad():
            base_u = policy_action_2d(policy, t_frac, x_t, y_t, cfg)

        recovered_u, lx, ly, bg, sg = recovered_action_batch_2d(
            policy=policy,
            x=x_t,
            y=y_t,
            t_value=float(step * cfg.dt),
            cfg=cfg,
            device=device,
            noise_bank=shared_noise,
        )

        base_u_np = base_u.cpu().numpy()
        recovered_u_np = recovered_u.cpu().numpy()
        lx_np = lx.cpu().numpy()
        ly_np = ly.cpu().numpy()
        bg_np = bg.cpu().numpy()
        sg_np = sg.cpu().numpy()

        lbl9 = classify_recovered_region(recovered_u_np[:, 0], recovered_u_np[:, 1], cfg.region_tol)

        baseline_u1[j, i] = base_u_np[:, 0]
        baseline_u2[j, i] = base_u_np[:, 1]
        recovered_u1[j, i] = recovered_u_np[:, 0]
        recovered_u2[j, i] = recovered_u_np[:, 1]
        lam_x[j, i] = lx_np
        lam_y1[j, i] = ly_np[:, 0]
        lam_y2[j, i] = ly_np[:, 1]
        buy_gap1[j, i] = bg_np[:, 0]
        sell_gap1[j, i] = sg_np[:, 0]
        buy_gap2[j, i] = bg_np[:, 1]
        sell_gap2[j, i] = sg_np[:, 1]
        recovered_region[j, i] = lbl9

    return {
        "time": np.array([float(step * cfg.dt)], dtype=np.float64),
        "pi1_grid": PI1,
        "pi2_grid": PI2,
        "x_grid": X,
        "y1_grid": Y1,
        "y2_grid": Y2,
        "valid_mask": valid.astype(np.uint8),
        "baseline_u1": baseline_u1,
        "baseline_u2": baseline_u2,
        "recovered_u1": recovered_u1,
        "recovered_u2": recovered_u2,
        "lam_x": lam_x,
        "lam_y1": lam_y1,
        "lam_y2": lam_y2,
        "buy_gap1": buy_gap1,
        "sell_gap1": sell_gap1,
        "buy_gap2": buy_gap2,
        "sell_gap2": sell_gap2,
        "recovered_region": recovered_region,
    }

def make_output_dirs(outdir: str) -> Dict:
    base = Path(outdir)
    figs_main = base / "figures_main"
    data_dir = base / "data"
    for p in [base, figs_main, data_dir]:
        ensure_dir(str(p))
    return {
        "base": str(base),
        "figures_main": str(figs_main),
        "data": str(data_dir),
    }

class DPBenchmarkConfig:
    # Model
    gamma: float = -2.0               # CRRA exponent. Risk aversion = 1-gamma = 3 when gamma=-2.
    T: float = 1.0
    r: float = 0.02
    mu: Tuple[float, float] = (0.08, 0.10)       # expected returns of risky assets
    sigma: Tuple[float, float] = (0.20, 0.25)
    rho: float = 0.30

    # Transaction costs
    # sell-only benchmark: buy_cost=0, sell_cost>0
    buy_cost: Tuple[float, float] = (0.0, 0.0)
    sell_cost: Tuple[float, float] = (0.03, 0.03)

    # DP discretization
    n_grid: int = 61                  # grid per axis before triangular filtering
    p1_max: float = 0.95
    p2_max: float = 0.95
    p_sum_max: float = 0.95           # no-borrowing/simplex cap: p1+p2 <= p_sum_max

    n_steps: int = 24                 # finite-horizon DP time steps
    gh_order: int = 5                 # Gauss-Hermite order per dimension. 5 => 25 nodes.

    # Numerical tolerances
    z_bisect_iter: int = 80
    z_upper_init: float = 2.0
    z_expand_max_iter: int = 20
    z_min: float = 1e-12

    # NTR classification
    # hold if q*=p exactly, or if utility loss from holding is below this tolerance
    hold_value_tol: float = 1e-10
    trade_dollar_tol: float = 1e-10

    # Output
    outdir: str = "finite_horizon_tc_dp_groundtruth"
    save_outputs: bool = True

    # Optional convergence sweep
    run_convergence_check: bool = False
    convergence_n_grids: Tuple[int, ...] = (41, 51, 61)
    convergence_n_steps: Tuple[int, ...] = (16, 24)

def covariance_matrix(cfg: DPBenchmarkConfig) -> np.ndarray:
    s1, s2 = cfg.sigma
    return np.array(
        [
            [s1 * s1, cfg.rho * s1 * s2],
            [cfg.rho * s1 * s2, s2 * s2],
        ],
        dtype=float,
    )

def merton_weight(cfg: DPBenchmarkConfig) -> np.ndarray:
    risk_aversion = 1.0 - cfg.gamma
    excess = np.asarray(cfg.mu, dtype=float) - cfg.r
    return np.linalg.solve(covariance_matrix(cfg), excess) / risk_aversion

def utility_power_multiplier(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    Returns x^gamma / gamma.
    Requires x>0.
    """
    return np.power(np.maximum(x, 1e-300), gamma) / gamma

def make_triangular_grid(cfg: DPBenchmarkConfig) -> np.ndarray:
    """
    Rectangular grid filtered by p1>=0, p2>=0, p1+p2<=p_sum_max.
    Returns array of shape [S,2].
    """
    p1 = np.linspace(0.0, cfg.p1_max, cfg.n_grid)
    p2 = np.linspace(0.0, cfg.p2_max, cfg.n_grid)

    pts = []
    for x in p1:
        for y in p2:
            if x >= -1e-14 and y >= -1e-14 and x + y <= cfg.p_sum_max + 1e-14:
                pts.append((float(x), float(y)))

    pts = np.array(pts, dtype=float)

    if len(pts) == 0:
        raise RuntimeError("Empty grid. Check p1_max, p2_max, p_sum_max, n_grid.")

    return pts

def liquidation_multiplier(p: np.ndarray, cfg: DPBenchmarkConfig) -> np.ndarray:
    """
    Terminal liquidation multiplier relative to current total wealth.

    For long-only p:
        cash = 1 - p1 - p2
        sell proceeds = (1-mu_i) p_i
        liquidation wealth = cash + sum_i (1-mu_i)p_i
                           = 1 - sum_i mu_i p_i
    """
    p = np.asarray(p, dtype=float)
    mu_sell = np.asarray(cfg.sell_cost, dtype=float)

    ell = 1.0 - p @ mu_sell
    return np.maximum(ell, 1e-300)

def self_financing_F(z: np.ndarray, P: np.ndarray, Q: np.ndarray, cfg: DPBenchmarkConfig) -> np.ndarray:
    """
    F(z)=0 encodes self-financing transition from pre-trade fraction p to post-trade fraction q.

    Pre-trade total wealth: W.
    Post-trade total wealth after transaction costs: z W.

    Pre risky dollar holdings: p_i W.
    Post risky dollar holdings: q_i z W.

    Trade dollar amount:
        Delta_i = z q_i - p_i.

    Cash equation:
        z(1-sum q)
        =
        1-sum p
        - sum_i (1+lambda_i) Delta_i^+
        + sum_i (1-mu_i) Delta_i^-,

    where Delta_i^- = max(-Delta_i,0) = max(p_i-zq_i,0).
    """
    lam = np.asarray(cfg.buy_cost, dtype=float)
    mu_sell = np.asarray(cfg.sell_cost, dtype=float)

    Psum = P[..., 0] + P[..., 1]
    Qsum = Q[..., 0] + Q[..., 1]

    Delta = z[..., None] * Q - P

    buy = np.maximum(Delta, 0.0)
    sell = np.maximum(-Delta, 0.0)

    rhs_cash = (
        1.0 - Psum
        - np.sum((1.0 + lam) * buy, axis=-1)
        + np.sum((1.0 - mu_sell) * sell, axis=-1)
    )

    lhs_cash = z * (1.0 - Qsum)

    return lhs_cash - rhs_cash

def compute_trade_multiplier_matrix(states: np.ndarray, actions: np.ndarray, cfg: DPBenchmarkConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute z[p_index, q_index] for all state-action pairs.

    Returns:
        z_mat: [S,A]
        feasible: [S,A]
    """
    S = states.shape[0]
    A = actions.shape[0]

    P = states[:, None, :]      # [S,1,2]
    Q = actions[None, :, :]     # [1,A,2]

    lo = np.full((S, A), cfg.z_min, dtype=float)
    hi = np.full((S, A), cfg.z_upper_init, dtype=float)

    F_lo = self_financing_F(lo, P, Q, cfg)
    F_hi = self_financing_F(hi, P, Q, cfg)

    # Expand upper bracket where needed
    for _ in range(cfg.z_expand_max_iter):
        bad = F_hi <= 0.0
        if not np.any(bad):
            break
        hi[bad] *= 2.0
        F_hi = self_financing_F(hi, P, Q, cfg)

    feasible = (F_lo <= 0.0) & (F_hi >= 0.0) & np.isfinite(F_lo) & np.isfinite(F_hi)

    # Bisection
    for _ in range(cfg.z_bisect_iter):
        mid = 0.5 * (lo + hi)
        F_mid = self_financing_F(mid, P, Q, cfg)

        go_right = F_mid < 0.0
        lo = np.where(go_right, mid, lo)
        hi = np.where(go_right, hi, mid)

    z = 0.5 * (lo + hi)
    z = np.where(feasible, z, np.nan)

    # Sanity: q=p should give z=1 up to numerical precision.
    return z, feasible

def gauss_hermite_correlated_nodes(cfg: DPBenchmarkConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns nodes and weights for E[f(Z)], Z~N(0, corr).
    Tensor-product Gauss-Hermite.

    For standard normal:
        E[f(Z)] = sum_k w_k / sqrt(pi) * f(sqrt(2) x_k)
    """
    x, w = hermgauss(cfg.gh_order)

    nodes_1d = np.sqrt(2.0) * x
    weights_1d = w / np.sqrt(np.pi)

    corr = np.array([[1.0, cfg.rho], [cfg.rho, 1.0]], dtype=float)
    L = np.linalg.cholesky(corr)

    nodes = []
    weights = []

    for i in range(cfg.gh_order):
        for j in range(cfg.gh_order):
            z_ind = np.array([nodes_1d[i], nodes_1d[j]], dtype=float)
            z_corr = L @ z_ind
            nodes.append(z_corr)
            weights.append(weights_1d[i] * weights_1d[j])

    return np.array(nodes, dtype=float), np.array(weights, dtype=float)

def next_fraction_and_gross(actions: np.ndarray, cfg: DPBenchmarkConfig, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For every action q and every quadrature node, compute:
        risky gross returns R_i
        portfolio gross return G
        next pre-trade risky fractions p_next

    Returns:
        p_next: [A,M,2]
        G:      [A,M]
        w:      [M]
    """
    q = actions
    A = q.shape[0]

    nodes, weights = gauss_hermite_correlated_nodes(cfg)
    M = nodes.shape[0]

    mu = np.asarray(cfg.mu, dtype=float)
    sig = np.asarray(cfg.sigma, dtype=float)

    drift = (mu - 0.5 * sig * sig) * dt

    R = np.empty((M, 2), dtype=float)
    for m in range(M):
        R[m, :] = np.exp(drift + sig * math.sqrt(dt) * nodes[m, :])

    Rf = math.exp(cfg.r * dt)

    qsum = q[:, 0] + q[:, 1]

    G = (
        (1.0 - qsum[:, None]) * Rf
        + q[:, 0:1] * R[None, :, 0]
        + q[:, 1:2] * R[None, :, 1]
    )

    G = np.maximum(G, 1e-300)

    p_next = np.empty((A, M, 2), dtype=float)
    p_next[:, :, 0] = q[:, 0:1] * R[None, :, 0] / G
    p_next[:, :, 1] = q[:, 1:2] * R[None, :, 1] / G

    return p_next, G, weights

def interpolate_value_on_states(states: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """
    Linear interpolation on irregular triangular state cloud.
    Falls back to nearest interpolation for rare outside/edge NaNs.
    """
    lin = LinearNDInterpolator(states, values, fill_value=np.nan)
    out = lin(query)

    if np.any(~np.isfinite(out)):
        near = NearestNDInterpolator(states, values)
        bad = ~np.isfinite(out)
        out[bad] = near(query[bad])

    return out

def solve_finite_horizon_dp(cfg: DPBenchmarkConfig) -> Dict[str, Any]:
    t0 = time.time()

    if abs(cfg.gamma) < 1e-14:
        raise NotImplementedError("This code currently handles gamma != 0 only.")

    states = make_triangular_grid(cfg)
    actions = states.copy()

    S = states.shape[0]
    A = actions.shape[0]
    dt = cfg.T / cfg.n_steps

    print("=" * 80)
    print("Finite-horizon transaction-cost DP/QVI benchmark")
    print("=" * 80)
    print(f"Number of states:  {S}")
    print(f"Number of actions: {A}")
    print(f"n_steps:           {cfg.n_steps}")
    print(f"dt:                {dt:.6f}")
    print(f"GH nodes:          {cfg.gh_order} x {cfg.gh_order} = {cfg.gh_order**2}")
    print(f"gamma:             {cfg.gamma}")
    print(f"risk aversion:     {1.0 - cfg.gamma}")
    print(f"buy_cost:          {cfg.buy_cost}")
    print(f"sell_cost:         {cfg.sell_cost}")
    print(f"rho:               {cfg.rho}")
    print(f"Merton weight:     {merton_weight(cfg)}")
    print("=" * 80)

    # Precompute self-financing trade multipliers
    print("Precomputing exact self-financing trade multipliers z(p -> q)...")
    z_mat, feasible = compute_trade_multiplier_matrix(states, actions, cfg)

    if np.any(~np.isfinite(np.diag(z_mat))):
        raise RuntimeError("Some no-trade z(p->p) entries are infeasible. Check grid/domain.")

    diag_err = np.nanmax(np.abs(np.diag(z_mat) - 1.0))
    print(f"max |z(p->p)-1| = {diag_err:.3e}")

    # For infeasible action-state pairs, set z_gamma to NaN; handled later.
    z_gamma = np.power(np.maximum(z_mat, 1e-300), cfg.gamma)
    z_gamma = np.where(feasible, z_gamma, np.nan)

    # Precompute q-driven transition p_next and gross return G
    print("Precomputing quadrature transitions for each post-trade action q...")
    p_next, G, gh_w = next_fraction_and_gross(actions, cfg, dt)

    A_, M_, _ = p_next.shape
    assert A_ == A

    # Terminal condition:
    #   Phi_T(p) = U(liquidation_multiplier(p))
    # because total wealth is normalized to 1.
    ell = liquidation_multiplier(states, cfg)
    Phi = utility_power_multiplier(ell, cfg.gamma)

    Phi_history = [Phi.copy()]
    policy_history = []
    value_hold_history = []
    value_best_history = []

    # Backward induction from T to 0.
    # Phi_n(p) = max_q z(p,q)^gamma * E[G(q,R)^gamma * Phi_{n+1}(p_next)]
    print("Running backward dynamic programming...")
    for step in range(cfg.n_steps - 1, -1, -1):
        # Interpolate next value at p_next for every q and quadrature node.
        query = p_next.reshape(-1, 2)
        Phi_next_query = interpolate_value_on_states(states, Phi, query).reshape(A, M_)

        continuation = np.sum(
            gh_w[None, :] * np.power(G, cfg.gamma) * Phi_next_query,
            axis=1,
        )  # [A]

        # Score[p,q]
        score = z_gamma * continuation[None, :]

        # Infeasible pairs -> very bad
        score = np.where(np.isfinite(score), score, -np.inf)

        best_idx = np.argmax(score, axis=1)
        best_val = score[np.arange(S), best_idx]

        hold_val = score[np.arange(S), np.arange(S)]

        Phi = best_val

        policy_history.append(best_idx.copy())
        value_hold_history.append(hold_val.copy())
        value_best_history.append(best_val.copy())
        Phi_history.append(Phi.copy())

        if (cfg.n_steps - step) % max(1, cfg.n_steps // 6) == 0 or step == 0:
            mean_gap = float(np.nanmean(best_val - hold_val))
            max_gap = float(np.nanmax(best_val - hold_val))
            print(
                f"  solved time index {step:03d} | "
                f"mean(best-hold)={mean_gap:.3e}, max(best-hold)={max_gap:.3e}"
            )

    # policy_history was appended backward.
    # Last appended corresponds to time 0.
    policy_t0 = policy_history[-1]
    best_val_t0 = value_best_history[-1]
    hold_val_t0 = value_hold_history[-1]

    q_star = actions[policy_t0]
    z_star = z_mat[np.arange(S), policy_t0]

    dollar_trade = z_star[:, None] * q_star - states

    # Discrete NTR:
    # hold is optimal if either best action is exactly p, or best-hold utility advantage is tiny.
    same_action = policy_t0 == np.arange(S)
    hold_gap = best_val_t0 - hold_val_t0

    hold_mask = same_action | (hold_gap <= cfg.hold_value_tol)

    # Action classification by dollar trade
    action1 = np.zeros(S, dtype=int)
    action2 = np.zeros(S, dtype=int)

    action1[dollar_trade[:, 0] > cfg.trade_dollar_tol] = 1
    action1[dollar_trade[:, 0] < -cfg.trade_dollar_tol] = -1

    action2[dollar_trade[:, 1] > cfg.trade_dollar_tol] = 1
    action2[dollar_trade[:, 1] < -cfg.trade_dollar_tol] = -1

    action1[hold_mask] = 0
    action2[hold_mask] = 0

    # 9-class code: (-1,0,1)^2 -> 0..8
    region_code = (action1 + 1) * 3 + (action2 + 1)

    elapsed = time.time() - t0

    result = {
        "cfg": dataclasses.asdict(cfg),
        "states": states,
        "actions": actions,
        "z_mat": z_mat,
        "feasible": feasible,
        "Phi_t0": Phi,
        "policy_t0": policy_t0,
        "q_star_t0": q_star,
        "z_star_t0": z_star,
        "dollar_trade_t0": dollar_trade,
        "hold_mask_t0": hold_mask,
        "hold_gap_t0": hold_gap,
        "best_val_t0": best_val_t0,
        "hold_val_t0": hold_val_t0,
        "action1_t0": action1,
        "action2_t0": action2,
        "region_code_t0": region_code,
        "merton": merton_weight(cfg),
        "elapsed_sec": elapsed,
        "diag_err": diag_err,
    }

    print("=" * 80)
    print(f"Done. elapsed = {elapsed:.2f} sec")
    print(f"NTR states at t=0: {int(hold_mask.sum())} / {S}")
    print("=" * 80)

    return result

def summarize_ntr_geometry(result: Dict[str, Any]) -> pd.DataFrame:
    states = result["states"]
    hold = result["hold_mask_t0"]
    m = result["merton"]

    if hold.sum() == 0:
        row = {
            "ntr_count": 0,
            "ntr_frac": 0.0,
            "ntr_xmin": np.nan,
            "ntr_xmax": np.nan,
            "ntr_ymin": np.nan,
            "ntr_ymax": np.nan,
            "merton_x": float(m[0]),
            "merton_y": float(m[1]),
            "merton_inside_discrete_ntr": False,
        }
        return pd.DataFrame([row])

    pts = states[hold]

    # nearest grid point to Merton
    d2 = np.sum((states - m[None, :]) ** 2, axis=1)
    im = int(np.argmin(d2))

    row = {
        "ntr_count": int(hold.sum()),
        "ntr_frac": float(hold.mean()),
        "ntr_xmin": float(pts[:, 0].min()),
        "ntr_xmax": float(pts[:, 0].max()),
        "ntr_ymin": float(pts[:, 1].min()),
        "ntr_ymax": float(pts[:, 1].max()),
        "ntr_centroid_x": float(pts[:, 0].mean()),
        "ntr_centroid_y": float(pts[:, 1].mean()),
        "merton_x": float(m[0]),
        "merton_y": float(m[1]),
        "nearest_grid_to_merton_x": float(states[im, 0]),
        "nearest_grid_to_merton_y": float(states[im, 1]),
        "merton_inside_discrete_ntr": bool(hold[im]),
        "merton_nearest_hold_gap": float(result["hold_gap_t0"][im]),
    }

    return pd.DataFrame([row])

def plot_dp_result(result: Dict[str, Any], outdir: Optional[Union[str, Path]] = None):
    states = result["states"]
    hold = result["hold_mask_t0"]
    region = result["region_code_t0"]
    qstar = result["q_star_t0"]
    dtrade = result["dollar_trade_t0"]
    hold_gap = result["hold_gap_t0"]
    m = result["merton"]

    tri = mtri.Triangulation(states[:, 0], states[:, 1])

    # Avoid plotting triangles outside simplex.
    xtri = states[tri.triangles, 0]
    ytri = states[tri.triangles, 1]
    tri_centers_sum = xtri.mean(axis=1) + ytri.mean(axis=1)
    tri.set_mask(tri_centers_sum > result["cfg"]["p_sum_max"] + 1e-10)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    # Panel 1: NTR
    ax = axes[0]
    cf = ax.tricontourf(tri, hold.astype(float), levels=[-0.5, 0.5, 1.5], alpha=0.75)
    ax.tricontour(tri, hold.astype(float), levels=[0.5], colors="black", linewidths=2.0)
    ax.plot(m[0], m[1], "ko", ms=6, label="Merton")
    ax.axvline(m[0], color="k", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(m[1], color="k", ls=":", lw=0.8, alpha=0.5)
    ax.set_title("Discrete finite-horizon NTR at t=0")
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend()

    # Panel 2: 9-class region
    ax = axes[1]
    cf = ax.tricontourf(tri, region.astype(float), levels=np.arange(-0.5, 9.5, 1.0), alpha=0.85)
    ax.tricontour(tri, hold.astype(float), levels=[0.5], colors="black", linewidths=2.0)
    ax.plot(m[0], m[1], "ko", ms=6, label="Merton")
    ax.set_title("9-class buy/hold/sell region")
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend()
    cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
    cbar.set_label("region code = (a1+1)*3 + (a2+1)")

    # Panel 3: hold optimality gap
    ax = axes[2]
    positive_gap = np.maximum(hold_gap, 0.0)
    cf = ax.tricontourf(tri, np.log10(positive_gap + 1e-16), levels=30, alpha=0.90)
    ax.tricontour(tri, hold.astype(float), levels=[0.5], colors="black", linewidths=2.0)
    ax.plot(m[0], m[1], "ko", ms=6, label="Merton")
    ax.set_title("log10(best value - hold value)")
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.colorbar(cf, ax=ax, shrink=0.85)

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / "finite_horizon_dp_ntr_regions.png", dpi=240)
        fig.savefig(outdir / "finite_horizon_dp_ntr_regions.pdf")

    plt.show()
    plt.close(fig)

    # Vector field / policy arrows, downsampled
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.6), constrained_layout=True)

    ax.tricontourf(tri, hold.astype(float), levels=[-0.5, 0.5, 1.5], alpha=0.35)
    ax.tricontour(tri, hold.astype(float), levels=[0.5], colors="black", linewidths=2.0)

    # Downsample arrows
    stride = max(1, len(states) // 500)
    idxs = np.arange(0, len(states), stride)

    U = qstar[idxs, 0] - states[idxs, 0]
    V = qstar[idxs, 1] - states[idxs, 1]

    ax.quiver(
        states[idxs, 0],
        states[idxs, 1],
        U,
        V,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0025,
        alpha=0.65,
    )

    ax.plot(m[0], m[1], "ko", ms=6, label="Merton")
    ax.set_title("Optimal post-trade policy q*(p) - p")
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend()

    if outdir is not None:
        fig.savefig(outdir / "finite_horizon_dp_policy_arrows.png", dpi=240)
        fig.savefig(outdir / "finite_horizon_dp_policy_arrows.pdf")

    plt.show()
    plt.close(fig)

def save_result(result: Dict[str, Any], outdir: Optional[Union[str, Path]] = None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg_dict = result["cfg"]

    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2)

    states = result["states"]

    df = pd.DataFrame({
        "p1": states[:, 0],
        "p2": states[:, 1],
        "hold": result["hold_mask_t0"].astype(int),
        "region_code": result["region_code_t0"],
        "action1": result["action1_t0"],
        "action2": result["action2_t0"],
        "q1_star": result["q_star_t0"][:, 0],
        "q2_star": result["q_star_t0"][:, 1],
        "dollar_trade1": result["dollar_trade_t0"][:, 0],
        "dollar_trade2": result["dollar_trade_t0"][:, 1],
        "hold_gap": result["hold_gap_t0"],
        "best_val": result["best_val_t0"],
        "hold_val": result["hold_val_t0"],
        "Phi_t0": result["Phi_t0"],
    })

    df.to_csv(outdir / "dp_policy_t0.csv", index=False)

    summary = summarize_ntr_geometry(result)
    summary["elapsed_sec"] = result["elapsed_sec"]
    summary["diag_max_abs_z_no_trade_minus_1"] = result["diag_err"]
    summary.to_csv(outdir / "summary.csv", index=False)

    np.savez_compressed(
        outdir / "dp_result_arrays.npz",
        states=result["states"],
        actions=result["actions"],
        Phi_t0=result["Phi_t0"],
        policy_t0=result["policy_t0"],
        q_star_t0=result["q_star_t0"],
        z_star_t0=result["z_star_t0"],
        dollar_trade_t0=result["dollar_trade_t0"],
        hold_mask_t0=result["hold_mask_t0"].astype(np.uint8),
        hold_gap_t0=result["hold_gap_t0"],
        best_val_t0=result["best_val_t0"],
        hold_val_t0=result["hold_val_t0"],
        action1_t0=result["action1_t0"],
        action2_t0=result["action2_t0"],
        region_code_t0=result["region_code_t0"],
        merton=result["merton"],
    )

    print(f"Saved outputs to: {outdir.resolve()}")
    display(summary)

def run_convergence_check(base_cfg: DPBenchmarkConfig) -> pd.DataFrame:
    rows = []

    print("=" * 80)
    print("Running convergence check")
    print("=" * 80)

    for ng in base_cfg.convergence_n_grids:
        for ns in base_cfg.convergence_n_steps:
            c = dataclasses.replace(base_cfg)
            c.n_grid = int(ng)
            c.n_steps = int(ns)
            c.run_convergence_check = False
            c.save_outputs = False

            print()
            print(f"[Convergence run] n_grid={ng}, n_steps={ns}")
            res = solve_finite_horizon_dp(c)
            summ = summarize_ntr_geometry(res).iloc[0].to_dict()

            row = {
                "n_grid": ng,
                "n_steps": ns,
                "n_states": len(res["states"]),
                "elapsed_sec": res["elapsed_sec"],
                **summ,
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    outdir = Path(base_cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "convergence_check.csv", index=False)

    print("=" * 80)
    print("Convergence summary")
    print("=" * 80)
    display(df)

    return df

def make_dp_config_from_direct_policy_cfg(cfg: Config2D) -> DPBenchmarkConfig:
    sell_cost = cfg.dp_sell_cost if cfg.dp_sell_cost is not None else (cfg.alpha, cfg.alpha)
    return DPBenchmarkConfig(
        gamma=1.0 - cfg.gamma,  # DP exponent: U(w)=w^gamma/gamma, risk aversion=1-gamma
        T=cfg.T,
        r=cfg.r,
        mu=cfg.mu,
        sigma=cfg.sigma,
        rho=cfg.rho,
        buy_cost=cfg.dp_buy_cost,
        sell_cost=sell_cost,
        n_grid=cfg.dp_n_grid,
        p1_max=cfg.dp_p1_max,
        p2_max=cfg.dp_p2_max,
        p_sum_max=cfg.dp_p_sum_max,
        n_steps=cfg.dp_n_steps,
        gh_order=cfg.dp_gh_order,
        hold_value_tol=cfg.dp_hold_value_tol,
        trade_dollar_tol=cfg.dp_trade_dollar_tol,
        outdir=os.path.join(cfg.outdir, "dp_qvi_benchmark"),
        save_outputs=True,
        run_convergence_check=cfg.dp_run_convergence_check,
        convergence_n_grids=cfg.dp_convergence_n_grids,
        convergence_n_steps=cfg.dp_convergence_n_steps,
    )

def solve_dp_qvi_benchmark_2d(cfg: Config2D) -> Dict:
    dp_cfg = make_dp_config_from_direct_policy_cfg(cfg)
    result = solve_finite_horizon_dp(dp_cfg)
    if dp_cfg.save_outputs:
        outdir = Path(dp_cfg.outdir)
        save_result(result, outdir)
        plot_dp_result(result, outdir=outdir)
    if dp_cfg.run_convergence_check:
        result["convergence_df"] = run_convergence_check(dp_cfg)
    return result

def dp_result_to_rectangular_snapshot(dp_result: Dict, cfg: Config2D) -> Dict:
    """Convert triangular finite-horizon DP/QVI output into the legacy snapshot shape
    used by the existing direct policy optimization plotting functions.
    """
    n = int(cfg.dp_snapshot_points)
    y1g = np.linspace(cfg.pi1_min, cfg.pi1_max, n)
    y2g = np.linspace(cfg.pi2_min, cfg.pi2_max, n)
    Y1, Y2 = np.meshgrid(y1g, y2g, indexing="ij")
    query = np.stack([Y1.ravel(), Y2.ravel()], axis=1)

    states = dp_result["states"]
    valid_simplex = (query[:, 0] >= -1e-14) & (query[:, 1] >= -1e-14) & (query[:, 0] + query[:, 1] <= dp_result["cfg"]["p_sum_max"] + 1e-14)

    def interp_array(values, default=np.nan):
        lin = LinearNDInterpolator(states, values, fill_value=np.nan)
        out = lin(query)
        bad = ~np.isfinite(out)
        if np.any(bad):
            near = NearestNDInterpolator(states, values)
            out[bad] = near(query[bad])
        out = np.where(valid_simplex, out, default)
        return out.reshape(n, n)

    hold_float = interp_array(dp_result["hold_mask_t0"].astype(float), default=np.nan)
    region_float = interp_array(dp_result["region_code_t0"].astype(float), default=np.nan)
    action1 = interp_array(dp_result["action1_t0"].astype(float), default=np.nan)
    action2 = interp_array(dp_result["action2_t0"].astype(float), default=np.nan)
    hold_gap = interp_array(dp_result["hold_gap_t0"].astype(float), default=np.nan)

    # Existing plotters expect ntr_mask and optional region9 on a rectangular mesh.
    # For invalid/outside-simplex cells, keep NaN to avoid contour/mesh artifacts.
    snap = {
        "time": 0.0,
        "ntr_mask": hold_float,
        "region9": region_float,
        "action1": action1,
        "action2": action2,
        "hold_gap": hold_gap,
        "Y1": Y1,
        "Y2": Y2,
    }
    return {
        "kind": "finite_horizon_dp_qvi",
        "dp_result": dp_result,
        "y1g": y1g,
        "y2g": y2g,
        "snapshots": {0: snap},
        "domain": {
            "y1_lo": float(y1g[0]),
            "y1_hi": float(y1g[-1]),
            "y2_lo": float(y2g[0]),
            "y2_hi": float(y2g[-1]),
        },
    }

FIG_DPI = 200

RECOVERED_REGION_CMAP = ListedColormap([
    "#d9d9d9",  # NT
    "#1f78b4",  # B1
    "#084081",  # S1
    "#33a02c",  # B2
    "#006d2c",  # S2
    "#e31a1c",  # B1B2
    "#7f0000",  # S1S2
    "#ff7f00",  # B1S2
    "#6a3d9a",  # S1B2
])

RECOVERED_REGION_NORM = BoundaryNorm(np.arange(-0.5, 9.5, 1.0), RECOVERED_REGION_CMAP.N)

def save_neurips_fig(fig, path, dpi=None):
    """
    Minimal save wrapper.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=(FIG_DPI if dpi is None else dpi), bbox_inches="tight")
    plt.close(fig)

def _nearest_benchmark_snapshot(dz_results: Dict, target_time: Optional[float] = None):
    """
    Return the benchmark snapshot closest to target_time.
    Supports legacy dz_results['snapshots'] structure.
    """
    if dz_results is None:
        return None

    snaps = dz_results.get("snapshots", None)
    if snaps is None or len(snaps) == 0:
        return None

    if target_time is None:
        key = min(snaps.keys())
        return snaps[key]

    keys = list(snaps.keys())
    best_key = min(
        keys,
        key=lambda k: abs(float(snaps[k].get("time", k)) - float(target_time))
    )
    return snaps[best_key]

def add_benchmark_ntr_boundary(
    ax,
    dz_results: Optional[Dict],
    cfg: "Config2D",
    target_time: Optional[float] = None,
    color: str = "black",
    linewidth: float = 1.15,
    linestyle: str = "--",
    label: str = "DP/QVI NTR boundary",
    zorder: int = 8,
):
    if dz_results is None:
        return None

    handle = None

    # Case 1: Gridded benchmark snapshots
    snap = _nearest_benchmark_snapshot(dz_results, target_time)
    if snap is not None and ("y1g" in dz_results) and ("y2g" in dz_results):
        if "ntr_mask" in snap:
            y1g = np.asarray(dz_results["y1g"])
            y2g = np.asarray(dz_results["y2g"])
            Y1g, Y2g = np.meshgrid(y1g, y2g, indexing="ij")
            Z = snap["ntr_mask"].astype(float)

            cs = ax.contour(
                Y1g, Y2g, Z,
                levels=[0.5],
                colors=[color],
                linewidths=[linewidth],
                linestyles=[linestyle],
                zorder=zorder,
            )
            # 수정한 부분: Line2D 프록시 아티팩트를 생성하여 범례에 전달
            if len(cs.collections) > 0:
                from matplotlib.lines import Line2D
                handle = Line2D([0], [0], color=color, lw=linewidth, 
                                linestyle=linestyle, label=label)
                # 현재 축에 핸들을 추가하는 대신, 함수 호출부에서 legend()가 이 핸들을 인식하게 함
                # 혹은 ax.legend() 호출 시 명시적으로 핸들을 넘겨줄 수도 있습니다.
                return handle

    # Case 2: Unstructured finite-horizon DP result at t=0
    if ("states" in dz_results) and ("hold_mask_t0" in dz_results):
        states = np.asarray(dz_results["states"])
        hold = np.asarray(dz_results["hold_mask_t0"]).astype(float)

        if states.ndim == 2 and states.shape[1] >= 2 and hold.size == states.shape[0]:
            tri = mtri.Triangulation(states[:, 0], states[:, 1])
            # (생략: 기존 mask 로직)
            cs = ax.tricontour(
                tri, hold,
                levels=[0.5],
                colors=[color],
                linewidths=[linewidth],
                linestyles=[linestyle],
                zorder=zorder,
            )
            if len(cs.collections) > 0:
                from matplotlib.lines import Line2D
                handle = Line2D([0], [0], color=color, lw=linewidth, 
                                linestyle=linestyle, label=label)
                return handle

    return None

def plot_recovered_region_pi(
    path: str,
    recovered_plane: Dict,
    dz_results: Optional[Dict],
    cfg: "Config2D"
) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.45))

    PI1 = recovered_plane["pi1_grid"]
    PI2 = recovered_plane["pi2_grid"]
    valid = recovered_plane["valid_mask"].astype(bool)
    arr = np.where(valid, recovered_plane["recovered_region"].astype(float), np.nan)
    t_val = float(recovered_plane["time"][0])

    im = ax.pcolormesh(
        PI1, PI2, arr,
        cmap=RECOVERED_REGION_CMAP,
        norm=RECOVERED_REGION_NORM,
        shading="nearest",
        rasterized=True
    )

    # Benchmark NTR boundary overlay
    h_bench =add_benchmark_ntr_boundary(
        ax=ax,
        dz_results=dz_results,
        cfg=cfg,
        target_time=t_val,
        color="black",
        linewidth=1.15,
        linestyle="--",
        label="M.Dai et al.(2010)",
        zorder=9,
    )

    #add_standard_pi_overlays(ax, cfg, add_merton=True)

    ax.set_xlim(cfg.pi1_min, cfg.pi1_max)
    ax.set_ylim(cfg.pi2_min, cfg.pi2_max)
    ax.set_xlabel(r"$y_1$")
    ax.set_ylabel(r"$y_2$")
    #ax.set_title(
    #    rf"Recovered trading region with benchmark NTR boundary, $t={t_val:.2f}$",
    #    fontsize=9
    #)
    ax.grid(alpha=0.2)

    #cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.035)
    #cbar.set_ticks(range(9))
    #cbar.set_ticklabels(
    #    ["No-Trade","Buy1","Sell1","Buy2","Sell2","Buy1-Buy2","Sell1-Sell2","Buy1-Sell2","Sell1-Buy2"],
    #    fontsize=15
    #)
    # 함수 내부 예시
    #h_bench = add_benchmark_ntr_boundary(ax=ax, ...)
    if h_bench:
        ax.legend(handles=[h_bench], loc="upper right", fontsize=9, frameon=True)
    #ax.legend(loc="upper right", fontsize=6, frameon=True)
    save_neurips_fig(fig, path)

def run_region_pipeline_2d(cfg: Config2D) -> Dict:
    """Train the direct policy and recover the final N=2 trading-region map."""
    set_seed(cfg.seed)
    device = choose_device(cfg.device)
    outdirs = make_output_dirs(cfg.outdir)
    save_json(asdict(cfg), os.path.join(outdirs["base"], "config.json"))

    policy = PolicyNet2D(hidden=cfg.hidden, depth=cfg.depth).to(device)
    train_history = train_direct_policy_2d(policy, cfg, device)

    dp_raw_result = solve_dp_qvi_benchmark_2d(cfg)
    benchmark_results = dp_result_to_rectangular_snapshot(dp_raw_result, cfg)

    recovered_plane = compute_recovered_region_plane_pi_at_time(
        policy, cfg, device, cfg.plane_time
    )

    plot_recovered_region_pi(
        os.path.join(outdirs["figures_main"], "recovered_region_pi.png"),
        recovered_plane,
        benchmark_results,
        cfg,
    )
    arrays = {k: v for k, v in recovered_plane.items() if isinstance(v, np.ndarray)}
    save_npz(
        os.path.join(outdirs["data"], "recovered_region_pi.npz"),
        **arrays,
    )

    return {
        "cfg": cfg,
        "device": str(device),
        "outdirs": outdirs,
        "policy": policy,
        "train_history": train_history,
        "dp_raw_result": dp_raw_result,
        "benchmark_results": benchmark_results,
        "recovered_plane": recovered_plane,
    }

if __name__ == "__main__":
    cfg = Config2D(
        T=1.0,
        n_steps=20,
        r=0.02,
        mu=(0.08, 0.10),
        sigma=(0.20, 0.25),
        rho=0.0,
        gamma=3.0,
        alpha=0.03,
        eps_quad=0.01,
        u_max=10.0,
        hidden=256,
        depth=3,
        batch_size=256,
        n_train_steps=1000,
        lr=1e-4,
        print_every=200,
        outer_eval_paths=128,
        inner_mc_paths=256,
        eval_chunk_size=64,
        x0_eval=0.30,
        y10_eval=0.35,
        y20_eval=0.35,
        plane_time=0.0,
        plane_points=101,
        plane_chunk_size=128,
        pi1_min=0.0,
        pi1_max=0.80,
        pi2_min=0.0,
        pi2_max=0.80,
        plane_ref_W=1.0,
        region_tol=5e-2,
        seed=80,
        device="auto",
        outdir="paper_2asset_region",
    )
    RESULTS = run_region_pipeline_2d(cfg)

