# ============================================================
# ONE-CELL, SELF-CONTAINED PIPELINE
# 3-ASSET PORTFOLIO OPTIMIZATION WITH TX COST
# MODIFIED AS REQUESTED:
#   - TRAINING FEATURES:
#         W = x + y1 + y2 + y3
#         p_i = y_i / W
#         feature = (t/T, log W, p1, p2, p3)
#   - VISUALIZATION AXES:
#         W = 1 cut, coordinates are (y1, y2, y3)
#         hence x = 1 - y1 - y2 - y3
#   - IMPORTANT:
#         x < 0 IS ALLOWED in visualization volume
#         so NO valid-mask restriction like x >= 0
#         domain is full [0,1]^3 in (y1,y2,y3)
# ============================================================

import os
import math
import json
import time
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import torch
import torch.nn as nn
import torch.optim as optim

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

def set_neurips_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
    })
    
# ============================================================
# Configuration
# ============================================================

@dataclass
class Config3D:
    T: float = 1.0
    n_steps: int = 40
    r: float = 0.02
    mu: Tuple[float, float, float] = (0.09, 0.10, 0.11)
    sigma: Tuple[float, float, float] = (0.2, 0.25, 0.30)
    corr: Tuple[Tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.00, 1.0),
    )
    gamma: float = 3.0

    alpha: float = 0.02
    eps_quad: float = 0.02
    u_max: float = 10.0

    hidden: int = 128
    depth: int = 3
    batch_size: int = 256
    n_train_steps: int = 1000
    lr: float = 1e-3
    print_every: int = 200

    init_pi_sum_high: float = 1.0
    logw_low: float = math.log(0.8)
    logw_high: float = math.log(1.2)

    outer_eval_paths: int = 256
    inner_mc_paths: int = 128
    eval_chunk_size: int = 64
    lambda_floor: float = 1e-10

    x0_eval: float = 0.25
    y0_eval: Tuple[float, float, float] = (0.25, 0.25, 0.25)

    use_common_random_numbers: bool = True
    use_antithetic: bool = True

    plane_time: float = 0.0
    plane_W_value: float = 1.0   # W = x + y1 + y2 + y3 = plane_W_value cut

    # visualization domain in (y1, y2, y3)
    y1_min: float = 0.0
    y1_max: float = 1.0
    y2_min: float = 0.0
    y2_max: float = 1.0
    y3_min: float = 0.0
    y3_max: float = 1.0

    volume_points: int = 31
    volume_chunk_size: int = 384

    pmp_u_region_tol: float = 0.005

    surface_alpha: float = 0.9
    zero_alpha: float = 0.5
    ntr_alpha: float = 0.9
    zero_level_eps: float = 1e-8

    seed: int = 7
    device: str = "auto"
    outdir: str = "paper_3asset_Wcut_y_axes"

    @property
    def dt(self) -> float:
        return self.T / self.n_steps

    @property
    def dim(self) -> int:
        return 3


# ============================================================
# Models
# ============================================================

class PolicyNet3D(nn.Module):
    def __init__(self, hidden: int = 96, depth: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = 5   # t/T, logW, p1, p2, p3
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.Tanh())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Utilities
# ============================================================

def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)

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

def utility(x: torch.Tensor, gamma: float) -> torch.Tensor:
    x = torch.clamp(x, min=1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return torch.log(x)
    return (x.pow(1.0 - gamma) - 1.0) / (1.0 - gamma)

# liquidation wealth: used in objective / feasibility / PMP scaling
def liquidation_wealth_3d(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    return x + (1.0 - alpha) * y.sum(dim=-1)

def liquidation_wealth_np_3d(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, y3: np.ndarray, alpha: float) -> np.ndarray:
    return x + (1.0 - alpha) * (y1 + y2 + y3)

# total wealth W: used in training features and W=1 visualization cut
def total_wealth_3d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y.sum(dim=-1)

def total_wealth_np_3d(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, y3: np.ndarray) -> np.ndarray:
    return x + y1 + y2 + y3

def corr_matrix_3d(cfg: Config3D) -> np.ndarray:
    return np.array(cfg.corr, dtype=np.float64)

def cov_matrix_3d(cfg: Config3D) -> np.ndarray:
    sigma = np.asarray(cfg.sigma, dtype=np.float64)
    D = np.diag(sigma)
    C = corr_matrix_3d(cfg)
    return D @ C @ D

def chol_corr_torch(cfg: Config3D, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    C = torch.tensor(corr_matrix_3d(cfg), dtype=dtype, device=device)
    return torch.linalg.cholesky(C)

def make_correlated_dW(z: torch.Tensor, cfg: Config3D) -> torch.Tensor:
    chol = chol_corr_torch(cfg, dtype=z.dtype, device=z.device)
    return z @ chol.T

def sample_standard_normals(shape: Tuple[int, ...], device: torch.device, seed: Optional[int] = None) -> torch.Tensor:
    if seed is None:
        return torch.randn(shape, device=device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.randn(shape, generator=gen, device="cpu").to(device)

def _base_noise_count(m: int, antithetic: bool) -> Tuple[int, int]:
    if not antithetic:
        return int(m), int(m)
    half = int(math.ceil(m / 2))
    total = 2 * half
    return half, total

def build_shared_noise_bank(start_step: int, cfg: Config3D, device: torch.device, m: Optional[int] = None) -> Optional[torch.Tensor]:
    if not cfg.use_common_random_numbers:
        return None
    remaining = cfg.n_steps - int(start_step)
    if remaining <= 0:
        return None
    m_eff = cfg.inner_mc_paths if m is None else int(m)
    base_count, total_count = _base_noise_count(m_eff, cfg.use_antithetic)
    z = sample_standard_normals((remaining, base_count, cfg.dim), device=device, seed=cfg.seed + 1000 * (start_step + 1))
    base = make_correlated_dW(z, cfg)
    if cfg.use_antithetic:
        noise = torch.cat([base, -base], dim=1)[:, :total_count, :]
    else:
        noise = base
    return noise

def build_forward_noise_bank(start_step: int, cfg: Config3D, device: torch.device, batch_size: int) -> Optional[torch.Tensor]:
    if not cfg.use_common_random_numbers:
        return None
    remaining = cfg.n_steps - int(start_step)
    if remaining <= 0:
        return None
    z = sample_standard_normals((remaining, int(batch_size), cfg.dim), device=device, seed=cfg.seed + 2000 * (start_step + 1) + 17)
    return make_correlated_dW(z, cfg)


# ============================================================
# Features / state sampling
# ============================================================

# MODIFIED:
# feature = (t/T, logW, p1, p2, p3), where W = x + y1 + y2 + y3, p_i = y_i / W
def state_features_3d(t_frac: torch.Tensor, x: torch.Tensor, y: torch.Tensor, cfg: Config3D) -> torch.Tensor:
    W = total_wealth_3d(x, y)
    logW = torch.log(torch.clamp(W, min=1e-12))
    p = y / torch.clamp(W.unsqueeze(-1), min=1e-12)
    return torch.cat([t_frac.unsqueeze(-1), logW.unsqueeze(-1), p], dim=-1)

def sample_initial_states_3d(cfg: Config3D, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    logw = torch.empty(batch_size, device=device).uniform_(cfg.logw_low, cfg.logw_high)
    W = torch.exp(logw)

    u = torch.rand(batch_size, cfg.dim, device=device)
    row_sum = torch.clamp(u.sum(dim=-1, keepdim=True), min=1e-12)
    direction = u / row_sum
    scale = torch.empty(batch_size, 1, device=device).uniform_(0.0, cfg.init_pi_sum_high)
    pi = direction * scale

    y = pi * W.unsqueeze(-1)
    x = W - y.sum(dim=-1)
    return x, y


# ============================================================
# Feasibility / dynamics
# ============================================================

def is_feasible_post_trade(x: torch.Tensor, y: torch.Tensor, u: torch.Tensor, cfg: Config3D) -> torch.Tensor:
    # NOTE:
    # current feasibility still uses liquidation wealth scaling and y>=0, L_trade>0.
    # this is separate from the modified input feature / visualization coordinates.
    L = liquidation_wealth_3d(x, y, cfg.alpha)
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
    cond = (y_trade >= -1e-12).all(dim=-1) & (L_trade > 1e-12)
    return cond

def project_rate_vector(u_raw: torch.Tensor, x: torch.Tensor, y: torch.Tensor, cfg: Config3D) -> torch.Tensor:
    u = torch.clamp(u_raw, min=-cfg.u_max, max=cfg.u_max)
    L = liquidation_wealth_3d(x, y, cfg.alpha)
    L_safe = torch.clamp(L, min=1e-12)

    lower = -y / torch.clamp(L_safe.unsqueeze(-1) * cfg.dt, min=1e-12)
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

def policy_action_3d(policy: PolicyNet3D, t_frac: torch.Tensor, x: torch.Tensor, y: torch.Tensor, cfg: Config3D) -> torch.Tensor:
    inp = state_features_3d(t_frac, x, y, cfg)
    raw = policy(inp)
    u = cfg.u_max * torch.tanh(raw)
    return project_rate_vector(u, x, y, cfg)

def step_dynamics_from_rate_3d(x: torch.Tensor, y: torch.Tensor, u: torch.Tensor, dW: torch.Tensor, cfg: Config3D) -> Tuple[torch.Tensor, torch.Tensor]:
    L = liquidation_wealth_3d(x, y, cfg.alpha)
    L_safe = torch.clamp(L, min=1e-12)

    u = project_rate_vector(u, x, y, cfg)
    buy = torch.relu(u)
    sell = torch.relu(-u)
    dt = cfg.dt

    x_trade = (
        x
        - L_safe * buy.sum(dim=-1) * dt
        + (1.0 - cfg.alpha) * L_safe * sell.sum(dim=-1) * dt
        - 0.5 * cfg.eps_quad * L_safe * (u ** 2).sum(dim=-1) * dt
    )
    y_trade = y + L_safe.unsqueeze(-1) * u * dt
    y_trade = torch.clamp(y_trade, min=0.0)

    x_next = x_trade * math.exp(cfg.r * dt)

    mu_t = torch.tensor(cfg.mu, dtype=y.dtype, device=y.device).view(1, cfg.dim)
    sigma_t = torch.tensor(cfg.sigma, dtype=y.dtype, device=y.device).view(1, cfg.dim)
    expo = (mu_t - 0.5 * sigma_t ** 2) * dt + sigma_t * math.sqrt(dt) * dW
    y_next = y_trade * torch.exp(expo)
    return x_next, y_next

def rollout_policy_3d(policy: PolicyNet3D, x0: torch.Tensor, y0: torch.Tensor, start_step: int, noise_bank: Optional[torch.Tensor], cfg: Config3D) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x0
    y = y0
    batch = x.shape[0]
    device = x.device
    remaining = cfg.n_steps - int(start_step)

    for j in range(remaining):
        step = int(start_step) + j
        t_frac = torch.full((batch,), float(step) / cfg.n_steps, device=device)
        u = policy_action_3d(policy, t_frac, x, y, cfg)

        if noise_bank is None:
            dW = make_correlated_dW(torch.randn(batch, cfg.dim, device=device), cfg)
        else:
            dW = noise_bank[j]

        x, y = step_dynamics_from_rate_3d(x, y, u, dW, cfg)

    return x, y


# ============================================================
# Warm-up training / evaluation
# ============================================================

def train_baseline_pg_dpo_3d(policy: PolicyNet3D, cfg: Config3D, device: torch.device) -> Dict[str, List[float]]:
    opt = optim.Adam(policy.parameters(), lr=cfg.lr)
    curve_steps, curve_vals = [], []

    for step in range(cfg.n_train_steps + 1):
        x0, y0 = sample_initial_states_3d(cfg, cfg.batch_size, device)
        xT, yT = rollout_policy_3d(policy, x0, y0, 0, None, cfg)
        terminal_liq = liquidation_wealth_3d(xT, yT, cfg.alpha)
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

def evaluate_baseline_vectorized_3d(policy: PolicyNet3D, cfg: Config3D, device: torch.device) -> Dict[str, float]:
    reps = int(cfg.outer_eval_paths)
    chunk_size = max(1, int(cfg.eval_chunk_size))
    liq_chunks = []

    for start in range(0, reps, chunk_size):
        bsz = min(chunk_size, reps - start)
        x0 = torch.full((bsz,), cfg.x0_eval, device=device)
        y0 = torch.tensor(cfg.y0_eval, dtype=torch.float32, device=device).view(1, cfg.dim).repeat(bsz, 1)
        outer_bank = build_forward_noise_bank(0, cfg, device=device, batch_size=bsz) if cfg.use_common_random_numbers else None
        xT, yT = rollout_policy_3d(policy, x0, y0, 0, outer_bank, cfg)
        liq_chunks.append(liquidation_wealth_3d(xT, yT, cfg.alpha).detach().cpu())

    liqT = torch.cat(liq_chunks, dim=0)
    return {
        "EU": float(utility(liqT, cfg.gamma).mean().item()),
        "mean_liq": float(liqT.mean().item()),
        "std_liq": float(liqT.std(unbiased=False).item()),
    }


# ============================================================
# PMP projector
# ============================================================

def _expand_shared_noise_3d(noise_bank: torch.Tensor, batch_states: int) -> torch.Tensor:
    remaining, m_paths, d = noise_bank.shape
    return noise_bank.repeat_interleave(batch_states, dim=1)

def pmp_projected_action_batch_3d(policy: PolicyNet3D, x: torch.Tensor, y: torch.Tensor, t_value: float, cfg: Config3D, device: torch.device, noise_bank: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    y_rep = y.view(1, batch_states, cfg.dim).repeat(m_paths, 1, 1).reshape(-1, cfg.dim).clone().detach().requires_grad_(True)

    if noise_bank is None:
        remaining = cfg.n_steps - start_step
        if remaining > 0:
            base_count, total_count = _base_noise_count(m_eff, cfg.use_antithetic)
            z = torch.randn((remaining, base_count, batch_states, cfg.dim), device=device)
            base = make_correlated_dW(z, cfg)
            if cfg.use_antithetic:
                noise = torch.cat([base, -base], dim=1)[:, :total_count, :, :]
            else:
                noise = base
            noise = noise.permute(0, 2, 1, 3).reshape(remaining, batch_states * m_paths, cfg.dim)
        else:
            noise = None
    else:
        noise = _expand_shared_noise_3d(noise_bank[:, :m_paths, :], batch_states)

    xT, yT = rollout_policy_3d(policy, x_rep, y_rep, start_step, noise, cfg)
    payoff = utility(liquidation_wealth_3d(xT, yT, cfg.alpha), cfg.gamma)
    grad_x, grad_y = torch.autograd.grad(payoff.sum(), [x_rep, y_rep], create_graph=False)

    lam_x = grad_x.view(m_paths, batch_states).mean(dim=0)
    lam_y = grad_y.view(m_paths, batch_states, cfg.dim).mean(dim=0)

    lam_x_safe = torch.clamp(lam_x, min=cfg.lambda_floor).unsqueeze(-1)
    buy_gap = lam_y - lam_x.unsqueeze(-1)
    sell_gap = (1.0 - cfg.alpha) * lam_x.unsqueeze(-1) - lam_y

    denom = torch.clamp(cfg.eps_quad * lam_x_safe, min=cfg.lambda_floor)
    u_buy = torch.relu(buy_gap / denom)
    u_sell = torch.relu(sell_gap / denom)
    u_pmp = u_buy - u_sell
    u_pmp = project_rate_vector(u_pmp, x.detach(), y.detach(), cfg)

    return u_pmp.detach(), lam_x.detach(), lam_y.detach(), buy_gap.detach(), sell_gap.detach()

def evaluate_pmp_vectorized_3d(policy: PolicyNet3D, cfg: Config3D, device: torch.device) -> Dict[str, float]:
    reps = int(cfg.outer_eval_paths)
    chunk_size = max(1, int(cfg.eval_chunk_size))
    liq_chunks = []

    for start in range(0, reps, chunk_size):
        bsz = min(chunk_size, reps - start)
        x = torch.full((bsz,), cfg.x0_eval, device=device)
        y = torch.tensor(cfg.y0_eval, dtype=torch.float32, device=device).view(1, cfg.dim).repeat(bsz, 1)
        outer_bank = build_forward_noise_bank(0, cfg, device=device, batch_size=bsz) if cfg.use_common_random_numbers else None

        for step in range(cfg.n_steps):
            t_val = step * cfg.dt
            inner_bank = build_shared_noise_bank(step, cfg, device=device, m=cfg.inner_mc_paths) if cfg.use_common_random_numbers else None
            u, _, _, _, _ = pmp_projected_action_batch_3d(policy, x, y, t_val, cfg, device, noise_bank=inner_bank)
            if outer_bank is None:
                dW = make_correlated_dW(torch.randn(bsz, cfg.dim, device=device), cfg)
            else:
                dW = outer_bank[step, :bsz, :]
            x, y = step_dynamics_from_rate_3d(x, y, u, dW, cfg)

        liq_chunks.append(liquidation_wealth_3d(x, y, cfg.alpha).detach().cpu())

    liqT = torch.cat(liq_chunks, dim=0)
    return {
        "EU": float(utility(liqT, cfg.gamma).mean().item()),
        "mean_liq": float(liqT.mean().item()),
        "std_liq": float(liqT.std(unbiased=False).item()),
    }


# ============================================================
# Region helper / W=1 cut volume on (y1, y2, y3)
# ============================================================

def sign3(v: np.ndarray, tol: float) -> np.ndarray:
    out = np.zeros_like(v, dtype=np.int64)
    out[v > tol] = 1
    out[v < -tol] = -1
    return out

def classify_pmp_u_region_27(u1: np.ndarray, u2: np.ndarray, u3: np.ndarray, tol: float) -> np.ndarray:
    a1 = sign3(u1, tol)
    a2 = sign3(u2, tol)
    a3 = sign3(u3, tol)
    return (a1 + 1) * 9 + (a2 + 1) * 3 + (a3 + 1)

# MODIFIED:
# visualization coordinates are (Y1, Y2, Y3) on W = 1 cut
# so X = W0 - (Y1 + Y2 + Y3)
def Wcut_y_to_x(Y1: np.ndarray, Y2: np.ndarray, Y3: np.ndarray, cfg: Config3D):
    W0 = float(cfg.plane_W_value)
    X = W0 - (Y1 + Y2 + Y3)
    return X

# MODIFIED:
# no valid mask restriction; full [0,1]^3 is used even if x < 0
def full_volume_mask(Y1: np.ndarray, Y2: np.ndarray, Y3: np.ndarray, cfg: Config3D) -> np.ndarray:
    return np.ones_like(Y1, dtype=bool)

def compute_pmp_volume_at_time(policy: PolicyNet3D, cfg: Config3D, device: torch.device, time_value: float) -> Dict[str, np.ndarray]:
    y1s = np.linspace(cfg.y1_min, cfg.y1_max, cfg.volume_points, dtype=np.float64)
    y2s = np.linspace(cfg.y2_min, cfg.y2_max, cfg.volume_points, dtype=np.float64)
    y3s = np.linspace(cfg.y3_min, cfg.y3_max, cfg.volume_points, dtype=np.float64)

    Y1, Y2, Y3 = np.meshgrid(y1s, y2s, y3s, indexing="ij")
    valid = full_volume_mask(Y1, Y2, Y3, cfg)
    X = Wcut_y_to_x(Y1, Y2, Y3, cfg)

    W_grid = total_wealth_np_3d(X, Y1, Y2, Y3)  # should be identically plane_W_value
    L_grid = liquidation_wealth_np_3d(X, Y1, Y2, Y3, cfg.alpha)

    shp = Y1.shape
    pmp_u1 = np.full(shp, np.nan, dtype=np.float64)
    pmp_u2 = np.full(shp, np.nan, dtype=np.float64)
    pmp_u3 = np.full(shp, np.nan, dtype=np.float64)
    lam_x = np.full(shp, np.nan, dtype=np.float64)
    lam_y1 = np.full(shp, np.nan, dtype=np.float64)
    lam_y2 = np.full(shp, np.nan, dtype=np.float64)
    lam_y3 = np.full(shp, np.nan, dtype=np.float64)
    buy_gap1 = np.full(shp, np.nan, dtype=np.float64)
    sell_gap1 = np.full(shp, np.nan, dtype=np.float64)
    buy_gap2 = np.full(shp, np.nan, dtype=np.float64)
    sell_gap2 = np.full(shp, np.nan, dtype=np.float64)
    buy_gap3 = np.full(shp, np.nan, dtype=np.float64)
    sell_gap3 = np.full(shp, np.nan, dtype=np.float64)
    trade_gap = np.full(shp, np.nan, dtype=np.float64)
    pmp_region = np.full(shp, -1, dtype=np.int64)

    coords = np.argwhere(valid)
    if coords.size == 0:
        raise RuntimeError("No points in plotting volume.")

    step = int(round(float(time_value) / cfg.dt))
    step = max(0, min(cfg.n_steps - 1, step))
    shared_noise = build_shared_noise_bank(step, cfg, device=device) if cfg.use_common_random_numbers else None

    for start in range(0, coords.shape[0], cfg.volume_chunk_size):
        chunk = coords[start:start + cfg.volume_chunk_size]
        i = chunk[:, 0]
        j = chunk[:, 1]
        k = chunk[:, 2]

        x_t = torch.tensor(X[i, j, k], dtype=torch.float32, device=device)
        y_t = torch.tensor(np.stack([Y1[i, j, k], Y2[i, j, k], Y3[i, j, k]], axis=-1), dtype=torch.float32, device=device)

        pmp_u, lx, ly, bg, sg = pmp_projected_action_batch_3d(
            policy=policy,
            x=x_t,
            y=y_t,
            t_value=float(step * cfg.dt),
            cfg=cfg,
            device=device,
            noise_bank=shared_noise,
        )

        pmp_u_np = pmp_u.cpu().numpy()
        lx_np = lx.cpu().numpy()
        ly_np = ly.cpu().numpy()
        bg_np = bg.cpu().numpy()
        sg_np = sg.cpu().numpy()

        max_bg = np.maximum.reduce([bg_np[:, 0], bg_np[:, 1], bg_np[:, 2]])
        max_sg = np.maximum.reduce([sg_np[:, 0], sg_np[:, 1], sg_np[:, 2]])
        tg = np.maximum(max_bg, max_sg)

        lbl27 = classify_pmp_u_region_27(
            pmp_u_np[:, 0],
            pmp_u_np[:, 1],
            pmp_u_np[:, 2],
            cfg.pmp_u_region_tol,
        )

        pmp_u1[i, j, k] = pmp_u_np[:, 0]
        pmp_u2[i, j, k] = pmp_u_np[:, 1]
        pmp_u3[i, j, k] = pmp_u_np[:, 2]
        lam_x[i, j, k] = lx_np
        lam_y1[i, j, k] = ly_np[:, 0]
        lam_y2[i, j, k] = ly_np[:, 1]
        lam_y3[i, j, k] = ly_np[:, 2]
        buy_gap1[i, j, k] = bg_np[:, 0]
        sell_gap1[i, j, k] = sg_np[:, 0]
        buy_gap2[i, j, k] = bg_np[:, 1]
        sell_gap2[i, j, k] = sg_np[:, 1]
        buy_gap3[i, j, k] = bg_np[:, 2]
        sell_gap3[i, j, k] = sg_np[:, 2]
        trade_gap[i, j, k] = tg
        pmp_region[i, j, k] = lbl27

        if start == 0 or ((start // cfg.volume_chunk_size) % 25 == 0):
            print(f"[volume] processed {min(start + len(chunk), len(coords))}/{len(coords)} grid points")

    return {
        "time": np.array([float(step * cfg.dt)], dtype=np.float64),
        "y1_grid": Y1,
        "y2_grid": Y2,
        "y3_grid": Y3,
        "x_grid": X,
        "W_grid": W_grid,
        "L_grid": L_grid,
        "valid_mask": valid.astype(np.uint8),   # all ones by construction
        "pmp_u1": pmp_u1,
        "pmp_u2": pmp_u2,
        "pmp_u3": pmp_u3,
        "lam_x": lam_x,
        "lam_y1": lam_y1,
        "lam_y2": lam_y2,
        "lam_y3": lam_y3,
        "buy_gap1": buy_gap1,
        "sell_gap1": sell_gap1,
        "buy_gap2": buy_gap2,
        "sell_gap2": sell_gap2,
        "buy_gap3": buy_gap3,
        "sell_gap3": sell_gap3,
        "trade_gap": trade_gap,
        "pmp_region": pmp_region,
    }


# ============================================================
# Isosurface extraction: marching tetrahedra on regular grid
# ============================================================

_TETRAS = np.array([
    [0, 5, 1, 6],
    [0, 1, 2, 6],
    [0, 2, 3, 6],
    [0, 3, 7, 6],
    [0, 7, 4, 6],
    [0, 4, 5, 6],
], dtype=np.int64)

_CUBE_CORNER_OFFSETS = np.array([
    [0, 0, 0],  # 0
    [1, 0, 0],  # 1
    [1, 1, 0],  # 2
    [0, 1, 0],  # 3
    [0, 0, 1],  # 4
    [1, 0, 1],  # 5
    [1, 1, 1],  # 6
    [0, 1, 1],  # 7
], dtype=np.int64)

_TET_EDGES = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

def _interp_iso_point(p1, p2, v1, v2, level):
    denom = (v2 - v1)
    if abs(denom) < 1e-14:
        t = 0.5
    else:
        t = (level - v1) / denom
    t = min(max(t, 0.0), 1.0)
    return p1 + t * (p2 - p1)

def _triangulate_tetra(coords4, vals4, level):
    inside = vals4 >= level
    n_inside = int(np.sum(inside))
    if n_inside == 0 or n_inside == 4:
        return []

    pts = []
    for a, b in _TET_EDGES:
        va, vb = vals4[a], vals4[b]
        cross = ((va >= level) and (vb < level)) or ((va < level) and (vb >= level))
        if cross:
            pts.append(_interp_iso_point(coords4[a], coords4[b], va, vb, level))

    if len(pts) < 3:
        return []
    if len(pts) == 3:
        return [np.stack(pts, axis=0)]
    if len(pts) == 4:
        return [
            np.stack([pts[0], pts[1], pts[2]], axis=0),
            np.stack([pts[0], pts[2], pts[3]], axis=0),
        ]
    out = []
    for i in range(1, len(pts)-1):
        out.append(np.stack([pts[0], pts[i], pts[i+1]], axis=0))
    return out

def extract_isosurface_regular_grid(Xg, Yg, Zg, F, level):
    assert Xg.shape == Yg.shape == Zg.shape == F.shape
    nx, ny, nz = F.shape
    tris = []

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                corner_idx = _CUBE_CORNER_OFFSETS + np.array([i, j, k])[None, :]
                coords8 = np.stack([
                    Xg[corner_idx[:,0], corner_idx[:,1], corner_idx[:,2]],
                    Yg[corner_idx[:,0], corner_idx[:,1], corner_idx[:,2]],
                    Zg[corner_idx[:,0], corner_idx[:,1], corner_idx[:,2]],
                ], axis=1)
                vals8 = F[corner_idx[:,0], corner_idx[:,1], corner_idx[:,2]]

                if np.any(np.isnan(vals8)):
                    continue
                vmin = np.min(vals8)
                vmax = np.max(vals8)
                if not (vmin <= level <= vmax):
                    continue

                for tet in _TETRAS:
                    coords4 = coords8[tet]
                    vals4 = vals8[tet]
                    tri_list = _triangulate_tetra(coords4, vals4, level)
                    tris.extend(tri_list)

    if len(tris) == 0:
        return np.zeros((0, 3, 3), dtype=np.float64)
    return np.stack(tris, axis=0)

def add_tri_mesh(ax, tris, alpha=0.25, facecolor="lightgray", edgecolor="none", linewidth=0.0):
    if tris.shape[0] == 0:
        return
    mesh = Poly3DCollection(tris, alpha=alpha, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_collection3d(mesh)

def set_axes_equal_3d(ax, xlim, ylim, zlim):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    xr = xlim[1] - xlim[0]
    yr = ylim[1] - ylim[0]
    zr = zlim[1] - zlim[0]
    r = max(xr, yr, zr)
    xm = 0.5 * (xlim[0] + xlim[1])
    ym = 0.5 * (ylim[0] + ylim[1])
    zm = 0.5 * (zlim[0] + zlim[1])

    ax.set_xlim((xm - r/2, xm + r/2))
    ax.set_ylim((ym - r/2, ym + r/2))
    ax.set_zlim((zm - r/2, zm + r/2))

def save_3d_isosurface(tris, cfg: Config3D, save_path: str, title: str,
                       facecolor="lightgray", alpha=0.22,
                       elev=24, azim=38):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    add_tri_mesh(ax, tris, alpha=alpha, facecolor=facecolor, edgecolor="none", linewidth=0.0)

    ax.set_xlabel(r"$y_1$")
    ax.set_ylabel(r"$y_2$")
    ax.set_zlabel(r"$y_3$")
    ax.set_title(title)

    set_axes_equal_3d(
        ax,
        (cfg.y1_min, cfg.y1_max),
        (cfg.y2_min, cfg.y2_max),
        (cfg.y3_min, cfg.y3_max),
    )
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)

def save_multi_isosurface(tris_list, cfg: Config3D, save_path: str, title: str,
                          facecolors=("lightgray", "gray", "dimgray"),
                          alphas=(0.20, 0.22, 0.20),
                          elev=24, azim=38):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    for tris, fc, aa in zip(tris_list, facecolors, alphas):
        add_tri_mesh(ax, tris, alpha=aa, facecolor=fc, edgecolor="none", linewidth=0.0)

    ax.set_xlabel(r"$y_1$")
    ax.set_ylabel(r"$y_2$")
    ax.set_zlabel(r"$y_3$")
    ax.set_title(title)

    set_axes_equal_3d(
        ax,
        (cfg.y1_min, cfg.y1_max),
        (cfg.y2_min, cfg.y2_max),
        (cfg.y3_min, cfg.y3_max),
    )
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)

def plot_training_curve(history: Dict[str, List[float]], save_path: str):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(history["steps"], history["values"])
    plt.xlabel("train step")
    plt.ylabel(r"$\mathbb{E}[U(L_T)]$")
    plt.title("Training curve")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)

def plot_signed_isosurfaces(volume: Dict[str, np.ndarray], cfg: Config3D, field_name: str, save_path: str):
    Y1 = volume["y1_grid"]
    Y2 = volume["y2_grid"]
    Y3 = volume["y3_grid"]
    F = volume[field_name]
    tol = cfg.pmp_u_region_tol

    tris_neg = extract_isosurface_regular_grid(Y1, Y2, Y3, F, -tol)
    tris_zero = extract_isosurface_regular_grid(Y1, Y2, Y3, F, 0.0)
    tris_pos = extract_isosurface_regular_grid(Y1, Y2, Y3, F, +tol)

    save_multi_isosurface(
        [tris_neg, tris_zero, tris_pos],
        cfg,
        save_path,
        title=rf"{field_name} signed isosurfaces on $W=1$ cut in $(y_1,y_2,y_3)$",
        facecolors=("silver", "gray", "black"),
        alphas=(0.16, 0.22, 0.16),
    )
    return {
        "neg": tris_neg.shape[0],
        "zero": tris_zero.shape[0],
        "pos": tris_pos.shape[0],
    }

def plot_zero_isosurface(volume: Dict[str, np.ndarray], cfg: Config3D, field_name: str, save_path: str):
    Y1 = volume["y1_grid"]
    Y2 = volume["y2_grid"]
    Y3 = volume["y3_grid"]
    F = volume[field_name]

    tris_zero = extract_isosurface_regular_grid(Y1, Y2, Y3, F, 0.0)
    save_3d_isosurface(
        tris_zero,
        cfg,
        save_path,
        title=rf"{field_name}=0 isosurface on $W=1$ cut in $(y_1,y_2,y_3)$",
        facecolor="lightgray",
        alpha=cfg.zero_alpha,
    )
    return tris_zero.shape[0]

def plot_ntr_boundary(volume: Dict[str, np.ndarray], cfg: Config3D, save_path: str):
    Y1 = volume["y1_grid"]
    Y2 = volume["y2_grid"]
    Y3 = volume["y3_grid"]
    u1 = volume["pmp_u1"]
    u2 = volume["pmp_u2"]
    u3 = volume["pmp_u3"]
    F = np.maximum.reduce([np.abs(u1), np.abs(u2), np.abs(u3)]) - cfg.pmp_u_region_tol

    tris = extract_isosurface_regular_grid(Y1, Y2, Y3, F, 0.0)
    save_3d_isosurface(
        tris,
        cfg,
        save_path,
        title=r"NTR boundary: $\max(|u_1|,|u_2|,|u_3|)=\mathrm{tol}$ on $W=1$ cut",
        facecolor="lightgray",
        alpha=cfg.ntr_alpha,
    )
    return tris.shape[0]

def save_volume_npz(volume: Dict[str, np.ndarray], path: str):
    np.savez_compressed(path, **volume)


# ============================================================
# Full pipeline
# ============================================================

set_neurips_style()

def run_full_pipeline_3d_paper_axis(cfg: Config3D):
    set_seed(cfg.seed)
    device = choose_device(cfg.device)

    outdir = cfg.outdir
    fig_dir = os.path.join(outdir, "figures")
    data_dir = os.path.join(outdir, "data")
    ensure_dir(outdir)
    ensure_dir(fig_dir)
    ensure_dir(data_dir)

    print(f"[info] device = {device}")
    print(f"[info] output dir = {outdir}")
    print(f"[info] visualization domain = [0,1]^3 in (y1,y2,y3)")
    print(f"[info] W-cut plane: W = x + y1 + y2 + y3 = {cfg.plane_W_value}")
    print(f"[info] x<0 allowed in volume construction")

    policy = PolicyNet3D(hidden=cfg.hidden, depth=cfg.depth).to(device)

    # 1) warm-up train
    history = train_baseline_pg_dpo_3d(policy, cfg, device)
    plot_training_curve(history, os.path.join(fig_dir, "training_curve.png"))

    # 2) evaluate
    baseline_metrics = evaluate_baseline_vectorized_3d(policy, cfg, device)
    pmp_metrics = evaluate_pmp_vectorized_3d(policy, cfg, device)

    print("[baseline metrics]", baseline_metrics)
    print("[pmp metrics]", pmp_metrics)

    # 3) volume at fixed time
    volume = compute_pmp_volume_at_time(policy, cfg, device, cfg.plane_time)
    save_volume_npz(volume, os.path.join(data_dir, "volume_Wcut_y_axes.npz"))

    # 4) figures
    tri_stats_u1 = plot_signed_isosurfaces(volume, cfg, "pmp_u1", os.path.join(fig_dir, "u1_signed_isosurfaces.png"))
    tri_stats_u2 = plot_signed_isosurfaces(volume, cfg, "pmp_u2", os.path.join(fig_dir, "u2_signed_isosurfaces.png"))
    tri_stats_u3 = plot_signed_isosurfaces(volume, cfg, "pmp_u3", os.path.join(fig_dir, "u3_signed_isosurfaces.png"))

    n_u1_zero = plot_zero_isosurface(volume, cfg, "pmp_u1", os.path.join(fig_dir, "u1_zero_only.png"))
    n_u2_zero = plot_zero_isosurface(volume, cfg, "pmp_u2", os.path.join(fig_dir, "u2_zero_only.png"))
    n_u3_zero = plot_zero_isosurface(volume, cfg, "pmp_u3", os.path.join(fig_dir, "u3_zero_only.png"))

    n_ntr = plot_ntr_boundary(volume, cfg, os.path.join(fig_dir, "ntr_boundary.png"))

    summary = {
        "config": asdict(cfg),
        "baseline_metrics": baseline_metrics,
        "pmp_metrics": pmp_metrics,
        "triangle_counts": {
            "u1_signed": tri_stats_u1,
            "u2_signed": tri_stats_u2,
            "u3_signed": tri_stats_u3,
            "u1_zero_only": int(n_u1_zero),
            "u2_zero_only": int(n_u2_zero),
            "u3_zero_only": int(n_u3_zero),
            "ntr_boundary": int(n_ntr),
        }
    }
    save_json(summary, os.path.join(outdir, "summary.json"))

    print("[saved]", os.path.join(fig_dir, "training_curve.png"))
    print("[saved]", os.path.join(fig_dir, "u1_signed_isosurfaces.png"))
    print("[saved]", os.path.join(fig_dir, "u2_signed_isosurfaces.png"))
    print("[saved]", os.path.join(fig_dir, "u3_signed_isosurfaces.png"))
    print("[saved]", os.path.join(fig_dir, "u1_zero_only.png"))
    print("[saved]", os.path.join(fig_dir, "u2_zero_only.png"))
    print("[saved]", os.path.join(fig_dir, "u3_zero_only.png"))
    print("[saved]", os.path.join(fig_dir, "ntr_boundary.png"))
    print("[saved]", os.path.join(data_dir, "volume_Wcut_y_axes.npz"))
    print("[saved]", os.path.join(outdir, "summary.json"))

    return {
        "policy": policy,
        "history": history,
        "baseline_metrics": baseline_metrics,
        "pmp_metrics": pmp_metrics,
        "volume": volume,
        "outdirs": {
            "root": outdir,
            "figures": fig_dir,
            "data": data_dir,
        }
    }


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    cfg = Config3D()
    results = run_full_pipeline_3d_paper_axis(cfg)

# ============================================================
# NEXT CELL:
# 3D HEATMAP VISUALIZATION FOR PMP u FIELDS
#   - Uses existing `results["volume"]` if available
#   - Otherwise loads saved npz file
#   - Produces 3D color heatmaps for pmp_u1, pmp_u2, pmp_u3
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ------------------------------------------------------------
# load volume
# ------------------------------------------------------------
if "results" in globals() and ("volume" in results):
    _volume = results["volume"]
    _outdir = results["outdirs"]["figures"] if ("outdirs" in results and "figures" in results["outdirs"]) else "."
elif "volume" in globals():
    _volume = volume
    _outdir = "."
else:
    npz_path = os.path.join("paper_3asset_Wcut_y_axes", "data", "volume_Wcut_y_axes.npz")
    if not os.path.exists(npz_path):
        raise RuntimeError(f"Cannot find volume data: {npz_path}")
    _npz = np.load(npz_path)
    _volume = {k: _npz[k] for k in _npz.files}
    _outdir = os.path.join("paper_3asset_Wcut_y_axes", "figures")

os.makedirs(_outdir, exist_ok=True)

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def _set_axes_equal_3d(ax, xlim, ylim, zlim):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    xr = xlim[1] - xlim[0]
    yr = ylim[1] - ylim[0]
    zr = zlim[1] - zlim[0]
    r = max(xr, yr, zr)

    xm = 0.5 * (xlim[0] + xlim[1])
    ym = 0.5 * (ylim[0] + ylim[1])
    zm = 0.5 * (zlim[0] + zlim[1])

    ax.set_xlim((xm - r / 2, xm + r / 2))
    ax.set_ylim((ym - r / 2, ym + r / 2))
    ax.set_zlim((zm - r / 2, zm + r / 2))

def _get_symmetric_vlim(F, q=0.995):
    vals = F[np.isfinite(F)]
    if vals.size == 0:
        return 1.0
    vmax = np.quantile(np.abs(vals), q)
    vmax = max(float(vmax), 1e-8)
    return vmax

def plot_3d_heatmap_scatter_panel(
    volume,
    field_names=("pmp_u1", "pmp_u2", "pmp_u3"),
    save_path=None,
    stride=1,
    max_points=18000,
    cmap="coolwarm",
    point_size=10,
    alpha=0.85,
    elev=24,
    azim=38,
    label_fontsize=18,
    tick_fontsize=12,
    cbar_fontsize=13,
):
    Y1 = volume["y1_grid"]
    Y2 = volume["y2_grid"]
    Y3 = volume["y3_grid"]

    fig = plt.figure(figsize=(25.0, 5.8))
    stats = {}

    for idx, field_name in enumerate(field_names):
        F = volume[field_name]

        if stride > 1:
            Y1s = Y1[::stride, ::stride, ::stride]
            Y2s = Y2[::stride, ::stride, ::stride]
            Y3s = Y3[::stride, ::stride, ::stride]
            Fs  = F [::stride, ::stride, ::stride]
        else:
            Y1s, Y2s, Y3s, Fs = Y1, Y2, Y3, F

        mask = np.isfinite(Fs)
        y1 = Y1s[mask]
        y2 = Y2s[mask]
        y3 = Y3s[mask]
        fv = Fs[mask]

        if y1.size == 0:
            raise RuntimeError(f"No finite values found in {field_name}.")

        n = y1.size
        if n > max_points:
            thin_idx = np.linspace(0, n - 1, max_points).astype(int)
            y1 = y1[thin_idx]
            y2 = y2[thin_idx]
            y3 = y3[thin_idx]
            fv = fv[thin_idx]

        vmax = _get_symmetric_vlim(fv, q=0.995)
        norm = Normalize(vmin=-vmax, vmax=vmax)

        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")

        sc = ax.scatter(
            y1, y2, y3,
            c=fv,
            cmap=cmap,
            norm=norm,
            s=point_size,
            alpha=alpha,
            linewidths=0,
            depthshade=False,
        )

        # No title
        ax.set_title("")

        # Larger axis labels
        ax.set_xlabel(r"$y_1$", fontsize=label_fontsize, labelpad=10)
        ax.set_ylabel(r"$y_2$", fontsize=label_fontsize, labelpad=10)
        ax.set_zlabel(r"$y_3$", fontsize=label_fontsize, labelpad=10)

        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.tick_params(axis="z", which="major", labelsize=tick_fontsize)

        _set_axes_equal_3d(ax, (0, 1), (0, 1), (0, 1))
        ax.view_init(elev=elev, azim=azim)

        cbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            pad=0.04,
            shrink=0.72,
        )
        cbar.set_label(rf"$u_{idx+1}$", fontsize=cbar_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)

        stats[field_name] = {
            "num_points": int(n),
            "num_points_plotted": int(y1.size),
            "vmax_abs": float(vmax),
        }

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return stats

def plot_3d_heatmap_with_slices(
    volume,
    field_name,
    save_path,
    slice_idx=None,
    cmap="coolwarm",
    alpha_scatter=0.18,
    point_size=6,
    elev=24,
    azim=38,
):
    Y1 = volume["y1_grid"]
    Y2 = volume["y2_grid"]
    Y3 = volume["y3_grid"]
    F  = volume[field_name]

    n1, n2, n3 = F.shape
    if slice_idx is None:
        slice_idx = (n1 // 2, n2 // 2, n3 // 2)

    i0, j0, k0 = slice_idx
    i0 = max(0, min(n1 - 1, int(i0)))
    j0 = max(0, min(n2 - 1, int(j0)))
    k0 = max(0, min(n3 - 1, int(k0)))

    mask = np.isfinite(F)
    y1 = Y1[mask]
    y2 = Y2[mask]
    y3 = Y3[mask]
    fv = F[mask]

    vmax = _get_symmetric_vlim(F, q=0.995)
    norm = Normalize(vmin=-vmax, vmax=vmax)

    fig = plt.figure(figsize=(9.0, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    # faint background cloud
    if y1.size > 20000:
        idx = np.linspace(0, y1.size - 1, 20000).astype(int)
        y1p, y2p, y3p, fvp = y1[idx], y2[idx], y3[idx], fv[idx]
    else:
        y1p, y2p, y3p, fvp = y1, y2, y3, fv

    ax.scatter(
        y1p, y2p, y3p,
        c=fvp, cmap=cmap, norm=norm,
        s=point_size, alpha=alpha_scatter, linewidths=0, depthshade=False
    )

    # y1 = const slice
    YY2, YY3 = Y2[i0, :, :], Y3[i0, :, :]
    YY1 = Y1[i0, :, :]
    FF1 = F[i0, :, :]
    ax.plot_surface(
        YY1, YY2, YY3,
        facecolors=plt.cm.get_cmap(cmap)(norm(FF1)),
        rstride=1, cstride=1, shade=False, antialiased=False, alpha=0.95
    )

    # y2 = const slice
    XX1, XX3 = Y1[:, j0, :], Y3[:, j0, :]
    XX2 = Y2[:, j0, :]
    FF2 = F[:, j0, :]
    ax.plot_surface(
        XX1, XX2, XX3,
        facecolors=plt.cm.get_cmap(cmap)(norm(FF2)),
        rstride=1, cstride=1, shade=False, antialiased=False, alpha=0.95
    )

    # y3 = const slice
    ZZ1, ZZ2 = Y1[:, :, k0], Y2[:, :, k0]
    ZZ3 = Y3[:, :, k0]
    FF3 = F[:, :, k0]
    ax.plot_surface(
        ZZ1, ZZ2, ZZ3,
        facecolors=plt.cm.get_cmap(cmap)(norm(FF3)),
        rstride=1, cstride=1, shade=False, antialiased=False, alpha=0.95
    )

    ax.set_xlabel(r"$y_1$")
    ax.set_ylabel(r"$y_2$")
    ax.set_zlabel(r"$y_3$")
    ax.set_title(
        rf"3D heatmap + orthogonal slices of {field_name} on $W=1$ cut"
    )

    _set_axes_equal_3d(ax, (0, 1), (0, 1), (0, 1))
    ax.view_init(elev=elev, azim=azim)

    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.08, shrink=0.82)
    cbar.set_label(field_name)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)

    return {
        "slice_idx": (int(i0), int(j0), int(k0)),
        "vmax_abs": float(vmax),
    }

# ------------------------------------------------------------
# generate figures
# ------------------------------------------------------------
panel_stats = plot_3d_heatmap_scatter_panel(
    _volume,
    field_names=("pmp_u1", "pmp_u2", "pmp_u3"),
    save_path=os.path.join(_outdir, "pmp_u123_3d_heatmap_scatter_panel.png"),
    stride=1,
    max_points=18000,
    cmap="coolwarm",
    point_size=10,
    alpha=0.85,
    elev=24,
    azim=38,
    label_fontsize=20,
    tick_fontsize=13,
    cbar_fontsize=14,
)

print("[saved]", os.path.join(_outdir, "pmp_u123_3d_heatmap_scatter_panel.png"))
print(panel_stats)
#stats_u1 = plot_3d_heatmap_scatter(
#    _volume,
#    "pmp_u1",
#    os.path.join(_outdir, "pmp_u1_3d_heatmap_scatter.png"),
#    stride=1,
#    max_points=18000,
#    cmap="coolwarm",
#    point_size=12,
#    alpha=0.85,
#)

#stats_u2 = plot_3d_heatmap_scatter(
#    _volume,
##    "pmp_u2",
 #   os.path.join(_outdir, "pmp_u2_3d_heatmap_scatter.png"),
 #   stride=1,
 #   max_points=18000,
 #   cmap="coolwarm",
 #   point_size=12,
 #   alpha=0.85,
#)

#stats_u3 = plot_3d_heatmap_scatter(
#    _volume,
#    "pmp_u3",
#    os.path.join(_outdir, "pmp_u3_3d_heatmap_scatter.png"),
#    stride=1,
#    max_points=18000,
#    cmap="coolwarm",
#    point_size=12,
#    alpha=0.85,
#)

slice_u1 = plot_3d_heatmap_with_slices(
    _volume,
    "pmp_u1",
    os.path.join(_outdir, "pmp_u1_3d_heatmap_with_slices.png"),
)

slice_u2 = plot_3d_heatmap_with_slices(
    _volume,
    "pmp_u2",
    os.path.join(_outdir, "pmp_u2_3d_heatmap_with_slices.png"),
)

slice_u3 = plot_3d_heatmap_with_slices(
    _volume,
    "pmp_u3",
    os.path.join(_outdir, "pmp_u3_3d_heatmap_with_slices.png"),
)

print("[saved]", os.path.join(_outdir, "pmp_u1_3d_heatmap_scatter.png"), stats_u1)
print("[saved]", os.path.join(_outdir, "pmp_u2_3d_heatmap_scatter.png"), stats_u2)
print("[saved]", os.path.join(_outdir, "pmp_u3_3d_heatmap_scatter.png"), stats_u3)
print("[saved]", os.path.join(_outdir, "pmp_u1_3d_heatmap_with_slices.png"), slice_u1)
print("[saved]", os.path.join(_outdir, "pmp_u2_3d_heatmap_with_slices.png"), slice_u2)
print("[saved]", os.path.join(_outdir, "pmp_u3_3d_heatmap_with_slices.png"), slice_u3)

# ============================================================
# NEXT CELL:
# 27-REGION 3D VISUALIZATION
#   - uses volume["pmp_region"]
#   - figure 1: full 3D scatter with all 27 regions
#   - figure 2: 27 subplots, one for each region
# ============================================================

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------------------------------------------
# load volume
# ------------------------------------------------------------
if "results" in globals() and ("volume" in results):
    _volume = results["volume"]
    _outdir = results["outdirs"]["figures"] if ("outdirs" in results and "figures" in results["outdirs"]) else "."
elif "volume" in globals():
    _volume = volume
    _outdir = "."
else:
    npz_path = os.path.join("paper_3asset_Wcut_y_axes", "data", "volume_Wcut_y_axes.npz")
    if not os.path.exists(npz_path):
        raise RuntimeError(f"Cannot find volume data: {npz_path}")
    _npz = np.load(npz_path)
    _volume = {k: _npz[k] for k in _npz.files}
    _outdir = os.path.join("paper_3asset_Wcut_y_axes", "figures")

os.makedirs(_outdir, exist_ok=True)

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def _set_axes_equal_3d(ax, xlim, ylim, zlim):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    xr = xlim[1] - xlim[0]
    yr = ylim[1] - ylim[0]
    zr = zlim[1] - zlim[0]
    r = max(xr, yr, zr)

    xm = 0.5 * (xlim[0] + xlim[1])
    ym = 0.5 * (ylim[0] + ylim[1])
    zm = 0.5 * (zlim[0] + zlim[1])

    ax.set_xlim((xm - r / 2, xm + r / 2))
    ax.set_ylim((ym - r / 2, ym + r / 2))
    ax.set_zlim((zm - r / 2, zm + r / 2))

def decode_region_id(region_id):
    # inverse of:
    # (a1 + 1) * 9 + (a2 + 1) * 3 + (a3 + 1)
    a1p = region_id // 9
    rem = region_id % 9
    a2p = rem // 3
    a3p = rem % 3
    a1 = a1p - 1
    a2 = a2p - 1
    a3 = a3p - 1
    return (a1, a2, a3)

def action_word(a):
    if a == -1:
        return "sell"
    if a == 0:
        return "hold"
    if a == 1:
        return "buy"
    return "?"

def region_label(region_id):
    a1, a2, a3 = decode_region_id(int(region_id))
    return f"({action_word(a1)}, {action_word(a2)}, {action_word(a3)})"

def region_short_label(region_id):
    a1, a2, a3 = decode_region_id(int(region_id))
    mp = {-1: "S", 0: "H", 1: "B"}
    return f"({mp[a1]},{mp[a2]},{mp[a3]})"

def build_region_colors():
    # stable 27-color palette from tab20 + tab20b + tab20c
    cmaps = [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c]
    colors = []
    for cmap in cmaps:
        for i in range(cmap.N):
            colors.append(cmap(i))
    # use first 27
    return colors[:27]

def subsample_points(y1, y2, y3, r, max_points=30000):
    n = len(y1)
    if n <= max_points:
        return y1, y2, y3, r
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return y1[idx], y2[idx], y3[idx], r[idx]

# ------------------------------------------------------------
# main plotting functions
# ------------------------------------------------------------
def plot_27_regions_3d_scatter(volume, save_path, max_points=30000, elev=24, azim=38):
    Y1 = volume["y1_grid"]
    Y2 = volume["y2_grid"]
    Y3 = volume["y3_grid"]
    R  = volume["pmp_region"]

    mask = np.isfinite(R) & (R >= 0)
    if not np.any(mask):
        raise RuntimeError("No valid region points found in pmp_region.")

    y1 = Y1[mask]
    y2 = Y2[mask]
    y3 = Y3[mask]
    r  = R[mask].astype(np.int64)

    y1, y2, y3, r = subsample_points(y1, y2, y3, r, max_points=max_points)

    colors = build_region_colors()

    fig = plt.figure(figsize=(15.5, 12.5))
    ax = fig.add_subplot(111, projection="3d")

    for rid in range(27):
        sel = (r == rid)
        if np.any(sel):
            ax.scatter(
                y1[sel], y2[sel], y3[sel],
                s=10,
                alpha=0.85,
                color=colors[rid],
                linewidths=0,
                depthshade=False,
                label=f"{rid}: {region_short_label(rid)}"
            )

    ax.set_xlabel(r"$y_1$")
    ax.set_ylabel(r"$y_2$")
    ax.set_zlabel(r"$y_3$")
    ax.set_title(r"27-region 3D scatter on $W=1$ cut in $(y_1,y_2,y_3)$")

    _set_axes_equal_3d(ax, (0, 1), (0, 1), (0, 1))
    ax.view_init(elev=elev, azim=azim)

    handles = []
    present_regions = sorted(np.unique(r).tolist())
    for rid in present_regions:
        handles.append(
            Line2D([0], [0],
                   marker='o',
                   color='w',
                   markerfacecolor=colors[rid],
                   markersize=7,
                   label=f"{rid}: {region_short_label(rid)}")
        )

    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=8,
        ncol=1,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "num_points_plotted": int(len(r)),
        "regions_present": present_regions,
    }

def plot_27_regions_subplots(
    volume,
    save_path,
    max_points_per_region=2500,
    elev=24,
    azim=38,
    title_fontsize=30,     # <-- region label 크게
    label_fontsize=11,     # <-- y1,y2,y3
    tick_fontsize=9
):
    Y1 = volume["y1_grid"]
    Y2 = volume["y2_grid"]
    Y3 = volume["y3_grid"]
    R  = volume["pmp_region"]

    mask = np.isfinite(R) & (R >= 0)
    if not np.any(mask):
        raise RuntimeError("No valid region points found in pmp_region.")

    y1_all = Y1[mask]
    y2_all = Y2[mask]
    y3_all = Y3[mask]
    r_all  = R[mask].astype(np.int64)

    colors = build_region_colors()

    fig = plt.figure(figsize=(46, 17))
    nrows, ncols = 3, 9

    stats = {}

    for rid in range(27):
        ax = fig.add_subplot(nrows, ncols, rid + 1, projection="3d")
        sel = (r_all == rid)

        count_total = int(np.sum(sel))
        stats[rid] = count_total

        if count_total > 0:
            y1 = y1_all[sel]
            y2 = y2_all[sel]
            y3 = y3_all[sel]

            if count_total > max_points_per_region:
                idx = np.linspace(0, count_total - 1, max_points_per_region).astype(int)
                y1 = y1[idx]
                y2 = y2[idx]
                y3 = y3[idx]

            ax.scatter(
                y1, y2, y3,
                s=8,
                alpha=0.85,
                color=colors[rid],
                linewidths=0,
                depthshade=False,
            )

        # -------------------------
        # 핵심: region label 크게
        # -------------------------
        ax.set_title(
            f"{region_short_label(rid)}",
            fontsize=title_fontsize,
            pad=2
        )

        # axis label 크게
        ax.set_xlabel(r"$y_1$", fontsize=label_fontsize, labelpad=2)
        ax.set_ylabel(r"$y_2$", fontsize=label_fontsize, labelpad=2)
        ax.set_zlabel(r"$y_3$", fontsize=label_fontsize, labelpad=2)

        ax.tick_params(labelsize=tick_fontsize)

        _set_axes_equal_3d(ax, (0, 1), (0, 1), (0, 1))
        ax.view_init(elev=elev, azim=azim)

    # -------------------------
    # 논문용: 전체 title 제거 (추천)
    # -------------------------
    # fig.suptitle(...)

    plt.tight_layout(pad=0.6)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return stats
def save_region_table_text(save_path_txt):
    lines = []
    lines.append("region_id -> (u1, u2, u3)")
    lines.append("where S=sell, H=hold, B=buy")
    lines.append("")
    for rid in range(27):
        lines.append(f"{rid:2d} -> {region_short_label(rid)}   {region_label(rid)}")
    with open(save_path_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ------------------------------------------------------------
# generate figures
# ------------------------------------------------------------
stats_main = plot_27_regions_3d_scatter(
    _volume,
    os.path.join(_outdir, "pmp_region_27_all_scatter.png"),
    max_points=30000,
)

stats_sub = plot_27_regions_subplots(
    _volume,
    os.path.join(_outdir, "pmp_region_27_subplots.png"),
    max_points_per_region=2500,
)

save_region_table_text(
    os.path.join(_outdir, "pmp_region_27_label_table.txt")
)

print("[saved]", os.path.join(_outdir, "pmp_region_27_all_scatter.png"), stats_main)
print("[saved]", os.path.join(_outdir, "pmp_region_27_subplots.png"))
print("[saved]", os.path.join(_outdir, "pmp_region_27_label_table.txt"))

print("\n[region counts]")
for rid in range(27):
    print(f"region {rid:2d} {region_short_label(rid):>9s} : {stats_sub[rid]}")
