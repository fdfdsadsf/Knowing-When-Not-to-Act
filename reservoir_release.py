# Non-financial transfer experiment:
# Stochastic two-reservoir level regulation with one-sided singular release
#
# UPDATED VERSION
#   - DP eta=0 discrete singular-control reference added
#   - Methods:
#       DP eta=0, DPO, P_eta^[1], P_eta^[2], P_eta^[4], P_eta^*
#   - No No-Release baseline
#   - No exact-inactivity panel
#   - No panel titles
#
# IMPORTANT
#   The DP line is an independently computed discrete-time/grid reference
#   for the original eta=0 singular problem under the SAME execution order:
#       release -> running cost -> exact drift + noise -> future value.
#   It is a numerical ground-truth reference for this low-dimensional
#   benchmark, not a certified continuous-time solution.
#
# ORIGINAL PROBLEM = MINIMIZATION

import math
import time
import random
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 0. Reproducibility / device

SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.set_float32_matmul_precision("high")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

print(f"[device] {DEVICE}")


# 1. Configuration

@dataclass
class CFG:
    # horizon
    T: float = 2.0
    n_steps: int = 32

    # original / regularized costs
    alpha: float = 0.28
    eta: float = 0.45

    # DPO policy
    hidden: int = 128
    depth: int = 3
    u_max: float = 2.5

    # Stage-I DPO
    train_steps: int = 2500
    batch_size: int = 384
    lr: float = 1.0e-3
    grad_clip: float = 10.0
    print_every: int = 250

    # initial-state sampling
    init_jitter_1: float = 0.08
    init_jitter_2: float = 0.08

    # feedback BPTT
    # even number for antithetic sampling
    inner_mc_paths: int = 24

    # same-time replay
    # <= 0 -> use physical dt
    sr_delta: float = -1.0
    sr_max_iterations: int = 24
    switching_margin: float = 2.0e-3
    same_time_tol: float = 1.0e-7

    # held-out outer evaluation
    outer_eval_paths: int = 64
    eval_seed: int = 2317

    # state feasibility for intervention
    h_min: float = 0.25

    # DP eta=0 reference
    # State grid = [h_min, dp_h_max]^2.
    # The next-state interpolation is clipped only at the grid box edge.
    # Choose dp_h_max large enough that clipping is negligible.
    dp_h_max: float = 2.25
    dp_n_grid: int = 101
    dp_gh_order: int = 5

    # output
    outdir: str = "reservoir_release_refinement"


cfg = CFG()


DT = cfg.T / cfg.n_steps
SR_DELTA = DT if cfg.sr_delta <= 0.0 else cfg.sr_delta

OUTDIR = Path(cfg.outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)


# 2. Coupled two-reservoir dynamics

A = torch.tensor(
    [
        [-0.35,  0.08],
        [ 0.18, -0.30],
    ],
    device=DEVICE,
    dtype=DTYPE,
)

b = torch.tensor(
    [0.22, 0.17],
    device=DEVICE,
    dtype=DTYPE,
)

SIGMA = torch.tensor(
    [
        [0.055, 0.000],
        [0.018, 0.050],
    ],
    device=DEVICE,
    dtype=DTYPE,
)

H_TARGET = torch.tensor(
    [0.85, 0.90],
    device=DEVICE,
    dtype=DTYPE,
)

H0_NOMINAL = torch.tensor(
    [1.65, 1.45],
    device=DEVICE,
    dtype=DTYPE,
)

INIT_JITTER = torch.tensor(
    [cfg.init_jitter_1, cfg.init_jitter_2],
    device=DEVICE,
    dtype=DTYPE,
)

Q = torch.tensor(
    [
        [1.20, 0.15],
        [0.15, 1.00],
    ],
    device=DEVICE,
    dtype=DTYPE,
)

QF = torch.tensor(
    [
        [3.00, 0.25],
        [0.25, 2.50],
    ],
    device=DEVICE,
    dtype=DTYPE,
)

# Exact affine drift step
AD = torch.matrix_exp(A * DT)
H_EQ = torch.linalg.solve(-A, b)

# NumPy copies for DP
A_np = A.detach().cpu().numpy().astype(np.float64)
b_np = b.detach().cpu().numpy().astype(np.float64)
SIGMA_np = SIGMA.detach().cpu().numpy().astype(np.float64)
H_TARGET_np = H_TARGET.detach().cpu().numpy().astype(np.float64)
Q_np = Q.detach().cpu().numpy().astype(np.float64)
QF_np = QF.detach().cpu().numpy().astype(np.float64)
AD_np = AD.detach().cpu().numpy().astype(np.float64)
H_EQ_np = H_EQ.detach().cpu().numpy().astype(np.float64)

print(f"[model] dt={DT:.5f}, equilibrium={H_EQ_np}")


# 3. Cost / dynamics helpers

def quad_cost(h, M):
    d = h - H_TARGET
    return 0.5 * torch.einsum("bi,ij,bj->b", d, M, d)


def running_state_cost(h):
    return quad_cost(h, Q)


def terminal_cost(h):
    return quad_cost(h, QF)


def deterministic_drift_step(h_post):
    return H_EQ + (h_post - H_EQ) @ AD.T


def diffusion_step(eps):
    return math.sqrt(DT) * (eps @ SIGMA.T)


def feasible_release(h, proposed_release):
    room = (h - cfg.h_min).clamp_min(0.0)
    return torch.minimum(proposed_release.clamp_min(0.0), room)


def seeded_randn(shape, seed):
    g = torch.Generator(device=str(DEVICE))
    g.manual_seed(int(seed))
    return torch.randn(*shape, generator=g, device=DEVICE, dtype=DTYPE)


def sample_initial(batch, seed=None):
    if seed is None:
        eps = torch.randn(batch, 2, device=DEVICE, dtype=DTYPE)
    else:
        eps = seeded_randn((batch, 2), seed)

    h = H0_NOMINAL + INIT_JITTER * eps
    return h.clamp_min(cfg.h_min + 0.05)


# 4. DP eta=0 reference
#
# Discrete Bellman recursion for the original singular problem:
#
#   V_k(h)
#   = min_{q <= h}
#       alpha * 1'(h-q)
#       + ell(q) dt
#       + E[V_{k+1}(F(q,eps))],
#
# where q is the post-release level.
#
# Rewrite:
#
#   V_k(h)
#   = alpha * 1'h
#     + min_{q <= h}
#       { -alpha * 1'q
#         + ell(q) dt
#         + E[V_{k+1}(F(q,eps))] }.
#
# On a rectangular grid, the constrained minimization is therefore a
# two-dimensional prefix minimum. This makes the low-dimensional DP cheap.

DP_GRID = np.linspace(
    cfg.h_min,
    cfg.dp_h_max,
    cfg.dp_n_grid,
    dtype=np.float64,
)
DP_DH = float(DP_GRID[1] - DP_GRID[0])
DP_N = len(DP_GRID)

DP_H1, DP_H2 = np.meshgrid(DP_GRID, DP_GRID, indexing="ij")
DP_STATES = np.stack([DP_H1.ravel(), DP_H2.ravel()], axis=1)


def np_quad_cost(states, M):
    d = states - H_TARGET_np[None, :]
    return 0.5 * np.einsum("bi,ij,bj->b", d, M, d)


def bilinear_interp_uniform(V, pts):
    """
    Bilinear interpolation on DP_GRID x DP_GRID.
    Values outside the numerical DP box are clipped to its boundary.
    """
    x = (pts[:, 0] - cfg.h_min) / DP_DH
    y = (pts[:, 1] - cfg.h_min) / DP_DH

    x = np.clip(x, 0.0, DP_N - 1.0)
    y = np.clip(y, 0.0, DP_N - 1.0)

    i0 = np.floor(x).astype(np.int64)
    j0 = np.floor(y).astype(np.int64)
    i1 = np.minimum(i0 + 1, DP_N - 1)
    j1 = np.minimum(j0 + 1, DP_N - 1)

    wx = x - i0
    wy = y - j0

    v00 = V[i0, j0]
    v10 = V[i1, j0]
    v01 = V[i0, j1]
    v11 = V[i1, j1]

    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )


def prefix_min_with_arg(G):
    """
    For every (i,j), compute

        min_{p<=i, q<=j} G[p,q]

    and store one minimizing grid index (p,q).
    """
    n1, n2 = G.shape
    pref = np.empty_like(G)
    arg_i = np.empty((n1, n2), dtype=np.int16)
    arg_j = np.empty((n1, n2), dtype=np.int16)

    for i in range(n1):
        for j in range(n2):
            best = G[i, j]
            bi, bj = i, j

            if i > 0 and pref[i - 1, j] < best:
                best = pref[i - 1, j]
                bi = int(arg_i[i - 1, j])
                bj = int(arg_j[i - 1, j])

            if j > 0 and pref[i, j - 1] < best:
                best = pref[i, j - 1]
                bi = int(arg_i[i, j - 1])
                bj = int(arg_j[i, j - 1])

            pref[i, j] = best
            arg_i[i, j] = bi
            arg_j[i, j] = bj

    return pref, arg_i, arg_j


def build_gauss_hermite_2d(order):
    """Tensor-product Gauss-Hermite for E[f(Z)], Z~N(0,I_2)."""
    x, w = np.polynomial.hermite.hermgauss(order)
    z = np.sqrt(2.0) * x
    w = w / np.sqrt(np.pi)

    nodes = []
    weights = []
    for i in range(order):
        for j in range(order):
            nodes.append([z[i], z[j]])
            weights.append(w[i] * w[j])

    return np.asarray(nodes, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def solve_dp_eta0():
    print("\n[DP eta=0] solving low-dimensional singular-control reference ...")
    t0 = time.perf_counter()

    gh_nodes, gh_weights = build_gauss_hermite_2d(cfg.dp_gh_order)

    # Terminal value
    V_next = np_quad_cost(DP_STATES, QF_np).reshape(DP_N, DP_N)

    # Optimal post-release grid indices at each calendar time/state
    post_i = np.empty((cfg.n_steps, DP_N, DP_N), dtype=np.int16)
    post_j = np.empty((cfg.n_steps, DP_N, DP_N), dtype=np.int16)

    running_grid = np_quad_cost(DP_STATES, Q_np).reshape(DP_N, DP_N)

    # Deterministic next-state means for every possible post-release q
    mu_next = H_EQ_np[None, :] + (DP_STATES - H_EQ_np[None, :]) @ AD_np.T

    for k in range(cfg.n_steps - 1, -1, -1):
        EV = np.zeros(DP_STATES.shape[0], dtype=np.float64)

        for node, wt in zip(gh_nodes, gh_weights):
            noise = math.sqrt(DT) * (SIGMA_np @ node)
            pts = mu_next + noise[None, :]
            EV += wt * bilinear_interp_uniform(V_next, pts)

        EV = EV.reshape(DP_N, DP_N)

        # G(q) in the prefix-min representation
        G = (
            running_grid * DT
            + EV
            - cfg.alpha * (DP_H1 + DP_H2)
        )

        pref, ai, aj = prefix_min_with_arg(G)

        # V_k(h) = alpha * (h1+h2) + prefix_min G
        V_curr = cfg.alpha * (DP_H1 + DP_H2) + pref

        post_i[k] = ai
        post_j[k] = aj
        V_next = V_curr

        if k == cfg.n_steps - 1 or k == 0 or k % 8 == 0:
            print(f"  [DP] k={k:02d}/{cfg.n_steps-1:02d}")

    elapsed = time.perf_counter() - t0
    print(f"[DP eta=0] finished in {elapsed:.2f}s")

    # Diagnostics: how often next-state interpolation touches the DP box edge
    # under the quadrature nodes, evaluated over all post-action grid states.
    gh_nodes, _ = build_gauss_hermite_2d(cfg.dp_gh_order)
    touched = 0
    total = 0
    for node in gh_nodes:
        noise = math.sqrt(DT) * (SIGMA_np @ node)
        pts = mu_next + noise[None, :]
        bad = (
            (pts[:, 0] < cfg.h_min)
            | (pts[:, 0] > cfg.dp_h_max)
            | (pts[:, 1] < cfg.h_min)
            | (pts[:, 1] > cfg.dp_h_max)
        )
        touched += int(bad.sum())
        total += len(bad)

    print(
        f"[DP eta=0] quadrature box-edge touch fraction = "
        f"{touched / max(total,1):.6e}"
    )

    return {
        "post_i": post_i,
        "post_j": post_j,
        "V0_grid": V_next,
        "solve_time_sec": elapsed,
    }


DP_SOL = solve_dp_eta0()


def dp_eta0_release(k, h):
    """
    Apply the discrete DP policy to continuous held-out states.

    We use the lower grid cell for state lookup so that any nonzero DP target
    is guaranteed not to exceed the actual pre-release state. If the DP action
    is 'hold' at that grid state, release is set exactly to zero rather than
    snapping the state to the grid.
    """
    h_np = h.detach().cpu().numpy().astype(np.float64)

    idx1 = np.floor((h_np[:, 0] - cfg.h_min) / DP_DH).astype(np.int64)
    idx2 = np.floor((h_np[:, 1] - cfg.h_min) / DP_DH).astype(np.int64)
    idx1 = np.clip(idx1, 0, DP_N - 1)
    idx2 = np.clip(idx2, 0, DP_N - 1)

    qi = DP_SOL["post_i"][k, idx1, idx2].astype(np.int64)
    qj = DP_SOL["post_j"][k, idx1, idx2].astype(np.int64)

    q = np.column_stack([DP_GRID[qi], DP_GRID[qj]])

    hold = (qi == idx1) & (qj == idx2)
    release_np = np.maximum(h_np - q, 0.0)
    release_np[hold] = 0.0

    release = torch.tensor(release_np, device=DEVICE, dtype=DTYPE)
    return feasible_release(h, release)


# 5. Stage-I DPO policy

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_dim = 3  # time + h1 + h2

        for _ in range(cfg.depth):
            layers += [nn.Linear(in_dim, cfg.hidden), nn.Tanh()]
            in_dim = cfg.hidden

        layers += [nn.Linear(in_dim, 2)]
        self.net = nn.Sequential(*layers)
        nn.init.constant_(self.net[-1].bias, -1.5)

    def forward(self, t, h):
        if not torch.is_tensor(t):
            tcol = torch.full((h.shape[0], 1), float(t), device=h.device, dtype=h.dtype)
        elif t.ndim == 0:
            tcol = t.expand(h.shape[0]).unsqueeze(1)
        elif t.ndim == 1:
            tcol = t.unsqueeze(1)
        else:
            tcol = t

        tnorm = 2.0 * tcol / cfg.T - 1.0
        hnorm = (h - H_TARGET) / 0.80
        x = torch.cat([tnorm, hnorm], dim=1)
        raw = self.net(x)

        # Smooth positive warm policy: no structural exact-zero threshold
        return cfg.u_max * torch.sigmoid(raw)


policy = PolicyNet().to(DEVICE)


# 6. eta-regularized Stage-I rollout

def rollout_regularized(policy, h0, eps_seq):
    h = h0
    cost = torch.zeros(h.shape[0], device=DEVICE, dtype=DTYPE)

    for k in range(cfg.n_steps):
        t = k * DT
        u = policy(t, h)

        release = feasible_release(h, u * DT)
        u_eff = release / DT
        h_post = h - release

        cost = cost + running_state_cost(h_post) * DT
        cost = cost + cfg.alpha * release.sum(dim=1)
        cost = cost + 0.5 * cfg.eta * u_eff.square().sum(dim=1) * DT

        h = deterministic_drift_step(h_post) + diffusion_step(eps_seq[k])

    return cost + terminal_cost(h)


# 7. Stage-I DPO training

optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
train_hist = []

policy.train()
t0_train = time.perf_counter()

for it in range(1, cfg.train_steps + 1):
    h0 = sample_initial(cfg.batch_size)
    eps_seq = torch.randn(
        cfg.n_steps,
        cfg.batch_size,
        2,
        device=DEVICE,
        dtype=DTYPE,
    )

    Jeta = rollout_regularized(policy, h0, eps_seq)
    loss = Jeta.mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
    optimizer.step()

    train_hist.append(float(loss.detach().cpu()))

    if it == 1 or it % cfg.print_every == 0 or it == cfg.train_steps:
        print(f"[train] {it:5d}/{cfg.train_steps}  J_eta={loss.item():.6f}")

train_time = time.perf_counter() - t0_train
print(f"[train] finished in {train_time:.2f}s")

# Freeze warm policy
policy.eval()
for p in policy.parameters():
    p.requires_grad_(False)


torch.save(
    {
        "policy_state_dict": policy.state_dict(),
        "cfg": cfg.__dict__,
        "A": A.detach().cpu(),
        "b": b.detach().cpu(),
        "Sigma": SIGMA.detach().cpu(),
        "target": H_TARGET.detach().cpu(),
    },
    OUTDIR / "reservoir_dpo_policy.pt",
)


# 8. Antithetic continuation noises

def make_antithetic_inner_noise(remaining_steps, n_query, n_inner, seed):
    assert n_inner % 2 == 0
    half = n_inner // 2

    base = seeded_randn((remaining_steps, n_query, half, 2), seed)
    eps = torch.cat([base, -base], dim=2)
    return eps.reshape(remaining_steps, n_query * n_inner, 2)


# 9. Feedback-BPTT continuation costate

def feedback_bptt_costate(k0, h_query, seed):
    """
    Closed-loop feedback BPTT.

    Policy parameters are frozen, but h -> u_theta(t,h) Jacobians are retained.
    """
    M = h_query.shape[0]
    K = cfg.inner_mc_paths
    remaining = cfg.n_steps - k0

    h0 = h_query.detach().clone().requires_grad_(True)
    h = h0[:, None, :].expand(M, K, 2).reshape(M * K, 2)

    eps_all = make_antithetic_inner_noise(remaining, M, K, seed)
    cost = torch.zeros(M * K, device=DEVICE, dtype=DTYPE)

    for local_j, k in enumerate(range(k0, cfg.n_steps)):
        t = k * DT

        # No detach: fixed-feedback state dependence is differentiated.
        u = policy(t, h)

        release = feasible_release(h, u * DT)
        u_eff = release / DT
        h_post = h - release

        cost = cost + running_state_cost(h_post) * DT
        cost = cost + cfg.alpha * release.sum(dim=1)
        cost = cost + 0.5 * cfg.eta * u_eff.square().sum(dim=1) * DT

        h = deterministic_drift_step(h_post) + diffusion_step(eps_all[local_j])

    cost = cost + terminal_cost(h)
    V_hat = cost.reshape(M, K).mean(dim=1)

    lam = torch.autograd.grad(
        V_hat.sum(),
        h0,
        create_graph=False,
        retain_graph=False,
    )[0]

    return lam.detach()


# 10. One-sided Yosida map

def reservoir_yosida_rate(lam):
    # Minimize over u_i >= 0:
    #   (-lambda_i + alpha)u_i + eta/2 u_i^2
    # => u_i^* = ReLU(lambda_i-alpha)/eta
    return torch.relu(lam - cfg.alpha) / cfg.eta


# 11. Finite / adaptive same-time maps

def refined_release(k, h, mode, seed_base):
    """
    mode:
        PG1    : up to 1 map
        PG2    : up to 2 maps
        PG4    : up to 4 maps
        PGstar : until first no-release entry, stall, or finite cap

    Same-time maps contain no physical drift/noise/running cost.
    The net release is then executed once under the original eta=0 semantics.
    """
    M = h.shape[0]
    virtual = h.detach().clone()
    net_release = torch.zeros_like(virtual)
    alive = torch.ones(M, dtype=torch.bool, device=DEVICE)

    if mode == "PG1":
        max_iter = 1
    elif mode == "PG2":
        max_iter = 2
    elif mode == "PG4":
        max_iter = 4
    elif mode == "PGstar":
        max_iter = cfg.sr_max_iterations
    else:
        raise ValueError(mode)

    for m in range(max_iter):
        idx = torch.nonzero(alive, as_tuple=False).flatten()
        if idx.numel() == 0:
            break

        lam = feedback_bptt_costate(
            k0=k,
            h_query=virtual[idx],
            seed=seed_base + 10007 * k + 97 * m,
        )

        excess = lam - cfg.alpha
        rate = reservoir_yosida_rate(lam)
        entered = (excess <= cfg.switching_margin).all(dim=1)

        delta = feasible_release(virtual[idx], rate * SR_DELTA)
        delta[entered] = 0.0

        virtual[idx] = virtual[idx] - delta
        net_release[idx] = net_release[idx] + delta

        stalled = delta.abs().sum(dim=1) <= cfg.same_time_tol
        done = entered | stalled
        if done.any():
            alive[idx[done]] = False

    return net_release


# 12. eta=0 held-out evaluation

METHOD_SPECS = [
    ("DP0",    r"DP $\eta=0$"),
    ("DPO",    r"$u_{\theta}$"),
    ("PG1",    r"$\mathcal{P}_{\eta}^{[1]}$"),
    ("PG2",    r"$\mathcal{P}_{\eta}^{[2]}$"),
    ("PG4",    r"$\mathcal{P}_{\eta}^{[4]}$"),
    ("PGstar", r"$\mathcal{P}_{\eta}^{*}$"),
]
METHOD_KEYS = [k for k, _ in METHOD_SPECS]

# Same inner-MC seed base for every PG depth.
# Therefore PG1 is literally the first prefix of PG2/PG4/PGstar whenever the
# queried state is the same, making the depth ablation much cleaner.
REFINEMENT_SEED_BASE = cfg.eval_seed + 100000


def evaluate_method(method, h0_eval, outer_eps):
    """
    Every method is evaluated under ORIGINAL eta=0 semantics:
      - no quadratic leakage
      - linear singular release cost only.
    """
    M = h0_eval.shape[0]
    h = h0_eval.detach().clone()

    cumulative_running = torch.zeros(M, device=DEVICE, dtype=DTYPE)
    cumulative_release_cost = torch.zeros(M, device=DEVICE, dtype=DTYPE)

    dist_trace = [
        torch.linalg.vector_norm(h - H_TARGET, dim=1).detach().cpu()
    ]
    release_cost_trace = [cumulative_release_cost.detach().cpu()]
    jstop_trace = [terminal_cost(h).detach().cpu()]

    t0 = time.perf_counter()

    for k in range(cfg.n_steps):
        t = k * DT

        if method == "DP0":
            release = dp_eta0_release(k, h)

        elif method == "DPO":
            with torch.no_grad():
                u = policy(t, h)
                release = feasible_release(h, u * DT)

        elif method in {"PG1", "PG2", "PG4", "PGstar"}:
            release = refined_release(
                k=k,
                h=h,
                mode=method,
                seed_base=REFINEMENT_SEED_BASE,
            )

        else:
            raise ValueError(method)

        # Execute the selected release once.
        h_post = h - release

        step_running = running_state_cost(h_post) * DT
        step_release_cost = cfg.alpha * release.sum(dim=1)

        cumulative_running += step_running
        cumulative_release_cost += step_release_cost

        # One physical stochastic step
        h = deterministic_drift_step(h_post) + diffusion_step(outer_eps[k])

        dist_trace.append(
            torch.linalg.vector_norm(h - H_TARGET, dim=1).detach().cpu()
        )
        release_cost_trace.append(cumulative_release_cost.detach().cpu())
        jstop_trace.append(
            (
                cumulative_running
                + cumulative_release_cost
                + terminal_cost(h)
            ).detach().cpu()
        )

    total_objective = cumulative_running + cumulative_release_cost + terminal_cost(h)
    eval_time = time.perf_counter() - t0

    return {
        "method": method,
        "objective": total_objective.detach().cpu().numpy(),
        "final_distance": torch.linalg.vector_norm(
            h - H_TARGET, dim=1
        ).detach().cpu().numpy(),
        "release_cost": cumulative_release_cost.detach().cpu().numpy(),
        "running_cost": cumulative_running.detach().cpu().numpy(),
        "dist_trace": torch.stack(dist_trace, dim=0).numpy(),
        "release_cost_trace": torch.stack(release_cost_trace, dim=0).numpy(),
        "jstop_trace": torch.stack(jstop_trace, dim=0).numpy(),
        "eval_time_sec": eval_time,
    }


# 13. Common held-out paths

h0_eval = sample_initial(cfg.outer_eval_paths, seed=cfg.eval_seed)
outer_eps = seeded_randn(
    (cfg.n_steps, cfg.outer_eval_paths, 2),
    cfg.eval_seed + 1,
)

results = {}

for method in METHOD_KEYS:
    print(f"[eval] {method}")
    results[method] = evaluate_method(method, h0_eval, outer_eps)

    vals = results[method]["objective"]
    se = vals.std(ddof=1) / math.sqrt(len(vals))

    print(
        f"       J0={vals.mean():.6f} "
        f"± {se:.3e} (SE), "
        f"time={results[method]['eval_time_sec']:.2f}s"
    )


# 14. Summary table / DP gaps

DP_MEAN = float(np.mean(results["DP0"]["objective"]))

rows = []
for method in METHOD_KEYS:
    r = results[method]
    obj = r["objective"]
    mean_obj = float(np.mean(obj))

    rows.append(
        {
            "method": method,
            "objective_mean": mean_obj,
            "objective_se": float(np.std(obj, ddof=1) / math.sqrt(len(obj))),
            "gap_to_DP": mean_obj - DP_MEAN,
            "release_cost_mean": float(np.mean(r["release_cost"])),
            "running_cost_mean": float(np.mean(r["running_cost"])),
            "final_distance_mean": float(np.mean(r["final_distance"])),
            "eval_time_sec": float(r["eval_time_sec"]),
        }
    )

summary = pd.DataFrame(rows).sort_values("objective_mean").reset_index(drop=True)

print("\n=== ORIGINAL eta=0 HELD-OUT SUMMARY (lower is better) ===")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

summary.to_csv(OUTDIR / "reservoir_summary.csv", index=False)


# 15. Pathwise export

path_rows = []

for method in METHOD_KEYS:
    r = results[method]
    for pth in range(cfg.outer_eval_paths):
        path_rows.append(
            {
                "method": method,
                "path": pth,
                "objective": r["objective"][pth],
                "release_cost": r["release_cost"][pth],
                "running_cost": r["running_cost"][pth],
                "final_distance": r["final_distance"][pth],
            }
        )

pd.DataFrame(path_rows).to_csv(
    OUTDIR / "reservoir_pathwise.csv",
    index=False,
)


# 16. Paper visualization
#     NO PANEL TITLES

plt.rcParams.update(
    {
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    }
)

STYLE = {
    "DP0":    dict(color="#222222", lw=2.6, ls="--"),
    "DPO":    dict(color="#ef3b2c", lw=2.2, ls="-"),
    "PG1":    dict(color="#a1d99b", lw=2.2, ls="-"),
    "PG2":    dict(color="#41ab5d", lw=2.2, ls="-"),
    "PG4":    dict(color="#238b45", lw=2.2, ls="-"),
    "PGstar": dict(color="#00441b", lw=2.8, ls="-"),
}

LABEL = dict(METHOD_SPECS)


def mean_iqr(arr):
    mean = np.mean(arr, axis=1)
    q1 = np.quantile(arr, 0.25, axis=1)
    q3 = np.quantile(arr, 0.75, axis=1)
    return mean, q1, q3


t_state = np.linspace(0.0, cfg.T, cfg.n_steps + 1)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(10.9, 3.25),
)

# Distance to target levels
ax = axes[0]
for method in METHOD_KEYS:
    mean, q1, q3 = mean_iqr(results[method]["dist_trace"])
    st = STYLE[method]

    ax.plot(t_state, mean, label=LABEL[method], **st)
    ax.fill_between(
        t_state,
        q1,
        q3,
        color=st["color"],
        alpha=0.10 if method == "DP0" else 0.12,
        linewidth=0,
    )

ax.set_xlabel("time")
ax.set_ylabel(r"$\|H_t-h^{\star}\|_2$")
ax.grid(alpha=0.22)

# Cumulative release cost
ax = axes[1]
for method in METHOD_KEYS:
    mean, q1, q3 = mean_iqr(results[method]["release_cost_trace"])
    st = STYLE[method]

    ax.plot(t_state, mean, label=LABEL[method], **st)
    ax.fill_between(
        t_state,
        q1,
        q3,
        color=st["color"],
        alpha=0.10 if method == "DP0" else 0.12,
        linewidth=0,
    )

ax.set_xlabel("time")
ax.set_ylabel(r"$\alpha\sum_{\tau<t}\|\Delta\Xi_\tau\|_1$")
ax.grid(alpha=0.22)

# Stopped-horizon / truncated objective trace
ax = axes[2]
for method in METHOD_KEYS:
    mean, q1, q3 = mean_iqr(results[method]["jstop_trace"])
    st = STYLE[method]

    ax.plot(t_state, mean, label=LABEL[method], **st)
    ax.fill_between(
        t_state,
        q1,
        q3,
        color=st["color"],
        alpha=0.10 if method == "DP0" else 0.12,
        linewidth=0,
    )

ax.set_xlabel("time")
ax.set_ylabel(r"$J_{\mathrm{stop}}(t)$")
ax.grid(alpha=0.22)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=6,
    frameon=False,
    bbox_to_anchor=(0.5, 1.08),
    handlelength=2.4,
)

fig.tight_layout(rect=[0, 0, 1, 0.96])

pdf_path = OUTDIR / "reservoir_release_refinement_main.pdf"
png_path = OUTDIR / "reservoir_release_refinement_main.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=240, bbox_inches="tight")
plt.show()


# 17. Training diagnostic

fig2, ax2 = plt.subplots(figsize=(5.4, 3.2))
ax2.plot(np.arange(1, len(train_hist) + 1), train_hist, lw=1.3)
ax2.set_xlabel("Stage-I update")
ax2.set_ylabel(r"training $J_\eta$")
ax2.grid(alpha=0.22)
fig2.tight_layout()
fig2.savefig(OUTDIR / "reservoir_training_curve.pdf", bbox_inches="tight")
plt.show()


# 18. Saved outputs

print("\n[saved]")
print(" ", pdf_path)
print(" ", png_path)
print(" ", OUTDIR / "reservoir_summary.csv")
print(" ", OUTDIR / "reservoir_pathwise.csv")
print(" ", OUTDIR / "reservoir_dpo_policy.pt")
