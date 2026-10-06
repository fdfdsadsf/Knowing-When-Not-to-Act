# CELL 2 — GLOBAL N-DIMENSIONAL NEURAL BASELINES
# on the frozen independent-CARA DP oracle from CELL 1
# IMPORTANT
#   * This cell NEVER solves DP.
#   * The oracle is assetwise/separable, but EVERY learned baseline below is
#     one GLOBAL N-output network for each N.
#   * The network is not given asset-block labels or separate per-asset models.
#   * There is ONE common cash account x; borrowing/lending is unrestricted.
#   * Risky holdings remain long-only (sales cannot exceed current y_i).
#   * No fixed-time refinement/replay and no policy-objective reporting.

import json, math, random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn


@dataclass
class MainConfig:
    oracle_dir: str = "cara_cara_uncorr_oracle"
    outdir: str = "cara_cara_uncorr_global_results"

    # Regularized continuation used by BPTT / P_eta.
    eta: float = 0.001

    # GLOBAL policy network / DPO
    hidden: int = 256
    depth: int = 3
    train_steps: int = 20000
    batch: int = 256  # retained for student/distillation mini-batches
    lr: float = 1e-4
    policy_order_scale: float = 0.8  # max absolute dollar order per decision

    # DPO quality improvements.
    # One uniformly random decision time k is sampled per optimizer step.
    # At that k, each training state is evaluated with 4 antithetic MC paths,
    # and the optimization target is the mean of statewise CARA CEs.
    policy_state_batch: int = 64
    policy_mc_paths: int = 4

    # Preserve the old boundary-state frequency (~1/7 of sampled states),
    # but randomize WHICH risky coordinate is placed exactly on y_i=0.
    boundary_state_prob: float = 1.0 / 7.0

    # GLOBAL CE-PINN.  We model the CARA certainty equivalent C rather than
    # the raw exponentially-small value V=-exp(-a C).
    pinn_hidden: int = 256
    pinn_depth: int = 3
    pinn_steps: int = 5000
    pinn_batch: int = 192
    pinn_lr: float = 4e-4
    pinn_hutchinson_samples: int = 2

    # BPTT / ratio distillation
    mc_paths: int = 2048
    teacher_size: int = 2000
    student_hidden: int = 192
    student_depth: int = 3
    student_steps: int = 10000
    student_lr: float = 5e-4
    inference_batch: int = 32

    # Training-state domain, independent of the frozen test set.
    train_y_max: float = 1.20
    train_cash_abs: float = 0.75

    # Coordinatewise baseline thresholds in DOLLAR ORDER units.
    magnitude_thresholds: tuple = (0.001, 0.01, 0.1)
    label_tol: float = 1e-8

    # Multi-seed experiment.
    # Each run_seed changes model initialization, training-state sampling,
    # DPO market noise, PINN sampling, and distillation teacher/student randomness.
    seeds: tuple = (20260924, 20260925, 20260926, 20260927, 20260928)

    # Keep the frozen-test BPTT Monte Carlo bank identical across run seeds.
    # This makes cross-seed std primarily measure training / initialization
    # sensitivity rather than adding independent evaluation-MC noise.
    eval_seed: int = 20269999

    device: str = "auto"


C = MainConfig()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(s: str):
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


DEVICE = choose_device(C.device)
seed_all(int(C.seeds[0]))
OUT = Path(C.outdir)
OUT.mkdir(parents=True, exist_ok=True)
ORACLE_DIR = Path(C.oracle_dir)
manifest_path = ORACLE_DIR / "oracle_manifest.json"
if not manifest_path.exists():
    raise FileNotFoundError(f"Run CELL 1 first; missing {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
om = manifest["oracle_config"]

DIMS = tuple(int(x) for x in om["test_dims"])
T = float(om["horizon"])
STEPS = int(om["steps"])
DT = T / STEPS
R = float(om["interest"])
ALPHA = float(om["alpha"])
CARA = float(om["cara"])
MU_PATTERN = tuple(float(x) for x in om["mu_pattern"])
VOL_PATTERN = tuple(float(x) for x in om["vol_pattern"])
Y_ORACLE_MAX = float(om["y_grid_max"])

if len(MU_PATTERN) != len(VOL_PATTERN):
    raise ValueError("Invalid oracle manifest: mu/vol pattern mismatch.")

print("device:", DEVICE)
print("oracle signature:", manifest["oracle_signature"])
print("dims:", DIMS)
print("state layout: [global_cash, y1, ..., yN]")


def asset_params(N: int, device, dtype):
    mu = torch.tensor([MU_PATTERN[i % len(MU_PATTERN)] for i in range(N)], device=device, dtype=dtype)
    vol = torch.tensor([VOL_PATTERN[i % len(VOL_PATTERN)] for i in range(N)], device=device, dtype=dtype)
    return mu, vol


class MLP(nn.Module):
    def __init__(self, inputs, outputs, hidden, depth):
        super().__init__()
        layers = []
        d = inputs
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers += [nn.Linear(d, outputs)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Global state / policy
def risky(z):
    return z[:, 1:]


def policy_features(t, z):
    # CARA cash-translation invariance is known analytically, so the policy
    # does not need cash as an input.  It still sees ALL N risky coordinates
    # jointly and outputs ALL N controls jointly.
    y = risky(z)
    return torch.cat([t[:, None], y / max(C.train_y_max, 1e-8)], dim=-1)


def project_long_only_orders(z, a):
    y = risky(z)
    sell = torch.minimum(torch.relu(-a), y.clamp_min(0.0))
    buy = torch.relu(a)  # unrestricted borrowing => no cash-budget projection
    return buy - sell


def policy_orders(net, t, z):
    raw = C.policy_order_scale * torch.tanh(net(policy_features(t, z)))
    return project_long_only_orders(z, raw)


def sample_training_states(N, n, device, interior=False, dtype=torch.float32):
    # One global cash coordinate, which may be negative.
    x = (2.0 * torch.rand((n, 1), device=device, dtype=dtype) - 1.0) * C.train_cash_abs
    lo = 0.03 if interior else 0.0
    y = lo + (C.train_y_max - lo) * torch.rand((n, N), device=device, dtype=dtype)

    if not interior and n > 0:
        # Random-coordinate boundary sampling.
        # Roughly boundary_state_prob of STATES get exactly one y_i=0.
        # This preserves the old boundary-state frequency while removing the
        # old coordinate-0 bias.  We intentionally do NOT zero each coordinate
        # independently, which would create O(N) zero coordinates per state.
        boundary_rows = torch.nonzero(
            torch.rand(n, device=device) < C.boundary_state_prob,
            as_tuple=False,
        ).flatten()
        if boundary_rows.numel() > 0:
            boundary_cols = torch.randint(
                low=0, high=N, size=(boundary_rows.numel(),), device=device
            )
            y[boundary_rows, boundary_cols] = 0.0

    return torch.cat([x, y], dim=-1)


def forward_step(z, order, normal, eta):
    """One physical step.

    Signed order q is a DOLLAR displacement during one decision interval.
    The regularizer is the cash cost

        eta/2 * |u|^2 dt = eta/(2 dt) * |q|^2,
        q = u dt,

    so P_eta(R)=(threshold excess)/eta is dimensionally consistent.
    """
    N = z.shape[1] - 1
    q = project_long_only_orders(z, order)
    x = z[:, 0]
    y = risky(z)

    x_post = (
        x
        - torch.relu(q).sum(-1)
        + (1.0 - ALPHA) * torch.relu(-q).sum(-1)
    )
    if eta > 0.0:
        x_post = x_post - eta / (2.0 * DT) * q.square().sum(-1)

    x_next = x_post * math.exp(R * DT)
    mu, vol = asset_params(N, z.device, z.dtype)
    growth = torch.exp((mu - 0.5 * vol.square()) * DT + vol * math.sqrt(DT) * normal)
    y_next = (y + q).clamp_min(0.0) * growth
    return torch.cat([x_next[:, None], y_next], dim=-1)


def rollout_terminal_wealth(net, k, z, eta, noises=None):
    N = z.shape[1] - 1
    state = z
    for j in range(k, STEPS):
        t = torch.full((len(state),), j / STEPS, device=state.device, dtype=state.dtype)
        q = policy_orders(net, t, state)
        eps = noises[j-k] if noises is not None else torch.randn((len(state), N), device=state.device, dtype=state.dtype)
        state = forward_step(state, q, eps, eta)
    return state[:, 0] + (1.0 - ALPHA) * risky(state).sum(-1)


def cara_certainty_equivalent_from_paths(W, dim=0):
    # CE = -(1/a) log E[e^{-a W}], computed stably.
    n = W.shape[dim]
    return -(torch.logsumexp(-CARA * W, dim=dim) - math.log(n)) / CARA


def policy_statewise_antithetic_objective(net, k, z, eta):
    """
    Mean statewise CARA certainty equivalent at a fixed random start time k.

    For each training state z_i, run exactly C.policy_mc_paths market paths.
    The noise paths are antithetic, so for M=4 they are

        eps^(1), eps^(2), -eps^(1), -eps^(2).

    We first compute a CARA CE separately for each initial state,

        CE_i = -(1/a) log [ (1/M) sum_m exp(-a W_{i,m}) ],

    and then maximize mean_i CE_i.  Hence one unusually bad initial state
    cannot reweight the whole training batch through a single pooled log-sum-exp.
    """
    M = int(C.policy_mc_paths)
    if M != 4:
        raise ValueError(
            f"This experiment is configured for exactly 4 DPO MC paths/state; got {M}."
        )

    B = len(z)
    N = z.shape[1] - 1
    remaining = STEPS - k

    # Path-major layout:
    #   [path0: all B states, path1: all B states, ...].
    rep = (
        z.unsqueeze(0)
        .expand(M, B, N + 1)
        .reshape(M * B, N + 1)
    )

    # Two iid base paths and their antithetic copies => exactly four paths.
    eps_half = torch.randn(
        (remaining, M // 2, B, N),
        device=z.device,
        dtype=z.dtype,
    )
    eps = torch.cat([eps_half, -eps_half], dim=1)
    noises = [eps[j].reshape(M * B, N) for j in range(remaining)]

    W = rollout_terminal_wealth(
        net, k, rep, eta, noises=noises
    ).reshape(M, B)

    # CE over market paths, independently for every sampled initial state.
    ce_by_state = cara_certainty_equivalent_from_paths(W, dim=0)
    return ce_by_state.mean()


def train_global_policy(N, eta, seed):
    seed_all(seed)
    net = MLP(N + 1, N, C.hidden, C.depth).to(DEVICE)  # input: time + all risky holdings
    opt = torch.optim.Adam(net.parameters(), lr=C.lr)

    for step in range(C.train_steps):
        # Random-k DPO training: one time slice per optimizer step, sampled
        # uniformly from every admissible decision time.  Over training this
        # directly covers the full (t, y_1, ..., y_N) policy domain rather than
        # relying only on states reached by a rollout starting from k=0.
        k = int(np.random.randint(0, STEPS))

        z = sample_training_states(
            N, C.policy_state_batch, DEVICE, interior=False
        )
        objective = policy_statewise_antithetic_objective(
            net, k, z, eta
        )

        opt.zero_grad(set_to_none=True)
        (-objective).backward()
        grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

        if step % max(C.train_steps // 5, 1) == 0 or step == C.train_steps - 1:
            print(
                f"    policy eta={eta:g} step={step:5d} "
                f"k={k:3d} statewise_CE={objective.item():.6f} "
                f"grad={float(grad_norm):.3e}"
            )

    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# Global BPTT ratio from the regularized global policy
def bptt_ratios(N, net, k, z, seed):
    rng = np.random.default_rng(seed)
    half = (C.mc_paths + 1) // 2
    e = rng.standard_normal((STEPS-k, half, N)).astype("float32")
    e = np.concatenate([e, -e], axis=1)[:, :C.mc_paths]

    out_r, out_ok = [], []
    for part in torch.split(z, C.inference_batch):
        b = len(part)
        rep = (
            part.detach().clone()[None]
            .expand(C.mc_paths, b, N + 1)
            .clone()
            .reshape(-1, N + 1)
            .requires_grad_(True)
        )
        ns = [
            torch.tensor(e[j], device=z.device).repeat_interleave(b, dim=0)
            for j in range(STEPS-k)
        ]
        W = rollout_terminal_wealth(net, k, rep, C.eta, ns).reshape(C.mc_paths, b)
        # Per-state CE; dCE_y/dCE_x has the same ratio as dV_y/dV_x for CARA.
        ce = cara_certainty_equivalent_from_paths(W, dim=0)
        g = torch.autograd.grad(ce.sum(), rep)[0].reshape(C.mc_paths, b, N + 1).sum(0)
        gx = g[:, 0]
        gy = g[:, 1:]
        good = torch.isfinite(g).all(-1) & (gx > 1e-9)
        ratio = gy / gx[:, None].clamp_min(1e-9)
        good = good & torch.isfinite(ratio).all(-1)
        out_r.append(ratio.detach())
        out_ok.append(good.detach())
    return torch.cat(out_r), torch.cat(out_ok)


# Global CARA certainty-equivalent PINN
class GlobalCEPINN(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.net = MLP(N + 1, 1, C.pinn_hidden, C.pinn_depth)  # time + all risky y
        nn.init.zeros_(self.net.net[-1].weight)
        nn.init.zeros_(self.net.net[-1].bias)

    def forward(self, t, z):
        x = z[:, 0]
        y = risky(z)
        tau = T * (1.0 - t)
        base = torch.exp(R * tau) * x + (1.0 - ALPHA) * y.sum(-1)
        corr = T * (1.0 - t) * self.net(policy_features(t, z))[:, 0]
        return base + corr


def pinn_residual(net, N, t, z):
    ce = net(t, z)
    g = torch.autograd.grad(ce.sum(), z, create_graph=True, retain_graph=True)[0]
    # t is normalized in [0,1], so divide by T for physical-time derivative.
    ce_t = torch.autograd.grad(ce.sum(), t, create_graph=True, retain_graph=True)[0] / T

    cx = g[:, 0]
    cy = g[:, 1:]
    y = risky(z)
    mu, vol = asset_params(N, z.device, z.dtype)

    drift = R * z[:, 0] * cx + (mu * y * cy).sum(-1)

    # Diagonal diffusion Hessian trace via Hutchinson, plus the exact
    # exponential-utility gradient-square term from the CE transformation.
    trace_terms = []
    for _ in range(max(1, C.pinn_hutchinson_samples)):
        eps = torch.empty_like(y).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        v = vol * y * eps
        directional = (cy * v).sum()
        Hz = torch.autograd.grad(directional, z, create_graph=True, retain_graph=True)[0][:, 1:]
        trace_terms.append((Hz * v).sum(-1))
    hess_trace = torch.stack(trace_terms).mean(0)
    grad_sq = ((vol * y) ** 2 * cy.square()).sum(-1)
    diffusion = 0.5 * (hess_trace - CARA * grad_sq)

    ratio = cy / cx[:, None].clamp_min(1e-9)
    u = (torch.relu(ratio - 1.0) - torch.relu(1.0 - ALPHA - ratio)) / C.eta

    # Long-only state constraint: suppress infinitesimal selling exactly at y=0.
    u = torch.where((y <= 1e-8) & (u < 0.0), torch.zeros_like(u), u)
    gain = torch.where(
        u >= 0.0,
        (cy - cx[:, None]) * u,
        (cy - (1.0 - ALPHA) * cx[:, None]) * u,
    )
    ham = (gain - 0.5 * C.eta * cx[:, None] * u.square()).sum(-1)
    residual = ce_t + drift + diffusion + ham
    return residual, cx


def train_global_pinn(N, seed):
    seed_all(seed)
    net = GlobalCEPINN(N).to(device=DEVICE, dtype=torch.float64)
    opt = torch.optim.Adam(net.parameters(), lr=C.pinn_lr)

    for step in range(C.pinn_steps):
        z = sample_training_states(N, C.pinn_batch, DEVICE, interior=True, dtype=torch.float64).requires_grad_(True)
        t = torch.rand((C.pinn_batch,), device=DEVICE, dtype=torch.float64, requires_grad=True)
        res, cx = pinn_residual(net, N, t, z)
        loss = res.square().mean() + 5.0 * torch.relu(1e-7 - cx).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

        if step % max(C.pinn_steps // 5, 1) == 0 or step == C.pinn_steps - 1:
            print(f"    PINN step={step:5d} loss={loss.item():.4e}")

    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def pinn_ratios(net, k, z):
    inp = z.detach().double().requires_grad_(True)
    t = torch.full((len(inp),), k / STEPS, device=inp.device, dtype=inp.dtype)
    ce = net(t, inp)
    g = torch.autograd.grad(ce.sum(), inp)[0]
    gx = g[:, 0]
    gy = g[:, 1:]
    valid = torch.isfinite(g).all(-1) & (gx > 1e-9)
    r = gy / gx[:, None].clamp_min(1e-9)
    valid = valid & torch.isfinite(r).all(-1)
    return r.float().detach(), valid.detach()


# Global ratio distillation
class GlobalStudent(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.net = MLP(N + 1, N, C.student_hidden, C.student_depth)

    def forward(self, t, z):
        # Midpoint-centered output, same structural convention as the paper.
        return (1.0 - ALPHA / 2.0) + C.eta * self.net(policy_features(t, z))


def train_global_student(N, regularized_policy, seed):
    seed_all(seed)
    rng = np.random.default_rng(seed)
    n = C.teacher_size
    z = sample_training_states(N, n, DEVICE, interior=True)
    ks = rng.integers(0, STEPS, size=n)
    target = torch.full((n, N), float("nan"), device=DEVICE)
    valid = torch.zeros(n, dtype=torch.bool, device=DEVICE)

    for k in range(STEPS):
        ids = np.flatnonzero(ks == k)
        if not len(ids):
            continue
        r, good = bptt_ratios(N, regularized_policy, k, z[ids], seed + k)
        target[ids] = r
        valid[ids] = good

    perm = rng.permutation(n)
    cut = int(0.8 * n)
    tr = np.array([i for i in perm[:cut] if valid[i].item()], dtype=int)
    va = np.array([i for i in perm[cut:] if valid[i].item()], dtype=int)
    if len(tr) < 16 or len(va) < 8:
        raise RuntimeError("Too few valid BPTT teacher states for distillation.")

    net = GlobalStudent(N).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=C.student_lr)
    tt = torch.tensor(ks / STEPS, device=DEVICE, dtype=torch.float32)
    best = float("inf")
    best_state = None

    for step in range(C.student_steps):
        net.train()
        idx = np.random.choice(tr, size=min(C.batch, len(tr)), replace=True)
        pred = net(tt[idx], z[idx])
        loss = ((pred - target[idx]) / C.eta).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

        if step % 25 == 0 or step == C.student_steps - 1:
            net.eval()
            with torch.no_grad():
                vl = ((net(tt[va], z[va]) - target[va]) / C.eta).square().mean().item()
            if vl < best:
                best = vl
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

    net.load_state_dict(best_state)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, {"student_val_loss_eta_scaled": best, "teacher_valid": int(valid.sum())}


# Regime labels and evaluation
def label_orders(a, tol):
    return torch.where(
        a > tol,
        torch.ones_like(a, dtype=torch.int8),
        torch.where(a < -tol, -torch.ones_like(a, dtype=torch.int8), torch.zeros_like(a, dtype=torch.int8)),
    )


def label_ratio(r):
    return torch.where(
        r > 1.0,
        torch.ones_like(r, dtype=torch.int8),
        torch.where(r < 1.0 - ALPHA, -torch.ones_like(r, dtype=torch.int8), torch.zeros_like(r, dtype=torch.int8)),
    )


def ratio_to_orders(z, r):
    rate = (torch.relu(r - 1.0) - torch.relu(1.0 - ALPHA - r)) / C.eta
    q = rate * DT
    return project_long_only_orders(z, q)


def predictions(N, models, k, z, seed):
    t = torch.full((len(z),), k / STEPS, device=z.device)
    with torch.no_grad():
        u0 = policy_orders(models["u0"], t, z)
        ue = policy_orders(models["ueta"], t, z)

    rb, vb = bptt_ratios(N, models["ueta"], k, z, seed)
    rp, vp = pinn_ratios(models["pinn"], k, z)
    with torch.no_grad():
        rs = models["student"](t, z)
    vs = torch.isfinite(rs).all(-1)

    acts = {
        "No-Action": torch.zeros_like(u0),
        "u_theta,0": u0,
        "u_theta,eta": ue,
    }

    # Coordinatewise magnitude masking baselines:
    # coordinate i is held iff |q_i| < tau, independently of all other coords.
    for tau in C.magnitude_thresholds:
        acts[f"Coordinate mask tau={tau:g} on u_theta,0"] = torch.where(
            u0.abs() >= tau, u0, torch.zeros_like(u0)
        )

    acts.update({
        "BPTT hold gate on u_theta,0": torch.where(
            label_ratio(rb) == 0, torch.zeros_like(u0), u0
        ),
        "P_eta(R_BPTT)": ratio_to_orders(z, rb),
        "P_eta(R_PINN)": ratio_to_orders(z, rp),
        "P_eta(R_theta)": ratio_to_orders(z, rs),
    })
    validity = {name: torch.ones(len(z), device=z.device, dtype=torch.bool) for name in acts}
    validity["BPTT hold gate on u_theta,0"] = vb
    validity["P_eta(R_BPTT)"] = vb
    validity["P_eta(R_PINN)"] = vp
    validity["P_eta(R_theta)"] = vs

    ratios = {
        "BPTT ratio [raw]": (rb, vb),
        "PINN ratio [raw]": (rp, vp),
        "Student ratio [raw]": (rs, vs),
    }
    return acts, validity, ratios


def metrics_from_labels(N, name, truth, pred, valid):
    mask = valid[:, None] & (pred != 99)
    pa = (pred != 0)[mask]
    ta = (truth != 0)[mask]
    pl = pred[mask]
    tl = truth[mask]

    tp = int((pa & ta).sum())
    tn = int(((~pa) & (~ta)).sum())
    fp = int((pa & (~ta)).sum())
    fn = int(((~pa) & ta).sum())
    action_recall = tp / max(tp + fn, 1)
    hold_recall = tn / max(tn + fp, 1)

    return {
        "N": N,
        "method": name,
        "n_state": len(truth),
        "n_coordinates": int(mask.sum()),
        "invalid_pct": 100.0 * (1.0 - float(mask.mean())),
        "action_inact_acc": float((pa == ta).mean()),
        "regime_acc": float((pl == tl).mean()),
        "hold_iou": float(tn / max(tn + fn + fp, 1)),
        "action_recall": float(action_recall),
        "hold_recall": float(hold_recall),
        "balanced_binary_acc": float(0.5 * (action_recall + hold_recall)),
        "full_vector_acc": float((((pred == truth) & mask).all(axis=1)).mean()),
        "dp_buy_prevalence": float((truth == 1).mean()),
        "dp_hold_prevalence": float((truth == 0).mean()),
        "dp_sell_prevalence": float((truth == -1).mean()),
    }


def evaluate_one_N(N, models, run_seed):
    data = np.load(ORACLE_DIR / manifest["datasets"][str(N)]["file"])
    if str(data["oracle_signature"].item()) != manifest["oracle_signature"]:
        raise RuntimeError("Frozen test set/oracle signature mismatch.")

    ks = data["time_index"].astype(int)
    z_np = data["state"].astype(np.float32)
    truth = data["truth"].astype(np.int8)
    z = torch.tensor(z_np, device=DEVICE)

    n = len(z_np)
    store = {}
    valid_store = {}

    for k in range(STEPS):
        ids = np.flatnonzero(ks == k)
        if not len(ids):
            continue

        # IMPORTANT: use a common evaluation-MC bank for every training seed.
        # If you instead want "full-pipeline" randomness in the reported std,
        # replace C.eval_seed below by run_seed.
        eval_mc_seed = C.eval_seed + 100000 * N + k
        acts, val, rat = predictions(
            N, models, k, z[ids], eval_mc_seed
        )

        for name, a in acts.items():
            store.setdefault(name, np.full((n, N), 99, dtype=np.int8))
            store[name][ids] = label_orders(a, C.label_tol).cpu().numpy()
            valid_store.setdefault(name, np.ones(n, dtype=bool))
            valid_store[name][ids] = val[name].cpu().numpy()

        for name, (r, v) in rat.items():
            store.setdefault(name, np.full((n, N), 99, dtype=np.int8))
            store[name][ids] = label_ratio(r).cpu().numpy()
            valid_store.setdefault(name, np.ones(n, dtype=bool))
            valid_store[name][ids] = v.cpu().numpy()

    rows = [
        metrics_from_labels(N, name, truth, pred, valid_store[name])
        for name, pred in store.items()
    ]
    for row in rows:
        row["seed"] = int(run_seed)

    # Seed-specific files: nothing is overwritten by later seeds.
    pd.DataFrame(rows).to_csv(
        OUT / f"N{N}_seed{run_seed}_regime_accuracy.csv",
        index=False,
    )
    np.savez_compressed(
        OUT / f"N{N}_seed{run_seed}_predictions.npz",
        time_index=ks,
        state=z_np,
        truth=truth,
        **{f"pred_{i}": p for i, p in enumerate(store.values())},
    )
    return rows


def aggregate_multiseed(df):
    """
    Aggregate stochastic performance metrics over run seeds.

    pandas std() uses ddof=1, i.e. SAMPLE standard deviation.
    Counts and DP class prevalences are frozen-test-set descriptors, so they
    remain in the raw per-seed CSV rather than being treated as stochastic
    performance metrics here.
    """
    metric_cols = [
        "invalid_pct",
        "action_inact_acc",
        "regime_acc",
        "hold_iou",
        "action_recall",
        "hold_recall",
        "balanced_binary_acc",
        "full_vector_acc",
    ]

    summary = (
        df.groupby(["N", "method"], as_index=False)[metric_cols]
        .agg(["mean", "std", "min", "max"])
    )

    # Flatten MultiIndex columns:
    # ('regime_acc', 'mean') -> 'regime_acc_mean'
    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]

    # groupby(..., as_index=False) + multi-agg can still yield names N_ / method_
    summary = summary.rename(columns={"N_": "N", "method_": "method"})

    # Defensive: with a single seed pandas sample std is NaN.
    # For >=2 seeds (the intended setting) these are ordinary sample stds.
    return summary


def plot_summary(summary):
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    methods = [
        "u_theta,0",
        "u_theta,eta",
        "Coordinate mask tau=0.001 on u_theta,0",
        "Coordinate mask tau=0.01 on u_theta,0",
        "Coordinate mask tau=0.1 on u_theta,0",
        "BPTT hold gate on u_theta,0",
        "P_eta(R_BPTT)",
        "P_eta(R_PINN)",
        "P_eta(R_theta)",
    ]
    specs = [
        ("action_inact_acc", "Action / inaction accuracy", "binary"),
        ("regime_acc", "Buy / hold / sell accuracy", "three_class"),
        ("hold_iou", "Hold IoU", "hold_iou"),
    ]

    for key, ylabel, fname in specs:
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        for m in methods:
            sub = summary[summary.method == m].sort_values("N")
            if not len(sub):
                continue

            x = sub["N"].to_numpy(dtype=float)
            mean = sub[f"{key}_mean"].to_numpy(dtype=float)
            std = sub[f"{key}_std"].fillna(0.0).to_numpy(dtype=float)

            # Mean curve with +/- 1 sample-standard-deviation band.
            line = ax.plot(x, mean, marker="o", label=m)[0]
            ax.fill_between(
                x,
                np.clip(mean - std, 0.0, 1.0),
                np.clip(mean + std, 0.0, 1.0),
                alpha=0.12,
                color=line.get_color(),
            )

        ax.set_xlabel("Number of risky assets N")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.01)
        ax.set_xticks(DIMS)
        ax.grid(ls=":", alpha=0.4)
        ax.legend(fontsize=8, ncol=2, loc="lower left")
        fig.tight_layout()
        fig.savefig(OUT / f"{fname}_multiseed.pdf")
        fig.savefig(OUT / f"{fname}_multiseed.png", dpi=220)
        plt.close(fig)


# Run all dimensions x all seeds
all_rows = []

for N in DIMS:
    print(f"\n===== GLOBAL neural experiment N={N} =====")
    print(f"state_dim={N+1}, network outputs={N}, covariance=diagonal")

    for run_idx, run_seed in enumerate(C.seeds, start=1):
        run_seed = int(run_seed)
        print(
            f"\n--- seed {run_idx}/{len(C.seeds)}: {run_seed} "
            f"(N={N}) ---"
        )

        print("  train global u_theta,0")
        u0 = train_global_policy(N, 0.0, run_seed + 10000 + N)

        print("  train global u_theta,eta")
        ue = train_global_policy(N, C.eta, run_seed + 20000 + N)

        print("  train global CARA CE-PINN")
        pn = train_global_pinn(N, run_seed + 30000 + N)

        print("  BPTT teacher -> global ratio distillation")
        st, diag = train_global_student(N, ue, run_seed + 40000 + N)

        models = {"u0": u0, "ueta": ue, "pinn": pn, "student": st}

        # Seed-specific checkpoints.
        torch.save(
            u0.state_dict(),
            OUT / f"N{N}_seed{run_seed}_u0_global.pt",
        )
        torch.save(
            ue.state_dict(),
            OUT / f"N{N}_seed{run_seed}_ueta_global.pt",
        )
        torch.save(
            pn.state_dict(),
            OUT / f"N{N}_seed{run_seed}_pinn_global.pt",
        )
        torch.save(
            st.state_dict(),
            OUT / f"N{N}_seed{run_seed}_student_global.pt",
        )
        (OUT / f"N{N}_seed{run_seed}_distill_diag.json").write_text(
            json.dumps(diag, indent=2),
            encoding="utf-8",
        )

        rows = evaluate_one_N(N, models, run_seed)
        all_rows.extend(rows)

        show = pd.DataFrame(rows)[
            [
                "method",
                "action_inact_acc",
                "regime_acc",
                "hold_iou",
                "balanced_binary_acc",
                "invalid_pct",
            ]
        ]
        print(show.to_string(index=False))

        # Free GPU memory before the next independent run.
        del models, u0, ue, pn, st
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Raw per-seed results + mean/std/min/max summary
df = pd.DataFrame(all_rows)
df = df[
    [
        "N",
        "seed",
        "method",
        "n_state",
        "n_coordinates",
        "invalid_pct",
        "action_inact_acc",
        "regime_acc",
        "hold_iou",
        "action_recall",
        "hold_recall",
        "balanced_binary_acc",
        "full_vector_acc",
        "dp_buy_prevalence",
        "dp_hold_prevalence",
        "dp_sell_prevalence",
    ]
]
df.to_csv(OUT / "regime_accuracy_all_seeds.csv", index=False)

summary = aggregate_multiseed(df)
summary.to_csv(
    OUT / "regime_accuracy_multiseed_summary.csv",
    index=False,
)

plot_summary(summary)
(OUT / "main_config.json").write_text(
    json.dumps(asdict(C), indent=2),
    encoding="utf-8",
)

# Compact console table for the main three reported metrics.
main_cols = [
    "N",
    "method",
    "action_inact_acc_mean",
    "action_inact_acc_std",
    "action_inact_acc_min",
    "action_inact_acc_max",
    "regime_acc_mean",
    "regime_acc_std",
    "regime_acc_min",
    "regime_acc_max",
    "hold_iou_mean",
    "hold_iou_std",
    "hold_iou_min",
    "hold_iou_max",
]

print("\n===== MULTI-SEED SUMMARY: mean / std / min / max =====")
print(summary[main_cols].to_string(index=False))

print("\nDONE:", OUT.resolve())
print("seeds:", tuple(int(s) for s in C.seeds))
print(
    "STD DEFINITION: sample standard deviation across run seeds (ddof=1). "
    "Frozen oracle/test states and BPTT evaluation MC bank are shared across seeds."
)
print(
    "DESIGN CHECK: DP truth is 1D-per-asset due to independent CARA structure; "
    "every learned baseline was trained as one GLOBAL N-dimensional network. "
    "DPO uses uniform random-k starts, 4-path statewise antithetic CARA CE, "
    "and random-coordinate boundary-state sampling."
)
