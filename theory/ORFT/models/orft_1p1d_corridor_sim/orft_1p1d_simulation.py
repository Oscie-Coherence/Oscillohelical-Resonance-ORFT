#!/usr/bin/env python3
"""
ORFT — 1+1D Corridor-Locked Phase Field Simulation 
=================================================================

This simulation is a minimal, reproducible numerical demo consistent with ORFT-style
constraints described in the ORFT whitepaper (corridor locking under noise, localized
structures, early warning via coherence-gradient growth).

Partner-safe posture:
- Offline, no network calls.
- Deterministic given a seed.
- No proprietary Oscie internals; reference-grade demo.
- Uses numpy only.

Model overview
--------------
We simulate a 1D lattice of phases theta[x] with periodic boundary conditions.

We compute a local circular coherence order parameter r(x):
    Z(x) = mean_{i in neighborhood} exp(i * theta[i])
    r(x) = |Z(x)| in [0,1]

A corridor potential V_band(r) is applied per site, centered at C_star (~0.605):
    V = kappa*(d)^2 + lam*(d)^4 + B*ReLU(|d|-delta)^p
where d = r - C_star.

Dynamics are overdamped Langevin-like:
    theta <- theta + dt * force(theta) + noise

Forces:
- Spatial smoothness via a Laplacian term (encourages coherent localized configurations).
- Corridor-driven "sync" term: nudges theta toward the local mean angle when raising coherence
  reduces corridor energy; nudges away when lowering coherence reduces corridor energy.

This is an interpretable approximation that produces the qualitative behaviors:
- corridor convergence
- resilience under sustained noise
- coherence-gradient growth as an early warning proxy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np


# -----------------------------
# Utilities
# -----------------------------
def wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def global_coherence(theta: np.ndarray) -> float:
    """Global circular coherence r = |mean exp(i theta)|."""
    z = np.mean(np.cos(theta) + 1j * np.sin(theta))
    return float(np.abs(z))


def local_coherence(theta: np.ndarray, window: int) -> np.ndarray:
    """
    Local circular coherence r(x) over a sliding window of radius `window`.
    Periodic boundary conditions.
    """
    n = theta.size
    w = int(window)
    if w < 1:
        return np.full(n, global_coherence(theta), dtype=np.float64)

    c = np.cos(theta)
    s = np.sin(theta)
    denom = float(2 * w + 1)

    r = np.zeros(n, dtype=np.float64)
    for x in range(n):
        idx = [(x + j) % n for j in range(-w, w + 1)]
        z_re = float(np.sum(c[idx]) / denom)
        z_im = float(np.sum(s[idx]) / denom)
        r[x] = math.sqrt(z_re * z_re + z_im * z_im)

    return r


def local_mean_angle(theta: np.ndarray, window: int) -> np.ndarray:
    """
    Local circular mean angle phi(x) over neighborhood radius `window`.
    """
    n = theta.size
    w = int(window)
    if w < 1:
        z = np.mean(np.cos(theta) + 1j * np.sin(theta))
        phi0 = math.atan2(z.imag, z.real)
        return np.full(n, phi0, dtype=np.float64)

    denom = float(2 * w + 1)
    phi = np.zeros(n, dtype=np.float64)
    for x in range(n):
        idx = [(x + j) % n for j in range(-w, w + 1)]
        z_re = float(np.sum(np.cos(theta[idx])) / denom)
        z_im = float(np.sum(np.sin(theta[idx])) / denom)
        phi[x] = math.atan2(z_im, z_re)
    return phi


# -----------------------------
# Corridor potential
# -----------------------------
@dataclass(frozen=True)
class CorridorConfig:
    C_star: float = 0.605
    delta: float = 0.015
    kappa: float = 10.0
    lam: float = 40.0
    B: float = 80.0
    p: float = 2.0


def v_band(C: np.ndarray, cfg: CorridorConfig) -> np.ndarray:
    d = C - cfg.C_star
    core = cfg.kappa * (d * d) + cfg.lam * (d ** 4)
    wall = cfg.B * (relu(np.abs(d) - cfg.delta) ** cfg.p)
    return core + wall


def dv_dC(C: np.ndarray, cfg: CorridorConfig) -> np.ndarray:
    d = C - cfg.C_star
    core = 2.0 * cfg.kappa * d + 4.0 * cfg.lam * (d ** 3)

    mask = (np.abs(d) > cfg.delta).astype(np.float64)
    wall_mag = relu(np.abs(d) - cfg.delta)

    if abs(cfg.p - 1.0) < 1e-12:
        wall_pow = np.ones_like(wall_mag)
    else:
        wall_pow = wall_mag ** (cfg.p - 1.0)

    wall = cfg.B * cfg.p * wall_pow * np.sign(d) * mask
    return core + wall


# -----------------------------
# Simulation config
# -----------------------------
@dataclass(frozen=True)
class SimConfig:
    n: int = 256
    steps: int = 8000
    dt: float = 0.02
    K: float = 0.40
    noise: float = 0.12
    damping: float = 1.0
    local_window: int = 6
    snapshot_every: int = 400
    seed: int = 7


def corridor_hit(Cg: float) -> int:
    """Global corridor indicator for [0.59, 0.62]."""
    return int(0.59 <= Cg <= 0.62)


def run_sim(cfg: SimConfig, ccfg: CorridorConfig, out_dir: str) -> Dict[str, float]:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    # Init: random phases + localized bump (seeds a stable structure under corridor lock)
    theta = wrap_pi(rng.uniform(-np.pi, np.pi, size=cfg.n).astype(np.float64))
    center = cfg.n // 2
    width = max(3, cfg.n // 30)
    bump = np.exp(-((np.arange(cfg.n) - center) ** 2) / (2.0 * width * width))
    theta = wrap_pi(theta + 0.8 * bump)

    # Write config
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"sim": asdict(cfg), "corridor": asdict(ccfg)}, f, indent=2, sort_keys=True)

    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step",
                "t",
                "C_global",
                "C_local_mean",
                "C_local_min",
                "C_local_max",
                "corridor_hit_global_059_062",
                "grad_energy",
                "band_energy_mean",
                "early_warning_coherence_grad_rms",
            ]
        )

        for step in range(cfg.steps + 1):
            t = step * cfg.dt

            C_local = local_coherence(theta, cfg.local_window)
            Cg = global_coherence(theta)

            dtheta = wrap_pi(np.roll(theta, -1) - theta)
            grad_E = float(0.5 * cfg.K * np.sum(dtheta * dtheta))

            band_E = v_band(C_local, ccfg)
            band_E_mean = float(np.mean(band_E))

            dC = np.roll(C_local, -1) - C_local
            ew_rms = float(np.sqrt(np.mean(dC * dC)))

            w.writerow(
                [
                    step,
                    t,
                    Cg,
                    float(np.mean(C_local)),
                    float(np.min(C_local)),
                    float(np.max(C_local)),
                    corridor_hit(Cg),
                    grad_E,
                    band_E_mean,
                    ew_rms,
                ]
            )

            if cfg.snapshot_every > 0 and step % cfg.snapshot_every == 0:
                np.savez_compressed(
                    os.path.join(out_dir, f"snapshot_{step:07d}.npz"),
                    theta=theta.astype(np.float32),
                    C_local=C_local.astype(np.float32),
                    step=np.int64(step),
                    t=np.float64(t),
                )

            if step == cfg.steps:
                break

            # -------- Forces (overdamped) --------
            # Spatial Laplacian smoothing
            lap = (2.0 * theta - np.roll(theta, -1) - np.roll(theta, 1))
            force_grad = -cfg.K * lap

            # Corridor-driven sync/desync via chain-rule approximation
            dV = dv_dC(C_local, ccfg)
            phi = local_mean_angle(theta, cfg.local_window)
            dphi = wrap_pi(theta - phi)

            # If -dV positive => synchronizing reduces energy => push dphi toward 0
            sync_gain = np.clip(-dV, -50.0, 50.0)
            force_band = -sync_gain * dphi

            # Noise
            noise_term = cfg.noise * np.sqrt(cfg.dt) * rng.normal(0.0, 1.0, size=cfg.n)

            theta = theta + cfg.damping * cfg.dt * (force_grad + force_band) + noise_term
            theta = wrap_pi(theta)

    summary = {
        "out_dir": out_dir,
        "final_C_global": float(global_coherence(theta)),
        "final_C_local_mean": float(np.mean(local_coherence(theta, cfg.local_window))),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ORFT 1+1D corridor-locked phase field sim (partner-safe).")
    # sim
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--K", type=float, default=0.40)
    p.add_argument("--noise", type=float, default=0.12)
    p.add_argument("--damping", type=float, default=1.0)
    p.add_argument("--local-window", type=int, default=6)
    p.add_argument("--snapshot-every", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=str, default="out")
    # corridor
    p.add_argument("--C-star", type=float, default=0.605)
    p.add_argument("--delta", type=float, default=0.015)
    p.add_argument("--kappa", type=float, default=10.0)
    p.add_argument("--lam", type=float, default=40.0)
    p.add_argument("--B", type=float, default=80.0)
    p.add_argument("--p", type=float, default=2.0)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    sim_cfg = SimConfig(
        n=args.n,
        steps=args.steps,
        dt=args.dt,
        K=args.K,
        noise=args.noise,
        damping=args.damping,
        local_window=args.local_window,
        snapshot_every=args.snapshot_every,
        seed=args.seed,
    )
    cor_cfg = CorridorConfig(
        C_star=args.C_star,
        delta=args.delta,
        kappa=args.kappa,
        lam=args.lam,
        B=args.B,
        p=args.p,
    )

    summary = run_sim(sim_cfg, cor_cfg, args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
