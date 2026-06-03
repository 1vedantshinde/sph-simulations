"""
comparison.py
=============
Validates the SPH solution against the exact Riemann solver,
computes L2 errors, and produces a publication-quality comparison plot.

Usage:
    python comparison.py [--N 800] [--t_end 0.2]
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ── Import SPH solver ─────────────────────────────────────────────────────────
from sph_solver import run_sph, GAMMA



# ══════════════════════════════════════════════════════════════════════════════
# Fallback: analytic Sod solution (Toro 2009, Ch. 4)
# ══════════════════════════════════════════════════════════════════════════════

def _sod_exact(x_grid, t=0.2):
    """
    Analytic solution to the standard Sod problem at time t.

    Left state : rho=1,     P=1,   v=0
    Right state: rho=0.125, P=0.1, v=0
    Discontinuity at x=0.5.

    Returns rho, v, P on x_grid.
    """
    gamma = GAMMA

    # ── Known Sod solution values at t=0.2 ───────────────────────────────
    # (from Toro Table 4.1 / Wikipedia Sod shock tube)
    rho_l, P_l, v_l = 1.0,   1.0,   0.0
    rho_r, P_r, v_r = 0.125, 0.1,   0.0
    x0 = 0.5

    # Sound speeds
    c_l = np.sqrt(gamma * P_l / rho_l)
    c_r = np.sqrt(gamma * P_r / rho_r)

    # ── Star-state values (Newton–Raphson, converged) ─────────────────────
    # These are the textbook converged values for gamma=1.4 Sod:
    P_star = 0.30313017805064745
    v_star = 0.92745262004895055 * 0.5   # ≈ 0.46376...
    # Actually recompute properly via iteration
    def f(P_s, rho_k, P_k, c_k):
        if P_s > P_k:   # shock
            A = 2.0 / ((gamma + 1) * rho_k)
            B = (gamma - 1) / (gamma + 1) * P_k
            return (P_s - P_k) * np.sqrt(A / (P_s + B))
        else:            # rarefaction
            return (2 * c_k / (gamma - 1)) * ((P_s / P_k)**((gamma-1)/(2*gamma)) - 1)

    def df(P_s, rho_k, P_k, c_k):
        if P_s > P_k:
            A = 2.0 / ((gamma + 1) * rho_k)
            B = (gamma - 1) / (gamma + 1) * P_k
            return np.sqrt(A / (P_s + B)) * (1 - (P_s - P_k) / (2 * (P_s + B)))
        else:
            return (1.0 / (rho_k * c_k)) * (P_s / P_k)**(-(gamma+1)/(2*gamma))

    P_s = 0.5 * (P_l + P_r)   # initial guess
    for _ in range(100):
        F  = f(P_s, rho_l, P_l, c_l) + f(P_s, rho_r, P_r, c_r) + (v_r - v_l)
        dF = df(P_s, rho_l, P_l, c_l) + df(P_s, rho_r, P_r, c_r)
        P_s -= F / dF
        if abs(F / dF) < 1e-12:
            break

    v_s = 0.5 * (v_l + v_r) + 0.5 * (f(P_s, rho_r, P_r, c_r)
                                       - f(P_s, rho_l, P_l, c_l))

    # Density in star regions
    rho_sl = rho_l * (P_s / P_l)**(1.0 / gamma)                     # rarefaction fan
    rho_sr = rho_r * ((P_s / P_r + (gamma-1)/(gamma+1))
                       / ((gamma-1)/(gamma+1) * P_s / P_r + 1))      # shock

    # Wave speeds
    S_shock = v_r + c_r * np.sqrt((gamma+1)/(2*gamma) * P_s/P_r
                                   + (gamma-1)/(2*gamma))             # right shock
    S_hl    = v_l - c_l                                               # left fan head
    c_sl    = c_l * (P_s / P_l)**((gamma-1)/(2*gamma))
    S_tl    = v_s - c_sl                                              # left fan tail

    # ── Sample solution on x_grid ─────────────────────────────────────────
    xi = (x_grid - x0) / t   # self-similar variable

    rho_sol = np.zeros_like(x_grid)
    v_sol   = np.zeros_like(x_grid)
    P_sol   = np.zeros_like(x_grid)

    for k, s in enumerate(xi):
        if s <= S_hl:                              # undisturbed left
            rho_sol[k] = rho_l; v_sol[k] = v_l; P_sol[k] = P_l
        elif s <= S_tl:                            # rarefaction fan
            u_fan = 2/(gamma+1) * (c_l + s)       # velocity inside fan (Toro 4.56)
            c_fan = c_l - (gamma-1)/2 * u_fan      # sound speed inside fan
            # wait—apply Toro's formula properly:
            _c   = 2/(gamma+1) * (c_l + (gamma-1)/2 * v_l + s)
            # Toro (4.56): v = 2/(γ+1)*(c_l + (γ-1)/2*v_l + ξ)
            _v   = 2/(gamma+1) * (c_l + (gamma-1)/2 * v_l + s)
            _c   = c_l - (gamma-1)/2 * (_v - v_l)
            _rho = rho_l * (_c / c_l)**(2/(gamma-1))
            _P   = P_l   * (_c / c_l)**(2*gamma/(gamma-1))
            rho_sol[k] = _rho; v_sol[k] = _v; P_sol[k] = _P
        elif s <= v_s:                             # left star region
            rho_sol[k] = rho_sl; v_sol[k] = v_s; P_sol[k] = P_s
        elif s <= S_shock:                         # right star region
            rho_sol[k] = rho_sr; v_sol[k] = v_s; P_sol[k] = P_s
        else:                                      # undisturbed right
            rho_sol[k] = rho_r; v_sol[k] = v_r; P_sol[k] = P_r

    return rho_sol, v_sol, P_sol


# ══════════════════════════════════════════════════════════════════════════════
# L2 error computation
# ══════════════════════════════════════════════════════════════════════════════

def interpolate_to_grid(x_particles, q_particles, x_grid):
    """
    Interpolate particle quantity q onto a uniform grid.

    Uses linear interpolation after sorting particles by position.
    This is correct as long as particles are not too disordered.

    Parameters
    ----------
    x_particles : (N,) sorted particle positions
    q_particles : (N,) particle quantity
    x_grid      : (M,) uniform grid positions

    Returns
    -------
    q_grid : (M,) interpolated values
    """
    idx    = np.argsort(x_particles)
    x_s    = x_particles[idx]
    q_s    = q_particles[idx]
    interp = interp1d(x_s, q_s, kind="linear", bounds_error=False,
                      fill_value=(q_s[0], q_s[-1]))
    return interp(x_grid)


def l2_error(q_sph, q_exact, dx):
    """
    Discrete L2 norm: sqrt( Σ (q_sph - q_exact)² · dx ) / sqrt( Σ q_exact² · dx )

    This is the normalised, volume-weighted RMS error.
    """
    num   = np.sqrt(np.sum((q_sph - q_exact)**2) * dx)
    denom = np.sqrt(np.sum(q_exact**2) * dx)
    return num / denom


#Comparison Plots
def make_comparison_plot(x_sph, rho_sph, v_sph, P_sph,
                         x_ex, rho_ex, v_ex, P_ex,
                         t_end, errors, save_path="sph_vs_exact.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"SPH vs Exact Riemann — Sod Shock Tube  (t = {t_end:.2f})",
                 fontsize=13, fontweight="bold")

    datasets = [
        (rho_sph, rho_ex, "Density ρ",   errors["rho"]),
        (v_sph,   v_ex,   "Velocity v",  errors["v"]),
        (P_sph,   P_ex,   "Pressure P",  errors["P"]),
    ]

    for ax, (sph_q, ex_q, label, err) in zip(axes, datasets):
        ax.plot(x_ex, ex_q, "k-",  lw=2,   label="Exact Riemann", zorder=3)
        ax.scatter(x_sph, sph_q,   s=4, c="#e74c3c", alpha=0.6,
                   label=f"SPH  (L2={err:.3e})", zorder=2)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {save_path}")




def main():
    parser = argparse.ArgumentParser(description="SPH vs Exact Riemann comparison")
    parser.add_argument("--N",     type=int,   default=800,  help="Particle count")
    parser.add_argument("--t_end", type=float, default=0.2,  help="End time")
    parser.add_argument("--alpha", type=float, default=1.0,  help="AV alpha")
    parser.add_argument("--beta",  type=float, default=2.0,  help="AV beta")
    parser.add_argument("--alpha_u", type=float, default=1.0,  help="Thermal conduction alpha_u")
    parser.add_argument("--eta",   type=float, default=1.5,  help="Smoothing length eta factor")
    args = parser.parse_args()

    print(f"Running SPH  N={args.N}  t_end={args.t_end}  alpha_u={args.alpha_u}  eta={args.eta} …")
    x_sph, rho_sph, v_sph, P_sph, u_sph, t_end, wt = run_sph(
        N=args.N, t_end=args.t_end,
        alpha=args.alpha, beta=args.beta,
        alpha_u=args.alpha_u, eta=args.eta, verbose=True
    )
    print(f"Wall time: {wt:.2f} s")

    # Exact solution on fine grid 
    x_grid    = np.linspace(0.0, 1.0, 2000)
    rho_ex, v_ex, P_ex = _sod_exact(x_grid, t=t_end)

    # Interpolate SPH onto same grid
    dx        = x_grid[1] - x_grid[0]
    rho_interp = interpolate_to_grid(x_sph, rho_sph, x_grid)
    v_interp   = interpolate_to_grid(x_sph, v_sph,   x_grid)
    P_interp   = interpolate_to_grid(x_sph, P_sph,   x_grid)

    #L2 errors
    errors = {
        "rho": l2_error(rho_interp, rho_ex, dx),
        "v":   l2_error(v_interp,   v_ex,   dx),
        "P":   l2_error(P_interp,   P_ex,   dx),
    }
    print("\n── L2 Errors ──────────────────────────────")
    for k, v in errors.items():
        print(f"  {k:4s}: {v:.4e}")

    # Plot 
    make_comparison_plot(x_sph, rho_sph, v_sph, P_sph,
                         x_grid, rho_ex, v_ex, P_ex,
                         t_end, errors)


if __name__ == "__main__":
    main()
