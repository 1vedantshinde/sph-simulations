"""
SPH Sod Shock Tube Solver
=========================
Fully Lagrangian SPH implementation for the 1D Sod shock tube problem.

Backend: NumPy (CPU) or CuPy (GPU) — controlled by USE_GPU flag.

Key numerical choices:
  - Cubic spline kernel  (Monaghan & Lattanzio 1985)
  - Dynamic smoothing length h_i tied to local particle spacing
  - Density by neighbour summation (vectorized offset loop)
  - Symmetric pressure-gradient force (Monaghan 1992, Eq. 3.7)
  - Symmetric energy equation (fully conserving total energy)
  - Monaghan (1992) artificial viscosity (alpha=1, beta=2)
  - Price (2008) artificial thermal conduction to smooth contact discontinuity
  - Leapfrog (kick-drift-kick) time integrator
  - Adaptive CFL timestep (C_CFL=0.3)
"""

import time
import numpy as np

#Gpu switch (currently the code has only been implemented for cpu which means this part is redundant)
USE_GPU = False   # flip to True to use CuPy

if USE_GPU:
    try:
        import cupy as xp
        _gpu_available = True
    except ImportError:
        print("CuPy not available – falling back to NumPy.")
        import numpy as xp
        _gpu_available = False
else:
    import numpy as xp
    _gpu_available = False

GAMMA = 1.4          # ideal gas adiabatic index (Sod standard)

# hard coded initial conditions
# Left state  (x < 0.5)
RHO_L, P_L, V_L = 1.0,  1.0,   0.0
# Right state (x ≥ 0.5)
RHO_R, P_R, V_R = 0.125, 0.1,  0.0



# 1.  KERNEL


def cubic_spline_kernel(r, h):
    """
    Normalised 1-D cubic spline kernel W(r, h).

    W(q) = sigma/h * { 1 - 1.5 q^2 + 0.75 q^3          0 ≤ q < 1
                      { 0.25 (2 - q)^3                   1 ≤ q < 2
                      { 0                                 q ≥ 2

    sigma = 2/3  (1-D normalisation)
    Support radius = 2h.
    """
    sigma = 2.0 / 3.0
    if xp.isscalar(h):
        h = xp.ones_like(r) * h
    q = r / h
    W = xp.zeros_like(q)

    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)

    W[m1] = sigma / h[m1] * (1.0 - 1.5 * q[m1]**2 + 0.75 * q[m1]**3)
    W[m2] = sigma / h[m2] * 0.25 * (2.0 - q[m2])**3
    return W


def cubic_spline_kernel_grad(dx, r, h):
    """
    Gradient dW/dx_i = (dW/dr)(x_i - x_j)/r  (scalar 1-D).

    dW/dq = sigma/h * { -3q + 2.25 q^2          0 ≤ q < 1
                       { -0.75 (2-q)^2           1 ≤ q < 2
                       { 0                        q ≥ 2

    dW/dx_i = (1/h) * dW/dq * sign(x_i - x_j)
    """
    sigma = 2.0 / 3.0
    if xp.isscalar(h):
        h = xp.ones_like(r) * h
    q = r / h
    dW = xp.zeros_like(q)

    m1 = (q > 0) & (q < 1.0)
    m2 = (q >= 1.0) & (q < 2.0)

    # dW/dq
    dWdq = xp.zeros_like(q)
    dWdq[m1] = sigma * (-3.0 * q[m1] + 2.25 * q[m1]**2)
    dWdq[m2] = sigma * (-0.75 * (2.0 - q[m2])**2)

    # chain rule: dW/dx_i = (dW/dq)(1/h)(dx/r)
    safe_r = xp.where(r > 0, r, 1.0)
    dW = dWdq / (h * h) * (dx / safe_r)
    return dW


# 2.  INITIAL CONDITIONS

def create_sod_initial_conditions(N=800):
    """
    Create 1-D Sod shock tube particle distribution on [0, 1].

    Particle spacing is adjusted so that the denser left half has
    4× as many particles as the right half (matching ρ_L / ρ_R = 8).
    This ensures equal mass per particle.
    """
    ratio = RHO_L / RHO_R    # = 8
    N_L = int(N * ratio / (1.0 + ratio))
    N_R = N - N_L

    x_L = np.linspace(0.0, 0.5, N_L, endpoint=False)
    x_R = np.linspace(0.5, 1.0, N_R, endpoint=False)
    x   = np.concatenate([x_L, x_R])

    rho = np.where(x < 0.5, RHO_L, RHO_R)
    P   = np.where(x < 0.5, P_L,   P_R)
    v   = np.zeros(N)
    # Specific internal energy from ideal gas EOS: u = P / (rho * (gamma-1))
    u   = P / (rho * (GAMMA - 1.0))

    # Equal mass per particle
    dx_L = 0.5 / N_L
    dx_R = 0.5 / N_R
    m_L  = RHO_L * dx_L
    m_R  = RHO_R * dx_R
    m    = np.where(x < 0.5, m_L, m_R)

    eta  = 1.5
    h    = eta * dx_L          # Uniform initial h returned for compatibility

    return (xp.asarray(x, dtype=xp.float64),
            xp.asarray(rho, dtype=xp.float64),
            xp.asarray(v, dtype=xp.float64),
            xp.asarray(u, dtype=xp.float64),
            xp.asarray(m, dtype=xp.float64),
            float(h))


# 3.  DYNAMIC SMOOTHING LENGTH

def compute_h_dynamic(x, eta=1.5, k_spacing=2):
    """
    Computes variable smoothing length h_i based on local particle spacing.
    In 1D, dx_i is computed using central difference spanning 2*k_spacing intervals.
    """
    N = len(x)
    dx = xp.zeros(N, dtype=xp.float64)
    dx[k_spacing:-k_spacing] = (x[2*k_spacing:] - x[:-2*k_spacing]) / (2 * k_spacing)
    # Fill boundary particles
    dx[:k_spacing] = dx[k_spacing]
    dx[-k_spacing:] = dx[-k_spacing-1]
    return eta * dx


# 4.  NEIGHBOUR SEARCH (retained for backward compatibility)

def find_neighbours(x, h):
    """
    Query-ball style neighbour search. Retained for compatibility.
    """
    from scipy.spatial import KDTree as CPUKDTree
    x_np = np.asarray(xp.asnumpy(x) if _gpu_available else x).reshape(-1, 1)
    tree = CPUKDTree(x_np)
    h_val = float(xp.mean(h)) if not isinstance(h, (float, int)) else float(h)
    neighbours = tree.query_ball_point(x_np, r=2.0 * h_val)
    return neighbours


# 5.  DENSITY ESTIMATION

def compute_density(x, m, h, neighbours=None):
    """
    Density by kernel summation: ρ_i = Σ_j m_j W(|x_i - x_j|, h_ij).
    Fully vectorized using offset slicing (since particles are 1D sorted).
    """
    N = len(x)
    if xp.isscalar(h):
        h = xp.ones_like(x) * h
        
    rho = xp.zeros(N, dtype=xp.float64)
    sigma = 2.0 / 3.0
    
    # Self contribution
    rho += m * (sigma / h)
    
    # Loop over neighboring offsets (k_max=10 is safe for eta <= 2.5)
    k_max = 10
    for k in range(1, k_max + 1):
        i = xp.arange(N - k)
        j = i + k
        
        dx = x[i] - x[j]
        r = xp.abs(dx)
        h_ij = 0.5 * (h[i] + h[j])
        
        q = r / h_ij
        W = xp.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        
        W[m1] = sigma / h_ij[m1] * (1.0 - 1.5 * q[m1]**2 + 0.75 * q[m1]**3)
        W[m2] = sigma / h_ij[m2] * 0.25 * (2.0 - q[m2])**3
        
        rho[i] += m[j] * W
        rho[j] += m[i] * W
        
    return rho


# 6.  EQUATION OF STATE

def compute_pressure(rho, u):
    """
    Ideal gas EOS:  P = (gamma - 1) * rho * u
    """
    return (GAMMA - 1.0) * rho * u


# 7.  FORCES & ENERGY DERIVATIVES

def compute_accelerations(x, v, rho, P, u, m, h, alpha=1.0, beta=2.0, alpha_u=1.0, eta_visc=0.01):
    """
    Compute acceleration dv/dt and internal energy rate du/dt.
    Vectorized using index offset loops. Incorporates:
      - Symmetric pressure-gradient force
      - Symmetric energy rate (conserving total energy)
      - Monaghan (1992) artificial viscosity
      - Price (2008) artificial thermal conduction
      - Signal velocity calculation for CFL timestep control
    """
    N = len(x)
    a = xp.zeros(N, dtype=xp.float64)
    du = xp.zeros(N, dtype=xp.float64)
    c = xp.sqrt(GAMMA * P / rho)
    v_sig = xp.copy(c)
    
    sigma = 2.0 / 3.0
    k_max = 10
    
    for k in range(1, k_max + 1):
        i = xp.arange(N - k)
        j = i + k
        
        dx = x[i] - x[j]
        r = xp.abs(dx)
        h_ij = 0.5 * (h[i] + h[j])
        
        mask = r < 2.0 * h_ij
        if not xp.any(mask):
            continue
            
        i_m = i[mask]
        j_m = j[mask]
        dx_m = dx[mask]
        r_m = r[mask]
        h_ij_m = h_ij[mask]
        
        q = r_m / h_ij_m
        dWdq = xp.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        
        dWdq[m1] = sigma * (-3.0 * q[m1] + 2.25 * q[m1]**2)
        dWdq[m2] = sigma * (-0.75 * (2.0 - q[m2])**2)
        
        safe_r = xp.where(r_m > 0, r_m, 1.0)
        dW = dWdq / (h_ij_m * h_ij_m) * (dx_m / safe_r)
        
        # Kernel radial derivative dW/dr
        dW_dr = dWdq / (h_ij_m * h_ij_m)
        
        dv = v[i_m] - v[j_m]
        vdotr = dv * dx_m
        
        c_bar = 0.5 * (c[i_m] + c[j_m])
        rho_bar = 0.5 * (rho[i_m] + rho[j_m])
        
        #Artificial viscosity (added because particles need to "collide")
        mu = xp.zeros_like(vdotr)
        neg_mask = vdotr < 0.0
        if xp.any(neg_mask):
            mu[neg_mask] = h_ij_m[neg_mask] * vdotr[neg_mask] / (r_m[neg_mask]**2 + (eta_visc * h_ij_m[neg_mask])**2)
            
        Pi = xp.zeros_like(mu)
        if xp.any(neg_mask):
            Pi[neg_mask] = (-alpha * c_bar[neg_mask] * mu[neg_mask] + beta * mu[neg_mask]**2) / rho_bar[neg_mask]
            
        # Signal speed update for CFL
        v_sig_pair = c[i_m] + c[j_m] - 3.0 * mu
        xp.maximum.at(v_sig, i_m, v_sig_pair)
        xp.maximum.at(v_sig, j_m, v_sig_pair)
        
        # Momentum acceleration=
        term_a = P[i_m]/rho[i_m]**2 + P[j_m]/rho[j_m]**2 + Pi
        xp.subtract.at(a, i_m, m[j_m] * term_a * dW)
        xp.add.at(a, j_m, m[i_m] * term_a * dW)
        
        # =Symmetric internal energy
        term_u = 0.5 * (P[i_m]/rho[i_m]**2 + P[j_m]/rho[j_m]**2 + Pi)
        xp.add.at(du, i_m, m[j_m] * term_u * dv * dW)
        xp.add.at(du, j_m, m[i_m] * term_u * dv * dW)
        
        #Price (2008) artificial thermal conduction 
        if alpha_u > 0.0:
            v_sig_u = xp.sqrt(xp.abs(P[i_m] - P[j_m]) / rho_bar)
            du_cond = (m[j_m] / rho_bar) * alpha_u * v_sig_u * (u[i_m] - u[j_m]) * dW_dr
            xp.add.at(du, i_m, du_cond)
            xp.add.at(du, j_m, (m[i_m] / rho_bar) * alpha_u * v_sig_u * (u[j_m] - u[i_m]) * dW_dr)
            
    return a, du, v_sig


# 8.  ADAPTIVE TIMESTEP

def compute_timestep(h, v_sig, C_cfl=0.3):
    """
    CFL-limited adaptive timestep.
    """
    return C_cfl * float(xp.min(h / v_sig))


# 9.  LEAPFROG (KICK-DRIFT-KICK) INTEGRATOR

def leapfrog_step(x, v, u, m, dt, alpha=1.0, beta=2.0, alpha_u=1.0, eta=1.5):
    """
    One full leapfrog KDK step.
    Ensure particles are kept sorted at all stages.
    """
    idx = xp.argsort(x)
    x = x[idx]
    v = v[idx]
    u = u[idx]
    m = m[idx]
    
    h = compute_h_dynamic(x, eta=eta)
    rho = compute_density(x, m, h)
    P = compute_pressure(rho, u)
    a, du, _ = compute_accelerations(x, v, rho, P, u, m, h, alpha, beta, alpha_u)
    
    v_half = v + 0.5 * dt * a
    u_half = u + 0.5 * dt * du
    x_new = x + dt * v_half
    
    idx_new = xp.argsort(x_new)
    x_new = x_new[idx_new]
    v_half = v_half[idx_new]
    u_half = u_half[idx_new]
    m = m[idx_new]
    
    h_new = compute_h_dynamic(x_new, eta=eta)
    rho_new = compute_density(x_new, m, h_new)
    P_new = compute_pressure(rho_new, u_half)
    a2, du2, _ = compute_accelerations(x_new, v_half, rho_new, P_new, u_half, m, h_new, alpha, beta, alpha_u)
    
    v_new = v_half + 0.5 * dt * a2
    u_new = u_half + 0.5 * dt * du2
    
    return x_new, v_new, u_new


# 10. MAIN SIMULATION LOOP

def run_sph(N=800, t_end=0.2, C_cfl=0.3, alpha=1.0, beta=2.0, alpha_u=1.0, eta=1.5, verbose=True):
    """
    Run the SPH Sod shock tube simulation.
    """
    x, rho, v, u, m, _ = create_sod_initial_conditions(N)

    t     = 0.0
    step  = 0
    t0    = time.perf_counter()

    while t < t_end:
        # Sort particles
        idx = xp.argsort(x)
        x = x[idx]
        v = v[idx]
        u = u[idx]
        m = m[idx]
        
        h = compute_h_dynamic(x, eta=eta)
        rho = compute_density(x, m, h)
        P   = compute_pressure(rho, u)

        # Retrieve exact signal velocity for CFL
        _, _, v_sig = compute_accelerations(x, v, rho, P, u, m, h, alpha, beta, alpha_u)
        
        dt = compute_timestep(h, v_sig, C_cfl)
        dt = min(dt, t_end - t)

        x, v, u = leapfrog_step(x, v, u, m, dt, alpha, beta, alpha_u, eta)
        t   += dt
        step += 1

        if verbose and step % 50 == 0:
            print(f"  step={step:4d}  t={t:.5f}  dt={dt:.2e}")

    # Final density and pressure
    idx = xp.argsort(x)
    x = x[idx]
    v = v[idx]
    u = u[idx]
    m = m[idx]
    h = compute_h_dynamic(x, eta=eta)
    rho = compute_density(x, m, h)
    P   = compute_pressure(rho, u)

    wall_time = time.perf_counter() - t0
    if verbose:
        print(f"\nDone: {step} steps, t={t:.4f}, wall time={wall_time:.2f}s")

    return (xp.asnumpy(x) if _gpu_available else np.asarray(x),
            xp.asnumpy(rho) if _gpu_available else np.asarray(rho),
            xp.asnumpy(v) if _gpu_available else np.asarray(v),
            xp.asnumpy(P) if _gpu_available else np.asarray(P),
            xp.asnumpy(u) if _gpu_available else np.asarray(u),
            t, wall_time)


# 11. ENTRY POINT (quick smoke test)

if __name__ == "__main__":
    print("Running SPH Sod shock tube (N=800, t=0.2) …")
    x, rho, v, P, u, t_end, wt = run_sph(N=800, verbose=True)
    print(f"Final time = {t_end:.4f},  wall time = {wt:.2f} s")
    print(f"rho range: [{rho.min():.4f}, {rho.max():.4f}]")
    print(f"v   range: [{v.min():.4f}, {v.max():.4f}]")
    print(f"P   range: [{P.min():.4f}, {P.max():.4f}]")
