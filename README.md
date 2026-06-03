# SPH Sod Shock Tube Simulation

A GPU-accelerated Smoothed Particle Hydrodynamics (SPH) solver for the 1-D Sod shock tube problem, validated against an exact Riemann solution.

---

## Project Structure

```
SPH-simulations/
├── sph_solver.py              # Core SPH engine (kernel, ICs, forces, integrator)
├── comparison.py              # L2 error analysis + comparison plot
├── sph_initial_conditions.py  # Import shim (delegates to sph_solver.py)
├── Riemann_Solver.ipynb       # Exact Riemann solver (reference, do not modify)
├── requirements.txt           # CPU dependencies
└── venv/                      # Python virtual environment
```

---

## Setup

### 1. Activate the virtual environment

```bash
cd /home/vs/Code/SPH-simulations
source venv/bin/activate
```

### 2. Install dependencies (first time only)

```bash
pip install -r requirements.txt
```

For GPU support (optional), uncomment the relevant `cupy` line in `requirements.txt` then re-run:

```bash
# Edit requirements.txt — uncomment the cupy line matching your CUDA version
pip install -r requirements.txt
```

---

## Running

### Smoke test — run the SPH solver only

Runs N=800 particles to t=0.2 and prints physical ranges to stdout.

```bash
python sph_solver.py
```

Expected output:

```
Running SPH Sod shock tube (N=800, t=0.2) …
  step=  50  t=0.00574  dt=1.09e-04
  ...
Done: 1826 steps, t=0.2000, wall time=123.12 s
rho range: [0.5548, 1.0018]
v   range: [-1.1660, 1.0905]
P   range: [0.4438, 1.0018]
```

---

### Full validation — SPH vs exact Riemann + L2 errors + plot

```bash
python comparison.py
```

This runs SPH, evaluates the exact Sod solution, computes normalised L2 errors for density, velocity, and pressure, and saves a comparison plot to `sph_vs_exact.png`.

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--N` | `800` | Number of SPH particles |
| `--t_end` | `0.2` | Simulation end time |
| `--alpha` | `1.0` | Artificial viscosity linear coefficient |
| `--beta` | `2.0` | Artificial viscosity quadratic coefficient |

```bash
# Standard run
python comparison.py --N 800 --t_end 0.2

# Higher resolution (slower, lower L2 error)
python comparison.py --N 3200 --t_end 0.2

# Adjust artificial viscosity
python comparison.py --N 800 --alpha 0.5 --beta 1.0
```

---

### GPU mode (CuPy)

Edit the top of `sph_solver.py`:

```python
USE_GPU = True   # line 29
```

Then run normally:

```bash
python comparison.py --N 800
```

> **Note:** Warm up CuPy with a small run first (`--N 200`) before benchmarking — the first call compiles CUDA kernels.

---

## Key Parameters

| Parameter | Location | Recommended | Effect |
|-----------|----------|-------------|--------|
| `N` | CLI `--N` | 800 | More particles → lower L2 error (~N^-0.5) |
| `h` | auto: `1.2 × dx_L` | — | Smoothing length; ~20–30 neighbours in 1-D |
| `alpha` | CLI `--alpha` | 1.0 | AV bulk damping; too low → post-shock ringing |
| `beta` | CLI `--beta` | 2.0 | AV quadratic; prevents particle penetration |
| `C_cfl` | `sph_solver.py` | 0.3 | CFL factor; >0.5 usually unstable |

---

## Expected L2 Errors (N=800, t=0.2)

| Quantity | L2 error |
|----------|----------|
| Density ρ | ~2 × 10⁻² |
| Velocity v | ~3 × 10⁻² |
| Pressure P | ~2 × 10⁻² |

The dominant error source is kernel broadening at the contact discontinuity and shock front — a fundamental SPH limitation, not a bug.
