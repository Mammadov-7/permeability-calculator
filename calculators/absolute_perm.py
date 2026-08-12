"""
Single-phase permeability history matching using Nelder-Mead optimization.

This module solves for the absolute permeability k of a 1D core sample by
matching the simulated steady-state pressure drop to a target value.

The numerical method is a 20-cell finite-difference scheme with implicit
(backward Euler) time stepping, solved each step with a sparse linear
solver (utils.fdm.build_tridiagonal_1d_diffusion).

Compressibility is provided by a pluggable EOS (utils.eos):
    - IncompressibleEOS  → constant c that the caller supplies
    - IdealGasEOS        → c = 1/P computed from local cell pressure

Objective function is squared error (SSE) between simulated and target
steady-state ΔP. This matches TwoPhase's objective and enables future
gradient-based optimizers.
"""

import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.optimize

from utils.fdm import build_tridiagonal_1d_diffusion


# ── Physical / numerical constants ──────────────────────────────────────────
N_CELLS  = 20                # spatial cells
MD_TO_M2 = 9.869233e-16      # 1 mD in m²


def _cell_mean_pressure(P):
    """Representative pressure for compressibility eval (arithmetic mean)."""
    return float(np.mean(P))


def _run_pressure_history(k_mD, N, dx, A, mu, phi, V_b, dt,
                          P_out, q_inj, n_steps, dt_mins,
                          eos, T_K):
    """
    Solve the 1D compressible-flow system for `n_steps` implicit time
    steps at permeability `k_mD` (in mD).

    Compressibility is queried from `eos` once per timestep at the
    mean cell pressure of the previous step. This is a lag-of-one
    linearisation — accurate to first order for our slow pressure
    changes, and avoids nonlinear iteration inside each step.

    Returns (time_min_list, inlet_dp_mbar_list).
    """
    k_m2  = k_mD * MD_TO_M2
    T     = (k_m2 * A) / (mu * dx)
    T_out = (k_m2 * A) / (mu * (dx / 2.0))

    P = np.ones(N) * P_out
    p_out_mbar = P_out / 100.0
    t_list  = []
    dp_list = []

    for step in range(n_steps):
        # Compressibility from EOS at the previous step's mean pressure.
        c_Pa  = eos.get_compressibility(_cell_mean_pressure(P), T_K)
        C     = (V_b * phi * c_Pa) / dt
        A_mat = build_tridiagonal_1d_diffusion(N, T, T_out, C)

        B = C * P.copy()
        B[0]  += q_inj
        B[-1] += T_out * P_out
        P = spsolve(A_mat, B)

        t_list.append((step + 1) * dt_mins)
        dp_list.append(P[0] / 100.0 - p_out_mbar)

    return t_list, dp_list


def match_permeability(L_cm, A_cm2, phi, mu_cP, eos, T_K,
                       q_ml_min, P_out_bar,
                       total_time_mins, dt_mins, target_dp_mbar):
    """
    Run history matching to find the permeability k (in mD) that produces
    a steady-state pressure drop matching `target_dp_mbar`.

    Parameters
    ----------
    L_cm, A_cm2, phi : core geometry
    mu_cP : phase viscosity
    eos : utils.eos.EOS instance
        Provides get_compressibility(P, T) each timestep.
    T_K : float
        Temperature in Kelvin (used by EOS models that need it; ignored
        by IncompressibleEOS and IdealGasEOS but reserved for later PR).
    q_ml_min, P_out_bar : boundary conditions
    total_time_mins, dt_mins : time discretisation
    target_dp_mbar : matching target

    Returns
    -------
    optimised_k : float [mD]
    time_plot : list[float]  (includes t=0 origin)
    delta_p_plot : list[float]  (includes 0 at t=0)
    n_iterations : int
    """
    # ── Unit conversions (input → SI) ───────────────────────────────────────
    L      = L_cm  / 100.0                    # m
    A      = A_cm2 / 10000.0                  # m²
    mu     = mu_cP * 1e-3                     # Pa·s
    q_inj  = (q_ml_min * 1e-6) / 60.0         # m³/s
    P_out  = P_out_bar * 1e5                  # Pa

    # ── Time / grid setup ──────────────────────────────────────────────────
    n_steps = int(total_time_mins / dt_mins)
    dt      = dt_mins * 60.0
    dx      = L / N_CELLS
    V_b     = A * dx

    iteration_counter = [0]

    def objective_function(k_array):
        k_guess = k_array[0]
        if k_guess <= 0:
            return 1e12
        _, dp_list = _run_pressure_history(
            k_guess, N_CELLS, dx, A, mu, phi, V_b, dt,
            P_out, q_inj, n_steps, dt_mins, eos, T_K,
        )
        # Squared error (matches TwoPhase objective, ready for gradient methods).
        residual = dp_list[-1] - target_dp_mbar
        iteration_counter[0] += 1
        return residual * residual

    # ── Nelder-Mead optimisation ───────────────────────────────────────────
    result = scipy.optimize.minimize(
        objective_function,
        x0=np.array([1500.0]),
        method="Nelder-Mead",
        options={
            "xatol":   1e-4,
            "fatol":   1e-8,   # tighter since we're now squared
            "maxiter": 500,
            "disp":    False,
        },
    )
    optimised_k = result.x[0]

    # ── Final run at the optimised k, for plotting ─────────────────────────
    time_results, dp_results = _run_pressure_history(
        optimised_k, N_CELLS, dx, A, mu, phi, V_b, dt,
        P_out, q_inj, n_steps, dt_mins, eos, T_K,
    )
    time_plot    = [0.0] + list(time_results)
    delta_p_plot = [0.0] + list(dp_results)

    return optimised_k, time_plot, delta_p_plot, iteration_counter[0]