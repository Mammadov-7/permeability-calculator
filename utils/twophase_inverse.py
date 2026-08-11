"""
Inverse fitting of Corey kr parameters from measured ΔP(t) data.

Provides three optimization strategies, dispatched by `optimizer_name`:

    "nelder-mead"            : local, derivative-free simplex method.
                               Runs with 8 multi-starts (random initial
                               guesses across the parameter bounds) and
                               log-space parameterization for kr_max
                               values. Best-of-8 result is returned.
    "differential-evolution" : global stochastic search. Robust to poor
                               initial guesses. Slower than N-M (~500
                               forward calls) but does not require multi-
                               start because it already explores globally.
    "levenberg-marquardt"    : local gradient-based least-squares. Fastest
                               when the initial guess is close to the true
                               parameters. Uses scipy's trust-region
                               reflective algorithm to respect bounds.

Each objective evaluation calls utils.twophase_solver.run_forward, so
fitting time = (number of forward calls) × (single forward time).

Pure NumPy / SciPy; no Streamlit.
"""

import copy

import numpy as np
from scipy.optimize import minimize, differential_evolution, least_squares


COREY_PARAM_NAMES = ["kr_inj_max", "n_inj", "kr_disp_max", "n_disp"]

DEFAULT_BOUNDS = {
    "kr_inj_max":  (0.01, 1.0),
    "n_inj":       (0.5,  10.0),
    "kr_disp_max": (0.01, 1.0),
    "n_disp":      (0.5,  10.0),
}

# Which parameters are fitted in log-space for Nelder-Mead multi-start.
# kr_max values benefit from log-space because their sensible range spans
# ~two orders of magnitude (0.01 - 1.0); Corey exponents n live on a linear
# range (0.5 - 10) where log-space would add no benefit.
LOG_SPACE_PARAMS = {"kr_inj_max": True, "n_inj": False,
                    "kr_disp_max": True, "n_disp": False}

OPTIMIZER_CHOICES = [
    "Differential Evolution",
    "Nelder-Mead",
    "Levenberg-Marquardt",
]


# ── Shared helpers ──────────────────────────────────────────────────────────
def _params_to_kr_dict(params):
    return {
        "inj_max":  params[0],
        "n_inj":    params[1],
        "disp_max": params[2],
        "n_disp":   params[3],
    }


def _apply_params(tp_inputs, x_free, free_indices, fixed_values):
    """Inject the free parameters into a copy of tp_inputs."""
    params = list(fixed_values)
    for j, i in enumerate(free_indices):
        params[i] = x_free[j]
    new_inputs = copy.deepcopy(tp_inputs)
    new_inputs["kr"] = _params_to_kr_dict(params)
    return new_inputs, params


def _import_run_forward():
    # Local import avoids a circular-import risk if the solver ever imports
    # anything from this module in the future.
    from utils.twophase_solver import run_forward
    return run_forward


def _sim_dp_at_times(tp_inputs, measured_t):
    """Run one forward simulation and interpolate ΔP at measured times."""
    run_forward = _import_run_forward()
    res = run_forward(tp_inputs)
    sim_dp = np.interp(measured_t, res["t_min"], res["dP_mbar"])
    return sim_dp, res


def _sse(sim_dp, measured_dp):
    return float(np.sum((sim_dp - measured_dp) ** 2))


# ── Parameter transforms for Nelder-Mead multi-start (log-space) ───────────
def _to_search_space(params, free_indices):
    """Map physical params → search space (log for kr_max, linear else)."""
    out = []
    for i in free_indices:
        name = COREY_PARAM_NAMES[i]
        v = params[i] if isinstance(params, list) else params[free_indices.index(i)]
        out.append(np.log(v) if LOG_SPACE_PARAMS[name] else v)
    return out


def _from_search_space(x_free, free_indices):
    """Map search-space vector → physical params for that vector."""
    out = []
    for j, i in enumerate(free_indices):
        name = COREY_PARAM_NAMES[i]
        out.append(np.exp(x_free[j]) if LOG_SPACE_PARAMS[name] else x_free[j])
    return out


def _search_bounds(free_indices):
    """Return bounds for each free param in the search space."""
    bnds = []
    for i in free_indices:
        name = COREY_PARAM_NAMES[i]
        lo, hi = DEFAULT_BOUNDS[name]
        if LOG_SPACE_PARAMS[name]:
            bnds.append((np.log(lo), np.log(hi)))
        else:
            bnds.append((lo, hi))
    return bnds


# ── Optimizer 1: Nelder-Mead with multi-start + log-space ─────────────────
N_MULTISTARTS = 8


def _fit_nelder_mead(tp_inputs, measured_t, measured_dp,
                     free_indices, fixed_values, x0_phys,
                     max_iter, on_iter):
    """
    Run Nelder-Mead from N_MULTISTARTS starting points and keep the best.

    Starting points:
        - 1st start uses the user's initial guess (x0_phys).
        - Remaining 7 are drawn from a Latin-hypercube-like uniform sample
          across the (log-space where applicable) search bounds.

    All optimizations happen in the search space (log for kr_max, linear
    for exponents). The result is converted back to physical params.
    """
    search_bnds = _search_bounds(free_indices)
    rng = np.random.default_rng(seed=42)

    # Generate diverse starting points in search space.
    starts = [_to_search_space(x0_phys, free_indices)]
    for _ in range(N_MULTISTARTS - 1):
        starts.append([
            rng.uniform(lo, hi) for (lo, hi) in search_bnds
        ])

    call_counter = {"i": 0}

    def objective_in_search(x_search):
        x_phys = _from_search_space(x_search, free_indices)
        new_inputs, params = _apply_params(
            tp_inputs, x_phys, free_indices, fixed_values,
        )
        try:
            sim_dp, _ = _sim_dp_at_times(new_inputs, measured_t)
        except Exception:
            call_counter["i"] += 1
            return 1e15
        sse = _sse(sim_dp, measured_dp)
        call_counter["i"] += 1
        if on_iter is not None:
            on_iter(call_counter["i"], sse,
                    {COREY_PARAM_NAMES[i]: params[i] for i in range(4)})
        return sse

    best = {"sse": np.inf, "x_phys": None, "message": "", "success": False}

    # Split the total iteration budget across the multi-starts.
    per_start_max = max(20, int(max_iter // N_MULTISTARTS))

    for k, x0_search in enumerate(starts):
        try:
            opt = minimize(
                objective_in_search, x0=x0_search, method="Nelder-Mead",
                bounds=search_bnds,
                options={"maxiter": per_start_max,
                         "xatol": 1e-3, "fatol": 1e-3, "disp": False},
            )
        except Exception as e:
            continue
        # Evaluate the final SSE cleanly (opt.fun is the last objective value).
        x_phys = _from_search_space(list(opt.x), free_indices)
        if opt.fun < best["sse"]:
            best.update({
                "sse":     float(opt.fun),
                "x_phys":  x_phys,
                "message": f"Best of {N_MULTISTARTS} restarts "
                           f"(start #{k + 1}): {opt.message}",
                "success": bool(opt.success),
            })

    if best["x_phys"] is None:
        raise RuntimeError("All Nelder-Mead multi-starts failed.")

    return best, call_counter["i"]


# ── Optimizer 2: Differential Evolution (global) ────────────────────────────
def _fit_differential_evolution(tp_inputs, measured_t, measured_dp,
                                free_indices, fixed_values,
                                max_iter, on_iter):
    """
    Global search via scipy.optimize.differential_evolution. No multi-start
    because DE is already population-based and globally exploratory.
    """
    bnds = [DEFAULT_BOUNDS[COREY_PARAM_NAMES[i]] for i in free_indices]
    call_counter = {"i": 0}

    def objective(x_free):
        new_inputs, params = _apply_params(
            tp_inputs, list(x_free), free_indices, fixed_values,
        )
        try:
            sim_dp, _ = _sim_dp_at_times(new_inputs, measured_t)
        except Exception:
            call_counter["i"] += 1
            return 1e15
        sse = _sse(sim_dp, measured_dp)
        call_counter["i"] += 1
        if on_iter is not None:
            on_iter(call_counter["i"], sse,
                    {COREY_PARAM_NAMES[i]: params[i] for i in range(4)})
        return sse

    # popsize * len(bnds) initial members; maxiter controls generations.
    # Total forward calls ≈ popsize * len(bnds) * (maxiter + 1).
    # With popsize=12, 4 free params, maxiter=25 → ~1300 calls worst-case
    # but DE typically converges well before hitting maxiter.
    opt = differential_evolution(
        objective, bounds=bnds,
        seed=42,
        maxiter=max(10, int(max_iter // 12)),
        popsize=12,
        tol=1e-4,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,       # final L-BFGS-B refinement inside DE
        init="sobol",      # quasi-random init for even coverage
        updating="deferred",
    )

    best = {
        "sse":     float(opt.fun),
        "x_phys":  list(opt.x),
        "message": f"Differential Evolution: {opt.message}",
        "success": bool(opt.success),
    }
    return best, call_counter["i"]


# ── Optimizer 3: Levenberg-Marquardt via least_squares ─────────────────────
def _fit_levenberg_marquardt(tp_inputs, measured_t, measured_dp,
                             free_indices, fixed_values, x0_phys,
                             max_iter, on_iter):
    """
    Bounded LM via trust-region reflective (`method="trf"`). Runs once from
    the user's initial guess. Uses finite differences for the Jacobian,
    which costs (N_free + 1) forward calls per iteration.
    """
    bnds = [DEFAULT_BOUNDS[COREY_PARAM_NAMES[i]] for i in free_indices]
    lo = [b[0] for b in bnds]
    hi = [b[1] for b in bnds]

    call_counter = {"i": 0}

    def residuals(x_free):
        new_inputs, params = _apply_params(
            tp_inputs, list(x_free), free_indices, fixed_values,
        )
        try:
            sim_dp, _ = _sim_dp_at_times(new_inputs, measured_t)
        except Exception:
            call_counter["i"] += 1
            # Return large residuals if forward failed.
            return np.full_like(measured_dp, 1e6, dtype=float)
        r = sim_dp - measured_dp
        call_counter["i"] += 1
        if on_iter is not None:
            sse = float(np.sum(r * r))
            on_iter(call_counter["i"], sse,
                    {COREY_PARAM_NAMES[i]: params[i] for i in range(4)})
        return r

    # Clip initial guess into bounds to avoid an immediate error.
    x0_clipped = [
        min(max(x0_phys[j], lo[j] + 1e-6), hi[j] - 1e-6)
        for j in range(len(x0_phys))
    ]

    opt = least_squares(
        residuals, x0=x0_clipped, bounds=(lo, hi),
        method="trf",             # bounded LM variant
        max_nfev=max(50, int(max_iter)),
        xtol=1e-6, ftol=1e-6, gtol=1e-6,
    )

    best = {
        "sse":     float(2.0 * opt.cost),   # least_squares returns 0.5·SSE
        "x_phys":  list(opt.x),
        "message": f"Levenberg-Marquardt (TRF): {opt.message}",
        "success": bool(opt.success),
    }
    return best, call_counter["i"]


# ── Main dispatcher ─────────────────────────────────────────────────────────
def fit_corey(tp_inputs,
              measured_t_min,
              measured_dp_mbar,
              fit_mask=(True, True, True, True),
              max_iter=80,
              on_iter=None,
              optimizer_name="Differential Evolution"):
    """
    Fit Corey parameters by minimizing SSE between simulated and measured
    ΔP(t). Dispatches to one of three optimization strategies based on
    `optimizer_name`.

    Parameters
    ----------
    tp_inputs : dict
        Full Step 3 input layout. The `kr` sub-dict provides:
          - initial guesses for the fitted parameters
          - fixed values for the non-fitted parameters
    measured_t_min, measured_dp_mbar : array-like
        Experimental data already in minutes and mbar.
    fit_mask : 4-tuple of bool
        Which of (kr_inj_max, n_inj, kr_disp_max, n_disp) are free.
    max_iter : int
        Rough budget for the optimizer's iteration count. Interpretation
        depends on the algorithm — see per-optimizer docstrings.
    on_iter : callable or None
        on_iter(call_index, sse, params_dict) called each forward
        evaluation. Use for progress display.
    optimizer_name : {"Differential Evolution",
                      "Nelder-Mead",
                      "Levenberg-Marquardt"}
        Which algorithm to run. Multi-start, log-space transforms, etc.
        are chosen automatically per algorithm.

    Returns
    -------
    dict with keys:
        fitted_params  : dict of all 4 Corey params
        sse            : final sum of squared errors [mbar²]
        n_calls        : number of forward evaluations
        converged      : bool from the optimizer
        results        : run_forward output at fitted params
        message        : optimizer convergence message
        optimizer_name : which algorithm was run
        fit_mask       : the fit_mask that was used
    """
    measured_t  = np.asarray(measured_t_min,  dtype=float)
    measured_dp = np.asarray(measured_dp_mbar, dtype=float)

    init_kr = tp_inputs["kr"]
    init_params = [
        init_kr["inj_max"], init_kr["n_inj"],
        init_kr["disp_max"], init_kr["n_disp"],
    ]

    free_indices = [i for i, m in enumerate(fit_mask) if m]
    fixed_values = list(init_params)
    x0_phys      = [init_params[i] for i in free_indices]

    if not free_indices:
        # Nothing to fit — just run the forward at the given params.
        new_inputs, params_final = _apply_params(
            tp_inputs, [], [], fixed_values,
        )
        run_forward = _import_run_forward()
        final_results = run_forward(new_inputs)
        sim_dp = np.interp(measured_t, final_results["t_min"],
                           final_results["dP_mbar"])
        return {
            "fitted_params": {
                COREY_PARAM_NAMES[i]: params_final[i] for i in range(4)
            },
            "sse":            _sse(sim_dp, measured_dp),
            "n_calls":        1,
            "converged":      True,
            "results":        final_results,
            "message":        "No free parameters selected.",
            "optimizer_name": optimizer_name,
            "fit_mask":       list(fit_mask),
        }

    # Dispatch to the chosen optimizer.
    if optimizer_name == "Nelder-Mead":
        best, n_calls = _fit_nelder_mead(
            tp_inputs, measured_t, measured_dp,
            free_indices, fixed_values, x0_phys, max_iter, on_iter,
        )
    elif optimizer_name == "Differential Evolution":
        best, n_calls = _fit_differential_evolution(
            tp_inputs, measured_t, measured_dp,
            free_indices, fixed_values, max_iter, on_iter,
        )
    elif optimizer_name == "Levenberg-Marquardt":
        best, n_calls = _fit_levenberg_marquardt(
            tp_inputs, measured_t, measured_dp,
            free_indices, fixed_values, x0_phys, max_iter, on_iter,
        )
    else:
        raise ValueError(
            f"Unknown optimizer '{optimizer_name}'. "
            f"Expected one of {OPTIMIZER_CHOICES}."
        )

    # Reconstruct full 4-param vector and run one final forward for plots.
    fitted_free = best["x_phys"]
    new_inputs, params_final = _apply_params(
        tp_inputs, fitted_free, free_indices, fixed_values,
    )
    run_forward = _import_run_forward()
    final_results = run_forward(new_inputs)
    final_sim_dp = np.interp(measured_t, final_results["t_min"],
                             final_results["dP_mbar"])
    final_sse = _sse(final_sim_dp, measured_dp)

    return {
        "fitted_params": {
            COREY_PARAM_NAMES[i]: params_final[i] for i in range(4)
        },
        "sse":            final_sse,
        "n_calls":        n_calls,
        "converged":      bool(best["success"]),
        "results":        final_results,
        "message":        best["message"],
        "optimizer_name": optimizer_name,
        "fit_mask":       list(fit_mask),
    }