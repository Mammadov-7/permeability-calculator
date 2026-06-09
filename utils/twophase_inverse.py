"""
Inverse fitting of Corey kr parameters from measured ΔP(t) data.

Uses scipy.optimize.minimize(method='Nelder-Mead') with simple bounds.
Each objective evaluation calls utils.twophase_solver.run_forward, so
fitting time = (number of forward calls) × (single forward time).
Expect ~50–200 forward calls per fit for 4 free parameters.

Pure NumPy / SciPy; no Streamlit.
"""

import copy

import numpy as np
from scipy.optimize import minimize

from utils.twophase_solver import run_forward


COREY_PARAM_NAMES = ["kr_inj_max", "n_inj", "kr_disp_max", "n_disp"]

DEFAULT_BOUNDS = {
    "kr_inj_max":  (0.01, 1.0),
    "n_inj":       (0.5,  10.0),
    "kr_disp_max": (0.01, 1.0),
    "n_disp":      (0.5,  10.0),
}


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


def fit_corey(tp_inputs,
              measured_t_min,
              measured_dp_mbar,
              fit_mask=(True, True, True, True),
              max_iter=80,
              xatol=1e-3,
              fatol=1e-3,
              on_iter=None):
    """
    Fit Corey parameters by minimizing SSE between simulated and
    measured ΔP(t).

    Parameters
    ----------
    tp_inputs : dict
        Step 3 layout. The kr sub-dict provides initial guesses for
        the fitted params (and fixed values for non-fitted ones).
    measured_t_min, measured_dp_mbar : array-like
        Experimental data, already in minutes and mbar.
    fit_mask : 4-tuple of bool
        Which of (kr_inj_max, n_inj, kr_disp_max, n_disp) are free.
    max_iter, xatol, fatol :
        Nelder–Mead stopping criteria.
    on_iter : callable or None
        Called as on_iter(call_index, sse, params_dict) each forward
        evaluation. Use for progress display.

    Returns
    -------
    dict with keys:
        fitted_params : dict of all 4 Corey params
        sse           : final sum of squared errors [mbar²]
        n_calls       : number of forward evaluations
        converged     : bool from scipy
        results       : run_forward output at fitted params
        message       : optimizer convergence message
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
    x0           = [init_params[i] for i in free_indices]
    bounds       = [DEFAULT_BOUNDS[COREY_PARAM_NAMES[i]]
                    for i in free_indices]

    counter = {"i": 0, "best": np.inf}

    def objective(x_free):
        new_inputs, params = _apply_params(
            tp_inputs, x_free, free_indices, fixed_values,
        )
        try:
            res = run_forward(new_inputs)
        except Exception:
            counter["i"] += 1
            if on_iter is not None:
                on_iter(counter["i"], 1e15,
                        {COREY_PARAM_NAMES[i]: params[i] for i in range(4)})
            return 1e15
        sim_dp_at_meas = np.interp(
            measured_t, res["t_min"], res["dP_mbar"],
        )
        sse = float(np.sum((sim_dp_at_meas - measured_dp) ** 2))

        counter["i"] += 1
        if sse < counter["best"]:
            counter["best"] = sse
        if on_iter is not None:
            on_iter(counter["i"], sse,
                    {COREY_PARAM_NAMES[i]: params[i] for i in range(4)})
        return sse

    if x0:
        opt = minimize(
            objective, x0=x0, method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": max_iter, "xatol": xatol, "fatol": fatol,
                     "disp": False},
        )
        fitted_free = list(opt.x)
        message     = str(opt.message)
        converged   = bool(opt.success)
    else:
        fitted_free = []
        message     = "No free parameters selected."
        converged   = True

    new_inputs, params_final = _apply_params(
        tp_inputs, fitted_free, free_indices, fixed_values,
    )
    final_results = run_forward(new_inputs)
    final_sse = float(np.sum((
        np.interp(measured_t, final_results["t_min"],
                  final_results["dP_mbar"]) - measured_dp
    ) ** 2))

    return {
        "fitted_params": {
            COREY_PARAM_NAMES[i]: params_final[i] for i in range(4)
        },
        "sse":         final_sse,
        "n_calls":     counter["i"],
        "converged":   converged,
        "results":     final_results,
        "message":     message,
        "fit_mask":    list(fit_mask),
    }