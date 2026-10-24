"""
Optimizer-comparison engine (Step 3.1).

Primary use
-----------
`run_optimizer_comparison` fits ONE dataset with several optimizers, from
the SAME initial guess, bounds and fit mask. Everything the runs see is
identical except the search algorithm, so any disagreement in the answers
is attributable to the optimizer or to the problem, not to the setup.

Why that is interesting: if DE, Nelder-Mead and Levenberg-Marquardt all
match the measured pressure history to within the same error but report
different kr parameters, then the pressure data does not determine those
parameters. That is the ill-posedness claim, shown rather than asserted.

Data sources
------------
Two, both supported:

    uploaded    a real (or exported) dP(t) CSV. No truth is known, so the
                runs can only be compared against each other.

    synthetic   generated here from the kr parameters on the Two-Phase
                page, with optional noise. Truth IS known, so each
                optimizer's answer can also be scored for accuracy.

Secondary use
-------------
`run_multistart` keeps the earlier experiment available: one optimizer,
many Latin-hypercube starting points. Not used by the page at present,
retained because it answers a complementary question — identical
algorithms that disagree remove the "maybe that optimizer is simply
worse" objection entirely.

Reporting
---------
RMSE (mbar) is always reported: the root-mean-square mismatch between
simulated and measured dP, in the same units as the data.

Reduced chi-squared is reported only when the noise level is known
(synthetic data), as

    chi2_red = SSE / (sigma^2 * (n_points - n_free))

which is ~1.0 when a fit is as good as the noise allows.

Pure NumPy / SciPy / pandas; no Streamlit.
"""

import copy
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc

from utils.kr_models import get_model_class
from utils.twophase_inverse import fit_kr, OPTIMIZER_CHOICES


# -- Input preparation ------------------------------------------------------
def prepare_inputs(tp_inputs, force_pc_off=True, n_cells=None):
    """
    Return a copy of tp_inputs adjusted for comparison runs.

    Capillary pressure is forced off by default. With Skjaeveland Pc a
    forward call costs roughly 1.5 s at N = 40 against about 0.15 s
    without, and Differential Evolution alone needs thousands of calls —
    enough to pass the hosted execution wall and lose the whole run.
    """
    inp = copy.deepcopy(tp_inputs)
    if force_pc_off:
        inp["pc"]["enabled"] = False
    if n_cells is not None:
        inp["numerical"]["N_cells"] = int(n_cells)
    return inp


# -- Synthetic data (optional data source) ----------------------------------
def generate_synthetic_data(inp, n_points=60, noise_pct=1.0, seed=0):
    """
    Run the forward model at the kr parameters in `inp`, subsample to
    `n_points` measurement times, and optionally add one Gaussian noise
    realisation.

    Noise is additive with CONSTANT sigma, set as a percentage of peak
    dP — a pressure transducer with fixed instrument precision. That also
    keeps the chi-squared interpretation clean, since fit_kr minimises
    unweighted SSE.

    Returns dict:
        t_min, dp_measured, dp_clean, sigma_mbar, n_points,
        forward_time_s, truth_params
    """
    from utils.twophase_solver import run_forward

    t0 = time.perf_counter()
    res = run_forward(inp)
    forward_time_s = time.perf_counter() - t0

    t_full = np.asarray(res["t_min"], dtype=float)
    dp_full = np.asarray(res["dP_mbar"], dtype=float)

    idx = np.unique(
        np.linspace(0, len(t_full) - 1, int(n_points)).astype(int)
    )
    t_min = t_full[idx]
    dp_clean = dp_full[idx]

    sigma = float(noise_pct) / 100.0 * float(np.nanmax(dp_full))
    rng = np.random.default_rng(int(seed))
    dp_measured = (dp_clean + rng.normal(0.0, sigma, size=dp_clean.shape)
                   if sigma > 0.0 else dp_clean.copy())

    return {
        "t_min":          t_min,
        "dp_measured":    dp_measured,
        "dp_clean":       dp_clean,
        "sigma_mbar":     sigma,
        "n_points":       int(len(t_min)),
        "forward_time_s": forward_time_s,
        "truth_params":   dict(inp["kr"]["params"]),
    }


def wrap_uploaded_data(t_min, dp_mbar):
    """
    Package an uploaded dP(t) series in the shape the comparison
    functions expect. No truth and no known noise level, so accuracy
    scoring and reduced chi-squared are unavailable.
    """
    return {
        "t_min":          np.asarray(t_min, dtype=float),
        "dp_measured":    np.asarray(dp_mbar, dtype=float),
        "dp_clean":       None,
        "sigma_mbar":     None,
        "n_points":       int(len(t_min)),
        "forward_time_s": None,
        "truth_params":   None,
    }


# -- Scoring ----------------------------------------------------------------
def _rmse(sse, n_points):
    return float(np.sqrt(float(sse) / max(int(n_points), 1)))


def _chi2_red(sse, sigma, n_points, n_free):
    if sigma is None or sigma <= 0.0:
        return float("nan")
    dof = max(int(n_points) - int(n_free), 1)
    return float(sse) / (sigma ** 2 * dof)


def _blank_run(label, optimizer, message, wall_s):
    return {
        "label": label, "optimizer": optimizer,
        "fitted_params": None,
        "sse": float("nan"), "rmse": float("nan"),
        "chi2_red": float("nan"),
        "n_calls": 0, "converged": False,
        "message": message, "wall_s": wall_s,
        "sim_t": None, "sim_dp": None,
    }


def _record(label, optimizer, fit, data, n_free, wall_s):
    return {
        "label": label, "optimizer": optimizer,
        "fitted_params": dict(fit["fitted_params"]),
        "sse":      float(fit["sse"]),
        "rmse":     _rmse(fit["sse"], data["n_points"]),
        "chi2_red": _chi2_red(fit["sse"], data["sigma_mbar"],
                              data["n_points"], n_free),
        "n_calls":   int(fit["n_calls"]),
        "converged": bool(fit["converged"]),
        "message":   fit["message"],
        "wall_s":    wall_s,
        "sim_t":     np.asarray(fit["results"]["t_min"], dtype=float),
        "sim_dp":    np.asarray(fit["results"]["dP_mbar"], dtype=float),
    }


# -- PRIMARY: three-optimizer comparison ------------------------------------
def run_optimizer_comparison(inp, data, fit_mask,
                             optimizers=None, max_iter=300,
                             on_run=None):
    """
    Fit the same dataset with each optimizer in `optimizers`, all from the
    identical initial guess held in inp["kr"]["params"].

    Note that Differential Evolution ignores the initial guess by design —
    it samples the whole bounded space — which is itself part of what the
    comparison reveals.

    Parameters
    ----------
    inp : dict
        Prepared tp_inputs. inp["kr"]["params"] supplies the shared
        initial guess for free parameters and the fixed values for the
        rest.
    data : dict
        From generate_synthetic_data or wrap_uploaded_data.
    fit_mask : list[bool]
        Free parameters, in model parameter_names order.
    optimizers : list[str] or None
        Subset of OPTIMIZER_CHOICES. None means all three.
    max_iter : int
        Shared effort budget. Each optimizer interprets it differently
        (DE as generations, LM as max function evaluations, N-M split
        across 8 restarts), which is why call counts are reported.
    on_run : callable or None
        on_run(index, total, name) before each run, for progress.

    Returns list of run dicts.
    """
    if optimizers is None:
        optimizers = list(OPTIMIZER_CHOICES)

    n_free = sum(1 for m in fit_mask if m)
    if n_free == 0:
        raise ValueError("At least one parameter must be free to compare.")

    runs = []
    total = len(optimizers)

    for i, name in enumerate(optimizers, start=1):
        if on_run is not None:
            on_run(i, total, name)

        run_inputs = copy.deepcopy(inp)      # identical starting state
        t0 = time.perf_counter()
        try:
            fit = fit_kr(
                run_inputs, data["t_min"], data["dp_measured"],
                fit_mask=fit_mask, max_iter=int(max_iter),
                on_iter=None, optimizer_name=name,
            )
        except Exception as e:               # one failure must not kill the set
            runs.append(_blank_run(name, name, f"FAILED: {e}",
                                   time.perf_counter() - t0))
            continue

        runs.append(_record(name, name, fit, data, n_free,
                            time.perf_counter() - t0))

    return runs


# -- SECONDARY: multi-start, one optimizer (retained, unused by the page) ----
def sample_starts(model_cls, free_indices, n_starts, seed=0):
    """
    Latin-hypercube sample of starting points across the free parameters'
    bounds, drawn in log space where the model flags a parameter as
    multi-decade. None is seeded at truth.
    """
    lows, highs, is_log = [], [], []
    for i in free_indices:
        name = model_cls.parameter_names[i]
        lo, hi = model_cls.parameter_bounds[name]
        lg = bool(model_cls.log_space_params[name])
        lows.append(np.log(lo) if lg else lo)
        highs.append(np.log(hi) if lg else hi)
        is_log.append(lg)

    sampler = qmc.LatinHypercube(d=len(free_indices), seed=int(seed))
    scaled = qmc.scale(sampler.random(int(n_starts)),
                       np.array(lows), np.array(highs))

    return [
        [float(np.exp(v)) if is_log[j] else float(v)
         for j, v in enumerate(row)]
        for row in scaled
    ]


def run_multistart(inp, data, fit_mask, n_starts=5, max_iter=300,
                   optimizer_name="Levenberg-Marquardt", seed=0,
                   on_run=None):
    """
    One optimizer, many starting points, same dataset. Complements the
    three-optimizer comparison: identical algorithms that disagree can
    only be reflecting a property of the problem.
    """
    model_cls = get_model_class(inp["kr"]["model"])
    param_names = list(model_cls.parameter_names)
    base_params = dict(inp["kr"]["params"])

    free_indices = [i for i, m in enumerate(fit_mask) if m]
    n_free = len(free_indices)
    if n_free == 0:
        raise ValueError("At least one parameter must be free to compare.")

    starts = sample_starts(model_cls, free_indices, n_starts, seed=seed)
    runs = []

    for k, start in enumerate(starts):
        label = f"{optimizer_name} #{k + 1}"
        if on_run is not None:
            on_run(k + 1, len(starts), label)

        run_inputs = copy.deepcopy(inp)
        params = dict(base_params)
        for j, i in enumerate(free_indices):
            params[param_names[i]] = start[j]
        run_inputs["kr"]["params"] = params

        t0 = time.perf_counter()
        try:
            fit = fit_kr(
                run_inputs, data["t_min"], data["dp_measured"],
                fit_mask=fit_mask, max_iter=int(max_iter),
                on_iter=None, optimizer_name=optimizer_name,
            )
        except Exception as e:
            runs.append(_blank_run(label, optimizer_name, f"FAILED: {e}",
                                   time.perf_counter() - t0))
            continue

        runs.append(_record(label, optimizer_name, fit, data, n_free,
                            time.perf_counter() - t0))

    return runs


# -- Reporting --------------------------------------------------------------
def results_table(runs, param_names, fit_mask, truth_params=None):
    """
    One row per run: label, fit quality, cost, and the fitted value of
    every free parameter. A TRUTH row is appended when truth is known.
    """
    free_names = [n for n, m in zip(param_names, fit_mask) if m]
    rows = []
    for r in runs:
        row = {
            "optimizer": r["label"],
            "RMSE_mbar": r["rmse"],
            "chi2_red":  r["chi2_red"],
            "calls":     r["n_calls"],
            "time_s":    round(r["wall_s"], 1),
            "converged": r["converged"],
        }
        fp = r["fitted_params"] or {}
        for n in free_names:
            row[n] = fp.get(n, float("nan"))
        rows.append(row)

    if truth_params is not None:
        truth_row = {"optimizer": "TRUTH", "RMSE_mbar": np.nan,
                     "chi2_red": np.nan, "calls": np.nan,
                     "time_s": np.nan, "converged": True}
        for n in free_names:
            truth_row[n] = truth_params[n]
        rows.append(truth_row)

    return pd.DataFrame(rows)


def disagreement_table(runs, param_names, fit_mask, truth_params=None):
    """
    Per-parameter disagreement across the runs that produced a fit.

    Spread as a percentage of the mean is the headline number: a large
    spread alongside similar RMSE values is the signature of a parameter
    the data does not determine. When truth is known, the largest error
    among the optimizers is included too.
    """
    free_names = [n for n, m in zip(param_names, fit_mask) if m]
    good = [r for r in runs if r["fitted_params"] is not None]
    if not good:
        return pd.DataFrame()

    rows = []
    for n in free_names:
        vals = np.array([r["fitted_params"][n] for r in good], dtype=float)
        mean = float(np.mean(vals))
        lo, hi = float(np.min(vals)), float(np.max(vals))
        row = {
            "parameter":          n,
            "min":                lo,
            "max":                hi,
            "spread_pct_of_mean": ((hi - lo) / abs(mean) * 100.0
                                   if mean != 0 else np.nan),
            "n_runs":             int(vals.size),
        }
        if truth_params is not None:
            t = float(truth_params[n])
            row["truth"] = t
            row["max_error_pct"] = (float(np.max(np.abs(vals - t)))
                                    / abs(t) * 100.0 if t != 0 else np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    if truth_params is not None:
        df = df[["parameter", "truth", "min", "max",
                 "spread_pct_of_mean", "max_error_pct", "n_runs"]]
    return df


def attach_kr_curves(runs, inp, truth_params=None):
    """
    Compute fitted kr curves for each successful run, attached under the
    key "kr". Returns the truth curves when truth_params is given, else
    None.
    """
    from utils.twophase import kr_curves

    for r in runs:
        if r["fitted_params"] is None:
            r["kr"] = None
            continue
        ci = copy.deepcopy(inp)
        ci["kr"]["params"] = dict(r["fitted_params"])
        r["kr"] = kr_curves(ci)

    if truth_params is None:
        return None
    ti = copy.deepcopy(inp)
    ti["kr"]["params"] = dict(truth_params)
    return kr_curves(ti)