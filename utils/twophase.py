"""
Two-phase relative permeability and capillary pressure functions.

Pure NumPy; Pyodide-compatible (no Streamlit, no I/O).
Used by:
    - utils/twophase_solver.py  (Step 5, forward simulator)
    - pages/2_TwoPhase.py       (Step 6, plotting kr / Pc curves)

Conventions
-----------
- S_inj    : saturation of the injected phase (0..1).
- S_inj_r  : residual / connate saturation of injected phase.
- S_disp_r : residual saturation of displaced phase.
- The injected phase is assumed to be the NON-WETTING phase (typical
  for gas / H2 / CO2 displacing brine). The displaced phase is wetting.
  Pc is defined as Pc = P_nw - P_w = P_inj - P_disp, positive on
  drainage.
- Saturations outside [S_inj_r, 1 - S_disp_r] are clipped to the
  endpoint values (kr = 0 or kr = kr_max).

Relative-permeability parameterization
--------------------------------------
The choice of kr model (Corey, LET, …) is delegated to utils/kr_models.
This module wraps the selected model with saturation normalization and
returns kr callables the solver can use. See utils/kr_models.py for
adding new parameterizations.

Capillary pressure remains hard-coded to Brooks–Corey for now; Phase
2.3 will make it pluggable using the same pattern as kr_models.
"""

import numpy as np

from utils.kr_models import make_kr_model


# ── Normalized saturation ───────────────────────────────────────────────────
def normalized_saturation(S_inj, S_inj_r, S_disp_r):
    """
    Map S_inj from [S_inj_r, 1 - S_disp_r] onto [0, 1] (clipped).
    Returns the normalized injected-phase saturation S_n.
    Accepts scalar or NumPy array.
    """
    denom = 1.0 - S_inj_r - S_disp_r
    if denom <= 0.0:
        raise ValueError(
            f"Invalid endpoints: S_inj_r ({S_inj_r}) + S_disp_r "
            f"({S_disp_r}) >= 1; no mobile saturation range."
        )
    S_n = (np.asarray(S_inj, dtype=float) - S_inj_r) / denom
    return np.clip(S_n, 0.0, 1.0)


# ── Brooks-Corey capillary pressure ─────────────────────────────────────────
def brooks_corey_pc(S_inj, S_inj_r, S_disp_r, P_entry, lam, S_eff_min=1e-4):
    """
    Brooks-Corey drainage Pc, in terms of wetting-phase effective
    saturation (assumes injected = non-wetting):

        S_w     = 1 - S_inj
        S_w_eff = (S_w - S_disp_r) / (1 - S_disp_r)
        Pc      = P_entry * S_w_eff ** (-1 / lambda)

    Pc shares units with P_entry (mbar in this project).
    S_eff_min clamps S_w_eff away from zero to avoid the singularity
    as the wetting phase approaches its residual saturation.
    Pc = P_entry at S_w_eff = 1 (entry/threshold pressure), rising
    monotonically as S_w decreases.
    """
    if 1.0 - S_disp_r <= 0.0:
        raise ValueError("S_disp_r must be < 1.")
    S_w     = 1.0 - np.asarray(S_inj, dtype=float)
    S_w_eff = (S_w - S_disp_r) / (1.0 - S_disp_r)
    S_w_eff = np.clip(S_w_eff, S_eff_min, 1.0)
    return P_entry * S_w_eff ** (-1.0 / lam)


# ── Convenience factories from tp_inputs dict ───────────────────────────────
def make_kr_functions(tp_inputs):
    """
    Returns (kr_inj_fn, kr_disp_fn): callables that take raw S_inj and
    return the corresponding relative permeability. Reads the model
    selection and its parameter values from tp_inputs["kr"]:

        tp_inputs["kr"] = {"model":  <display name, e.g. "Corey" or "LET">,
                           "params": {<name>: <value>, …}}

    Solver-side code (utils.twophase_solver) doesn't know or care which
    kr model is in use — it only calls the returned callables.
    """
    s_ir  = tp_inputs["S_inj_r"]
    s_dr  = tp_inputs["S_disp_r"]
    model = make_kr_model(tp_inputs["kr"]["model"],
                          tp_inputs["kr"]["params"])

    def kr_inj_fn(S_inj):
        S_n = normalized_saturation(S_inj, s_ir, s_dr)
        return model.kr_inj(S_n)

    def kr_disp_fn(S_inj):
        S_n = normalized_saturation(S_inj, s_ir, s_dr)
        return model.kr_disp(S_n)

    return kr_inj_fn, kr_disp_fn


def make_pc_function(tp_inputs):
    """
    Returns a Pc(S_inj) callable. If capillary pressure is disabled,
    returns a function that yields zeros. Output units = mbar.
    """
    pc = tp_inputs["pc"]
    if not pc["enabled"]:
        return lambda S_inj: np.zeros_like(np.asarray(S_inj, dtype=float))

    s_ir = tp_inputs["S_inj_r"]
    s_dr = tp_inputs["S_disp_r"]
    P_e  = pc["P_entry_mbar"]
    lam  = pc["lambda"]

    def pc_fn(S_inj):
        return brooks_corey_pc(S_inj, s_ir, s_dr, P_e, lam)

    return pc_fn


# ── Sampled curves for plotting (Step 6 will use these) ─────────────────────
def kr_curves(tp_inputs, n_points=200):
    """
    Sample kr_inj and kr_disp on a uniform grid over S_inj ∈ [0, 1].
    Returns dict with arrays ready for Plotly. The dict also carries
    the model name so callers (e.g. chart titles) can label the plot.
    """
    S = np.linspace(0.0, 1.0, n_points)
    kr_inj_fn, kr_disp_fn = make_kr_functions(tp_inputs)
    return {
        "S_inj":      S,
        "kr_inj":     kr_inj_fn(S),
        "kr_disp":    kr_disp_fn(S),
        "S_inj_r":    tp_inputs["S_inj_r"],
        "S_disp_r":   tp_inputs["S_disp_r"],
        "model_name": tp_inputs["kr"]["model"],
    }


def pc_curve(tp_inputs, n_points=200):
    """
    Sample Pc on a uniform grid over S_inj ∈ [0, 1]. Returns dict with
    arrays ready for Plotly. If Pc is disabled, Pc_mbar is all zeros.
    """
    S = np.linspace(0.0, 1.0, n_points)
    pc_fn = make_pc_function(tp_inputs)
    return {
        "S_inj":   S,
        "Pc_mbar": pc_fn(S),
        "enabled": tp_inputs["pc"]["enabled"],
    }