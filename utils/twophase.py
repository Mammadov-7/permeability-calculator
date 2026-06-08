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
"""

import numpy as np


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


# ── Corey relative permeabilities ───────────────────────────────────────────
def corey_kr_inj(S_inj, S_inj_r, S_disp_r, kr_max, n):
    """kr of injected phase via Corey model: kr = kr_max * S_n^n."""
    S_n = normalized_saturation(S_inj, S_inj_r, S_disp_r)
    return kr_max * S_n ** n


def corey_kr_disp(S_inj, S_inj_r, S_disp_r, kr_max, n):
    """kr of displaced phase via Corey model: kr = kr_max * (1 - S_n)^n."""
    S_n = normalized_saturation(S_inj, S_inj_r, S_disp_r)
    return kr_max * (1.0 - S_n) ** n


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
    Returns (kr_inj_fn, kr_disp_fn): callables that take S_inj and
    return the corresponding relative permeability. Reads from the
    same dict that pages/2_TwoPhase.py stashes in session_state.
    """
    s_ir = tp_inputs["S_inj_r"]
    s_dr = tp_inputs["S_disp_r"]
    kr   = tp_inputs["kr"]

    def kr_inj_fn(S_inj):
        return corey_kr_inj(S_inj, s_ir, s_dr, kr["inj_max"], kr["n_inj"])

    def kr_disp_fn(S_inj):
        return corey_kr_disp(S_inj, s_ir, s_dr, kr["disp_max"], kr["n_disp"])

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
    Returns dict with arrays ready for Plotly.
    """
    S = np.linspace(0.0, 1.0, n_points)
    kr_inj_fn, kr_disp_fn = make_kr_functions(tp_inputs)
    return {
        "S_inj":    S,
        "kr_inj":   kr_inj_fn(S),
        "kr_disp":  kr_disp_fn(S),
        "S_inj_r":  tp_inputs["S_inj_r"],
        "S_disp_r": tp_inputs["S_disp_r"],
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