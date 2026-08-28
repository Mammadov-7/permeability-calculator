"""
Two-phase relative permeability and capillary pressure functions.

Pure NumPy; Pyodide-compatible (no Streamlit, no I/O).
Used by:
    - utils/twophase_solver.py  (Step 5, forward simulator)
    - pages/2_TwoPhase.py       (Step 6, plotting kr / Pc curves)

Process-type awareness (Phase 2.4)
----------------------------------
The calculator supports two physical processes:

    drainage   -- injected phase is NON-WETTING; displaces the wetting
                  phase (e.g. gas / H2 / CO2 into brine-saturated core).
                  S_inj = S_nw = 1 - S_w. Pc = P_nw - P_w > 0 always
                  in Brooks-Corey / van Genuchten drainage models.

    imbibition -- injected phase is WETTING; displaces the non-wetting
                  phase (e.g. brine flooding a gas-saturated core, or
                  waterflood after primary drainage). S_inj = S_w. Pc =
                  P_nw - P_w can go negative in mixed-wet systems --
                  Skjaeveland is the standard model for that.

In BOTH cases the solver tracks the *injected-phase* saturation S_inj
on a cell grid and calls kr_inj_fn(S_inj), kr_disp_fn(S_inj), and
pc_fn(S_inj). The mapping S_inj -> S_w is done here in
make_pc_function per process type, so the Pc model itself always
receives normalized wetting-phase saturation S_w_eff and returns Pc
in the standard convention P_nw - P_w. The solver applies the
appropriate sign to the capillary flux term based on process type
(see utils/twophase_solver.py).

Kr models are process-type agnostic. Corey and LET are symmetric in
"injected vs displaced": swapping the phase labels and residual
values recovers the mirrored curves. So the kr framework does not
branch on process type; users pick physically sensible parameters
for whichever process they've selected.

Conventions
-----------
- S_inj    : saturation of the injected phase (0..1).
- S_inj_r  : residual / connate saturation of injected phase. Under
             imbibition this equals S_wr; under drainage it is the
             trapped injected-phase saturation.
- S_disp_r : residual saturation of displaced phase. Under drainage
             this equals S_wr; under imbibition it equals S_or.
- Pc = P_nw - P_w > 0 on drainage, may cross zero on imbibition
  (mixed-wet). Units: mbar.

Parameterization
----------------
Both relative permeability and capillary pressure use pluggable model
frameworks; the choice of parameterization is delegated to:
    - utils/kr_models  (Corey, LET, ...)
    - utils/pc_models  (Brooks-Corey, van Genuchten, Skjaeveland, ...)

This module wraps the selected models with saturation normalization
and returns callables the solver can use. See kr_models.py /
pc_models.py for adding new parameterizations.
"""

import numpy as np

from utils.kr_models import make_kr_model
from utils.pc_models import make_pc_model


# -- Normalized saturation --------------------------------------------------
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


# -- Convenience factories from tp_inputs dict ------------------------------
def make_kr_functions(tp_inputs):
    """
    Returns (kr_inj_fn, kr_disp_fn): callables that take raw S_inj and
    return the corresponding relative permeability. Reads the model
    selection and its parameter values from tp_inputs["kr"]:

        tp_inputs["kr"] = {"model":  <display name, e.g. "Corey" or "LET">,
                           "params": {<name>: <value>, ...}}

    Solver-side code (utils.twophase_solver) doesn't know or care which
    kr model is in use -- it only calls the returned callables. The
    kr functions are also process-type agnostic: kr_inj always means
    "kr of the injected phase" and kr_disp always means "kr of the
    displaced phase," whichever wettability role each phase plays.
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
    Returns a Pc(S_inj) callable in the standard convention
    Pc = P_nonwetting - P_wetting, units mbar. Reads the Pc model
    selection and its parameters from tp_inputs["pc"]:

        tp_inputs["pc"] = {"enabled": True|False,
                           "model":   <display name>,
                           "params":  {<name>: <value>, ...}}

    Process-type branch
    -------------------
    Reads tp_inputs.get("process_type", "drainage"):

        drainage   -- injected = non-wetting, so S_w = 1 - S_inj.
                      One-sided normalization referenced to S_disp_r
                      (which equals S_wr under drainage):
                          S_w_eff = (S_w - S_disp_r) / (1 - S_disp_r)
                      Preserves the Phase 2.3 Brooks-Corey and van
                      Genuchten conventions bit-for-bit; at initial
                      S_w = 1 the model receives S_w_eff = 1 and BC
                      returns P_entry (the entry pressure) as
                      designed.

        imbibition -- injected = wetting, so S_w = S_inj directly.
                      Two-sided normalization between both residuals
                      (Skjaeveland's two terms each need their own
                      singularity to map onto the effective coordinate):
                          S_w_eff = (S_w - S_inj_r) / (1 - S_inj_r - S_disp_r)
                      where S_inj_r plays the role of S_wr and
                      S_disp_r plays the role of S_or.

    If capillary pressure is disabled, returns a function that yields
    zeros. Solver-side code doesn't know or care which Pc model is
    in use -- it only calls the returned callable and applies the
    process-type-appropriate sign to the capillary flux term.
    """
    pc_cfg = tp_inputs["pc"]
    if not pc_cfg["enabled"]:
        return lambda S_inj: np.zeros_like(np.asarray(S_inj, dtype=float))

    process_type = tp_inputs.get("process_type", "drainage")
    s_ir  = tp_inputs["S_inj_r"]
    s_dr  = tp_inputs["S_disp_r"]
    model = make_pc_model(pc_cfg["model"], pc_cfg["params"])

    if process_type == "drainage":
        # Drainage: injected = non-wetting, so S_w = 1 - S_inj.
        # One-sided normalization preserved from Phase 2.3.
        if 1.0 - s_dr <= 0.0:
            raise ValueError("S_disp_r must be < 1 for drainage Pc.")

        def pc_fn(S_inj):
            S_w = 1.0 - np.asarray(S_inj, dtype=float)
            S_w_eff = (S_w - s_dr) / (1.0 - s_dr)
            # Clipping to (0, 1] happens inside model.pc().
            return model.pc(S_w_eff)

        return pc_fn

    elif process_type == "imbibition":
        # Imbibition: injected = wetting, so S_w = S_inj.
        # Two-sided normalization so both Skjaeveland singularities
        # (water-wet term at S_w_eff -> 0, oil-wet term at
        # S_w_eff -> 1) map onto the effective coordinate.
        denom = 1.0 - s_ir - s_dr
        if denom <= 0.0:
            raise ValueError(
                f"Invalid endpoints for imbibition: S_inj_r "
                f"({s_ir}) + S_disp_r ({s_dr}) >= 1."
            )

        def pc_fn(S_inj):
            S_w = np.asarray(S_inj, dtype=float)
            S_w_eff = (S_w - s_ir) / denom
            return model.pc(S_w_eff)

        return pc_fn

    else:
        raise ValueError(
            f"Unknown process_type '{process_type}'. "
            f"Expected 'drainage' or 'imbibition'."
        )


# -- Sampled curves for plotting (Step 6 will use these) --------------------
def kr_curves(tp_inputs, n_points=200):
    """
    Sample kr_inj and kr_disp on a uniform grid over S_inj in [0, 1].
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
    Sample Pc on a uniform grid over S_inj in [0, 1]. Returns dict
    with arrays ready for Plotly. If Pc is disabled, Pc_mbar is all
    zeros and model_name is None. Includes process_type so the
    plotting layer can choose linear vs log y-scale and label axes
    with the physically-appropriate saturation interpretation.
    """
    S = np.linspace(0.0, 1.0, n_points)
    pc_fn = make_pc_function(tp_inputs)
    return {
        "S_inj":        S,
        "Pc_mbar":      pc_fn(S),
        "enabled":      tp_inputs["pc"]["enabled"],
        "model_name":   tp_inputs["pc"].get("model") if tp_inputs["pc"]["enabled"] else None,
        "process_type": tp_inputs.get("process_type", "drainage"),
    }