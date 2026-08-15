"""
Pluggable relative-permeability parameterizations.

Each model is a class exposing a common interface so the forward
solver, the inverse fitter, and the UI can treat them uniformly:

    class SomeModel(KrModel):
        name                    : str          -- display name (UI)
        parameter_names         : list[str]    -- ordered param names
        parameter_bounds        : dict         -- {name: (lo, hi)}
        log_space_params        : dict         -- {name: bool}
        defaults                : dict         -- {name: default value}
        injected_param_names    : list[str]    -- subset for UI column
        displaced_param_names   : list[str]    -- subset for UI column
        param_display           : dict         -- per-param UI hints

        def kr_inj(self, S_n):  ...   # normalized-sat -> kr injected
        def kr_disp(self, S_n): ...   # normalized-sat -> kr displaced

Model instances hold their own parameter values, so the same interface
works whether the caller has 4 (Corey) or 8 (LET) parameters. Saturation
normalization is done *outside* the model (in utils/twophase's
make_kr_functions) so each model receives S_n ∈ [0, 1] and returns
kr ∈ [0, kr_max].

Adding a new parameterization: subclass KrModel, set the class
attributes, implement kr_inj/kr_disp, and register it in KR_MODELS.
Nothing else in the codebase needs to know about the new model.

Pure NumPy; Pyodide-compatible.
"""

import numpy as np


# ── Base class ──────────────────────────────────────────────────────────────
class KrModel:
    """
    Base class for all kr parameterizations.

    Subclasses set the class attributes below and implement kr_inj /
    kr_disp on *normalized* saturation S_n ∈ [0, 1]. The caller does
    the normalization; models are purely mathematical.
    """

    name = ""                       # display name shown in the UI dropdown
    parameter_names = []            # ordered list — the canonical parameter order
    parameter_bounds = {}           # {name: (lo, hi)}
    log_space_params = {}           # {name: bool} — for N-M multi-start transforms
    defaults = {}                   # {name: default value} — used to seed the UI
    injected_param_names = []       # names shown in the "Injected phase" UI column
    displaced_param_names = []      # names shown in the "Displaced phase" UI column
    param_display = {}              # {name: {"label": str, "step": float, "fmt": str}}

    def __init__(self, params):
        """
        params : dict of {name: value} covering all parameter_names.
        Missing names raise ValueError; extras are ignored.
        """
        missing = set(self.parameter_names) - set(params.keys())
        if missing:
            raise ValueError(
                f"{self.name}: missing parameter(s) {sorted(missing)}"
            )
        self.params = {k: float(params[k]) for k in self.parameter_names}

    def kr_inj(self, S_n):
        raise NotImplementedError

    def kr_disp(self, S_n):
        raise NotImplementedError


# ── Corey ───────────────────────────────────────────────────────────────────
class CoreyKr(KrModel):
    """
    Classical Corey (Corey 1954, Brooks & Corey 1966).

        kr_inj  = kr_inj_max  * S_n ** n_inj
        kr_disp = kr_disp_max * (1 - S_n) ** n_disp

    Two parameters per phase; four total. Widely used industry default;
    physically legible parameters.
    """

    name = "Corey"

    parameter_names = [
        "kr_inj_max", "n_inj",
        "kr_disp_max", "n_disp",
    ]

    parameter_bounds = {
        "kr_inj_max":  (0.01, 1.0),
        "n_inj":       (0.5,  10.0),
        "kr_disp_max": (0.01, 1.0),
        "n_disp":      (0.5,  10.0),
    }

    log_space_params = {
        "kr_inj_max":  True,
        "n_inj":       False,
        "kr_disp_max": True,
        "n_disp":      False,
    }

    defaults = {
        "kr_inj_max":  0.6,
        "n_inj":       2.0,
        "kr_disp_max": 1.0,
        "n_disp":      3.0,
    }

    injected_param_names  = ["kr_inj_max",  "n_inj"]
    displaced_param_names = ["kr_disp_max", "n_disp"]

    param_display = {
        "kr_inj_max":  {"label": "End-point kr_max", "step": 0.01, "fmt": "%.3f",
                        "help": "kr of injected phase at S_inj = 1 − S_r,disp (max sweep)."},
        "n_inj":       {"label": "Corey exponent n", "step": 0.1,  "fmt": "%.2f",
                        "help": "Curvature of kr curve: n=1 linear, n=2 quadratic, "
                                "higher = sharper front."},
        "kr_disp_max": {"label": "End-point kr_max", "step": 0.01, "fmt": "%.3f",
                        "help": "kr of displaced phase at S_inj = S_r,inj (no sweep)."},
        "n_disp":      {"label": "Corey exponent n", "step": 0.1,  "fmt": "%.2f",
                        "help": "Curvature of kr curve: n=1 linear, n=2 quadratic, "
                                "higher = sharper front."},
    }

    def kr_inj(self, S_n):
        S_n = np.clip(np.asarray(S_n, dtype=float), 0.0, 1.0)
        return self.params["kr_inj_max"] * S_n ** self.params["n_inj"]

    def kr_disp(self, S_n):
        S_n = np.clip(np.asarray(S_n, dtype=float), 0.0, 1.0)
        return self.params["kr_disp_max"] * (1.0 - S_n) ** self.params["n_disp"]


# ── LET ─────────────────────────────────────────────────────────────────────
class LETKr(KrModel):
    """
    Lomeland-Ebeltoft-Thomas (Lomeland et al. 2005, SCA).

        kr_inj  = kr_inj_max  * S_n^L      / (S_n^L      + E*(1 - S_n)^T)
        kr_disp = kr_disp_max * (1 - S_n)^L / ((1 - S_n)^L + E*S_n^T)

    Four parameters per phase; eight total. L governs low-saturation
    behaviour of that phase, T governs high-saturation behaviour, E
    controls the mid-region transition width. Superset of Corey:
    L = T = n, E = 1 recovers a Corey-like curve (not identical but
    very close).

    Trade-off: doubles the parameter count vs Corey. The inverse
    problem becomes measurably harder — SSE landscape is flatter,
    identifiability is worse. Recommended workflow: fix as many
    parameters as possible from independent constraints (steady-state
    endpoints, imbibition experiments) rather than freeing all eight.
    """

    name = "LET"

    parameter_names = [
        "kr_inj_max",  "L_inj",  "E_inj",  "T_inj",
        "kr_disp_max", "L_disp", "E_disp", "T_disp",
    ]

    # E spans several orders of magnitude in practice; kr_max spans two.
    # L and T live on a moderate linear range.
    parameter_bounds = {
        "kr_inj_max":  (0.01, 1.0),
        "L_inj":       (0.5,  8.0),
        "E_inj":       (0.01, 100.0),
        "T_inj":       (0.5,  8.0),
        "kr_disp_max": (0.01, 1.0),
        "L_disp":      (0.5,  8.0),
        "E_disp":      (0.01, 100.0),
        "T_disp":      (0.5,  8.0),
    }

    log_space_params = {
        "kr_inj_max":  True,
        "L_inj":       False,
        "E_inj":       True,
        "T_inj":       False,
        "kr_disp_max": True,
        "L_disp":      False,
        "E_disp":      True,
        "T_disp":      False,
    }

    defaults = {
        "kr_inj_max":  0.6,
        "L_inj":       2.0,
        "E_inj":       1.0,
        "T_inj":       2.0,
        "kr_disp_max": 1.0,
        "L_disp":      2.0,
        "E_disp":      1.0,
        "T_disp":      2.0,
    }

    injected_param_names  = ["kr_inj_max",  "L_inj",  "E_inj",  "T_inj"]
    displaced_param_names = ["kr_disp_max", "L_disp", "E_disp", "T_disp"]

    param_display = {
        "kr_inj_max":  {"label": "End-point kr_max", "step": 0.01, "fmt": "%.3f",
                        "help": "kr of injected phase at S_inj = 1 − S_r,disp (max sweep)."},
        "L_inj":       {"label": "LET exponent L",   "step": 0.1,  "fmt": "%.2f",
                        "help": "Controls low-saturation behaviour of injected phase. "
                                "Larger L = flatter approach off zero saturation."},
        "E_inj":       {"label": "LET parameter E",  "step": 0.01, "fmt": "%.3f",
                        "help": "Controls mid-region transition of injected phase. "
                                "E = 1 gives Corey-like symmetric curve; E > 1 shifts "
                                "the transition; large E ⇒ sharper front."},
        "T_inj":       {"label": "LET exponent T",   "step": 0.1,  "fmt": "%.2f",
                        "help": "Controls high-saturation approach of injected phase "
                                "to kr_max. Larger T = later, sharper approach."},
        "kr_disp_max": {"label": "End-point kr_max", "step": 0.01, "fmt": "%.3f",
                        "help": "kr of displaced phase at S_inj = S_r,inj (no sweep)."},
        "L_disp":      {"label": "LET exponent L",   "step": 0.1,  "fmt": "%.2f",
                        "help": "Controls low-saturation behaviour of displaced phase. "
                                "'Low' here means low displaced-phase saturation ⇒ high S_inj."},
        "E_disp":      {"label": "LET parameter E",  "step": 0.01, "fmt": "%.3f",
                        "help": "Controls mid-region transition of displaced phase. "
                                "E = 1 recovers Corey-like symmetry."},
        "T_disp":      {"label": "LET exponent T",   "step": 0.1,  "fmt": "%.2f",
                        "help": "Controls how the displaced-phase kr approaches its "
                                "residual endpoint."},
    }

    def kr_inj(self, S_n):
        S_n = np.clip(np.asarray(S_n, dtype=float), 0.0, 1.0)
        L = self.params["L_inj"]
        E = self.params["E_inj"]
        T = self.params["T_inj"]
        Sn_L      = S_n ** L
        one_m_T   = (1.0 - S_n) ** T
        denom     = Sn_L + E * one_m_T
        kr = np.where(denom > 0.0, Sn_L / denom, 0.0)
        return self.params["kr_inj_max"] * kr

    def kr_disp(self, S_n):
        # Symmetric form: displaced-phase kr uses (1 - S_n) in the L position
        # and S_n in the T position, so it monotonically decreases in S_inj.
        S_n = np.clip(np.asarray(S_n, dtype=float), 0.0, 1.0)
        L = self.params["L_disp"]
        E = self.params["E_disp"]
        T = self.params["T_disp"]
        one_m_L   = (1.0 - S_n) ** L
        Sn_T      = S_n ** T
        denom     = one_m_L + E * Sn_T
        kr = np.where(denom > 0.0, one_m_L / denom, 0.0)
        return self.params["kr_disp_max"] * kr


# ── Registry ────────────────────────────────────────────────────────────────
# Keys are display names (used directly by the Streamlit dropdown and stored
# in tp_inputs["kr"]["model"]). Adding a model = adding an entry here.
KR_MODELS = {
    "Corey": CoreyKr,
    "LET":   LETKr,
}

KR_MODEL_CHOICES = list(KR_MODELS.keys())


def get_model_class(model_name):
    """Look up a model class by its display name. Raises on unknown."""
    if model_name not in KR_MODELS:
        raise ValueError(
            f"Unknown kr model '{model_name}'. "
            f"Available: {KR_MODEL_CHOICES}"
        )
    return KR_MODELS[model_name]


def make_kr_model(model_name, params):
    """Instantiate a model of the given display name with the given params."""
    return get_model_class(model_name)(params)


def default_params(model_name):
    """Return the default parameter dict for a given model. Used to seed UI."""
    return dict(get_model_class(model_name).defaults)