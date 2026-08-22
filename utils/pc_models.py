"""
Pluggable capillary pressure parameterizations.

Mirrors the design of utils/kr_models.py. Each model is a class
exposing a common interface so the forward solver, the plotting
layer, and the UI can treat them uniformly:

    class SomePc(PcModel):
        name             : str            -- display name (UI dropdown)
        parameter_names  : list[str]      -- ordered param names
        parameter_bounds : dict           -- {name: (lo, hi)}
        defaults         : dict           -- {name: default value}
        param_display    : dict           -- per-param UI hints

        def pc(self, S_w_eff): ...        -- normalized wetting-phase sat -> Pc [mbar]

Model instances hold their own parameter values, so the same
interface works whether the model has 2 (Brooks-Corey, van Genuchten)
or more parameters. Saturation normalization (S_inj -> S_w_eff) is
done *outside* the model in utils/twophase.py's make_pc_function,
so the model receives normalized wetting-phase saturation
S_w_eff ∈ (0, 1] and returns Pc in mbar.

Adding a new parameterization: subclass PcModel, set the class
attributes, implement pc(), and register it in PC_MODELS. Nothing
else in the codebase needs to know about the new model.

Pure NumPy; Pyodide-compatible.
"""

import numpy as np


# ── Base class ──────────────────────────────────────────────────────────────
class PcModel:
    """
    Base class for all Pc parameterizations.

    Subclasses set the class attributes below and implement pc() on
    *normalized wetting-phase* saturation S_w_eff ∈ (0, 1]. The caller
    handles S_inj → S_w_eff mapping; models are purely mathematical.

    Sign convention: Pc = P_nonwetting − P_wetting > 0 on drainage.
    Units of Pc are mbar (matches P_entry_mbar in tp_inputs).
    """

    name = ""                       # display name shown in the UI dropdown
    parameter_names = []            # ordered — canonical parameter order
    parameter_bounds = {}           # {name: (lo, hi)}
    defaults = {}                   # {name: default} — seeds the UI
    param_display = {}              # {name: {"label", "step", "fmt", "help"}}

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

    def pc(self, S_w_eff):
        raise NotImplementedError


# ── Brooks-Corey ────────────────────────────────────────────────────────────
class BrooksCoreyPc(PcModel):
    """
    Brooks-Corey drainage capillary pressure (Brooks & Corey 1966).

        Pc(S_w_eff) = P_entry · S_w_eff ^ (−1/λ)

    Pc = P_entry at S_w_eff = 1 (threshold entry pressure), rising
    monotonically as the wetting phase drains. Sharp step behaviour
    at the threshold: unwettable rock until Pc exceeds P_entry, then
    invasion begins. Two parameters. Industry-standard default,
    physically legible (P_entry ↔ largest connected pore, λ ↔ pore-size
    distribution width).
    """

    name = "Brooks-Corey"

    parameter_names = ["P_entry_mbar", "lambda"]

    parameter_bounds = {
        "P_entry_mbar": (1.0, 5000.0),
        "lambda":       (0.1, 10.0),
    }

    defaults = {
        "P_entry_mbar": 100.0,
        "lambda":       2.0,
    }

    param_display = {
        "P_entry_mbar": {
            "label": "Entry pressure P_e [mbar]", "step": 5.0, "fmt": "%.1f",
            "help":  "Brooks-Corey threshold entry pressure of the "
                     "non-wetting (injected) phase. Pc = P_e at full "
                     "wetting saturation; rises as the wetting phase drains.",
        },
        "lambda": {
            "label": "Pore-size index λ", "step": 0.1, "fmt": "%.2f",
            "help":  "Brooks-Corey pore-size distribution index. "
                     "Smaller λ ⇒ wider pore-size distribution ⇒ more gradual "
                     "Pc rise. Typical range: 0.5 (heterogeneous) to 5 (uniform).",
        },
    }

    def pc(self, S_w_eff):
        S_w_eff = np.clip(np.asarray(S_w_eff, dtype=float), 1e-4, 1.0)
        return self.params["P_entry_mbar"] * S_w_eff ** (-1.0 / self.params["lambda"])


# ── Van Genuchten (Mualem-linked) ───────────────────────────────────────────
class VanGenuchtenPc(PcModel):
    """
    Van Genuchten drainage capillary pressure (van Genuchten 1980),
    with the Mualem constraint m = 1 − 1/n baked in.

    Given the saturation form S_w_eff = [1 + (α·Pc)^n]^(-m), solving
    for Pc:

        Pc(S_w_eff) = (1/α) · (S_w_eff^(-1/m) − 1)^(1/n)   with m = 1 − 1/n

    Two shape parameters (α, n); three by convention if m is freed, but
    m-linked is standard in soil physics and most reservoir simulators.
    Unlike Brooks-Corey, van Genuchten Pc → 0 smoothly as S_w_eff → 1
    (no sharp entry threshold), which makes it preferred for
    wettability-transitional rocks and any rock where BC's step at
    P_entry doesn't match observed data.

    Parameter α has units [1/mbar]. Small α ⇒ high characteristic
    pressure ⇒ harder for non-wetting phase to invade. Rough analog:
    1/α is a pressure scale comparable to (though not identical with)
    Brooks-Corey's P_entry.
    """

    name = "van Genuchten"

    parameter_names = ["alpha_vg", "n_vg"]

    # α range: 0.001–0.5 [1/mbar] covers ~2–1000 mbar characteristic pressure.
    # n range: > 1 required (n=1 makes m=0, degenerate). Upper bound 10 covers
    # very-uniform-pore-size cases.
    parameter_bounds = {
        "alpha_vg": (1e-3, 0.5),
        "n_vg":     (1.05, 10.0),
    }

    defaults = {
        "alpha_vg": 0.01,   # ⇒ characteristic pressure ~100 mbar
        "n_vg":     2.5,
    }

    param_display = {
        "alpha_vg": {
            "label": "Inverse pressure α [1/mbar]", "step": 0.001, "fmt": "%.4f",
            "help":  "Van Genuchten α parameter. Small α ⇒ high "
                     "characteristic pressure. Rough analog: 1/α gives a "
                     "pressure scale comparable to Brooks-Corey's P_entry. "
                     "Default 0.01 corresponds to ~100 mbar characteristic pressure.",
        },
        "n_vg": {
            "label": "Pore-size index n", "step": 0.1, "fmt": "%.2f",
            "help":  "Van Genuchten pore-size distribution index. "
                     "Larger n ⇒ sharper transition. n > 1 is required. "
                     "m auto-computed as 1 − 1/n (Mualem constraint). "
                     "Typical range: 1.5 (heterogeneous) to 5 (uniform).",
        },
    }

    def pc(self, S_w_eff):
        # Clip S_w_eff away from 0 (singularity) but allow exact 1 (Pc=0 there).
        S_w_eff = np.clip(np.asarray(S_w_eff, dtype=float), 1e-4, 1.0)
        alpha = self.params["alpha_vg"]
        n     = self.params["n_vg"]
        m     = 1.0 - 1.0 / n
        # (S^(-1/m) - 1) ≥ 0 always inside [0, 1], so the outer power is safe.
        inner = S_w_eff ** (-1.0 / m) - 1.0
        # Guard against numerical negatives from floating-point at S_w_eff = 1.
        inner = np.maximum(inner, 0.0)
        return (inner ** (1.0 / n)) / alpha


# ── Registry ────────────────────────────────────────────────────────────────
# Keys are display names (used directly by the Streamlit dropdown and stored
# in tp_inputs["pc"]["model"]). Adding a model = adding an entry here.
PC_MODELS = {
    "Brooks-Corey":  BrooksCoreyPc,
    "van Genuchten": VanGenuchtenPc,
}

PC_MODEL_CHOICES = list(PC_MODELS.keys())


def get_pc_model_class(model_name):
    """Look up a Pc model class by its display name. Raises on unknown."""
    if model_name not in PC_MODELS:
        raise ValueError(
            f"Unknown Pc model '{model_name}'. "
            f"Available: {PC_MODEL_CHOICES}"
        )
    return PC_MODELS[model_name]


def make_pc_model(model_name, params):
    """Instantiate a Pc model of the given display name with the given params."""
    return get_pc_model_class(model_name)(params)


def default_pc_params(model_name):
    """Return the default parameter dict for a given Pc model."""
    return dict(get_pc_model_class(model_name).defaults)