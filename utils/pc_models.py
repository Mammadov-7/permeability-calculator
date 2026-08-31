"""
Pluggable capillary pressure parameterizations.

Mirrors the design of utils/kr_models.py. Each model is a class
exposing a common interface so the forward solver, the plotting
layer, and the UI can treat them uniformly:

    class SomePc(PcModel):
        name             : str            -- display name (UI dropdown)
        process_types    : list[str]      -- {"drainage", "imbibition"}
        parameter_names  : list[str]      -- ordered param names
        parameter_bounds : dict           -- {name: (lo, hi)}
        defaults         : dict           -- {name: default value}
        param_display    : dict           -- per-param UI hints

        def pc(self, S_w_eff): ...        -- normalized wetting-phase sat -> Pc [mbar]

Model instances hold their own parameter values, so the same
interface works whether the model has 2 (Brooks-Corey, van Genuchten)
or 4 (Skjaeveland) parameters. Saturation normalization (S_inj -> S_w_eff)
is done *outside* the model in utils/twophase.py's make_pc_function,
so the model receives normalized wetting-phase saturation
S_w_eff in (0, 1] and returns Pc in mbar.

Sign convention
---------------
    Pc = P_nonwetting - P_wetting

Drainage models (Brooks-Corey, van Genuchten) return Pc > 0 always.
Imbibition models (Skjaeveland) can return Pc < 0 in mixed-wet
saturation ranges -- the whole point of the model.

Process-type filtering
----------------------
Each model declares which physical processes it validly represents
via the class attribute `process_types`. The UI uses
`pc_models_for_process(process_type)` to filter the dropdown so the
user cannot select Brooks-Corey for imbibition or Skjaeveland for
drainage. Physically meaningless combinations are prevented at the
UI layer rather than caught at solver time.

Adding a new parameterization: subclass PcModel, set the class
attributes (including process_types), implement pc(), and register
it in PC_MODELS. Nothing else in the codebase needs to know about
the new model.

Pure NumPy; Pyodide-compatible.
"""

import numpy as np


# -- Base class -------------------------------------------------------------
class PcModel:
    """
    Base class for all Pc parameterizations.

    Subclasses set the class attributes below and implement pc() on
    *normalized wetting-phase* saturation S_w_eff in (0, 1]. The caller
    handles S_inj -> S_w_eff mapping; models are purely mathematical.

    Sign convention: Pc = P_nonwetting - P_wetting.
    Drainage models return Pc > 0; imbibition models may return Pc < 0.
    Units of Pc are mbar (matches P_entry_mbar in tp_inputs).
    """

    name = ""                       # display name shown in the UI dropdown
    process_types = []              # subset of {"drainage", "imbibition"}
    parameter_names = []            # ordered -- canonical parameter order
    parameter_bounds = {}           # {name: (lo, hi)}
    defaults = {}                   # {name: default} -- seeds the UI
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


# -- Brooks-Corey (drainage) ------------------------------------------------
class BrooksCoreyPc(PcModel):
    """
    Brooks-Corey drainage capillary pressure (Brooks & Corey 1966).

        Pc(S_w_eff) = P_entry * S_w_eff ^ (-1/lambda)

    Pc = P_entry at S_w_eff = 1 (threshold entry pressure), rising
    monotonically as the wetting phase drains. Sharp step behaviour
    at the threshold: unwettable rock until Pc exceeds P_entry, then
    invasion begins. Two parameters. Industry-standard default,
    physically legible (P_entry ~ largest connected pore, lambda ~
    pore-size distribution width). Drainage only -- for the reverse
    process the wettability transition and possible sign change require
    an imbibition model such as Skjaeveland.
    """

    name = "Brooks-Corey"
    process_types = ["drainage", "imbibition"]

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
            "label": "Pore-size index lambda", "step": 0.1, "fmt": "%.2f",
            "help":  "Brooks-Corey pore-size distribution index. "
                     "Smaller lambda ==> wider pore-size distribution ==> "
                     "more gradual Pc rise. Typical range: 0.5 (heterogeneous) "
                     "to 5 (uniform).",
        },
    }

    def pc(self, S_w_eff):
        S_w_eff = np.clip(np.asarray(S_w_eff, dtype=float), 1e-4, 1.0)
        return self.params["P_entry_mbar"] * S_w_eff ** (-1.0 / self.params["lambda"])


# -- Van Genuchten (drainage, Mualem-linked) --------------------------------
class VanGenuchtenPc(PcModel):
    """
    Van Genuchten drainage capillary pressure (van Genuchten 1980),
    with the Mualem constraint m = 1 - 1/n baked in.

    Given the saturation form S_w_eff = [1 + (alpha*Pc)^n]^(-m), solving
    for Pc:

        Pc(S_w_eff) = (1/alpha) * (S_w_eff^(-1/m) - 1)^(1/n),  m = 1 - 1/n

    Two shape parameters (alpha, n); three by convention if m is freed,
    but m-linked is standard in soil physics and most reservoir
    simulators. Unlike Brooks-Corey, van Genuchten Pc -> 0 smoothly as
    S_w_eff -> 1 (no sharp entry threshold), which makes it preferred
    for wettability-transitional rocks and any rock where BC's step at
    P_entry doesn't match observed data. Drainage only.

    Parameter alpha has units [1/mbar]. Small alpha ==> high
    characteristic pressure ==> harder for non-wetting phase to invade.
    Rough analog: 1/alpha is a pressure scale comparable to (though not
    identical with) Brooks-Corey's P_entry.
    """

    name = "van Genuchten"
    process_types = ["drainage", "imbibition"]

    parameter_names = ["alpha_vg", "n_vg"]

    # alpha range: 0.001-0.5 [1/mbar] covers ~2-1000 mbar characteristic pressure.
    # n range: > 1 required (n=1 makes m=0, degenerate). Upper bound 10 covers
    # very-uniform-pore-size cases.
    parameter_bounds = {
        "alpha_vg": (1e-3, 0.5),
        "n_vg":     (1.05, 10.0),
    }

    defaults = {
        "alpha_vg": 0.01,   # ==> characteristic pressure ~100 mbar
        "n_vg":     2.5,
    }

    param_display = {
        "alpha_vg": {
            "label": "Inverse pressure alpha [1/mbar]", "step": 0.001, "fmt": "%.4f",
            "help":  "Van Genuchten alpha parameter. Small alpha ==> high "
                     "characteristic pressure. Rough analog: 1/alpha gives a "
                     "pressure scale comparable to Brooks-Corey's P_entry. "
                     "Default 0.01 corresponds to ~100 mbar characteristic pressure.",
        },
        "n_vg": {
            "label": "Pore-size index n", "step": 0.1, "fmt": "%.2f",
            "help":  "Van Genuchten pore-size distribution index. "
                     "Larger n ==> sharper transition. n > 1 is required. "
                     "m auto-computed as 1 - 1/n (Mualem constraint). "
                     "Typical range: 1.5 (heterogeneous) to 5 (uniform).",
        },
    }

    def pc(self, S_w_eff):
        # Clip S_w_eff away from 0 (singularity) but allow exact 1 (Pc=0 there).
        S_w_eff = np.clip(np.asarray(S_w_eff, dtype=float), 1e-4, 1.0)
        alpha = self.params["alpha_vg"]
        n     = self.params["n_vg"]
        m     = 1.0 - 1.0 / n
        # (S^(-1/m) - 1) >= 0 always inside [0, 1], so the outer power is safe.
        inner = S_w_eff ** (-1.0 / m) - 1.0
        # Guard against numerical negatives from floating-point at S_w_eff = 1.
        inner = np.maximum(inner, 0.0)
        return (inner ** (1.0 / n)) / alpha


# -- Skjaeveland (imbibition, mixed-wet) ------------------------------------
class SkjaevelandPc(PcModel):
    """
    Skjaeveland imbibition capillary pressure (Skjaeveland et al. 2000).

    Handles mixed-wet systems where Pc can smoothly transition from
    positive (water-wet contribution dominant, low S_w) to negative
    (oil-wet contribution dominant, high S_w). This is the standard
    imbibition model for carbonate reservoirs and aged sandstones
    where wettability has been altered by crude-oil interactions.

    Two-term sum, one per wettability character:

        Pc(S_w) = c_w / [(S_w - S_wr) / (1 - S_wr)]^a_w
               + c_o / [(S_o - S_or) / (1 - S_or)]^a_o

    where S_o = 1 - S_w, and by Skjaeveland's convention c_w > 0 and
    c_o < 0. The first term is positive and diverges as S_w -> S_wr
    (water-wet contribution: capillary suction holds the last of the
    water in place). The second term is negative and diverges as
    S_o -> S_or (oil-wet contribution: capillarity resists further
    water invasion into oil-wet pore fractions).

    Physical interpretation of the parameters:

        c_w > 0   Magnitude of the water-wet character. Large c_w ==>
                  strongly water-wet, Pc stays positive across most of
                  the saturation range. c_w -> 0 ==> no water-wet
                  contribution (pure oil-wet).

        c_o < 0   Magnitude of the oil-wet character (negative by
                  convention). Large |c_o| ==> strongly oil-wet
                  fraction, Pc goes deeply negative at high S_w.
                  c_o -> 0 ==> no oil-wet contribution (pure water-wet
                  imbibition curve, all positive).

        a_w, a_o  Shape exponents controlling how sharply each term
                  decays away from its residual singularity. Larger
                  exponents ==> sharper divergence, more localized
                  effect near the residual.

    Wettability tuning: choosing c_w and c_o lets you dial any
    wettability from strongly water-wet (large c_w, small |c_o|)
    through neutral (comparable magnitudes, zero-crossing near mid
    saturation) to strongly oil-wet (small c_w, large |c_o|).

    In the calculator's coordinate system the model receives
    normalized wetting-phase saturation S_w_eff in (0, 1]. Inside pc()
    we recover the physical S_w by inverting the normalization the
    caller applied, so the two Skjaeveland terms can each reference
    their own natural residual singularity. See make_pc_function() in
    utils/twophase.py for the calling convention.
    """

    name = "Skjaeveland"
    process_types = ["imbibition", "drainage"]

    parameter_names = ["c_w", "a_w", "c_o", "a_o"]

    # c_w > 0 (water-wet contribution magnitude, in mbar).
    # c_o < 0 (oil-wet contribution magnitude, in mbar).
    # Bounds chosen to cover the mbar-scale Pc values consistent with
    # the rest of the calculator (P_entry_mbar range 1-5000).
    parameter_bounds = {
        "c_w": (0.1, 2000.0),
        "a_w": (0.1, 5.0),
        "c_o": (-2000.0, -0.1),
        "a_o": (0.1, 5.0),
    }

    defaults = {
        "c_w": 50.0,     # moderate water-wet contribution
        "a_w": 0.5,      # gentle divergence near S_wr
        "c_o": -30.0,    # weaker oil-wet contribution ==> mildly mixed-wet
        "a_o": 0.5,      # gentle divergence near S_or
    }

    param_display = {
        "c_w": {
            "label": "Water-wet magnitude c_w [mbar]", "step": 5.0, "fmt": "%.1f",
            "help": "Amplitude of the water-wet Pc contribution "
                    "(positive by convention). Large c_w ==> strongly "
                    "water-wet: Pc stays positive across most of the "
                    "saturation range. Small c_w ==> no water-wet "
                    "contribution.",
        },
        "a_w": {
            "label": "Water-wet exponent a_w", "step": 0.05, "fmt": "%.2f",
            "help": "Shape exponent for the water-wet term. Larger a_w "
                    "==> sharper divergence as S_w approaches S_wr. "
                    "Typical range 0.2 to 1.0.",
        },
        "c_o": {
            "label": "Oil-wet magnitude c_o [mbar]", "step": 5.0, "fmt": "%.1f",
            "help": "Amplitude of the oil-wet Pc contribution "
                    "(negative by Skjaeveland's convention). Large |c_o| "
                    "==> strongly oil-wet fraction: Pc goes deeply "
                    "negative at high water saturation. Small |c_o| "
                    "==> no oil-wet contribution (pure water-wet "
                    "imbibition, Pc all positive).",
        },
        "a_o": {
            "label": "Oil-wet exponent a_o", "step": 0.05, "fmt": "%.2f",
            "help": "Shape exponent for the oil-wet term. Larger a_o "
                    "==> sharper divergence as S_o approaches S_or "
                    "(equivalently, as S_w approaches 1 - S_or). "
                    "Typical range 0.2 to 1.0.",
        },
    }

    def pc(self, S_w_eff):
        # S_w_eff arrives as normalized wetting-phase saturation in (0, 1].
        # For Skjaeveland's two-term form we work on the *effective*
        # scaled coordinates directly:
        #   term_w  singular at S_w_eff -> 0   (water residual limit)
        #   term_o  singular at S_w_eff -> 1   (oil residual limit)
        # Because make_pc_function has already mapped physical S_w onto
        # (0, 1] via the same normalization used by BC and vG, the two
        # effective coordinates are:
        #   x_w = S_w_eff                         (goes to 0 at S_wr)
        #   x_o = 1 - S_w_eff                     (goes to 0 at S_or)
        # These are exactly the terms Skjaeveland's formula wants when
        # written on normalized saturation.
        S_w_eff = np.clip(np.asarray(S_w_eff, dtype=float), 1e-4, 1.0 - 1e-4)
        x_w = S_w_eff
        x_o = 1.0 - S_w_eff
        term_w = self.params["c_w"] / (x_w ** self.params["a_w"])
        term_o = self.params["c_o"] / (x_o ** self.params["a_o"])
        return term_w + term_o


# -- Registry ---------------------------------------------------------------
# Keys are display names (used directly by the Streamlit dropdown and stored
# in tp_inputs["pc"]["model"]). Adding a model = adding an entry here.
PC_MODELS = {
    "Brooks-Corey":  BrooksCoreyPc,
    "van Genuchten": VanGenuchtenPc,
    "Skjaeveland":   SkjaevelandPc,
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


def pc_models_for_process(process_type):
    """
    Return the list of Pc-model display names valid for a given
    process type ("drainage" or "imbibition"). Used by the UI to
    filter the model dropdown so that physically meaningless choices
    (Brooks-Corey for imbibition, Skjaeveland for drainage) are
    prevented at the widget level.
    """
    return [name for name, cls in PC_MODELS.items()
            if process_type in cls.process_types]