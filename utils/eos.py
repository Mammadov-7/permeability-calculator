"""
Equation of State (EOS) models for fluid compressibility.

Every phase in phases.json declares an `eos_model` field ("incompressible"
or "ideal_gas") which selects one of the classes below. The solver calls
`phase_eos.get_compressibility(P_Pa, T_K)` per cell per timestep to obtain
a compressibility appropriate for the local pressure and temperature.

The architecture is intentionally pluggable so that Phase 3 can add a
`PengRobinsonEOS` subclass without touching the solver.

Conventions
-----------
All EOS methods take and return SI:
    - Pressure P in Pa
    - Temperature T in K
    - Compressibility c in 1/Pa
"""


class EOS:
    """Abstract base class for EOS models."""

    name = "abstract"

    def get_compressibility(self, P_Pa, T_K):
        """Return isothermal compressibility [1/Pa] at (P, T)."""
        raise NotImplementedError

    def describe(self, P_Pa, T_K):
        """
        Short human-readable summary of this EOS's state at (P, T),
        for UI readouts. Subclasses can override for richer detail.
        """
        c = self.get_compressibility(P_Pa, T_K)
        return f"{self.name}: c = {c:.3e} /Pa"


class IncompressibleEOS(EOS):
    """
    Constant compressibility — the value the user types in the UI.
    Used for liquids at core-flood conditions where c ~ 4.5e-10 /Pa
    barely changes over the pressure range.
    """

    name = "incompressible"

    def __init__(self, c_Pa=4.5e-10):
        self.c_user = float(c_Pa)

    def get_compressibility(self, P_Pa, T_K):
        return self.c_user


class IdealGasEOS(EOS):
    """
    Ideal-gas isothermal compressibility: c = 1/P.

    Derivation:
        For an ideal gas PV = nRT, at fixed T:
            ρ = PM/(RT)  →  ∂ρ/∂P = M/(RT) = ρ/P
            c = (1/ρ)(∂ρ/∂P) = 1/P

    Notes
    -----
    - c depends only on local pressure, not temperature (at fixed T).
    - Diverges as P → 0; solver should never see P ≤ 0 in practice
      because we enforce back-pressure P_out > 0.
    - Overestimates c for real gases at high pressure (Z-factor effect
      not captured). Phase 3 will add a `PengRobinsonEOS` subclass for
      rigorous non-ideality.
    """

    name = "ideal_gas"

    def get_compressibility(self, P_Pa, T_K):
        if P_Pa <= 0.0:
            # Numerical guard — should never occur in a physical run.
            return 1e-3
        return 1.0 / P_Pa


# ── Factory ────────────────────────────────────────────────────────────────
EOS_REGISTRY = {
    "incompressible": IncompressibleEOS,
    "ideal_gas":      IdealGasEOS,
}


def make_eos(eos_model, c_user_Pa=None):
    """
    Instantiate an EOS from a model name string.

    Parameters
    ----------
    eos_model : str
        One of the keys in EOS_REGISTRY.
    c_user_Pa : float or None
        User-supplied compressibility, used only by IncompressibleEOS
        (or when an override is enabled — the caller decides).
    """
    if eos_model not in EOS_REGISTRY:
        raise ValueError(
            f"Unknown eos_model '{eos_model}'. "
            f"Known: {list(EOS_REGISTRY.keys())}"
        )
    cls = EOS_REGISTRY[eos_model]
    if cls is IncompressibleEOS:
        return cls(c_Pa=c_user_Pa if c_user_Pa is not None else 4.5e-10)
    return cls()