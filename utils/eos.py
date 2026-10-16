"""
Equation of State (EOS) models for fluid compressibility.

Every phase in phases.json declares an `eos_model` field which selects one
of the classes below. The OnePhase solver calls
`eos.get_compressibility(P_Pa, T_K)` once per timestep, at the mean cell
pressure of the previous step (a lag-of-one linearisation), to obtain a
compressibility appropriate for the current pressure and temperature.

Models
------
    incompressible  constant c supplied by the user (liquids)
    ideal_gas       c = 1/P
    peng_robinson   real-gas c from the Peng-Robinson cubic

Conventions
-----------
All EOS methods take and return SI:
    - Pressure P in Pa
    - Temperature T in K
    - Compressibility c in 1/Pa
    - Density in kg/m3
"""

import math

import numpy as np


R_GAS = 8.314462618          # J/(mol·K)
SQRT2 = math.sqrt(2.0)


class EOS:
    """Abstract base class for EOS models."""

    name = "abstract"

    def get_compressibility(self, P_Pa, T_K):
        """Return isothermal compressibility [1/Pa] at (P, T)."""
        raise NotImplementedError

    def get_z_factor(self, P_Pa, T_K):
        """
        Return the compressibility FACTOR Z = PV/(nRT) [-].

        Not to be confused with get_compressibility, which returns the
        isothermal compressibility c [1/Pa]. Z is dimensionless and equals
        1.0 for an ideal gas.
        """
        return 1.0

    def get_density(self, P_Pa, T_K):
        """
        Return mass density [kg/m3], or None if this EOS has no basis for
        computing one. Nothing in the OnePhase solver consumes density
        (single-phase horizontal flow has no gravity term) — this exists
        for the UI diagnostic and for the Phase 4 centrifuge body force.
        """
        return None

    def describe(self, P_Pa, T_K):
        """Short human-readable summary of this EOS's state at (P, T)."""
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
            rho = PM/(RT)  ->  d(rho)/dP = M/(RT) = rho/P
            c = (1/rho)(d(rho)/dP) = 1/P

    Notes
    -----
    - c depends only on local pressure, not temperature (at fixed T).
    - Diverges as P -> 0; the solver should never see P <= 0 because we
      enforce a positive back-pressure.
    - Overestimates c for real gases at high pressure. Use
      PengRobinsonEOS when the Z-factor deviation matters.
    """

    name = "ideal_gas"

    def get_compressibility(self, P_Pa, T_K):
        if P_Pa <= 0.0:
            return 1e-3        # numerical guard, never hit physically
        return 1.0 / P_Pa


class PengRobinsonEOS(EOS):
    """
    Peng-Robinson (1976) real-gas EOS.

        P = RT/(V-b) - a*alpha / (V^2 + 2bV - b^2)

    In compressibility-factor form the cubic is

        Z^3 - (1-B) Z^2 + (A - 3B^2 - 2B) Z - (AB - B^2 - B^3) = 0

    with

        a     = 0.45724 R^2 Tc^2 / Pc
        b     = 0.07780 R Tc / Pc
        kappa = 0.37464 + 1.54226 w - 0.26992 w^2
        alpha = [1 + kappa (1 - sqrt(T/Tc))]^2
        A     = a*alpha*P / (RT)^2
        B     = b*P / (RT)

    Root selection
    --------------
    The cubic has one or three real roots. Roots with Z <= B are
    unphysical (molar volume below the co-volume) and are discarded. If
    two physical roots survive, the one with the lower molar Gibbs energy
    is selected, compared via the fugacity coefficient

        ln(phi) = Z - 1 - ln(Z-B)
                  - A/(2 sqrt(2) B) * ln[(Z+(1+sqrt2)B)/(Z+(1-sqrt2)B)]

    This matches established practice: MRST exposes the same choice as a
    `selectGibbsMinimum` flag, and Gibbs minimisation is the only
    criterion that is both necessary and sufficient for stability. The
    middle root is always thermodynamically unstable and is never chosen.

    Two real physical roots means the operating point sits inside the
    two-phase envelope. `root_info()` reports this so the UI can warn.
    At lab temperature only CO2 (Tc = 304 K) can reach that state; H2,
    N2 and CH4 are supercritical and always give a single root.

    Compressibility
    ---------------
        c = -(1/V)(dV/dP)_T  with  V = ZRT/P
          = 1/P - (1/Z)(dZ/dP)_T

    dZ/dP is evaluated by a central difference with a relative step of
    1e-5 (see `_DP_REL`). Chosen over the analytical derivative for
    verifiability: the finite difference can be checked by hand against
    two Z values printed from the diagnostic readout.

    Accuracy caveat
    ---------------
    Hydrogen has a NEGATIVE acentric factor (w = -0.219). The standard
    kappa correlation above was regressed on hydrocarbons with w roughly
    in [0, 0.5], so PR is extrapolating for H2. At lab temperature
    (Tr ~ 8.8) alpha is near unity and the error is small, but PR is not
    a reference-quality H2 model. For that, use the Muzny-Huber-Kazakov
    (2013) correlation as implemented in NIST REFPROP.
    """

    name = "peng_robinson"

    _DP_REL = 1.0e-5          # relative step for the numerical dZ/dP

    def __init__(self, Tc_K, Pc_Pa, omega, molar_mass_g_mol=None):
        if Tc_K is None or Pc_Pa is None or omega is None:
            raise ValueError(
                "PengRobinsonEOS requires Tc_K, Pc_Pa and omega. "
                "Add them to the phase entry in data/phases.json."
            )
        self.Tc = float(Tc_K)
        self.Pc = float(Pc_Pa)
        self.omega = float(omega)
        self.M = (float(molar_mass_g_mol) / 1000.0
                  if molar_mass_g_mol is not None else None)   # kg/mol

        self.a = 0.45724 * R_GAS ** 2 * self.Tc ** 2 / self.Pc
        self.b = 0.07780 * R_GAS * self.Tc / self.Pc
        self.kappa = (0.37464
                      + 1.54226 * self.omega
                      - 0.26992 * self.omega ** 2)

    # ── Dimensionless groups ───────────────────────────────────────────────
    def _alpha(self, T_K):
        Tr = T_K / self.Tc
        return (1.0 + self.kappa * (1.0 - math.sqrt(Tr))) ** 2

    def _A_B(self, P_Pa, T_K):
        RT = R_GAS * T_K
        A = self.a * self._alpha(T_K) * P_Pa / (RT ** 2)
        B = self.b * P_Pa / RT
        return A, B

    # ── Root selection ─────────────────────────────────────────────────────
    @staticmethod
    def _ln_phi(Z, A, B):
        """Log fugacity coefficient — the Gibbs-energy comparator."""
        if Z <= B:
            return float("inf")
        return (Z - 1.0
                - math.log(Z - B)
                - A / (2.0 * SQRT2 * B)
                * math.log((Z + (1.0 + SQRT2) * B)
                           / (Z + (1.0 - SQRT2) * B)))

    def root_info(self, P_Pa, T_K):
        """
        Solve the cubic and report which root was taken and why.

        Returns dict with keys:
            Z            selected compressibility factor
            n_real       number of real roots of the cubic
            n_physical   number of real roots with Z > B
            two_phase    True if two physical roots existed (inside the
                         two-phase envelope — UI should warn)
            phase_guess  "vapour-like", "liquid-like" or "single"
        """
        A, B = self._A_B(P_Pa, T_K)
        coeffs = [1.0,
                  -(1.0 - B),
                  A - 3.0 * B ** 2 - 2.0 * B,
                  -(A * B - B ** 2 - B ** 3)]
        roots = np.roots(coeffs)
        real = np.sort(roots[np.abs(roots.imag) < 1e-10].real)
        physical = [z for z in real if z > B]

        if not physical:
            # Below the co-volume everywhere — take the largest real root
            # and let the caller see n_physical == 0.
            return {"Z": float(max(real)), "n_real": len(real),
                    "n_physical": 0, "two_phase": False,
                    "phase_guess": "unphysical"}

        if len(physical) == 1:
            return {"Z": float(physical[0]), "n_real": len(real),
                    "n_physical": 1, "two_phase": False,
                    "phase_guess": "single"}

        # Two or three physical roots: smallest is liquid-like, largest is
        # vapour-like, any middle root is unstable and ignored.
        Z_liq, Z_vap = float(physical[0]), float(physical[-1])
        g_liq = self._ln_phi(Z_liq, A, B)
        g_vap = self._ln_phi(Z_vap, A, B)
        if g_liq < g_vap:
            Z, guess = Z_liq, "liquid-like"
        else:
            Z, guess = Z_vap, "vapour-like"
        return {"Z": Z, "n_real": len(real), "n_physical": len(physical),
                "two_phase": True, "phase_guess": guess}

    def get_z_factor(self, P_Pa, T_K):
        return self.root_info(P_Pa, T_K)["Z"]

    # ── Properties ─────────────────────────────────────────────────────────
    def get_compressibility(self, P_Pa, T_K):
        if P_Pa <= 0.0:
            return 1e-3        # numerical guard, matches IdealGasEOS

        h = max(P_Pa * self._DP_REL, 1.0)
        P_lo = max(P_Pa - h, 1.0)
        P_hi = P_Pa + h

        Z = self.get_z_factor(P_Pa, T_K)
        dZdP = ((self.get_z_factor(P_hi, T_K)
                 - self.get_z_factor(P_lo, T_K))
                / (P_hi - P_lo))

        c = 1.0 / P_Pa - dZdP / Z
        # A negative c is unphysical and can only come from a root
        # discontinuity straddling the phase boundary. Fall back to the
        # ideal value rather than handing the solver a bad number.
        return c if c > 0.0 else 1.0 / P_Pa

    def get_density(self, P_Pa, T_K):
        if self.M is None:
            return None
        Z = self.get_z_factor(P_Pa, T_K)
        return P_Pa * self.M / (Z * R_GAS * T_K)

    def describe(self, P_Pa, T_K):
        info = self.root_info(P_Pa, T_K)
        c = self.get_compressibility(P_Pa, T_K)
        c_ideal = 1.0 / P_Pa if P_Pa > 0 else float("nan")
        dev = (c / c_ideal - 1.0) * 100.0
        return (f"peng_robinson: Z = {info['Z']:.4f}, "
                f"c = {c:.3e} /Pa ({dev:+.2f}% vs ideal)")


# ── Factory ────────────────────────────────────────────────────────────────
EOS_REGISTRY = {
    "incompressible": IncompressibleEOS,
    "ideal_gas":      IdealGasEOS,
    "peng_robinson":  PengRobinsonEOS,
}

# Human-readable labels for UI dropdowns.
EOS_LABELS = {
    "incompressible": "Incompressible — constant c below.",
    "ideal_gas":      "Ideal gas — c = 1/P.",
    "peng_robinson":  "Peng-Robinson — real-gas c from the cubic.",
}


def has_critical_properties(phase):
    """True if a phase dict carries everything PengRobinsonEOS needs."""
    return all(phase.get(k) is not None
               for k in ("Tc_K", "Pc_Pa", "omega"))


def make_eos(eos_model, c_user_Pa=None, phase=None):
    """
    Instantiate an EOS from a model name string.

    Parameters
    ----------
    eos_model : str
        One of the keys in EOS_REGISTRY.
    c_user_Pa : float or None
        User-supplied compressibility, used only by IncompressibleEOS.
    phase : dict or None
        The full phase dict. Required for peng_robinson, which reads
        Tc_K, Pc_Pa, omega and (optionally) molar_mass_g_mol from it.
    """
    if eos_model not in EOS_REGISTRY:
        raise ValueError(
            f"Unknown eos_model '{eos_model}'. "
            f"Known: {list(EOS_REGISTRY.keys())}"
        )
    cls = EOS_REGISTRY[eos_model]

    if cls is IncompressibleEOS:
        return cls(c_Pa=c_user_Pa if c_user_Pa is not None else 4.5e-10)

    if cls is PengRobinsonEOS:
        if phase is None:
            raise ValueError(
                "peng_robinson needs the phase dict for its critical "
                "properties."
            )
        return cls(
            Tc_K=phase.get("Tc_K"),
            Pc_Pa=phase.get("Pc_Pa"),
            omega=phase.get("omega"),
            molar_mass_g_mol=phase.get("molar_mass_g_mol"),
        )

    return cls()