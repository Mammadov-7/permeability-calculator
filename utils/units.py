"""
Unit-conversion helpers for the permeability calculator.

Each table maps a unit string to the multiplier that converts a value
expressed in that unit into the base unit used by the solvers.

Base units:
    Length          -> cm
    Area            -> cm²
    Porosity        -> fraction (0..1)
    Viscosity       -> cP
    Density         -> kg/m³
    Compressibility -> 1/Pa
    Volume          -> ml
    Time            -> min
    Pressure        -> bar
    Pressure drop   -> mbar
    Permeability    -> mD
"""

# ── Length → cm ──────────────────────────────────────────────────────────────
LENGTH_TO_CM = {
    "cm": 1.0,
    "m":  100.0,
    "mm": 0.1,
    "in": 2.54,
    "ft": 30.48,
}

# ── Area → cm² ───────────────────────────────────────────────────────────────
AREA_TO_CM2 = {
    "cm²": 1.0,
    "m²":  10_000.0,
    "mm²": 0.01,
    "in²": 6.4516,
    "ft²": 929.0304,
}

# ── Porosity → fraction ──────────────────────────────────────────────────────
POROSITY_TO_FRACTION = {
    "fraction": 1.0,
    "%":        0.01,
}

# ── Viscosity → cP ───────────────────────────────────────────────────────────
VISCOSITY_TO_CP = {
    "cP":    1.0,
    "Pa·s":  1000.0,
    "mPa·s": 1.0,
}

# ── Density → kg/m³ ─────────────────────────────────────────────────────────
DENSITY_TO_KGM3 = {
    "kg/m³":  1.0,
    "g/cm³":  1000.0,
    "g/L":    1.0,
    "lb/ft³": 16.0185,
}

# ── Compressibility → 1/Pa ───────────────────────────────────────────────────
COMPRESSIBILITY_TO_INV_PA = {
    "1/Pa":  1.0,
    "1/psi": 1.0 / 6894.757,
    "1/bar": 1.0 / 1e5,
}

# ── Volume → ml ──────────────────────────────────────────────────────────────
VOLUME_TO_ML = {
    "ml":  1.0,
    "cm³": 1.0,
    "l":   1000.0,
    "m³":  1e6,
    "bbl": 158_987.295,
}

# ── Time → min ───────────────────────────────────────────────────────────────
TIME_TO_MIN = {
    "s":   1.0 / 60.0,
    "min": 1.0,
    "hr":  60.0,
    "day": 1440.0,
}

# ── Pressure → bar ───────────────────────────────────────────────────────────
PRESSURE_TO_BAR = {
    "bar": 1.0,
    "psi": 0.0689476,
    "Pa":  1e-5,
    "kPa": 1e-2,
    "atm": 1.01325,
}

# ── Pressure drop → mbar ─────────────────────────────────────────────────────
DP_TO_MBAR = {
    "mbar": 1.0,
    "bar":  1000.0,
    "psi":  68.9476,
    "Pa":   1e-2,
    "kPa":  10.0,
}

# ── Permeability → mD ────────────────────────────────────────────────────────
# 1 darcy = 9.869233e-13 m² => 1 m² = 1.01325e15 mD
PERMEABILITY_TO_MD = {
    "mD": 1.0,
    "D":  1000.0,
    "m²": 1.01325e15,
}


def convert(value: float, unit: str, table: dict) -> float:
    """Convert `value` from `unit` to the base unit defined by `table`."""
    if unit not in table:
        raise ValueError(
            f"Unknown unit '{unit}'. Allowed: {list(table.keys())}"
        )
    return value * table[unit]


def convert_injection_rate(value: float, volume_unit: str,
                           time_unit: str) -> float:
    """Convert an injection rate (volume / time) to ml/min."""
    volume_in_ml = convert(value, volume_unit, VOLUME_TO_ML)
    time_in_min  = convert(1.0, time_unit, TIME_TO_MIN)
    return volume_in_ml / time_in_min