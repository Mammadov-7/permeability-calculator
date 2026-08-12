"""
Phase library and phase-picker UI for CoreFlood Lab.

- Built-in phases loaded from data/phases.json
- Custom phases stored in st.session_state (persist for the session)
- Optional JSON download/upload to persist across sessions
- phase_picker() — reusable widget for picking one phase

Each returned phase dict includes: name, type, density_kg_m3, viscosity_cP,
eos_model, compressibility_1_per_Pa (may be None for gases).
"""

import json
from pathlib import Path

import streamlit as st

from utils.units import DENSITY_TO_KGM3, VISCOSITY_TO_CP, convert
from utils.ui import input_row


# ── Data access ─────────────────────────────────────────────────────────────
_BUILTIN_PATH = Path(__file__).resolve().parent.parent / "data" / "phases.json"


def load_builtin_phases():
    with open(_BUILTIN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _session_customs():
    return st.session_state.setdefault("custom_phases", {})


def get_all_phases():
    phases = load_builtin_phases()
    phases.update(_session_customs())
    return phases


def add_custom_phase(name, type_, density_kg_m3, viscosity_cP,
                     eos_model="incompressible",
                     compressibility_1_per_Pa=None):
    _session_customs()[name] = {
        "type": type_,
        "density_kg_m3": float(density_kg_m3),
        "viscosity_cP": float(viscosity_cP),
        "eos_model": eos_model,
        "compressibility_1_per_Pa": (
            float(compressibility_1_per_Pa)
            if compressibility_1_per_Pa is not None else None
        ),
        "_custom": True,
    }


# ── Main reusable widget ────────────────────────────────────────────────────
def phase_picker(key_prefix, default=None):
    """
    Render dropdown + editable density/viscosity for one phase.
    Returns dict: name, type, density_kg_m3, viscosity_cP, eos_model,
    compressibility_1_per_Pa (may be None for gases with no user override).
    """
    all_phases = get_all_phases()
    options = list(all_phases.keys()) + ["+ Other..."]
    idx = options.index(default) if default in options else 0

    choice = st.selectbox(
        "phase", options, index=idx, key=f"{key_prefix}_choice",
        label_visibility="collapsed",
    )

    if choice == "+ Other...":
        return _custom_phase_form(key_prefix)

    base = all_phases[choice]
    rho, rho_u = input_row(
        "Density", base["density_kg_m3"], DENSITY_TO_KGM3,
        f"{key_prefix}_rho", default_unit="kg/m³",
    )
    mu, mu_u = input_row(
        "Viscosity", base["viscosity_cP"], VISCOSITY_TO_CP,
        f"{key_prefix}_mu", default_unit="cP",
    )

    return {
        "name": choice,
        "type": base["type"],
        "density_kg_m3": convert(rho, rho_u, DENSITY_TO_KGM3),
        "viscosity_cP":  convert(mu,  mu_u,  VISCOSITY_TO_CP),
        "eos_model":     base.get("eos_model", "incompressible"),
        "compressibility_1_per_Pa": base.get(
            "compressibility_1_per_Pa", None
        ),
    }


def _custom_phase_form(key_prefix):
    name = st.text_input(
        "Phase name", value="", placeholder="e.g. Argon",
        key=f"{key_prefix}_custom_name",
    )
    type_ = st.radio(
        "Type", ["gas", "liquid"], horizontal=True,
        key=f"{key_prefix}_custom_type",
    )
    rho, rho_u = input_row(
        "Density", 1.0, DENSITY_TO_KGM3,
        f"{key_prefix}_custom_rho", default_unit="kg/m³",
    )
    mu, mu_u = input_row(
        "Viscosity", 0.01, VISCOSITY_TO_CP,
        f"{key_prefix}_custom_mu", default_unit="cP",
    )
    rho_kgm3 = convert(rho, rho_u, DENSITY_TO_KGM3)
    mu_cP    = convert(mu,  mu_u,  VISCOSITY_TO_CP)

    # Default EOS choice based on type; can override for exotic cases.
    eos_default = "ideal_gas" if type_ == "gas" else "incompressible"
    eos_model = st.selectbox(
        "EOS model", ["incompressible", "ideal_gas"],
        index=["incompressible", "ideal_gas"].index(eos_default),
        key=f"{key_prefix}_custom_eos",
        help="Incompressible = constant compressibility you supply. "
             "Ideal gas = c computed from local pressure (c = 1/P).",
    )

    # Compressibility field only needed for incompressible EOS.
    c_val = None
    if eos_model == "incompressible":
        c_val = st.number_input(
            "Compressibility [1/Pa]", value=4.5e-10, format="%.2e",
            key=f"{key_prefix}_custom_c",
        )

    save = st.checkbox(
        "Save this phase for this session",
        key=f"{key_prefix}_save",
    )
    if save and name.strip():
        add_custom_phase(name.strip(), type_, rho_kgm3, mu_cP,
                         eos_model=eos_model,
                         compressibility_1_per_Pa=c_val)

    return {
        "name": name.strip() or "Custom",
        "type": type_,
        "density_kg_m3": rho_kgm3,
        "viscosity_cP": mu_cP,
        "eos_model": eos_model,
        "compressibility_1_per_Pa": c_val,
    }


# ── Download / upload custom library ────────────────────────────────────────
def io_buttons():
    customs = _session_customs()
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇  Download my phases (.json)",
            data=json.dumps(customs, indent=2) if customs else "{}",
            file_name="my_phases.json",
            mime="application/json",
            key="phases_download",
            disabled=not customs,
        )
    with c2:
        up = st.file_uploader(
            "Upload phases (.json)", type=["json"],
            key="phases_upload", label_visibility="collapsed",
        )
        if up is not None:
            try:
                loaded = json.load(up)
                for nm, d in loaded.items():
                    add_custom_phase(
                        nm, d.get("type", "liquid"),
                        d["density_kg_m3"], d["viscosity_cP"],
                        eos_model=d.get("eos_model", "incompressible"),
                        compressibility_1_per_Pa=d.get(
                            "compressibility_1_per_Pa"),
                    )
                st.success(f"Loaded {len(loaded)} custom phase(s).")
            except Exception as e:
                st.error(f"Couldn't read file: {e}")