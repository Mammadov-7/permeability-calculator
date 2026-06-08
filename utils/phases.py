"""
Phase library and phase-picker UI for the two-phase calculator.

- Built-in phases loaded from data/phases.json
- Custom phases stored in st.session_state (persist for the session)
- Optional JSON download/upload to persist across sessions
- phase_picker() — reusable widget for picking one phase
"""

import json
from pathlib import Path

import streamlit as st

from utils.units import DENSITY_TO_KGM3, VISCOSITY_TO_CP, convert


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


def add_custom_phase(name, type_, density_kg_m3, viscosity_cP):
    _session_customs()[name] = {
        "type": type_,
        "density_kg_m3": float(density_kg_m3),
        "viscosity_cP": float(viscosity_cP),
        "_custom": True,
    }


# ── Internal UI helper (mirrors app.py input_row) ───────────────────────────
def _input_row(label, default, units, key_prefix, fmt="%g", default_unit=None):
    c1, c2, c3 = st.columns([1.2, 1.5, 1])
    with c1:
        st.markdown(f'<div class="row-label">{label}</div>',
                    unsafe_allow_html=True)
    with c2:
        v = st.number_input(
            label, value=float(default), key=f"{key_prefix}_val",
            label_visibility="collapsed", format=fmt,
        )
    with c3:
        ulist = list(units.keys())
        idx = ulist.index(default_unit) if default_unit in ulist else 0
        u = st.selectbox(
            label, ulist, index=idx, key=f"{key_prefix}_unit",
            label_visibility="collapsed",
        )
    return v, u


# ── Main reusable widget ────────────────────────────────────────────────────
def phase_picker(key_prefix, default=None):
    """
    Render dropdown + editable density/viscosity for one phase.
    Returns dict: name, type, density_kg_m3, viscosity_cP.
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
    rho, rho_u = _input_row(
        "Density", base["density_kg_m3"], DENSITY_TO_KGM3,
        f"{key_prefix}_rho", default_unit="kg/m³",
    )
    mu, mu_u = _input_row(
        "Viscosity", base["viscosity_cP"], VISCOSITY_TO_CP,
        f"{key_prefix}_mu", default_unit="cP",
    )

    return {
        "name": choice,
        "type": base["type"],
        "density_kg_m3": convert(rho, rho_u, DENSITY_TO_KGM3),
        "viscosity_cP":  convert(mu,  mu_u,  VISCOSITY_TO_CP),
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
    rho, rho_u = _input_row(
        "Density", 1.0, DENSITY_TO_KGM3,
        f"{key_prefix}_custom_rho", default_unit="kg/m³",
    )
    mu, mu_u = _input_row(
        "Viscosity", 0.01, VISCOSITY_TO_CP,
        f"{key_prefix}_custom_mu", default_unit="cP",
    )
    rho_kgm3 = convert(rho, rho_u, DENSITY_TO_KGM3)
    mu_cP    = convert(mu,  mu_u,  VISCOSITY_TO_CP)

    save = st.checkbox(
        "Save this phase for this session",
        key=f"{key_prefix}_save",
    )
    if save and name.strip():
        add_custom_phase(name.strip(), type_, rho_kgm3, mu_cP)

    return {
        "name": name.strip() or "Custom",
        "type": type_,
        "density_kg_m3": rho_kgm3,
        "viscosity_cP": mu_cP,
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
                    )
                st.success(f"Loaded {len(loaded)} custom phase(s).")
            except Exception as e:
                st.error(f"Couldn't read file: {e}")