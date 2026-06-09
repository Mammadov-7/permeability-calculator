"""
CoreFlood Lab — Two-Phase (Relative Permeability) Tool
Step 6: kr/Pc preview + forward-simulation visualizations.
"""

import math

import streamlit as st
import streamlit.components.v1 as components

from utils.phases import phase_picker, io_buttons
from utils.units import (
    LENGTH_TO_CM, POROSITY_TO_FRACTION,
    VOLUME_TO_ML, TIME_TO_MIN, PRESSURE_TO_BAR, DP_TO_MBAR,
    convert, convert_injection_rate,
)
from utils.twophase import kr_curves, pc_curve
from utils.twophase_solver import run_forward
from utils.plotting import (
    build_kr_chart, build_pc_chart,
    build_dp_time_chart, build_profile_animation,
    render_chart_html,
)

# Stop-gap: permeability units (promote into utils/units.py in a later step)
PERMEABILITY_TO_MD = {
    "mD": 1.0,
    "D":  1000.0,
    "m²": 1.01325e15,
}

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CoreFlood Lab — Two-Phase",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling (matches PermCalc) ──────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebarNav"]  {display: none;}
[data-testid="stSidebar"]     {display: none;}
#MainMenu                     {visibility: hidden;}
header                        {visibility: hidden;}
footer                        {visibility: hidden;}
.stDeployButton               {display: none;}
[data-testid="stToolbar"]     {visibility: hidden;}
[data-testid="stDecoration"]  {display: none;}

html, body, [class*="css"] { font-family: 'Courier New', monospace; }

.section-label {
    color: #6B7785; font-size: 12px;
    letter-spacing: 0.12em; margin: 1rem 0 0.5rem 0;
}
.group-label {
    color: #FB923C; font-size: 13px; margin: 0.75rem 0 0.25rem 0;
}
.row-label {
    color: #9CA3AF; font-size: 13px; padding-top: 0.5rem;
}
.app-header {
    display: flex; align-items: center; gap: 10px;
    padding-bottom: 14px; border-bottom: 1px solid #1F2A33;
    margin-bottom: 1rem;
}
.dot-teal  { color: #2DD4BF; }
.app-title { color: #F0F4F8; font-size: 16px; }

.debug-box {
    background: #0F1A1F; border-left: 4px solid #2DD4BF;
    padding: 18px 22px; margin: 0.6rem 0;
    color: #C9D1D9 !important;
    font-size: 13px !important; line-height: 1.7 !important;
}
.debug-box code { color: #2DD4BF !important; }
.debug-box b    { color: #F0F4F8 !important; }
.warn-box {
    background: #2A1F0F; border-left: 3px solid #FB923C;
    padding: 10px 14px; margin: 0.5rem 0;
    color: #FB923C; font-size: 12px;
}
.metric-card {
    background: #0F1A1F; border-left: 4px solid #2DD4BF;
    padding: 14px 18px; margin: 0.4rem 0;
    color: #C9D1D9; font-size: 13px;
}
.metric-card b { color: #F0F4F8; }
.metric-card code { color: #2DD4BF; }
</style>
""", unsafe_allow_html=True)


# ── Input helpers (mirrors phases.py / app.py pattern) ─────────────────────
def _input_row(label, default, units, key_prefix, fmt="%g",
               default_unit=None, help=None):
    c1, c2, c3 = st.columns([1.2, 1.5, 1])
    with c1:
        st.markdown(f'<div class="row-label">{label}</div>',
                    unsafe_allow_html=True)
    with c2:
        v = st.number_input(
            label, value=float(default), key=f"{key_prefix}_val",
            label_visibility="collapsed", format=fmt, help=help,
        )
    with c3:
        ulist = list(units.keys())
        idx = ulist.index(default_unit) if default_unit in ulist else 0
        u = st.selectbox(
            label, ulist, index=idx, key=f"{key_prefix}_unit",
            label_visibility="collapsed",
        )
    return v, u


def _dim_row(label, default, key_prefix, fmt="%g",
             min_value=None, max_value=None, step=None, help=None):
    """Dimensionless input row (no unit selector)."""
    c1, c2 = st.columns([1.2, 2.5])
    with c1:
        st.markdown(f'<div class="row-label">{label}</div>',
                    unsafe_allow_html=True)
    with c2:
        kwargs = dict(
            value=float(default), key=f"{key_prefix}_val",
            label_visibility="collapsed", format=fmt, help=help,
        )
        if min_value is not None: kwargs["min_value"] = float(min_value)
        if max_value is not None: kwargs["max_value"] = float(max_value)
        if step is not None:      kwargs["step"] = float(step)
        return st.number_input(label, **kwargs)


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <span class="dot-teal">●</span>
  <span class="app-title">CoreFlood Lab — Two-Phase (Relative Permeability)</span>
</div>
""", unsafe_allow_html=True)

# ── Phase library import/export ─────────────────────────────────────────────
with st.expander("Phase library (download / upload custom phases)",
                 expanded=False):
    io_buttons()

# ── Phase selection ─────────────────────────────────────────────────────────
st.markdown('<div class="section-label">▌ PHASE SELECTION</div>',
            unsafe_allow_html=True)
col_inj, col_res = st.columns(2)
with col_inj:
    st.markdown('<div class="group-label">Injected phase</div>',
                unsafe_allow_html=True)
    injected = phase_picker("inj", default="H2")
with col_res:
    st.markdown('<div class="group-label">Displaced phase</div>',
                unsafe_allow_html=True)
    displaced = phase_picker("disp", default="Brine (3% NaCl)")

if injected["type"] == displaced["type"]:
    st.markdown(
        f'<div class="warn-box">⚠ Unusual fluid pair — both phases are '
        f'<b>{injected["type"]}</b>. Typical relative-permeability tests '
        f'use one gas + one liquid.</div>',
        unsafe_allow_html=True,
    )

# ── Saturation endpoints ────────────────────────────────────────────────────
st.markdown('<div class="section-label">▌ SATURATION ENDPOINTS</div>',
            unsafe_allow_html=True)
col_si, col_sd = st.columns(2)
with col_si:
    st.markdown('<div class="group-label">Injected phase</div>',
                unsafe_allow_html=True)
    s_inj_r_val, s_inj_r_u = _input_row(
        "S_r (residual)", 0.0, POROSITY_TO_FRACTION,
        "s_inj_r", default_unit="fraction",
        help="Residual saturation of injected phase; kr_inj = 0 below this.",
    )
with col_sd:
    st.markdown('<div class="group-label">Displaced phase</div>',
                unsafe_allow_html=True)
    s_disp_r_val, s_disp_r_u = _input_row(
        "S_r (residual)", 0.20, POROSITY_TO_FRACTION,
        "s_disp_r", default_unit="fraction",
        help="Residual saturation of displaced phase that cannot be swept.",
    )

s_inj_r  = convert(s_inj_r_val,  s_inj_r_u,  POROSITY_TO_FRACTION)
s_disp_r = convert(s_disp_r_val, s_disp_r_u, POROSITY_TO_FRACTION)
mobile_range = 1.0 - s_inj_r - s_disp_r

if mobile_range <= 0:
    st.markdown(
        f'<div class="warn-box">⚠ S_r,inj + S_r,disp = '
        f'<b>{s_inj_r + s_disp_r:.3f}</b> ≥ 1.0 — no mobile saturation '
        f'range remains. Reduce one of them.</div>',
        unsafe_allow_html=True,
    )

# ── Corey kr parameters ─────────────────────────────────────────────────────
st.markdown('<div class="section-label">▌ COREY kr PARAMETERS</div>',
            unsafe_allow_html=True)
col_ki, col_kd = st.columns(2)
with col_ki:
    st.markdown('<div class="group-label">Injected phase</div>',
                unsafe_allow_html=True)
    kr_inj_max = _dim_row(
        "End-point kr_max", 0.6, "kr_inj_max",
        min_value=0.0, max_value=1.0, step=0.01, fmt="%.3f",
        help="kr of injected phase at S_inj = 1 − S_r,disp.",
    )
    n_inj = _dim_row(
        "Corey exponent n", 2.0, "n_inj",
        min_value=0.5, max_value=10.0, step=0.1, fmt="%.2f",
        help="Curvature of kr curve. n=1 linear, n=2 quadratic.",
    )
with col_kd:
    st.markdown('<div class="group-label">Displaced phase</div>',
                unsafe_allow_html=True)
    kr_disp_max = _dim_row(
        "End-point kr_max", 1.0, "kr_disp_max",
        min_value=0.0, max_value=1.0, step=0.01, fmt="%.3f",
        help="kr of displaced phase at S_disp = 1 − S_r,inj.",
    )
    n_disp = _dim_row(
        "Corey exponent n", 3.0, "n_disp",
        min_value=0.5, max_value=10.0, step=0.1, fmt="%.2f",
        help="Curvature of kr curve. n=1 linear, n=2 quadratic.",
    )

# ── Capillary pressure ──────────────────────────────────────────────────────
st.markdown('<div class="section-label">▌ CAPILLARY PRESSURE</div>',
            unsafe_allow_html=True)
pc_enabled = st.checkbox(
    "Include capillary pressure (Brooks–Corey)",
    value=False, key="pc_enabled",
)
pc_entry_mbar = None
pc_lambda     = None
if pc_enabled:
    col_pe, col_pl = st.columns(2)
    with col_pe:
        pc_entry_val, pc_entry_u = _input_row(
            "Entry pressure P_e", 100.0, DP_TO_MBAR,
            "pc_entry", default_unit="mbar",
        )
        pc_entry_mbar = convert(pc_entry_val, pc_entry_u, DP_TO_MBAR)
    with col_pl:
        pc_lambda = _dim_row(
            "Pore-size index λ", 2.0, "pc_lambda",
            min_value=0.1, max_value=10.0, step=0.1, fmt="%.2f",
        )

# ── Rock & core geometry ────────────────────────────────────────────────────
st.markdown('<div class="section-label">▌ ROCK & CORE GEOMETRY</div>',
            unsafe_allow_html=True)
L_val, L_u     = _input_row("Length L",         10.0,  LENGTH_TO_CM,
                            "core_L", default_unit="cm")
D_val, D_u     = _input_row("Diameter D",       3.8,   LENGTH_TO_CM,
                            "core_D", default_unit="cm")
phi_val, phi_u = _input_row("Porosity φ",       20.0,  POROSITY_TO_FRACTION,
                            "core_phi", default_unit="%")
k_val, k_u     = _input_row("Absolute perm. k", 100.0, PERMEABILITY_TO_MD,
                            "core_k", default_unit="mD")

L_cm  = convert(L_val,   L_u,   LENGTH_TO_CM)
D_cm  = convert(D_val,   D_u,   LENGTH_TO_CM)
phi   = convert(phi_val, phi_u, POROSITY_TO_FRACTION)
k_mD  = convert(k_val,   k_u,   PERMEABILITY_TO_MD)
A_cm2 = math.pi * (D_cm ** 2) / 4.0

# ── Operating conditions ────────────────────────────────────────────────────
st.markdown('<div class="section-label">▌ OPERATING CONDITIONS</div>',
            unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([1.2, 1.5, 0.5, 0.5])
with c1:
    st.markdown('<div class="row-label">Injection rate q</div>',
                unsafe_allow_html=True)
with c2:
    q_val = st.number_input(
        "Injection rate", value=1.0, key="q_val",
        label_visibility="collapsed", format="%g",
    )
with c3:
    q_vol_u = st.selectbox(
        "vol", list(VOLUME_TO_ML.keys()), index=0, key="q_vol_u",
        label_visibility="collapsed",
    )
with c4:
    q_time_u = st.selectbox(
        "time", list(TIME_TO_MIN.keys()), index=1, key="q_time_u",
        label_visibility="collapsed",
    )
q_ml_min = convert_injection_rate(q_val, q_vol_u, q_time_u)

p_back_val, p_back_u   = _input_row("Back pressure P_out", 1.0,
                                    PRESSURE_TO_BAR, "op_pback",
                                    default_unit="bar")
t_total_val, t_total_u = _input_row("Total time", 60.0, TIME_TO_MIN,
                                    "op_ttotal", default_unit="min")
p_back_bar  = convert(p_back_val,  p_back_u,  PRESSURE_TO_BAR)
t_total_min = convert(t_total_val, t_total_u, TIME_TO_MIN)

# ── Numerical ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">▌ NUMERICAL</div>',
            unsafe_allow_html=True)
n_cells = st.number_input(
    "Grid cells N",
    min_value=20, max_value=500, value=100, step=10,
    key="n_cells",
    help="Number of 1D finite-difference cells. More cells = sharper "
         "saturation front but slower runs. 100 is a good default; use "
         "200+ for publication-grade plots.",
)

# ── Stash everything in session_state for the simulator ────────────────────
st.session_state["tp_inputs"] = {
    "injected":  injected,
    "displaced": displaced,
    "S_inj_r":   s_inj_r,
    "S_disp_r":  s_disp_r,
    "kr": {
        "inj_max":  kr_inj_max,
        "n_inj":    n_inj,
        "disp_max": kr_disp_max,
        "n_disp":   n_disp,
    },
    "pc": {
        "enabled":      pc_enabled,
        "P_entry_mbar": pc_entry_mbar,
        "lambda":       pc_lambda,
    },
    "core": {
        "L_cm":   L_cm,
        "D_cm":   D_cm,
        "A_cm2":  A_cm2,
        "phi":    phi,
        "k_mD":   k_mD,
    },
    "operating": {
        "q_ml_min":    q_ml_min,
        "P_back_bar":  p_back_bar,
        "t_total_min": t_total_min,
    },
    "numerical": {
        "N_cells": int(n_cells),
    },
}

# ── Collapsed debug readout ─────────────────────────────────────────────────
with st.expander("Collected inputs (debug)", expanded=False):
    pc_line = (
        f'<b>Pc:</b> Brooks–Corey, '
        f'P_e = <code>{pc_entry_mbar:.2f} mbar</code>, '
        f'λ = <code>{pc_lambda:.2f}</code>'
        if pc_enabled else
        '<b>Pc:</b> <code>off</code>'
    )
    st.markdown(
        f'<div class="debug-box">'
        f'<b>Injected:</b> {injected["name"]} ({injected["type"]}) — '
        f'ρ = <code>{injected["density_kg_m3"]:.3f} kg/m³</code>, '
        f'μ = <code>{injected["viscosity_cP"]:.4f} cP</code><br>'
        f'<b>Displaced:</b> {displaced["name"]} ({displaced["type"]}) — '
        f'ρ = <code>{displaced["density_kg_m3"]:.3f} kg/m³</code>, '
        f'μ = <code>{displaced["viscosity_cP"]:.4f} cP</code><br>'
        f'<b>Saturations:</b> S_r,inj = <code>{s_inj_r:.3f}</code>, '
        f'S_r,disp = <code>{s_disp_r:.3f}</code>, '
        f'mobile = <code>{mobile_range:.3f}</code><br>'
        f'<b>kr:</b> kr_max,inj=<code>{kr_inj_max:.3f}</code>, '
        f'n_inj=<code>{n_inj:.2f}</code>; '
        f'kr_max,disp=<code>{kr_disp_max:.3f}</code>, '
        f'n_disp=<code>{n_disp:.2f}</code><br>'
        f'{pc_line}<br>'
        f'<b>Core:</b> L=<code>{L_cm:.2f} cm</code>, '
        f'D=<code>{D_cm:.2f} cm</code> '
        f'(A=<code>{A_cm2:.3f} cm²</code>), '
        f'φ=<code>{phi:.3f}</code>, k=<code>{k_mD:.2f} mD</code><br>'
        f'<b>Operating:</b> q=<code>{q_ml_min:.4f} ml/min</code>, '
        f'P_back=<code>{p_back_bar:.3f} bar</code>, '
        f't_total=<code>{t_total_min:.2f} min</code><br>'
        f'<b>Numerical:</b> N=<code>{int(n_cells)}</code> cells'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── kr / Pc preview (live) ──────────────────────────────────────────────────
st.markdown('<div class="section-label">▌ kr / Pc PREVIEW</div>',
            unsafe_allow_html=True)

if mobile_range <= 0:
    st.markdown(
        '<div class="warn-box">Cannot draw curves: residual saturations '
        'leave no mobile range.</div>',
        unsafe_allow_html=True,
    )
else:
    kr_data = kr_curves(st.session_state["tp_inputs"])
    fig_kr  = build_kr_chart(
        kr_data, inj_name=injected["name"], disp_name=displaced["name"],
    )
    components.html(render_chart_html(fig_kr), height=410, scrolling=False)

    if pc_enabled:
        pc_data = pc_curve(st.session_state["tp_inputs"])
        fig_pc  = build_pc_chart(pc_data, s_inj_r, s_disp_r)
        components.html(render_chart_html(fig_pc), height=410, scrolling=False)

# ── Forward simulation ──────────────────────────────────────────────────────
st.markdown('<div class="section-label">▌ FORWARD SIMULATION</div>',
            unsafe_allow_html=True)

if mobile_range <= 0:
    st.markdown(
        '<div class="warn-box">Cannot simulate: residual saturations '
        'leave no mobile range.</div>',
        unsafe_allow_html=True,
    )
else:
    run_clicked = st.button(
        "▶  Run Forward Simulation", type="primary", key="run_forward",
    )

    if run_clicked:
        with st.spinner("Running 1D IMPES forward simulation…"):
            try:
                st.session_state["tp_results"] = run_forward(
                    st.session_state["tp_inputs"]
                )
                st.session_state["tp_results_error"] = None
            except Exception as e:
                st.session_state["tp_results"] = None
                st.session_state["tp_results_error"] = str(e)

    if st.session_state.get("tp_results_error"):
        st.error(f"Simulation failed: {st.session_state['tp_results_error']}")

    results = st.session_state.get("tp_results")
    if results is not None:
        bt_pvi  = results["BT_PVI"]
        bt_time = results["BT_time_min"]
        dp_init = float(results["dP_mbar"][0])
        dp_end  = float(results["dP_mbar"][-1])
        bt_line = (
            f'BT at <code>{bt_time:.2f} min</code> '
            f'(PVI=<code>{bt_pvi:.3f}</code>)'
            if bt_pvi is not None else
            '<code>no breakthrough within t_total</code>'
        )
        st.markdown(
            f'<div class="metric-card">'
            f'<b>Run summary:</b> '
            f'PV=<code>{results["PV_ml"]:.2f} ml</code>, '
            f'N=<code>{results["N"]}</code>, '
            f'inner steps=<code>{results["n_steps"]}</code>, '
            f'snapshots=<code>{len(results["t_min"])}</code><br>'
            f'<b>Breakthrough:</b> {bt_line}<br>'
            f'<b>ΔP:</b> initial=<code>{dp_init:.2f} mbar</code>, '
            f'final=<code>{dp_end:.2f} mbar</code>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ΔP vs time (animated, autoplay)
        fig_dp = build_dp_time_chart(results)
        components.html(
            render_chart_html(fig_dp, autoplay=True),
            height=450, scrolling=False,
        )

        # Saturation profile (animated, with slider)
        fig_prof = build_profile_animation(results)
        components.html(
            render_chart_html(fig_prof, autoplay=False),
            height=480, scrolling=False,
        )

        st.caption(
            "Charts above reflect the last ‘Run’ click. "
            "Change inputs and click Run again to refresh."
        )
    else:
        st.caption(
            "Click ‘Run Forward Simulation’ to generate ΔP history "
            "and saturation profile."
        )