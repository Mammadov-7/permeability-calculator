"""
CoreFlood Lab — Two-Phase (Relative Permeability) Tool
Step 7: Forward + Inverse modes via tabs.
"""

import io
import math

import numpy as np
import pandas as pd
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
from utils.twophase_inverse import fit_corey, COREY_PARAM_NAMES
from utils.plotting import (
    build_kr_chart, build_pc_chart,
    build_dp_time_chart, build_profile_animation,
    build_history_match_chart,
    render_chart_html,
)

# Stop-gap: permeability units (promote into utils/units.py later)
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

# ── Styling ─────────────────────────────────────────────────────────────────
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
.metric-card b    { color: #F0F4F8; }
.metric-card code { color: #2DD4BF; }
</style>
""", unsafe_allow_html=True)


# ── Input helpers ───────────────────────────────────────────────────────────
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

# ── Phase library expander ──────────────────────────────────────────────────
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
    )
with col_sd:
    st.markdown('<div class="group-label">Displaced phase</div>',
                unsafe_allow_html=True)
    s_disp_r_val, s_disp_r_u = _input_row(
        "S_r (residual)", 0.20, POROSITY_TO_FRACTION,
        "s_disp_r", default_unit="fraction",
    )
s_inj_r  = convert(s_inj_r_val,  s_inj_r_u,  POROSITY_TO_FRACTION)
s_disp_r = convert(s_disp_r_val, s_disp_r_u, POROSITY_TO_FRACTION)
mobile_range = 1.0 - s_inj_r - s_disp_r
if mobile_range <= 0:
    st.markdown(
        f'<div class="warn-box">⚠ S_r,inj + S_r,disp = '
        f'<b>{s_inj_r + s_disp_r:.3f}</b> ≥ 1.0 — no mobile range.</div>',
        unsafe_allow_html=True,
    )

# ── Corey kr parameters ─────────────────────────────────────────────────────
st.markdown('<div class="section-label">▌ COREY kr PARAMETERS '
            '(initial guesses for Inverse)</div>',
            unsafe_allow_html=True)
col_ki, col_kd = st.columns(2)
with col_ki:
    st.markdown('<div class="group-label">Injected phase</div>',
                unsafe_allow_html=True)
    kr_inj_max = _dim_row("End-point kr_max", 0.6, "kr_inj_max",
                          min_value=0.0, max_value=1.0, step=0.01, fmt="%.3f")
    n_inj      = _dim_row("Corey exponent n", 2.0, "n_inj",
                          min_value=0.5, max_value=10.0, step=0.1, fmt="%.2f")
with col_kd:
    st.markdown('<div class="group-label">Displaced phase</div>',
                unsafe_allow_html=True)
    kr_disp_max = _dim_row("End-point kr_max", 1.0, "kr_disp_max",
                           min_value=0.0, max_value=1.0, step=0.01, fmt="%.3f")
    n_disp      = _dim_row("Corey exponent n", 3.0, "n_disp",
                           min_value=0.5, max_value=10.0, step=0.1, fmt="%.2f")

# ── Capillary pressure ──────────────────────────────────────────────────────
st.markdown('<div class="section-label">▌ CAPILLARY PRESSURE</div>',
            unsafe_allow_html=True)
pc_enabled = st.checkbox("Include capillary pressure (Brooks–Corey)",
                         value=False, key="pc_enabled")
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
        pc_lambda = _dim_row("Pore-size index λ", 2.0, "pc_lambda",
                             min_value=0.1, max_value=10.0, step=0.1,
                             fmt="%.2f")

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
    q_val = st.number_input("Injection rate", value=1.0, key="q_val",
                            label_visibility="collapsed", format="%g")
with c3:
    q_vol_u = st.selectbox("vol", list(VOLUME_TO_ML.keys()), index=0,
                           key="q_vol_u", label_visibility="collapsed")
with c4:
    q_time_u = st.selectbox("time", list(TIME_TO_MIN.keys()), index=1,
                            key="q_time_u", label_visibility="collapsed")
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
n_cells = st.number_input("Grid cells N", min_value=20, max_value=500,
                          value=100, step=10, key="n_cells")

# ── Stash inputs in session_state ───────────────────────────────────────────
st.session_state["tp_inputs"] = {
    "injected":  injected,
    "displaced": displaced,
    "S_inj_r":   s_inj_r,
    "S_disp_r":  s_disp_r,
    "kr": {"inj_max": kr_inj_max, "n_inj": n_inj,
           "disp_max": kr_disp_max, "n_disp": n_disp},
    "pc": {"enabled": pc_enabled, "P_entry_mbar": pc_entry_mbar,
           "lambda": pc_lambda},
    "core": {"L_cm": L_cm, "D_cm": D_cm, "A_cm2": A_cm2,
             "phi": phi, "k_mD": k_mD},
    "operating": {"q_ml_min": q_ml_min, "P_back_bar": p_back_bar,
                  "t_total_min": t_total_min},
    "numerical": {"N_cells": int(n_cells)},
}

# Collapsed debug
with st.expander("Collected inputs (debug)", expanded=False):
    pc_line = (
        f'<b>Pc:</b> Brooks–Corey, '
        f'P_e=<code>{pc_entry_mbar:.2f} mbar</code>, '
        f'λ=<code>{pc_lambda:.2f}</code>'
        if pc_enabled else '<b>Pc:</b> <code>off</code>'
    )
    st.markdown(
        f'<div class="debug-box">'
        f'<b>Injected:</b> {injected["name"]} — '
        f'μ=<code>{injected["viscosity_cP"]:.4f} cP</code><br>'
        f'<b>Displaced:</b> {displaced["name"]} — '
        f'μ=<code>{displaced["viscosity_cP"]:.4f} cP</code><br>'
        f'<b>S:</b> S_r,inj=<code>{s_inj_r:.3f}</code>, '
        f'S_r,disp=<code>{s_disp_r:.3f}</code>, '
        f'mobile=<code>{mobile_range:.3f}</code><br>'
        f'<b>kr (initial):</b> kr_max,inj=<code>{kr_inj_max:.3f}</code>, '
        f'n_inj=<code>{n_inj:.2f}</code>, '
        f'kr_max,disp=<code>{kr_disp_max:.3f}</code>, '
        f'n_disp=<code>{n_disp:.2f}</code><br>'
        f'{pc_line}<br>'
        f'<b>Core:</b> L=<code>{L_cm:.2f} cm</code>, '
        f'D=<code>{D_cm:.2f} cm</code>, '
        f'φ=<code>{phi:.3f}</code>, '
        f'k=<code>{k_mD:.2f} mD</code><br>'
        f'<b>Op:</b> q=<code>{q_ml_min:.4f} ml/min</code>, '
        f't_total=<code>{t_total_min:.2f} min</code>, '
        f'N=<code>{int(n_cells)}</code>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════
# TABS — Forward simulation  /  Inverse fitting
# ════════════════════════════════════════════════════════════════════════════
tab_fwd, tab_inv = st.tabs(["⚡  Forward simulation",
                            "🔬  Inverse fitting"])

# ── FORWARD TAB ─────────────────────────────────────────────────────────────
with tab_fwd:
    st.markdown('<div class="section-label">▌ kr / Pc PREVIEW</div>',
                unsafe_allow_html=True)

    if mobile_range <= 0:
        st.markdown(
            '<div class="warn-box">No mobile range — adjust residual '
            'saturations to view curves.</div>',
            unsafe_allow_html=True,
        )
    else:
        kr_data = kr_curves(st.session_state["tp_inputs"])
        fig_kr  = build_kr_chart(
            kr_data, inj_name=injected["name"], disp_name=displaced["name"],
        )
        components.html(render_chart_html(fig_kr),
                        height=410, scrolling=False)
        if pc_enabled:
            pc_data = pc_curve(st.session_state["tp_inputs"])
            fig_pc  = build_pc_chart(pc_data, s_inj_r, s_disp_r)
            components.html(render_chart_html(fig_pc),
                            height=410, scrolling=False)

    st.markdown('<div class="section-label">▌ RUN FORWARD</div>',
                unsafe_allow_html=True)

    if mobile_range <= 0:
        st.markdown('<div class="warn-box">Cannot simulate — no mobile '
                    'range.</div>', unsafe_allow_html=True)
    else:
        run_clicked = st.button("▶  Run Forward Simulation",
                                type="primary", key="run_forward")
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
            st.error(f"Simulation failed: "
                     f"{st.session_state['tp_results_error']}")

        results = st.session_state.get("tp_results")
        if results is not None:
            bt_pvi  = results["BT_PVI"]
            bt_time = results["BT_time_min"]
            dp_init = float(results["dP_mbar"][0])
            dp_end  = float(results["dP_mbar"][-1])
            bt_line = (
                f'BT at <code>{bt_time:.2f} min</code> '
                f'(PVI=<code>{bt_pvi:.3f}</code>)'
                if bt_pvi is not None
                else '<code>no breakthrough in t_total</code>'
            )
            st.markdown(
                f'<div class="metric-card">'
                f'<b>Run:</b> PV=<code>{results["PV_ml"]:.2f} ml</code>, '
                f'N=<code>{results["N"]}</code>, '
                f'steps=<code>{results["n_steps"]}</code><br>'
                f'<b>BT:</b> {bt_line}<br>'
                f'<b>ΔP:</b> '
                f'initial=<code>{dp_init:.2f} mbar</code>, '
                f'final=<code>{dp_end:.2f} mbar</code>'
                f'</div>',
                unsafe_allow_html=True,
            )
            components.html(
                render_chart_html(build_dp_time_chart(results),
                                  autoplay=True),
                height=450, scrolling=False,
            )
            components.html(
                render_chart_html(build_profile_animation(results),
                                  autoplay=False),
                height=480, scrolling=False,
            )
        else:
            st.caption("Click ‘Run Forward Simulation’ to generate ΔP "
                       "history and saturation profile.")

# ── INVERSE TAB ─────────────────────────────────────────────────────────────
with tab_inv:
    st.markdown('<div class="section-label">▌ UPLOAD ΔP(t) DATA</div>',
                unsafe_allow_html=True)

    st.caption("Expected CSV: at least two columns — time and pressure "
               "drop. Column names and units are configurable below.")

    uploaded = st.file_uploader("Upload CSV", type=["csv", "txt"],
                                key="inv_csv",
                                label_visibility="collapsed")

    measured_t  = None
    measured_dp = None
    df          = None

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")

    if df is not None and len(df.columns) >= 2:
        cols = list(df.columns)
        c1, c2, c3, c4 = st.columns([1.4, 1.0, 1.4, 1.0])
        with c1:
            t_col = st.selectbox("Time column", cols, index=0,
                                 key="inv_tcol")
        with c2:
            t_unit = st.selectbox("Time unit", list(TIME_TO_MIN.keys()),
                                  index=list(TIME_TO_MIN.keys()).index("min"),
                                  key="inv_tunit")
        with c3:
            dp_col = st.selectbox("ΔP column", cols,
                                  index=1 if len(cols) > 1 else 0,
                                  key="inv_dpcol")
        with c4:
            dp_unit = st.selectbox("ΔP unit",
                                   list(DP_TO_MBAR.keys()),
                                   index=list(DP_TO_MBAR.keys()).index("mbar"),
                                   key="inv_dpunit")
        try:
            raw_t  = pd.to_numeric(df[t_col],  errors="coerce").to_numpy()
            raw_dp = pd.to_numeric(df[dp_col], errors="coerce").to_numpy()
            mask = np.isfinite(raw_t) & np.isfinite(raw_dp)
            measured_t  = raw_t[mask]  * TIME_TO_MIN[t_unit]   # → min
            measured_dp = raw_dp[mask] * DP_TO_MBAR[dp_unit]   # → mbar
            order = np.argsort(measured_t)
            measured_t, measured_dp = measured_t[order], measured_dp[order]
        except Exception as e:
            st.error(f"Failed to parse selected columns: {e}")
            measured_t, measured_dp = None, None

        if (measured_t is not None and len(measured_t) > 1):
            st.markdown(
                f'<div class="metric-card">'
                f'<b>Loaded:</b> <code>{len(measured_t)}</code> rows, '
                f't=<code>{measured_t[0]:.2f}</code> → '
                f'<code>{measured_t[-1]:.2f} min</code>, '
                f'ΔP=<code>{np.min(measured_dp):.2f}</code> → '
                f'<code>{np.max(measured_dp):.2f} mbar</code>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if measured_t[-1] > t_total_min:
                st.markdown(
                    f'<div class="warn-box">⚠ Measured t_max '
                    f'(<b>{measured_t[-1]:.2f} min</b>) exceeds the '
                    f'OPERATING ▸ Total time '
                    f'(<b>{t_total_min:.2f} min</b>). Increase '
                    f'<i>Total time</i> above to cover your data.</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="section-label">▌ FIT OPTIONS</div>',
                unsafe_allow_html=True)
    cF1, cF2, cF3, cF4 = st.columns(4)
    with cF1:
        fit_kim = st.checkbox("Fit kr_max,inj",  value=True, key="fit_kim")
    with cF2:
        fit_ni  = st.checkbox("Fit n_inj",       value=True, key="fit_ni")
    with cF3:
        fit_kdm = st.checkbox("Fit kr_max,disp", value=True, key="fit_kdm")
    with cF4:
        fit_nd  = st.checkbox("Fit n_disp",      value=True, key="fit_nd")
    fit_mask = (fit_kim, fit_ni, fit_kdm, fit_nd)

    max_iter = st.number_input("Max optimizer iterations", min_value=10,
                               max_value=500, value=80, step=10,
                               key="inv_maxiter",
                               help="Each iteration costs ~1 forward "
                                    "simulation. Pc enabled = much slower.")

    if pc_enabled:
        st.markdown(
            '<div class="warn-box">⚠ Capillary pressure is ON. Each '
            'forward call is much slower, so optimization can take '
            'tens of minutes. Disable Pc for initial fits, then '
            're-enable for refinement.</div>',
            unsafe_allow_html=True,
        )

    can_fit = (
        measured_t is not None
        and len(measured_t) > 1
        and mobile_range > 0
        and any(fit_mask)
    )

    if not can_fit:
        reasons = []
        if measured_t is None or len(measured_t or []) <= 1:
            reasons.append("upload a CSV with at least 2 valid rows")
        if mobile_range <= 0:
            reasons.append("fix saturation endpoints")
        if not any(fit_mask):
            reasons.append("select at least one parameter to fit")
        st.caption("To run the fit: " + ", ".join(reasons) + ".")

    run_fit = st.button("▶  Run Inverse Fit", type="primary",
                        disabled=not can_fit, key="run_inverse")

    if run_fit:
        progress_box = st.empty()
        bar          = st.progress(0.0)

        def _progress(i, sse, params):
            bar.progress(min(i / max(max_iter * 2, 1), 1.0))
            progress_box.markdown(
                f'<div class="metric-card">'
                f'<b>Iter {i}</b> — SSE=<code>{sse:.3f}</code><br>'
                f"kr_max,inj=<code>{params['kr_inj_max']:.3f}</code>, "
                f"n_inj=<code>{params['n_inj']:.2f}</code>, "
                f"kr_max,disp=<code>{params['kr_disp_max']:.3f}</code>, "
                f"n_disp=<code>{params['n_disp']:.2f}</code>"
                f'</div>',
                unsafe_allow_html=True,
            )

        with st.spinner("Running Nelder–Mead optimization…"):
            try:
                st.session_state["tp_fit"] = fit_corey(
                    st.session_state["tp_inputs"],
                    measured_t, measured_dp,
                    fit_mask=fit_mask,
                    max_iter=int(max_iter),
                    on_iter=_progress,
                )
                st.session_state["tp_measured_t"]  = measured_t
                st.session_state["tp_measured_dp"] = measured_dp
                st.session_state["tp_fit_error"]   = None
            except Exception as e:
                st.session_state["tp_fit"] = None
                st.session_state["tp_fit_error"] = str(e)
        bar.empty()
        progress_box.empty()

    if st.session_state.get("tp_fit_error"):
        st.error(f"Fit failed: {st.session_state['tp_fit_error']}")

    fit = st.session_state.get("tp_fit")
    if fit is not None:
        fp = fit["fitted_params"]
        mt = st.session_state["tp_measured_t"]
        md = st.session_state["tp_measured_dp"]

        st.markdown(
            f'<div class="metric-card">'
            f'<b>Converged:</b> '
            f'<code>{"yes" if fit["converged"] else "no"}</code> '
            f'({fit["n_calls"]} forward calls) — '
            f'<b>SSE</b>=<code>{fit["sse"]:.3f} mbar²</code><br>'
            f'<b>kr_max,inj</b>=<code>{fp["kr_inj_max"]:.4f}</code>, '
            f'<b>n_inj</b>=<code>{fp["n_inj"]:.3f}</code>, '
            f'<b>kr_max,disp</b>=<code>{fp["kr_disp_max"]:.4f}</code>, '
            f'<b>n_disp</b>=<code>{fp["n_disp"]:.3f}</code><br>'
            f'<b>Optimizer:</b> {fit["message"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # History-match plot
        components.html(
            render_chart_html(
                build_history_match_chart(mt, md, fit["results"])
            ),
            height=460, scrolling=False,
        )

        # Fitted kr curves
        fitted_inputs = dict(st.session_state["tp_inputs"])
        fitted_inputs["kr"] = {
            "inj_max":  fp["kr_inj_max"],
            "n_inj":    fp["n_inj"],
            "disp_max": fp["kr_disp_max"],
            "n_disp":   fp["n_disp"],
        }
        kr_fit = kr_curves(fitted_inputs)
        components.html(
            render_chart_html(build_kr_chart(
                kr_fit, inj_name=injected["name"],
                disp_name=displaced["name"],
            )),
            height=410, scrolling=False,
        )

        # CSV download of fitted curves + history match
        out_df = pd.DataFrame({
            "S_inj":   kr_fit["S_inj"],
            "kr_inj":  kr_fit["kr_inj"],
            "kr_disp": kr_fit["kr_disp"],
        })
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇  Download fitted kr curves (.csv)",
            data=csv_bytes,
            file_name="fitted_kr_curves.csv",
            mime="text/csv",
            key="inv_download",
        )
    else:
        st.caption("Run a fit to see fitted Corey parameters and "
                   "history-match plot here.")