"""
CoreFlood Lab — Two-Phase (Relative Permeability) Tool.

A 1D IMPES forward simulator and multi-algorithm inverse fitter for
relative permeability and capillary pressure analysis of core-flood
experiments. Inputs are collected at the top of the page; results
appear in the Forward and Inverse tabs below.

Module map
----------
utils.kr_models         : Pluggable kr parameterizations (Corey, LET).
utils.twophase          : kr / Pc factories using the selected model.
utils.twophase_solver   : 1D IMPES forward simulator.
utils.twophase_inverse  : Model-agnostic optimizer dispatcher.
utils.plotting          : Plotly chart builders.
utils.phases            : Phase library + phase-picker widget.
utils.units             : Unit-conversion tables and helpers.
utils.ui                : Shared CSS + input-row helpers.
"""

import copy
import hashlib
import json
import math

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils.phases import phase_picker, io_buttons
from utils.ui import inject_shared_css, render_header, input_row, dim_row
from utils.units import (
    LENGTH_TO_CM, POROSITY_TO_FRACTION, PERMEABILITY_TO_MD,
    VOLUME_TO_ML, TIME_TO_MIN, PRESSURE_TO_BAR, DP_TO_MBAR,
    convert, convert_injection_rate,
)
from utils.kr_models import KR_MODEL_CHOICES, get_model_class
from utils.twophase import kr_curves, pc_curve
from utils.twophase_solver import run_forward
from utils.twophase_inverse import fit_kr, OPTIMIZER_CHOICES
from utils.plotting import (
    build_kr_chart, build_pc_chart,
    build_dp_time_chart, build_profile_animation,
    build_history_match_chart,
    render_chart_html,
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CoreFlood Lab — Two-Phase",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_shared_css()
render_header("CoreFlood Lab — Two-Phase (Relative Permeability)")


def _inputs_hash(d: dict) -> str:
    """Short stable hash of the inputs dict for stale-results detection."""
    return hashlib.md5(
        json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:10]


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
    s_inj_r_val, s_inj_r_u = input_row(
        "S_r (residual)", 0.0, POROSITY_TO_FRACTION,
        "s_inj_r", default_unit="fraction",
        help="Residual saturation of the injected phase; kr_inj = 0 "
             "at and below this saturation.",
    )
with col_sd:
    st.markdown('<div class="group-label">Displaced phase</div>',
                unsafe_allow_html=True)
    s_disp_r_val, s_disp_r_u = input_row(
        "S_r (residual)", 0.20, POROSITY_TO_FRACTION,
        "s_disp_r", default_unit="fraction",
        help="Residual saturation of the displaced phase that cannot "
             "be swept by injection.",
    )
s_inj_r  = convert(s_inj_r_val,  s_inj_r_u,  POROSITY_TO_FRACTION)
s_disp_r = convert(s_disp_r_val, s_disp_r_u, POROSITY_TO_FRACTION)
mobile_range = 1.0 - s_inj_r - s_disp_r

# ── Relative permeability model + parameters (initial guesses) ──────────────
st.markdown('<div class="section-label">▌ RELATIVE PERMEABILITY MODEL '
            '(initial guesses for Inverse)</div>',
            unsafe_allow_html=True)

# Model dropdown as first row.
col_m_lbl, col_m_sel = st.columns([1.2, 2.8])
with col_m_lbl:
    st.markdown('<div class="row-label">Model</div>',
                unsafe_allow_html=True)
with col_m_sel:
    selected_model_name = st.selectbox(
        "Model", KR_MODEL_CHOICES, index=0,
        key="kr_model", label_visibility="collapsed",
        help=(
            "• Corey — 4 parameters (kr_max and n per phase). Industry "
            "default, physically legible, well-constrained by transient "
            "data.\n"
            "• LET — 8 parameters (kr_max, L, E, T per phase). More "
            "flexible; needed for wettability-altered or non-Corey rocks. "
            "The inverse problem is meaningfully harder; expect broader "
            "parameter uncertainty from single-rate transient data."
        ),
    )
model_cls   = get_model_class(selected_model_name)
param_names = list(model_cls.parameter_names)

# Parameter fields, two-column layout by phase, using each model's defaults
# and display metadata. Widget keys are param-name scoped so state carries
# over sensibly when the user switches between models that share names
# (e.g. kr_inj_max exists in both Corey and LET).
kr_param_values = {}
col_ki, col_kd = st.columns(2)
with col_ki:
    st.markdown('<div class="group-label">Injected phase</div>',
                unsafe_allow_html=True)
    for name in model_cls.injected_param_names:
        disp = model_cls.param_display[name]
        lo, hi = model_cls.parameter_bounds[name]
        kr_param_values[name] = dim_row(
            disp["label"], model_cls.defaults[name], f"kr_{name}",
            min_value=lo, max_value=hi,
            step=disp["step"], fmt=disp["fmt"],
            help=disp["help"],
        )
with col_kd:
    st.markdown('<div class="group-label">Displaced phase</div>',
                unsafe_allow_html=True)
    for name in model_cls.displaced_param_names:
        disp = model_cls.param_display[name]
        lo, hi = model_cls.parameter_bounds[name]
        kr_param_values[name] = dim_row(
            disp["label"], model_cls.defaults[name], f"kr_{name}",
            min_value=lo, max_value=hi,
            step=disp["step"], fmt=disp["fmt"],
            help=disp["help"],
        )

# ── Capillary pressure ──────────────────────────────────────────────────────
st.markdown('<div class="section-label">▌ CAPILLARY PRESSURE</div>',
            unsafe_allow_html=True)
pc_enabled = st.checkbox(
    "Include capillary pressure (Brooks–Corey)",
    value=False, key="pc_enabled",
    help="When enabled, the simulator includes a capillary-diffusion "
         "term that smooths the saturation front. Each forward call "
         "becomes significantly slower.",
)
pc_entry_mbar = None
pc_lambda     = None
if pc_enabled:
    col_pe, col_pl = st.columns(2)
    with col_pe:
        pc_entry_val, pc_entry_u = input_row(
            "Entry pressure P_e", 100.0, DP_TO_MBAR,
            "pc_entry", default_unit="mbar",
            help="Brooks–Corey threshold entry pressure of the "
                 "non-wetting (injected) phase.",
        )
        pc_entry_mbar = convert(pc_entry_val, pc_entry_u, DP_TO_MBAR)
    with col_pl:
        pc_lambda = dim_row(
            "Pore-size index λ", 2.0, "pc_lambda",
            min_value=0.1, max_value=10.0, step=0.1, fmt="%.2f",
            help="Brooks–Corey pore-size distribution index. "
                 "Smaller λ ⇒ wider pore-size distribution.",
        )

# ── Rock & core geometry ────────────────────────────────────────────────────
st.markdown('<div class="section-label">▌ ROCK & CORE GEOMETRY</div>',
            unsafe_allow_html=True)
L_val, L_u     = input_row("Length L",         10.0,  LENGTH_TO_CM,
                           "core_L", default_unit="cm")
D_val, D_u     = input_row("Diameter D",       3.8,   LENGTH_TO_CM,
                           "core_D", default_unit="cm")
phi_val, phi_u = input_row("Porosity φ",       20.0,  POROSITY_TO_FRACTION,
                           "core_phi", default_unit="%")
k_val, k_u     = input_row("Absolute perm. k", 100.0, PERMEABILITY_TO_MD,
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
p_back_val, p_back_u   = input_row("Back pressure P_out", 1.0,
                                   PRESSURE_TO_BAR, "op_pback",
                                   default_unit="bar")
t_total_val, t_total_u = input_row("Total time", 60.0, TIME_TO_MIN,
                                   "op_ttotal", default_unit="min")
p_back_bar  = convert(p_back_val,  p_back_u,  PRESSURE_TO_BAR)
t_total_min = convert(t_total_val, t_total_u, TIME_TO_MIN)

# ── Numerical ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">▌ NUMERICAL</div>',
            unsafe_allow_html=True)
n_cells = st.number_input(
    "Grid cells N", min_value=20, max_value=500, value=100, step=10,
    key="n_cells",
    help="Number of 1D finite-difference cells. More cells = sharper "
         "front but slower runs. 100 is a sensible default; use 200+ "
         "for publication-grade plots.",
)

# ── Stash inputs in session_state ───────────────────────────────────────────
st.session_state["tp_inputs"] = {
    "injected":  injected,
    "displaced": displaced,
    "S_inj_r":   s_inj_r,
    "S_disp_r":  s_disp_r,
    "kr": {"model":  selected_model_name,
           "params": dict(kr_param_values)},
    "pc": {"enabled": pc_enabled, "P_entry_mbar": pc_entry_mbar,
           "lambda": pc_lambda},
    "core": {"L_cm": L_cm, "D_cm": D_cm, "A_cm2": A_cm2,
             "phi": phi, "k_mD": k_mD},
    "operating": {"q_ml_min": q_ml_min, "P_back_bar": p_back_bar,
                  "t_total_min": t_total_min},
    "numerical": {"N_cells": int(n_cells)},
}
cur_hash = _inputs_hash(st.session_state["tp_inputs"])

# ── Validation ──────────────────────────────────────────────────────────────
input_errors = []
if mobile_range <= 0:
    input_errors.append(
        f"S_r,inj + S_r,disp = {s_inj_r + s_disp_r:.3f} ≥ 1.0 — "
        f"no mobile saturation range."
    )
if L_cm    <= 0: input_errors.append("Length must be > 0.")
if D_cm    <= 0: input_errors.append("Diameter must be > 0.")
if phi <= 0 or phi >= 1:
    input_errors.append("Porosity must lie strictly between 0 and 1.")
if k_mD       <= 0: input_errors.append("Absolute permeability must be > 0.")
if q_ml_min   <= 0: input_errors.append("Injection rate must be > 0.")
if t_total_min <= 0: input_errors.append("Total time must be > 0.")
if injected["viscosity_cP"]  <= 0 or displaced["viscosity_cP"] <= 0:
    input_errors.append("Phase viscosities must be > 0.")
if pc_enabled and pc_entry_mbar is not None and pc_entry_mbar <= 0:
    input_errors.append("Capillary entry pressure must be > 0.")
if pc_enabled and pc_lambda is not None and pc_lambda <= 0:
    input_errors.append("Pore-size index λ must be > 0.")

ready_to_run = len(input_errors) == 0

if input_errors:
    st.markdown(
        '<div class="error-box">'
        + "<br>".join("⛔ " + e for e in input_errors)
        + "</div>",
        unsafe_allow_html=True,
    )

# ── Collapsed debug readout ─────────────────────────────────────────────────
with st.expander("Collected inputs (debug)", expanded=False):
    pc_line = (
        f'<b>Pc:</b> Brooks–Corey, '
        f'P_e=<code>{pc_entry_mbar:.2f} mbar</code>, '
        f'λ=<code>{pc_lambda:.2f}</code>'
        if pc_enabled else '<b>Pc:</b> <code>off</code>'
    )
    kr_str = ", ".join(
        f"{n}=<code>{kr_param_values[n]:.4g}</code>" for n in param_names
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
        f'<b>kr model:</b> <code>{selected_model_name}</code> '
        f'({len(param_names)} params)<br>'
        f'<b>kr (initial):</b> {kr_str}<br>'
        f'{pc_line}<br>'
        f'<b>Core:</b> L=<code>{L_cm:.2f} cm</code>, '
        f'D=<code>{D_cm:.2f} cm</code>, '
        f'φ=<code>{phi:.3f}</code>, k=<code>{k_mD:.2f} mD</code><br>'
        f'<b>Op:</b> q=<code>{q_ml_min:.4f} ml/min</code>, '
        f't_total=<code>{t_total_min:.2f} min</code>, '
        f'N=<code>{int(n_cells)}</code> · '
        f'<b>hash:</b> <code>{cur_hash}</code>'
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
        st.markdown('<div class="warn-box">No mobile range — adjust '
                    'residual saturations to view curves.</div>',
                    unsafe_allow_html=True)
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

    run_clicked = st.button(
        "▶  Run Forward Simulation", type="primary",
        key="run_forward", disabled=not ready_to_run,
    )
    if run_clicked:
        with st.spinner("Running 1D IMPES forward simulation…"):
            try:
                st.session_state["tp_results"] = run_forward(
                    st.session_state["tp_inputs"]
                )
                st.session_state["tp_results_hash"]  = cur_hash
                st.session_state["tp_results_error"] = None
            except Exception as e:
                st.session_state["tp_results"] = None
                st.session_state["tp_results_error"] = str(e)

    if st.session_state.get("tp_results_error"):
        st.error(f"Simulation failed: "
                 f"{st.session_state['tp_results_error']}")

    results = st.session_state.get("tp_results")
    if results is not None:
        is_stale = st.session_state.get("tp_results_hash") != cur_hash
        if is_stale:
            st.markdown(
                '<div class="warn-box">⚠ Results below were computed '
                'with different inputs. Click <b>Run Forward Simulation</b> '
                'to refresh.</div>',
                unsafe_allow_html=True,
            )

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
            f'<b>ΔP:</b> initial=<code>{dp_init:.2f} mbar</code>, '
            f'final=<code>{dp_end:.2f} mbar</code>'
            f'</div>',
            unsafe_allow_html=True,
        )
        export_df = pd.DataFrame({
            "time_min": results["t_min"],
            "dP_mbar":  results["dP_mbar"],
        })
        st.download_button(
            "⬇  Download ΔP(t) history (.csv)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"forward_dP_history_{cur_hash}.csv",
            mime="text/csv",
            key="fwd_download_dp",
            help="Export this forward run's ΔP(t) as a CSV. Load it on the "
                 "Inverse tab to demonstrate the back-fitting workflow.",
            disabled=is_stale,
        )
        components.html(
            render_chart_html(build_dp_time_chart(results), autoplay=True),
            height=450, scrolling=False,
        )
        components.html(
            render_chart_html(build_profile_animation(results),
                              autoplay=False),
            height=480, scrolling=False,
        )
    elif ready_to_run:
        st.caption("Click 'Run Forward Simulation' to generate ΔP "
                   "history and saturation profile.")

# ── INVERSE TAB ─────────────────────────────────────────────────────────────
with tab_inv:
    st.markdown('<div class="section-label">▌ UPLOAD ΔP(t) DATA</div>',
                unsafe_allow_html=True)
    st.caption("Expected CSV: at least two columns (time and pressure "
               "drop). Column names and units are configurable below.")
    uploaded = st.file_uploader(
        "Upload CSV", type=["csv", "txt"], key="inv_csv",
        label_visibility="collapsed",
    )

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
            t_unit = st.selectbox(
                "Time unit", list(TIME_TO_MIN.keys()),
                index=list(TIME_TO_MIN.keys()).index("min"),
                key="inv_tunit",
            )
        with c3:
            dp_col = st.selectbox(
                "ΔP column", cols,
                index=1 if len(cols) > 1 else 0, key="inv_dpcol",
            )
        with c4:
            dp_unit = st.selectbox(
                "ΔP unit", list(DP_TO_MBAR.keys()),
                index=list(DP_TO_MBAR.keys()).index("mbar"),
                key="inv_dpunit",
            )
        try:
            raw_t  = pd.to_numeric(df[t_col],  errors="coerce").to_numpy()
            raw_dp = pd.to_numeric(df[dp_col], errors="coerce").to_numpy()
            mask = np.isfinite(raw_t) & np.isfinite(raw_dp)
            measured_t  = raw_t[mask]  * TIME_TO_MIN[t_unit]
            measured_dp = raw_dp[mask] * DP_TO_MBAR[dp_unit]
            order = np.argsort(measured_t)
            measured_t, measured_dp = measured_t[order], measured_dp[order]
        except Exception as e:
            st.error(f"Failed to parse selected columns: {e}")
            measured_t, measured_dp = None, None

        if measured_t is not None and len(measured_t) > 1:
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

    col_opt_label, col_opt_sel = st.columns([1.2, 2.8])
    with col_opt_label:
        st.markdown('<div class="row-label">Optimizer</div>',
                    unsafe_allow_html=True)
    with col_opt_sel:
        optimizer_name = st.selectbox(
            "Optimizer", OPTIMIZER_CHOICES,
            index=0,
            key="inv_optimizer",
            label_visibility="collapsed",
            help=(
                "• Differential Evolution — global, robust to poor initial "
                "guesses. Best default for LET where identifiability is "
                "harder.\n"
                "• Nelder-Mead — local simplex with 8 automatic multi-starts "
                "and log-space transforms; fast when initial guess is decent.\n"
                "• Levenberg-Marquardt — local gradient-based least-squares; "
                "fastest when the initial guess is already close."
            ),
        )

    # ── Fit-mask UI: "Fit all" master + Advanced expander ───────────────────
    # Initialize master toggle default once.
    if "cb_fit_all" not in st.session_state:
        st.session_state["cb_fit_all"] = True

    fit_all = st.checkbox(
        "Fit all model parameters", key="cb_fit_all",
        help="When on, every parameter of the selected model is treated "
             "as free. Open Advanced (or turn this off) to hold specific "
             "parameters fixed at their initial guesses above.",
    )

    # If the master is on, force every individual key to True BEFORE the
    # widgets render, so opening Advanced shows a consistent state.
    if fit_all:
        for name in param_names:
            st.session_state[f"cb_fit_{name}"] = True

    # Unchecking any individual checkbox turns the master off.
    def _turn_off_fit_all():
        st.session_state["cb_fit_all"] = False

    with st.expander("Advanced — select individual parameters",
                     expanded=not fit_all):
        st.caption(
            "Uncheck a parameter to hold it fixed at its initial guess. "
            "This automatically turns off 'Fit all'. Useful when a "
            "parameter is known from an independent measurement "
            "(e.g. steady-state endpoint) and shouldn't be varied."
        )
        n_cols = min(4, len(param_names))
        cols = st.columns(n_cols)
        individual_mask = []
        for i, name in enumerate(param_names):
            with cols[i % n_cols]:
                individual_mask.append(
                    st.checkbox(
                        f"Fit {name}", value=True, key=f"cb_fit_{name}",
                        on_change=_turn_off_fit_all,
                    )
                )

    fit_mask = ([True] * len(param_names)) if fit_all else individual_mask

    max_iter = st.number_input(
        "Max optimizer iterations", min_value=10, max_value=2000,
        value=200, step=10, key="inv_maxiter",
        help="Rough budget for the optimizer. Nelder-Mead splits this "
             "across 8 restarts; DE uses it to set max generations; LM "
             "uses it as max function evaluations. LET (8 params) needs "
             "meaningfully more budget than Corey (4).",
    )

    if pc_enabled:
        st.markdown(
            '<div class="warn-box">⚠ Capillary pressure is ON. Each '
            'forward call is much slower, so optimization can take '
            'tens of minutes. Disable Pc for initial fits, then '
            're-enable for refinement.</div>',
            unsafe_allow_html=True,
        )

    can_fit = (
        ready_to_run
        and measured_t is not None
        and len(measured_t) > 1
        and any(fit_mask)
    )
    if not can_fit:
        reasons = []
        if not ready_to_run:
            reasons.append("fix the input errors above")
        if measured_t is None or len(measured_t or []) <= 1:
            reasons.append("upload a CSV with ≥ 2 valid rows")
        if not any(fit_mask):
            reasons.append("select at least one parameter to fit")
        st.caption("To run the fit: " + ", ".join(reasons) + ".")

    run_fit = st.button("▶  Run Inverse Fit", type="primary",
                        disabled=not can_fit, key="run_inverse")

    if run_fit:
        progress_box = st.empty()
        bar          = st.progress(0.0)

        def _progress(i, sse, params_dict):
            bar.progress(min(i / max(int(max_iter) * 2, 1), 1.0))
            param_str = ", ".join(
                f"{n}=<code>{v:.4g}</code>"
                for n, v in params_dict.items()
            )
            progress_box.markdown(
                f'<div class="metric-card">'
                f'<b>Iter {i}</b> — SSE=<code>{sse:.3f}</code><br>'
                f'{param_str}'
                f'</div>',
                unsafe_allow_html=True,
            )

        with st.spinner(f"Running {optimizer_name} optimization "
                        f"({selected_model_name}, "
                        f"{sum(fit_mask)}/{len(fit_mask)} free)…"):
            try:
                st.session_state["tp_fit"] = fit_kr(
                    st.session_state["tp_inputs"],
                    measured_t, measured_dp,
                    fit_mask=fit_mask,
                    max_iter=int(max_iter),
                    on_iter=_progress,
                    optimizer_name=optimizer_name,
                )
                st.session_state["tp_fit_hash"]    = cur_hash
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
        is_stale = st.session_state.get("tp_fit_hash") != cur_hash
        if is_stale:
            st.markdown(
                '<div class="warn-box">⚠ Fit below was computed with '
                'different inputs. Re-upload data and click <b>Run '
                'Inverse Fit</b> to refresh.</div>',
                unsafe_allow_html=True,
            )

        fp = fit["fitted_params"]
        mt = st.session_state["tp_measured_t"]
        md = st.session_state["tp_measured_dp"]

        fp_lines = ", ".join(
            f'<b>{n}</b>=<code>{fp[n]:.4g}</code>'
            for n in fit["param_names"]
        )
        st.markdown(
            f'<div class="metric-card">'
            f'<b>Model:</b> <code>{fit.get("model_name", "?")}</code> · '
            f'<b>Optimizer:</b> <code>{fit.get("optimizer_name", "?")}</code> · '
            f'<b>Converged:</b> '
            f'<code>{"yes" if fit["converged"] else "no"}</code> '
            f'({fit["n_calls"]} forward calls) — '
            f'<b>SSE</b>=<code>{fit["sse"]:.3f} mbar²</code><br>'
            f'{fp_lines}<br>'
            f'<b>Message:</b> {fit["message"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

        components.html(
            render_chart_html(
                build_history_match_chart(mt, md, fit["results"])
            ),
            height=460, scrolling=False,
        )

        # Rebuild fitted-kr curves by injecting fitted params into the
        # current tp_inputs (preserving the model choice).
        fitted_inputs = copy.deepcopy(st.session_state["tp_inputs"])
        fitted_inputs["kr"]["params"] = dict(fp)
        kr_fit = kr_curves(fitted_inputs)
        components.html(
            render_chart_html(build_kr_chart(
                kr_fit, inj_name=injected["name"],
                disp_name=displaced["name"],
            )),
            height=410, scrolling=False,
        )

        out_df = pd.DataFrame({
            "S_inj":   kr_fit["S_inj"],
            "kr_inj":  kr_fit["kr_inj"],
            "kr_disp": kr_fit["kr_disp"],
        })
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇  Download fitted kr curves (.csv)",
            data=csv_bytes,
            file_name=f"fitted_kr_curves_{fit['model_name']}.csv",
            mime="text/csv",
            key="inv_download",
        )
    elif can_fit:
        st.caption("Click 'Run Inverse Fit' to fit model parameters "
                   "against the uploaded data.")

# ── Clear cached results (full-page utility) ────────────────────────────────
st.markdown("---")
clear_col_l, clear_col_r = st.columns([3, 1])
with clear_col_r:
    if st.button("🗑  Clear cached results", key="clear_results"):
        for key in ("tp_results", "tp_results_hash", "tp_results_error",
                    "tp_fit", "tp_fit_hash", "tp_fit_error",
                    "tp_measured_t", "tp_measured_dp"):
            st.session_state.pop(key, None)
        st.rerun()