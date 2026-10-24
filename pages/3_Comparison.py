"""
CoreFlood Lab — Optimizer Comparison (Step 3.1).

Fits ONE dataset with Differential Evolution, Nelder-Mead and
Levenberg-Marquardt, from the SAME initial guess, bounds and fit mask.
Everything the three runs see is identical except the search algorithm.

The question it answers: do different optimizers agree about the kr
parameters? If they match the pressure history equally well but report
different parameters, the data does not determine those parameters.

Inputs come from the Two-Phase page. Configure the core, phases, kr model
and parameters there first; this page reads st.session_state["tp_inputs"]
and adds only what the comparison needs.
"""

import copy

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils.ui import inject_shared_css, render_header
from utils.kr_models import get_model_class
from utils.plotting import render_chart_html
from utils.units import TIME_TO_MIN, DP_TO_MBAR
from utils.twophase_inverse import OPTIMIZER_CHOICES
from utils.comparison import (
    prepare_inputs, generate_synthetic_data, wrap_uploaded_data,
    run_optimizer_comparison, results_table, disagreement_table,
    attach_kr_curves,
)
from utils.comparison_plots import (
    build_history_overlay, build_kr_overlay, build_parameter_chart,
)

st.set_page_config(
    page_title="CoreFlood Lab — Optimizer Comparison",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_shared_css()
render_header("CoreFlood Lab — Optimizer Comparison")

# -- Require inputs from the Two-Phase page ---------------------------------
tp_inputs = st.session_state.get("tp_inputs")
if tp_inputs is None:
    st.markdown(
        '<div class="warn-box">⚠ No inputs found. Open the '
        '<b>Two-Phase</b> page first and configure the core, phases, kr '
        'model and parameters. This page reads them from there.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

model_cls = get_model_class(tp_inputs["kr"]["model"])
param_names = list(model_cls.parameter_names)
base_params = dict(tp_inputs["kr"]["params"])

st.markdown(
    '<div class="info-box">The same dataset is fitted by each selected '
    'optimizer, from the same initial guess and bounds. Only the search '
    'algorithm differs. If they agree on the pressure history but '
    'disagree on the parameters, the data does not determine those '
    'parameters.</div>',
    unsafe_allow_html=True,
)

params_str = ", ".join(f"<b>{n}</b>=<code>{base_params[n]:.4g}</code>"
                       for n in param_names)
st.markdown(
    f'<div class="metric-card">'
    f'<b>Model:</b> <code>{tp_inputs["kr"]["model"]}</code> · '
    f'<b>Process:</b> <code>{tp_inputs["process_type"]}</code> · '
    f'<b>Injected:</b> {tp_inputs["injected"]["name"]} · '
    f'<b>Displaced:</b> {tp_inputs["displaced"]["name"]}<br>'
    f'<b>Two-Phase kr parameters:</b> {params_str}'
    f'</div>',
    unsafe_allow_html=True,
)

# -- Data source ------------------------------------------------------------
st.markdown('<div class="section-label">▌ DATA</div>',
            unsafe_allow_html=True)

source = st.radio(
    "Data source",
    ["Upload ΔP(t) CSV", "Generate synthetic (truth known)"],
    index=0, horizontal=True, key="cmp_source",
    help="Upload: fit real or exported data; the optimizers can only be "
         "compared against each other. Synthetic: the Two-Phase kr "
         "parameters are treated as truth and data is generated from "
         "them, so each optimizer's answer can also be scored for "
         "accuracy.",
)
use_synthetic = source.startswith("Generate")

measured_t = measured_dp = None
noise_pct = 0.0
n_points = 60
seed = 0
guess_offset_pct = 25.0

if use_synthetic:
    c1, c2, c3 = st.columns(3)
    with c1:
        noise_pct = st.slider(
            "Noise level [% of peak ΔP]", 0.0, 5.0, 1.0, 0.1,
            key="cmp_noise",
            help="Gaussian noise with constant sigma, added once and "
                 "shared by every optimizer. At 0% the problem has a "
                 "unique zero-objective minimum — useful as a control.",
        )
        n_points = st.number_input(
            "Measurement points", 10, 400, 60, 10, key="cmp_npoints",
        )
    with c2:
        guess_offset_pct = st.number_input(
            "Initial guess offset from truth [%]", 0.0, 90.0, 25.0, 5.0,
            key="cmp_offset",
            help="The initial guess is set to truth × (1 + offset), "
                 "clipped to the parameter bounds. Without an offset "
                 "the local optimizers would start at the answer and "
                 "the test would prove nothing. DE ignores the initial "
                 "guess either way.",
        )
    with c3:
        seed = st.number_input("Random seed", 0, 9999, 0, 1,
                               key="cmp_seed")
else:
    uploaded = st.file_uploader("Upload CSV", type=["csv", "txt"],
                                key="cmp_csv", label_visibility="collapsed")
    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")

    if df is not None and len(df.columns) >= 2:
        cols = list(df.columns)
        c1, c2, c3, c4 = st.columns([1.4, 1.0, 1.4, 1.0])
        with c1:
            t_col = st.selectbox("Time column", cols, 0, key="cmp_tcol")
        with c2:
            t_unit = st.selectbox(
                "Time unit", list(TIME_TO_MIN.keys()),
                list(TIME_TO_MIN.keys()).index("min"), key="cmp_tunit")
        with c3:
            dp_col = st.selectbox("ΔP column", cols,
                                  1 if len(cols) > 1 else 0,
                                  key="cmp_dpcol")
        with c4:
            dp_unit = st.selectbox(
                "ΔP unit", list(DP_TO_MBAR.keys()),
                list(DP_TO_MBAR.keys()).index("mbar"), key="cmp_dpunit")
        try:
            raw_t = pd.to_numeric(df[t_col], errors="coerce").to_numpy()
            raw_dp = pd.to_numeric(df[dp_col], errors="coerce").to_numpy()
            mask = np.isfinite(raw_t) & np.isfinite(raw_dp)
            measured_t = raw_t[mask] * TIME_TO_MIN[t_unit]
            measured_dp = raw_dp[mask] * DP_TO_MBAR[dp_unit]
            order = np.argsort(measured_t)
            measured_t, measured_dp = measured_t[order], measured_dp[order]
            st.markdown(
                f'<div class="metric-card">'
                f'<b>Loaded:</b> <code>{len(measured_t)}</code> rows, '
                f't=<code>{measured_t[0]:.2f}</code> → '
                f'<code>{measured_t[-1]:.2f} min</code>, '
                f'ΔP=<code>{np.min(measured_dp):.2f}</code> → '
                f'<code>{np.max(measured_dp):.2f} mbar</code></div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Failed to parse selected columns: {e}")
            measured_t = measured_dp = None

# -- Free parameters --------------------------------------------------------
st.markdown('<div class="section-label">▌ FREE PARAMETERS</div>',
            unsafe_allow_html=True)
st.caption("Unchecked parameters are held fixed at the Two-Phase value "
           "for every optimizer.")

for _n in param_names:
    st.session_state.setdefault(f"cmp_fit_{_n}", True)

n_cols = min(4, len(param_names))
cols = st.columns(n_cols)
fit_mask = []
for i, name in enumerate(param_names):
    with cols[i % n_cols]:
        fit_mask.append(st.checkbox(f"Fit {name}", key=f"cmp_fit_{name}"))

free_names = [n for n, m in zip(param_names, fit_mask) if m]
n_free = len(free_names)

# -- Settings ---------------------------------------------------------------
st.markdown('<div class="section-label">▌ SETTINGS</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    optimizers = st.multiselect(
        "Optimizers", OPTIMIZER_CHOICES, default=list(OPTIMIZER_CHOICES),
        key="cmp_optimizers",
        help="Differential Evolution is by far the most expensive — "
             "deselect it for a quick run.",
    )
with c2:
    max_iter = st.number_input(
        "Effort budget (shared)", 50, 2000, 300, 50, key="cmp_effort",
        help="Each optimizer interprets this differently: DE as "
             "generations, LM as max function evaluations, N-M split "
             "across 8 restarts. Actual call counts are reported.",
    )
with c3:
    n_cells = st.number_input(
        "Grid cells N (override)", 20, 200,
        min(40, int(tp_inputs["numerical"]["N_cells"])), 10,
        key="cmp_ncells",
        help="40 is the tested value for staying inside the hosted "
             "execution wall.",
    )

force_pc_off = st.checkbox(
    "Force capillary pressure off", value=True, key="cmp_pc_off",
    help="Strongly recommended. Skjaeveland Pc multiplies the cost of "
         "every forward call by roughly ten, which puts a DE run past "
         "the execution wall.",
)

ready = (n_free > 0 and len(optimizers) > 0
         and (use_synthetic or measured_t is not None))
if n_free == 0:
    st.markdown('<div class="error-box">⛔ Select at least one free '
                'parameter.</div>', unsafe_allow_html=True)
if not optimizers:
    st.markdown('<div class="error-box">⛔ Select at least one '
                'optimizer.</div>', unsafe_allow_html=True)
if not use_synthetic and measured_t is None:
    st.caption("Upload a CSV with at least two valid rows to continue.")

run_clicked = st.button("▶  Run Comparison", type="primary",
                        disabled=not ready, key="cmp_run")

# -- Execute ----------------------------------------------------------------
if run_clicked:
    inp = prepare_inputs(tp_inputs, force_pc_off=force_pc_off,
                         n_cells=int(n_cells))
    status = st.empty()
    bar = st.progress(0.0)

    if use_synthetic:
        status.info("Generating the synthetic dataset at the truth "
                    "parameters…")
        try:
            data = generate_synthetic_data(
                inp, n_points=int(n_points), noise_pct=float(noise_pct),
                seed=int(seed))
        except Exception as e:
            bar.empty(); status.empty()
            st.error(f"Forward run failed: {e}")
            st.stop()
        truth_params = dict(data["truth_params"])

        # Displace the shared initial guess away from truth, clipped to
        # bounds, so the local optimizers do not start at the answer.
        guess = dict(truth_params)
        f = 1.0 + float(guess_offset_pct) / 100.0
        for n in free_names:
            lo, hi = model_cls.parameter_bounds[n]
            guess[n] = float(np.clip(truth_params[n] * f,
                                     lo + 1e-9, hi - 1e-9))
        inp = copy.deepcopy(inp)
        inp["kr"]["params"] = guess
    else:
        data = wrap_uploaded_data(measured_t, measured_dp)
        truth_params = None

    total = len(optimizers)

    def _on_run(i, n, name):
        bar.progress(min((i - 1) / n, 1.0))
        status.info(f"Run {i}/{n} — {name}")

    try:
        runs = run_optimizer_comparison(
            inp, data, fit_mask, optimizers=optimizers,
            max_iter=int(max_iter), on_run=_on_run,
        )
    except Exception as e:
        bar.empty(); status.empty()
        st.error(f"Comparison failed: {e}")
        st.stop()

    truth_kr = attach_kr_curves(runs, inp, truth_params=truth_params)

    bar.empty(); status.empty()
    st.session_state["cmp_data"] = data
    st.session_state["cmp_runs"] = runs
    st.session_state["cmp_truth_kr"] = truth_kr
    st.session_state["cmp_truth"] = truth_params
    st.session_state["cmp_mask"] = list(fit_mask)
    st.session_state["cmp_guess"] = dict(inp["kr"]["params"])

# -- Results ----------------------------------------------------------------
runs = st.session_state.get("cmp_runs")
if runs:
    data = st.session_state["cmp_data"]
    truth_kr = st.session_state["cmp_truth_kr"]
    truth_now = st.session_state["cmp_truth"]
    mask = st.session_state["cmp_mask"]
    guess = st.session_state["cmp_guess"]
    free_now = [n for n, m in zip(param_names, mask) if m]

    ok = [r for r in runs if r["fitted_params"] is not None]
    rmses = np.array([r["rmse"] for r in ok], dtype=float)

    st.markdown('<div class="section-label">▌ RESULT</div>',
                unsafe_allow_html=True)

    guess_str = ", ".join(f"<b>{n}</b>=<code>{guess[n]:.4g}</code>"
                          for n in free_now)
    if rmses.size:
        st.markdown(
            f'<div class="metric-card">'
            f'<b>Runs:</b> <code>{len(ok)}/{len(runs)}</code> completed · '
            f'<b>RMSE:</b> <code>{rmses.min():.3f}</code> → '
            f'<code>{rmses.max():.3f} mbar</code><br>'
            f'<b>Shared initial guess:</b> {guess_str}<br>'
            f'Optimizers with similar RMSE but different parameters '
            f'indicate the data does not determine those parameters.'
            f'</div>',
            unsafe_allow_html=True,
        )

    components.html(render_chart_html(build_history_overlay(data, runs)),
                    height=460, scrolling=False)
    components.html(
        render_chart_html(build_kr_overlay(
            runs, truth_kr,
            inj_name=tp_inputs["injected"]["name"],
            disp_name=tp_inputs["displaced"]["name"])),
        height=480, scrolling=False)
    components.html(
        render_chart_html(build_parameter_chart(
            runs, free_now, model_cls, truth_params=truth_now)),
        height=450, scrolling=False)

    st.markdown('<div class="section-label">▌ RUNS</div>',
                unsafe_allow_html=True)
    table = results_table(runs, param_names, mask,
                          truth_params=truth_now)
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">▌ DISAGREEMENT</div>',
                unsafe_allow_html=True)
    st.dataframe(
        disagreement_table(runs, param_names, mask,
                           truth_params=truth_now),
        use_container_width=True, hide_index=True,
    )

    st.download_button(
        "⬇  Download comparison runs (.csv)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="optimizer_comparison_runs.csv",
        mime="text/csv", key="cmp_download",
    )

    with st.expander("Optimizer messages", expanded=False):
        for r in runs:
            st.markdown(f"**{r['label']}** — {r['n_calls']} calls, "
                        f"{r['wall_s']:.1f} s — {r['message']}")