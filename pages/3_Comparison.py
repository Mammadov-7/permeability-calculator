"""
CoreFlood Lab — Optimizer Comparison (Step 3.1).

Demonstrates ill-posedness of the SCAL inverse problem by fitting one
shared noisy dataset many times and showing that the runs disagree about
the parameters while agreeing about the data.

Inputs come from the Two-Phase page. Configure the core, phases, kr
model and truth parameters there first; this page reads
st.session_state["tp_inputs"] and adds only what the comparison itself
needs (noise level, number of starts, optimizer effort).

The kr parameters currently set on the Two-Phase page are treated as
TRUTH here — the synthetic dataset is generated from them, and the runs
try to recover them from noisy data without being told the answer.
"""

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from utils.ui import inject_shared_css, render_header
from utils.kr_models import get_model_class
from utils.plotting import render_chart_html
from utils.comparison import (
    prepare_inputs, generate_synthetic_data, run_comparison,
    results_table, spread_summary, attach_kr_curves,
)
from utils.comparison_plots import (
    build_comparison_history_chart, build_comparison_kr_chart,
    build_parameter_spread_chart,
)

st.set_page_config(
    page_title="CoreFlood Lab — Optimizer Comparison",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

#inject_shared_css()
render_header("CoreFlood Lab — Optimizer Comparison")

# ── Require inputs from the Two-Phase page ─────────────────────────────────
tp_inputs = st.session_state.get("tp_inputs")
if tp_inputs is None:
    st.markdown(
        '<div class="warn-box">⚠ No inputs found. Open the '
        '<b>Two-Phase</b> page first and configure the core, phases, kr '
        'model and parameters. Those parameters are used here as the '
        '<b>truth</b> the comparison tries to recover.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

model_cls   = get_model_class(tp_inputs["kr"]["model"])
param_names = list(model_cls.parameter_names)
truth       = dict(tp_inputs["kr"]["params"])

st.markdown(
    '<div class="info-box">The kr parameters set on the Two-Phase page '
    'are the <b>truth</b>. A noisy synthetic dataset is generated from '
    'them, then fitted repeatedly from random starting points. None of '
    'the runs is told the answer.</div>',
    unsafe_allow_html=True,
)

truth_str = ", ".join(f"<b>{n}</b>=<code>{truth[n]:.4g}</code>"
                      for n in param_names)
st.markdown(
    f'<div class="metric-card">'
    f'<b>Model:</b> <code>{tp_inputs["kr"]["model"]}</code> · '
    f'<b>Process:</b> <code>{tp_inputs["process_type"]}</code> · '
    f'<b>Injected:</b> {tp_inputs["injected"]["name"]} · '
    f'<b>Displaced:</b> {tp_inputs["displaced"]["name"]}<br>'
    f'<b>Truth:</b> {truth_str}'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Which parameters are free ──────────────────────────────────────────────
st.markdown('<div class="section-label">▌ FREE PARAMETERS</div>',
            unsafe_allow_html=True)
st.caption("Parameters left unchecked are held at their truth value. "
           "Holding a parameter fixed at the exact truth is generous to "
           "the fit — the realistic case frees everything.")

for _n in param_names:
    _k = f"cmp_fit_{_n}"
    if _k not in st.session_state:
        st.session_state[_k] = True

n_cols = min(4, len(param_names))
cols = st.columns(n_cols)
fit_mask = []
for i, name in enumerate(param_names):
    with cols[i % n_cols]:
        fit_mask.append(st.checkbox(f"Fit {name}", key=f"cmp_fit_{name}"))

free_names = [n for n, m in zip(param_names, fit_mask) if m]
n_free = len(free_names)

# ── Comparison settings ────────────────────────────────────────────────────
st.markdown('<div class="section-label">▌ COMPARISON SETTINGS</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    noise_pct = st.slider(
        "Noise level [% of peak ΔP]", min_value=0.0, max_value=5.0,
        value=1.0, step=0.1, key="cmp_noise",
        help="Gaussian noise with constant sigma, added once and shared "
             "by every run. At 0% the problem has a unique zero-objective "
             "minimum and every run recovers truth — useful as a control, "
             "but it is the noise that makes the problem ill-posed.",
    )
    n_points = st.number_input(
        "Measurement points", min_value=10, max_value=400, value=60,
        step=10, key="cmp_npoints",
        help="How many times the ΔP history is sampled at. Real "
             "core-flood logs are coarser than the solver's timestep.",
    )
with c2:
    n_starts = st.number_input(
        "Multi-start LM runs", min_value=2, max_value=20, value=5,
        step=1, key="cmp_nstarts",
        help="Independent Levenberg-Marquardt fits from Latin-hypercube "
             "sampled starting points across the parameter bounds.",
    )
    lm_effort = st.number_input(
        "LM effort (per run)", min_value=50, max_value=2000, value=300,
        step=50, key="cmp_lm_effort",
        help="Passed to fit_kr as max_nfev. 300 is comfortable for a "
             "4-parameter Corey fit.",
    )
with c3:
    include_de = st.checkbox(
        "Add a Differential Evolution control run", value=False,
        key="cmp_include_de",
        help="One global-optimizer run on the same data, to show that a "
             "global method does not resolve the degeneracy. Expensive — "
             "typically several times the cost of all the LM runs "
             "combined.",
    )
    de_effort = st.number_input(
        "DE effort", min_value=50, max_value=2000, value=300, step=50,
        key="cmp_de_effort", disabled=not include_de,
    )

c4, c5, c6 = st.columns(3)
with c4:
    force_pc_off = st.checkbox(
        "Force capillary pressure off", value=True, key="cmp_pc_off",
        help="Recommended. The degeneracy does not depend on Pc, and "
             "Skjaeveland Pc multiplies the cost of every forward call "
             "by roughly five.",
    )
with c5:
    n_cells = st.number_input(
        "Grid cells N (override)", min_value=20, max_value=200,
        value=min(40, int(tp_inputs["numerical"]["N_cells"])), step=10,
        key="cmp_ncells",
        help="Overrides the Two-Phase page's N for comparison runs. "
             "40 is the tested value for staying inside the hosted "
             "execution wall.",
    )
with c6:
    seed = st.number_input(
        "Random seed", min_value=0, max_value=9999, value=0, step=1,
        key="cmp_seed",
        help="Controls both the noise realisation and the sampled "
             "starting points. Change it to confirm the conclusion is "
             "not an artefact of one draw.",
    )

# ── Cost guidance ──────────────────────────────────────────────────────────
est_calls_lm = int(n_starts) * min(int(lm_effort), (n_free + 1) * 25)
est_calls_de = (int(de_effort) // 12) * 12 * max(n_free, 1) + 100 \
    if include_de else 0
st.caption(
    f"Rough budget: ~{est_calls_lm + est_calls_de} forward calls "
    f"({n_free} free parameter(s), {int(n_starts)} LM run(s)"
    + (", 1 DE run" if include_de else "") + "). "
    "Per-call cost is measured when the dataset is generated, and the "
    "projected total is shown before the fits start."
)

if n_free == 0:
    st.markdown('<div class="error-box">⛔ Select at least one free '
                'parameter.</div>', unsafe_allow_html=True)

run_clicked = st.button("▶  Run Comparison", type="primary",
                        disabled=(n_free == 0), key="cmp_run")

# ── Execute ────────────────────────────────────────────────────────────────
if run_clicked:
    inp = prepare_inputs(tp_inputs, force_pc_off=force_pc_off,
                         n_cells=int(n_cells))

    status = st.empty()
    bar    = st.progress(0.0)

    status.info("Generating the synthetic dataset at the truth "
                "parameters…")
    try:
        data = generate_synthetic_data(
            inp, n_points=int(n_points), noise_pct=float(noise_pct),
            seed=int(seed),
        )
    except Exception as e:
        bar.empty()
        status.empty()
        st.error(f"Forward run failed: {e}")
        st.stop()

    per_call = data["forward_time_s"]
    projected_s = per_call * (est_calls_lm + est_calls_de)
    status.info(
        f"Forward call ≈ {per_call:.2f} s · projected total "
        f"≈ {projected_s / 60.0:.1f} min · noise σ = "
        f"{data['sigma_mbar']:.3f} mbar"
    )

    if projected_s > 20 * 60:
        bar.empty()
        st.error(
            f"Projected runtime is {projected_s / 60.0:.0f} minutes, "
            f"which risks hitting the hosted execution wall and losing "
            f"the whole run. Reduce the number of starts, lower N, turn "
            f"off the DE control, or free fewer parameters, then try "
            f"again."
        )
        st.stop()

    total_runs = int(n_starts) + (1 if include_de else 0)

    def _on_run(i, total, label):
        bar.progress(min((i - 1) / total, 1.0))
        status.info(f"Run {i}/{total} — {label} "
                    f"(forward call ≈ {per_call:.2f} s)")

    try:
        runs = run_comparison(
            inp, data, fit_mask,
            n_starts=int(n_starts), max_iter=int(lm_effort),
            include_de=bool(include_de), de_max_iter=int(de_effort),
            seed=int(seed), on_run=_on_run,
        )
    except Exception as e:
        bar.empty()
        status.empty()
        st.error(f"Comparison failed: {e}")
        st.stop()

    truth_kr = attach_kr_curves(runs, inp)

    bar.empty()
    status.empty()

    st.session_state["cmp_data"]     = data
    st.session_state["cmp_runs"]     = runs
    st.session_state["cmp_truth_kr"] = truth_kr
    st.session_state["cmp_mask"]     = list(fit_mask)
    st.session_state["cmp_truth"]    = dict(truth)

# ── Results ────────────────────────────────────────────────────────────────
runs = st.session_state.get("cmp_runs")
if runs:
    data      = st.session_state["cmp_data"]
    truth_kr  = st.session_state["cmp_truth_kr"]
    mask      = st.session_state["cmp_mask"]
    truth_now = st.session_state["cmp_truth"]
    free_now  = [n for n, m in zip(param_names, mask) if m]

    ok = [r for r in runs if r["fitted_params"] is not None]
    chi2s = np.array([r["chi2_red"] for r in ok], dtype=float)
    chi2s = chi2s[np.isfinite(chi2s)]

    st.markdown('<div class="section-label">▌ RESULT</div>',
                unsafe_allow_html=True)

    if chi2s.size:
        st.markdown(
            f'<div class="metric-card">'
            f'<b>Runs:</b> <code>{len(ok)}/{len(runs)}</code> completed · '
            f'<b>noise σ:</b> <code>{data["sigma_mbar"]:.3f} mbar</code> · '
            f'<b>reduced χ²:</b> <code>{chi2s.min():.2f}</code> → '
            f'<code>{chi2s.max():.2f}</code><br>'
            f'A reduced χ² near 1 means the run fits the data as well as '
            f'the noise permits. Runs that agree here but disagree on '
            f'parameters are the evidence of ill-posedness.'
            f'</div>',
            unsafe_allow_html=True,
        )

    components.html(
        render_chart_html(build_comparison_history_chart(data, runs)),
        height=460, scrolling=False,
    )
    components.html(
        render_chart_html(build_comparison_kr_chart(
            truth_kr, runs,
            inj_name=tp_inputs["injected"]["name"],
            disp_name=tp_inputs["displaced"]["name"],
        )),
        height=470, scrolling=False,
    )
    components.html(
        render_chart_html(build_parameter_spread_chart(
            runs, truth_now, free_now, model_cls,
        )),
        height=450, scrolling=False,
    )

    st.markdown('<div class="section-label">▌ RUNS</div>',
                unsafe_allow_html=True)
    table = results_table(runs, truth_now, param_names, mask)
    st.dataframe(table, use_container_width=True, hide_index=True)

    chi2_tol = st.number_input(
        "Spread summary — include runs with reduced χ² at or below",
        min_value=0.5, max_value=100.0, value=2.0, step=0.5,
        key="cmp_chi2_tol",
        help="A run that fitted badly is not evidence of degeneracy, "
             "just of a failed fit. This filters the spread summary to "
             "runs that actually matched the data.",
    )
    spread = spread_summary(runs, truth_now, param_names, mask,
                            chi2_tol=float(chi2_tol))
    st.dataframe(spread, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇  Download comparison runs (.csv)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="optimizer_comparison_runs.csv",
        mime="text/csv", key="cmp_download",
    )

    with st.expander("Optimizer messages", expanded=False):
        for r in runs:
            st.markdown(
                f"**{r['label']}** — {r['n_calls']} calls, "
                f"{r['wall_s']:.1f} s — {r['message']}"
            )