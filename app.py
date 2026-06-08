"""
CoreFlood Lab — Permeability Matching Tool
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.io as pio

from calculators.absolute_perm import match_permeability
from utils.units import (
    LENGTH_TO_CM, AREA_TO_CM2, POROSITY_TO_FRACTION,
    VISCOSITY_TO_CP, COMPRESSIBILITY_TO_INV_PA,
    VOLUME_TO_ML, TIME_TO_MIN, PRESSURE_TO_BAR, DP_TO_MBAR,
    convert, convert_injection_rate,
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CoreFlood Lab — Permeability",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Hide Streamlit chrome + custom CSS ──────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] {display: none;}
    [data-testid="stSidebar"]    {display: none;}
    #MainMenu          {visibility: hidden;}
    header             {visibility: hidden;}
    footer             {visibility: hidden;}
    .stDeployButton    {display: none;}
    [data-testid="stToolbar"]    {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}

    html, body, [class*="css"] {
        font-family: 'Courier New', monospace;
    }

    .section-label {
        color: #6B7785;
        font-size: 12px;
        letter-spacing: 0.12em;
        margin: 1rem 0 0.5rem 0;
    }
    .group-label {
        color: #FB923C;
        font-size: 13px;
        margin: 0.75rem 0 0.25rem 0;
    }
    .row-label {
        color: #9CA3AF;
        font-size: 13px;
        padding-top: 0.5rem;
    }
    .result-box {
        background: #0F1A1F;
        border-left: 3px solid #2DD4BF;
        padding: 12px 16px;
        margin-bottom: 14px;
    }
    .result-label {
        color: #6B7785;
        font-size: 11px;
        letter-spacing: 0.1em;
    }
    .result-k {
        color: #2DD4BF;
        font-size: 26px;
        font-weight: 500;
        margin-top: 4px;
    }
    .status-pill {
        display: inline-block;
        padding: 8px 14px;
        background: #0F1A1F;
        border: 1px solid #14392E;
        color: #2DD4BF;
        font-size: 12px;
        letter-spacing: 0.08em;
    }
    .status-ready  {color: #6B7785 !important; border-color: #1F2A33 !important;}

    .app-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-bottom: 14px;
        border-bottom: 1px solid #1F2A33;
        margin-bottom: 1rem;
    }
    .dot-teal {color: #2DD4BF;}
    .app-title {color: #F0F4F8; font-size: 16px;}

    .stButton > button[kind="primary"] {
        background: #2DD4BF;
        color: #0B1014;
        border: none;
        font-family: 'Courier New', monospace;
        font-weight: 600;
        letter-spacing: 0.08em;
    }
    .stButton > button[kind="primary"]:hover {
        background: #14B8A6;
        color: #0B1014;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
      <span class="dot-teal">●</span>
      <span class="app-title">CoreFlood Lab — Permeability Matching Tool</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_abs, tab_about = st.tabs(["Absolute Permeability", "About Model"])


# ── Input row helpers ───────────────────────────────────────────────────────
def input_row(label, default, units, key_prefix, fmt="%g", default_unit=None):
    c1, c2, c3 = st.columns([1.2, 1.5, 1])
    with c1:
        st.markdown(f'<div class="row-label">{label}</div>', unsafe_allow_html=True)
    with c2:
        value = st.number_input(
            label, value=float(default), key=f"{key_prefix}_val",
            label_visibility="collapsed", format=fmt,
        )
    with c3:
        unit_list = list(units.keys())
        idx = unit_list.index(default_unit) if default_unit and default_unit in unit_list else 0
        unit = st.selectbox(
            label, unit_list, index=idx, key=f"{key_prefix}_unit",
            label_visibility="collapsed",
        )
    return value, unit


def injection_rate_row(label, default, vol_units, time_units, key_prefix,
                       default_vol="ml", default_time="min"):
    c1, c2, c3, c4 = st.columns([1.2, 1.5, 0.75, 0.75])
    with c1:
        st.markdown(f'<div class="row-label">{label}</div>', unsafe_allow_html=True)
    with c2:
        value = st.number_input(
            label, value=float(default), key=f"{key_prefix}_val",
            label_visibility="collapsed", format="%g",
        )
    with c3:
        vol_list = list(vol_units.keys())
        vol_idx = vol_list.index(default_vol) if default_vol in vol_list else 0
        vol_unit = st.selectbox(
            "vol", vol_list, index=vol_idx, key=f"{key_prefix}_vol",
            label_visibility="collapsed",
        )
    with c4:
        time_list = list(time_units.keys())
        time_idx = time_list.index(default_time) if default_time in time_list else 0
        time_unit = st.selectbox(
            "per", time_list, index=time_idx, key=f"{key_prefix}_time",
            label_visibility="collapsed",
        )
    return value, vol_unit, time_unit


# ── Plotly chart builders ───────────────────────────────────────────────────
AXIS_TITLE_FONT = dict(color="#0B1014", family="Courier New", size=15)
AXIS_TICK_FONT  = dict(color="#0B1014", family="Courier New", size=13)


def _apply_layout(fig, t_max, y_max, title):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Courier New", color="#111827", size=14)),
        xaxis=dict(
            title=dict(text="<b>Time [min]</b>", font=AXIS_TITLE_FONT),
            tickfont=AXIS_TICK_FONT,
            gridcolor="#D1D5DB",
            zerolinecolor="#0B1014",
            linecolor="#0B1014",
            range=[0, t_max],
        ),
        yaxis=dict(
            title=dict(text="<b>ΔP [mbar]</b>", font=AXIS_TITLE_FONT),
            tickfont=AXIS_TICK_FONT,
            gridcolor="#D1D5DB",
            zerolinecolor="#0B1014",
            linecolor="#0B1014",
            range=[0, y_max],
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Courier New"),
        height=420,
        margin=dict(l=80, r=40, t=50, b=70),
        showlegend=False,
    )


def build_empty_chart(t_max, target_dp):
    fig = go.Figure()
    fig.add_hline(
        y=target_dp,
        line=dict(color="#DC2626", width=2, dash="dash"),
        annotation_text=f"<b>target {target_dp:.2f} mbar</b>",
        annotation_position="top right",
        annotation_font=dict(color="#DC2626", family="Courier New", size=12),
    )
    _apply_layout(fig, t_max, max(target_dp * 1.25, 1.0), "Awaiting simulation…")
    return fig


def build_animated_chart(t_arr, dp_arr, t_max, target_dp, k_value):
    n_total = len(t_arr)
    n_frames = 60
    step = max(1, n_total // n_frames)

    frame_indices = list(range(2, n_total, step))
    if not frame_indices or frame_indices[-1] != n_total:
        frame_indices.append(n_total)

    frames = [
        go.Frame(
            data=[go.Scatter(
                x=list(t_arr[:i]),
                y=list(dp_arr[:i]),
                mode="lines",
                line=dict(color="#16A34A", width=2.8),
            )],
            name=str(i),
        )
        for i in frame_indices
    ]

    fig = go.Figure(
        data=[go.Scatter(
            x=[t_arr[0]], y=[dp_arr[0]],
            mode="lines",
            line=dict(color="#16A34A", width=2.8),
            name="simulated",
            hovertemplate="t = %{x:.2f} min<br>ΔP = %{y:.2f} mbar<extra></extra>",
        )],
        frames=frames,
    )

    fig.add_hline(
        y=target_dp,
        line=dict(color="#DC2626", width=2, dash="dash"),
        annotation_text=f"<b>target {target_dp:.2f} mbar</b>",
        annotation_position="top right",
        annotation_font=dict(color="#DC2626", family="Courier New", size=12),
    )

    y_top = max(target_dp * 1.25, max(dp_arr) * 1.1)
    _apply_layout(fig, t_max, y_top, f"History-Matched Result — k = {k_value:.2f} mD")
    return fig


def render_chart_html(fig, autoplay=False):
    """Convert a Plotly figure to standalone HTML; optionally autoplay frames."""
    html_str = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        config={"displayModeBar": False, "responsive": True},
    )
    if autoplay:
        html_str += """
<script>
(function(){
    function tryStart(attempts){
        var plots = document.getElementsByClassName('plotly-graph-div');
        if ((plots.length === 0 || typeof Plotly === 'undefined') && attempts < 30) {
            setTimeout(function(){ tryStart(attempts + 1); }, 100);
            return;
        }
        if (plots.length > 0 && typeof Plotly !== 'undefined') {
            Plotly.animate(plots[plots.length - 1], null, {
                frame: {duration: 30, redraw: true},
                transition: {duration: 0},
                mode: 'immediate'
            });
        }
    }
    setTimeout(function(){ tryStart(0); }, 200);
})();
</script>
"""
    return html_str


# ── Absolute Permeability tab ───────────────────────────────────────────────
with tab_abs:
    left, right = st.columns([1.4, 1])

    with left:
        st.markdown('<div class="section-label">▌ INPUT PARAMETERS</div>', unsafe_allow_html=True)

        st.markdown('<div class="group-label">Core Geometry</div>', unsafe_allow_html=True)
        L_val,   L_unit   = input_row("Length",   6.0,    LENGTH_TO_CM,        "L")
        A_val,   A_unit   = input_row("Area",     11.4,   AREA_TO_CM2,         "A")
        phi_val, phi_unit = input_row("Porosity", 0.20,   POROSITY_TO_FRACTION,"phi")

        st.markdown('<div class="group-label">Fluid Properties</div>', unsafe_allow_html=True)
        mu_val, mu_unit = input_row("Viscosity",       1.0,     VISCOSITY_TO_CP,           "mu")
        c_val,  c_unit  = input_row("Compressibility", 4.5e-10, COMPRESSIBILITY_TO_INV_PA, "c", fmt="%.2e")

        st.markdown('<div class="group-label">Boundary Conditions</div>', unsafe_allow_html=True)
        q_val, q_vol_unit, q_time_unit = injection_rate_row(
            "Injection rate", 1.0, VOLUME_TO_ML, TIME_TO_MIN, "q",
        )
        p_val,  p_unit  = input_row("Back pressure", 10.0, PRESSURE_TO_BAR, "p")
        dp_val, dp_unit = input_row("Target ΔP",     50.0, DP_TO_MBAR,      "dp")
        t_val,  t_unit  = input_row("Sim time",      60.0, TIME_TO_MIN,     "t", default_unit="min")

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("▶ RUN MATCH", type="primary")

    # ── Compute (only if Run pressed) ───────────────────────────────────────
    result = None
    if run:
        try:
            L_cm           = convert(L_val,   L_unit,   LENGTH_TO_CM)
            A_cm2          = convert(A_val,   A_unit,   AREA_TO_CM2)
            phi            = convert(phi_val, phi_unit, POROSITY_TO_FRACTION)
            mu_cP          = convert(mu_val,  mu_unit,  VISCOSITY_TO_CP)
            c_Pa           = convert(c_val,   c_unit,   COMPRESSIBILITY_TO_INV_PA)
            q_ml_min       = convert_injection_rate(q_val, q_vol_unit, q_time_unit)
            P_out_bar      = convert(p_val,   p_unit,   PRESSURE_TO_BAR)
            target_dp_mbar = convert(dp_val,  dp_unit,  DP_TO_MBAR)
            total_time_min = convert(t_val,   t_unit,   TIME_TO_MIN)
        except ValueError as e:
            st.error(f"Unit error: {e}")
            st.stop()

        with st.spinner("Running Nelder-Mead optimisation…"):
            k, t_arr, dp_arr, n_iter = match_permeability(
                L_cm=L_cm, A_cm2=A_cm2, phi=phi,
                mu_cP=mu_cP, c_Pa=c_Pa, q_ml_min=q_ml_min,
                P_out_bar=P_out_bar,
                total_time_mins=total_time_min, dt_mins=1.0,
                target_dp_mbar=target_dp_mbar,
            )

        final_dp  = dp_arr[-1]
        error_pct = abs(final_dp - target_dp_mbar) / target_dp_mbar * 100.0
        result = {
            "k": k, "t_arr": t_arr, "dp_arr": dp_arr,
            "n_iter": n_iter, "final_dp": final_dp, "error_pct": error_pct,
            "target_dp": target_dp_mbar, "total_time": total_time_min,
        }

    # ── Right column: results ───────────────────────────────────────────────
    with right:
        st.markdown('<div class="section-label">▌ MATCHING RESULT</div>', unsafe_allow_html=True)

        if result:
            st.markdown(
                f"""
                <div class="result-box">
                  <div class="result-label">OPTIMIZED PERMEABILITY</div>
                  <div class="result-k">k = {result['k']:.2f} mD</div>
                </div>
                <div style="font-size: 13px; line-height: 1.9;">
                  <div style="display: flex; justify-content: space-between;"><span style="color:#9CA3AF;">Target ΔP</span><span>{result['target_dp']:.2f} mbar</span></div>
                  <div style="display: flex; justify-content: space-between;"><span style="color:#9CA3AF;">Final ΔP</span><span>{result['final_dp']:.2f} mbar</span></div>
                  <div style="display: flex; justify-content: space-between;"><span style="color:#9CA3AF;">Error</span><span style="color:#2DD4BF;">{result['error_pct']:.2f} %</span></div>
                  <div style="display: flex; justify-content: space-between;"><span style="color:#9CA3AF;">Iterations</span><span>{result['n_iter']}</span></div>
                </div>
                <br>
                <div class="status-pill">● STATUS: MATCH ACHIEVED</div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-box">
                  <div class="result-label">OPTIMIZED PERMEABILITY</div>
                  <div class="result-k">— mD</div>
                </div>
                <div class="status-pill status-ready">○ STATUS: READY</div>
                """,
                unsafe_allow_html=True,
            )

    # ── Chart (full-width, below the columns) ───────────────────────────────
    st.markdown('<div class="section-label">▌ DIFFERENTIAL PRESSURE vs TIME</div>', unsafe_allow_html=True)

    if result:
        fig = build_animated_chart(
            result["t_arr"], result["dp_arr"],
            result["total_time"], result["target_dp"], result["k"],
        )
        components.html(render_chart_html(fig, autoplay=True), height=460)
    else:
        try:
            t_max_init     = convert(t_val,  t_unit,  TIME_TO_MIN)
            dp_target_init = convert(dp_val, dp_unit, DP_TO_MBAR)
        except ValueError:
            t_max_init     = 60.0
            dp_target_init = 50.0
        fig = build_empty_chart(t_max_init, dp_target_init)
        components.html(render_chart_html(fig, autoplay=False), height=460)


# ── About Model tab ──────────────────────────────────────────────────────────
with tab_about:
    st.markdown(
        """
        ### About the Model

        This tool performs **automated history matching** for single-phase, slightly
        compressible flow through a 1D porous core sample.

        **Governing equation — Darcy's Law:**

        $$ q = -\\,\\frac{k\\,A}{\\mu}\\,\\frac{\\Delta P}{L} $$

        **Numerical method:**
        - 1D finite-difference discretisation (20 cells)
        - Implicit time stepping (backward Euler)
        - Sparse linear solver per time step (`scipy.sparse.linalg.spsolve`)

        **History matching:**
        Given a target ΔP measured in the lab, the **Nelder-Mead** optimiser searches
        for the permeability *k* that produces a steady-state ΔP matching the target
        within a 1e-4 mbar tolerance.

        **Units:**
        All inputs are converted internally to the solver's expected units
        (cm, cm², cP, 1/Pa, ml/min, bar, mbar, min).
        """
    )
