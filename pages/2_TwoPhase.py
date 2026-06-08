"""
CoreFlood Lab — Two-Phase (Relative Permeability) Tool
Step 2: Phase selection. Simulation logic in upcoming steps.
"""

import streamlit as st

from utils.phases import phase_picker, io_buttons

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
    padding: 24px 28px; margin: 0.6rem 0;
    color: #C9D1D9 !important;
    font-size: 22px !important; line-height: 1.9 !important;
}
.debug-box * {
    font-size: 22px !important;
    line-height: 1.9 !important;
}
.debug-box code { color: #2DD4BF !important; }
.debug-box b    { color: #F0F4F8 !important; }
.debug-box code { color: #2DD4BF; }
.warn-box {
    background: #2A1F0F; border-left: 3px solid #FB923C;
    padding: 10px 14px; margin: 0.5rem 0;
    color: #FB923C; font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <span class="dot-teal">●</span>
  <span class="app-title">CoreFlood Lab — Two-Phase (Relative Permeability)</span>
</div>
""", unsafe_allow_html=True)

# ── Optional: custom phase library import/export ────────────────────────────
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

# ── Fluid pair sanity check ─────────────────────────────────────────────────
if injected["type"] == displaced["type"]:
    st.markdown(
        f'<div class="warn-box">⚠ Unusual fluid pair — both phases are '
        f'<b>{injected["type"]}</b>. Typical relative-permeability tests '
        f'use one gas + one liquid.</div>',
        unsafe_allow_html=True,
    )

# ── Debug readout (temporary — verifies Step 2 works) ───────────────────────
st.markdown('<div class="section-label">▌ SELECTED (debug readout)</div>',
            unsafe_allow_html=True)
st.markdown(
    f'<div class="debug-box">'
    f'<b>Injected:</b> {injected["name"]} ({injected["type"]}) — '
    f'ρ = <code>{injected["density_kg_m3"]:.3f} kg/m³</code>, '
    f'μ = <code>{injected["viscosity_cP"]:.4f} cP</code><br>'
    f'<b>Displaced:</b> {displaced["name"]} ({displaced["type"]}) — '
    f'ρ = <code>{displaced["density_kg_m3"]:.3f} kg/m³</code>, '
    f'μ = <code>{displaced["viscosity_cP"]:.4f} cP</code>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Next-step note ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">▌ STATUS</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="debug-box">'
    'Step 2 complete. Coming next: Corey kr inputs, capillary-pressure '
    'toggle, core geometry, boundary conditions, then forward / inverse '
    'modes.'
    '</div>',
    unsafe_allow_html=True,
)