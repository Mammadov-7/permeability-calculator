"""
CoreFlood Lab — Two-Phase (Relative Permeability) Tool
Skeleton placeholder. Simulation logic added in subsequent build steps.
"""

import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CoreFlood Lab — Two-Phase",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Hide Streamlit chrome + custom CSS (matches PermCalc) ───────────────────
st.markdown(
    """
    <style>
    /* Hide auto-generated multipage sidebar */
    [data-testid="stSidebarNav"]  {display: none;}
    [data-testid="stSidebar"]     {display: none;}

    /* Hide Streamlit default chrome */
    #MainMenu                     {visibility: hidden;}
    header                        {visibility: hidden;}
    footer                        {visibility: hidden;}
    .stDeployButton               {display: none;}
    [data-testid="stToolbar"]     {visibility: hidden;}
    [data-testid="stDecoration"]  {display: none;}

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
    .app-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-bottom: 14px;
        border-bottom: 1px solid #1F2A33;
        margin-bottom: 1rem;
    }
    .dot-teal     {color: #2DD4BF;}
    .app-title    {color: #F0F4F8; font-size: 16px;}

    .placeholder-box {
        background: #0F1A1F;
        border-left: 3px solid #2DD4BF;
        padding: 16px 20px;
        margin: 1rem 0;
        color: #9CA3AF;
        font-size: 13px;
        line-height: 1.8;
    }
    .placeholder-box b {color: #2DD4BF;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
      <span class="dot-teal">●</span>
      <span class="app-title">CoreFlood Lab — Two-Phase (Relative Permeability)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Placeholder content ─────────────────────────────────────────────────────
st.markdown('<div class="section-label">▌ STATUS</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="placeholder-box">
      <b>● UNDER CONSTRUCTION</b><br><br>
      The skeleton page is live and routing works.<br>
      Upcoming build steps will add:<br>
      &nbsp;&nbsp;– Phase library (injected / resident phase pickers)<br>
      &nbsp;&nbsp;– Corey relative permeability parameters<br>
      &nbsp;&nbsp;– Optional capillary pressure (Brooks-Corey)<br>
      &nbsp;&nbsp;– 1D two-phase IMPES simulator<br>
      &nbsp;&nbsp;– Forward and Inverse modes
    </div>
    """,
    unsafe_allow_html=True,
)