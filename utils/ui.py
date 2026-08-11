"""
Shared UI helpers for CoreFlood Lab pages.

Provides:
    - inject_shared_css()  : Streamlit CSS block used by every page.
    - render_header(title) : the teal-dot page header.
    - input_row(...)       : the standard "label / number / unit" row.
    - dim_row(...)         : the "label / number" row (no unit selector).
    - injection_rate_row() : the "label / number / volume / time" row.

None of these functions perform unit conversion themselves — they just
return the raw numeric values and unit strings for the caller to convert
via utils.units.convert / convert_injection_rate.
"""

import streamlit as st


# ── CSS block ──────────────────────────────────────────────────────────────
_SHARED_CSS = """
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
.status-ready  { color: #6B7785 !important; border-color: #1F2A33 !important; }

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
.error-box {
    background: #2A0F0F; border-left: 3px solid #DC2626;
    padding: 10px 14px; margin: 0.5rem 0;
    color: #FCA5A5; font-size: 12px;
}
.metric-card {
    background: #0F1A1F; border-left: 4px solid #2DD4BF;
    padding: 14px 18px; margin: 0.4rem 0;
    color: #C9D1D9; font-size: 13px;
}
.metric-card b    { color: #F0F4F8; }
.metric-card code { color: #2DD4BF; }

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
"""


def inject_shared_css():
    """Emit the shared CSS block. Call once, near the top of each page."""
    st.markdown(_SHARED_CSS, unsafe_allow_html=True)


def render_header(title):
    """Render the teal-dot header bar used by every page."""
    st.markdown(
        f"""
        <div class="app-header">
          <span class="dot-teal">●</span>
          <span class="app-title">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Input rows ──────────────────────────────────────────────────────────────
def input_row(label, default, units, key_prefix, fmt="%g",
              default_unit=None, help=None):
    """
    Render a row with (label, number input, unit selector).

    Returns (value, unit_string). Caller is responsible for converting
    `value` using the unit and its base table.
    """
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


def dim_row(label, default, key_prefix, fmt="%g",
            min_value=None, max_value=None, step=None, help=None):
    """
    Render a row with (label, number input). No unit selector — used for
    dimensionless quantities (Corey exponents, ratios, integer counts).
    """
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


def injection_rate_row(label, default, vol_units, time_units, key_prefix,
                       default_vol="ml", default_time="min"):
    """
    Render a row with (label, number, volume-unit, time-unit) — for an
    injection rate expressed as volume/time.

    Returns (value, volume_unit, time_unit).
    """
    c1, c2, c3, c4 = st.columns([1.2, 1.5, 0.75, 0.75])
    with c1:
        st.markdown(f'<div class="row-label">{label}</div>',
                    unsafe_allow_html=True)
    with c2:
        v = st.number_input(
            label, value=float(default), key=f"{key_prefix}_val",
            label_visibility="collapsed", format="%g",
        )
    with c3:
        vlist = list(vol_units.keys())
        vidx = vlist.index(default_vol) if default_vol in vlist else 0
        vu = st.selectbox(
            "vol", vlist, index=vidx, key=f"{key_prefix}_vol",
            label_visibility="collapsed",
        )
    with c4:
        tlist = list(time_units.keys())
        tidx = tlist.index(default_time) if default_time in tlist else 0
        tu = st.selectbox(
            "per", tlist, index=tidx, key=f"{key_prefix}_time",
            label_visibility="collapsed",
        )
    return v, vu, tu