"""
Plotly chart builders for the optimizer-comparison page (Step 3.1).

Kept separate from utils/plotting.py so the comparison work can evolve
without touching the charts the Forward and Inverse tabs depend on.
Style constants and the shared layout helper are imported from
utils.plotting so the house aesthetic stays in one place.

Three figures, in the order the argument is made:

    build_comparison_history_chart  every run's dP(t) over the noisy
                                    data. If the curves are visually
                                    indistinguishable, the data cannot
                                    tell the parameter sets apart.

    build_comparison_kr_chart       every run's kr curves over truth.
                                    This is the money figure: same fit
                                    quality, different curves.

    build_parameter_spread_chart    where each run landed per parameter,
                                    normalised to the parameter's own
                                    bounds so they share one axis.
"""

import numpy as np
import plotly.graph_objects as go

from utils.plotting import (
    COLOR_INJ, COLOR_DISP, COLOR_REF, COLOR_AXIS,
    TITLE_FONT, AXIS_TITLE_FONT, AXIS_TICK_FONT,
    _apply_layout,
)

COLOR_RUN = "#6B7280"       # grey — individual runs, deliberately quiet
COLOR_DE  = "#7C3AED"       # violet — the DE control, distinguishable


def _run_color(run):
    return COLOR_DE if run["optimizer"].startswith("Differential") \
        else COLOR_RUN


# -- 1. History match: all runs over the noisy data -------------------------
def build_comparison_history_chart(data, runs):
    """Noisy measurements as markers; every run's simulated dP as a line."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["t_min"], y=data["dp_noisy"], mode="markers",
        marker=dict(color=COLOR_REF, size=6,
                    line=dict(color=COLOR_AXIS, width=0.5)),
        name="measured (noisy)",
        hovertemplate="t=%{x:.2f} min<br>ΔP=%{y:.2f} mbar<extra></extra>",
    ))

    for r in runs:
        if r.get("sim_dp") is None:
            continue
        fig.add_trace(go.Scatter(
            x=r["sim_t"], y=r["sim_dp"], mode="lines",
            line=dict(color=_run_color(r), width=1.6),
            opacity=0.75,
            name=f"{r['label']} (χ²ᵣ={r['chi2_red']:.2f})",
            hovertemplate="t=%{x:.2f} min<br>ΔP=%{y:.2f} mbar<extra></extra>",
        ))

    all_dp = [data["dp_noisy"]] + [r["sim_dp"] for r in runs
                                   if r.get("sim_dp") is not None]
    y_max = max(float(np.nanmax(a)) for a in all_dp) * 1.10
    x_max = float(np.nanmax(data["t_min"]))
    _apply_layout(
        fig, "<b>Time [min]</b>", "<b>ΔP [mbar]</b>",
        (0.0, x_max), (0.0, y_max),
        "History match — every run against the same noisy data",
        showlegend=True, height=430,
    )
    return fig


# -- 2. kr curves: all runs over truth --------------------------------------
def build_comparison_kr_chart(truth_kr, runs,
                              inj_name="Injected", disp_name="Displaced"):
    """
    Fitted kr curves from every run, drawn faint, with the truth curves
    bold on top. Runs must already carry a "kr" entry (see
    utils.comparison.attach_kr_curves).
    """
    fig = go.Figure()
    shown_legend = {"run": False, "de": False}

    for r in runs:
        kr = r.get("kr")
        if kr is None:
            continue
        is_de = r["optimizer"].startswith("Differential")
        key = "de" if is_de else "run"
        col = _run_color(r)
        for series in ("kr_inj", "kr_disp"):
            fig.add_trace(go.Scatter(
                x=kr["S_inj"], y=kr[series], mode="lines",
                line=dict(color=col, width=1.3),
                opacity=0.55,
                name=("DE control" if is_de else "individual fits"),
                legendgroup=key,
                showlegend=(series == "kr_inj"
                            and not shown_legend[key]),
                hovertemplate=(f"{r['label']}<br>S_inj=%{{x:.3f}}"
                               f"<br>kr=%{{y:.4f}}<extra></extra>"),
            ))
        shown_legend[key] = True

    fig.add_trace(go.Scatter(
        x=truth_kr["S_inj"], y=truth_kr["kr_inj"], mode="lines",
        line=dict(color=COLOR_INJ, width=3.2, dash="dash"),
        name=f"truth kr_inj ({inj_name})",
        hovertemplate="S_inj=%{x:.3f}<br>kr=%{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=truth_kr["S_inj"], y=truth_kr["kr_disp"], mode="lines",
        line=dict(color=COLOR_DISP, width=3.2, dash="dash"),
        name=f"truth kr_disp ({disp_name})",
        hovertemplate="S_inj=%{x:.3f}<br>kr=%{y:.4f}<extra></extra>",
    ))

    y_candidates = [float(np.nanmax(truth_kr["kr_inj"])),
                    float(np.nanmax(truth_kr["kr_disp"]))]
    for r in runs:
        if r.get("kr") is not None:
            y_candidates.append(float(np.nanmax(r["kr"]["kr_inj"])))
            y_candidates.append(float(np.nanmax(r["kr"]["kr_disp"])))
    y_max = max(max(y_candidates), 1e-3) * 1.05

    _apply_layout(
        fig, "<b>S_inj  [-]</b>", "<b>kr  [-]</b>",
        (0.0, 1.0), (0.0, y_max),
        "Fitted kr curves — all runs vs truth",
        showlegend=True, height=440,
    )
    return fig


# -- 3. Parameter spread ----------------------------------------------------
def build_parameter_spread_chart(runs, truth_params, free_names,
                                 model_cls):
    """
    One column per free parameter. Each run is a dot at its fitted value,
    normalised to (value - lo) / (hi - lo) using that parameter's own
    bounds, so every parameter shares a single 0-1 axis. Truth is a red
    cross. Tight clusters mean the parameter is identifiable; vertical
    scatter means it is not.
    """
    fig = go.Figure()
    x_positions = list(range(len(free_names)))

    def _norm(name, v):
        lo, hi = model_cls.parameter_bounds[name]
        return (float(v) - lo) / (hi - lo) if hi > lo else 0.5

    seen = set()
    for r in runs:
        fp = r.get("fitted_params")
        if fp is None:
            continue
        is_de = r["optimizer"].startswith("Differential")
        key = "de" if is_de else "run"
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=[_norm(n, fp[n]) for n in free_names],
            mode="markers",
            marker=dict(color=_run_color(r), size=11, opacity=0.75,
                        line=dict(color=COLOR_AXIS, width=0.6)),
            name=("DE control" if is_de else "individual fits"),
            legendgroup=key,
            showlegend=key not in seen,
            text=[f"{r['label']}  {n}={fp[n]:.4g}" for n in free_names],
            hovertemplate="%{text}<extra></extra>",
        ))
        seen.add(key)

    fig.add_trace(go.Scatter(
        x=x_positions,
        y=[_norm(n, truth_params[n]) for n in free_names],
        mode="markers",
        marker=dict(color=COLOR_REF, size=16, symbol="x",
                    line=dict(width=2)),
        name="truth",
        text=[f"truth  {n}={truth_params[n]:.4g}" for n in free_names],
        hovertemplate="%{text}<extra></extra>",
    ))

    _apply_layout(
        fig, "<b>parameter</b>",
        "<b>position within bounds  [-]</b>",
        (-0.5, len(free_names) - 0.5), (-0.05, 1.05),
        "Where each run landed (normalised to each parameter's bounds)",
        showlegend=True, height=420,
    )
    fig.update_layout(xaxis=dict(
        tickmode="array",
        tickvals=x_positions,
        ticktext=free_names,
        title=dict(text="<b>parameter</b>", font=AXIS_TITLE_FONT),
        tickfont=AXIS_TICK_FONT,
        gridcolor="#D1D5DB", zerolinecolor=COLOR_AXIS,
        linecolor=COLOR_AXIS, range=[-0.5, len(free_names) - 0.5],
    ))
    return fig