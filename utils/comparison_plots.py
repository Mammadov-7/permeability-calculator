"""
Plotly chart builders for the optimizer-comparison page (Step 3.1).

Kept separate from utils/plotting.py so the comparison work can evolve
without touching the charts the Forward and Inverse tabs depend on.
Style constants and the shared layout helper are imported from
utils.plotting so the house aesthetic stays in one place.

Three figures, in the order the argument is made:

    build_history_overlay     every optimizer's dP(t) over the measured
                              data. If the curves are visually
                              indistinguishable, the data cannot tell
                              the parameter sets apart.

    build_kr_overlay          every optimizer's kr curves, with truth
                              dashed when it is known. The money figure:
                              same fit quality, different curves.

    build_parameter_chart     where each optimizer landed per parameter,
                              normalised to that parameter's own bounds
                              so they share one axis.

Each optimizer keeps one colour across all three charts, so a reader can
follow a single algorithm from figure to figure.
"""

import numpy as np
import plotly.graph_objects as go

from utils.plotting import (
    COLOR_REF, COLOR_AXIS, AXIS_TITLE_FONT, AXIS_TICK_FONT,
    _apply_layout,
)

# One colour per optimizer, held constant across every chart.
OPTIMIZER_COLORS = {
    "Differential Evolution":  "#7C3AED",   # violet
    "Nelder-Mead":             "#FB923C",   # orange
    "Levenberg-Marquardt":     "#2DD4BF",   # teal
}
COLOR_FALLBACK = "#6B7280"                  # grey, for multi-start labels
COLOR_TRUTH = COLOR_REF                     # red


def _color(run):
    return OPTIMIZER_COLORS.get(run["optimizer"], COLOR_FALLBACK)


# -- 1. History overlay -----------------------------------------------------
def build_history_overlay(data, runs):
    """Measured dP as markers; each run's simulated dP as a line."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["t_min"], y=data["dp_measured"], mode="markers",
        marker=dict(color=COLOR_TRUTH, size=6,
                    line=dict(color=COLOR_AXIS, width=0.5)),
        name="measured",
        hovertemplate="t=%{x:.2f} min<br>dP=%{y:.2f} mbar<extra></extra>",
    ))

    for r in runs:
        if r.get("sim_dp") is None:
            continue
        fig.add_trace(go.Scatter(
            x=r["sim_t"], y=r["sim_dp"], mode="lines",
            line=dict(color=_color(r), width=2.0),
            opacity=0.85,
            name=f"{r['label']} (RMSE {r['rmse']:.2f})",
            hovertemplate="t=%{x:.2f} min<br>dP=%{y:.2f} mbar<extra></extra>",
        ))

    series = [data["dp_measured"]] + [r["sim_dp"] for r in runs
                                      if r.get("sim_dp") is not None]
    y_max = max(float(np.nanmax(a)) for a in series) * 1.10
    x_max = float(np.nanmax(data["t_min"]))
    _apply_layout(
        fig, "<b>Time [min]</b>", "<b>dP [mbar]</b>",
        (0.0, x_max), (0.0, y_max),
        "History match — every optimizer against the same data",
        showlegend=True, height=430,
    )
    return fig


# -- 2. kr overlay ----------------------------------------------------------
def build_kr_overlay(runs, truth_kr=None,
                     inj_name="Injected", disp_name="Displaced"):
    """
    Fitted kr curves from each run — solid for the injected phase, dotted
    for the displaced phase, one colour per optimizer. Truth, when known,
    is drawn thick and dashed in red on top.

    Runs must already carry a "kr" entry (see
    utils.comparison.attach_kr_curves).
    """
    fig = go.Figure()

    for r in runs:
        kr = r.get("kr")
        if kr is None:
            continue
        col = _color(r)
        fig.add_trace(go.Scatter(
            x=kr["S_inj"], y=kr["kr_inj"], mode="lines",
            line=dict(color=col, width=2.0),
            name=f"{r['label']} — inj",
            hovertemplate=(f"{r['label']}<br>S_inj=%{{x:.3f}}"
                           f"<br>kr_inj=%{{y:.4f}}<extra></extra>"),
        ))
        fig.add_trace(go.Scatter(
            x=kr["S_inj"], y=kr["kr_disp"], mode="lines",
            line=dict(color=col, width=2.0, dash="dot"),
            name=f"{r['label']} — disp",
            hovertemplate=(f"{r['label']}<br>S_inj=%{{x:.3f}}"
                           f"<br>kr_disp=%{{y:.4f}}<extra></extra>"),
        ))

    y_candidates = []
    for r in runs:
        if r.get("kr") is not None:
            y_candidates.append(float(np.nanmax(r["kr"]["kr_inj"])))
            y_candidates.append(float(np.nanmax(r["kr"]["kr_disp"])))

    if truth_kr is not None:
        fig.add_trace(go.Scatter(
            x=truth_kr["S_inj"], y=truth_kr["kr_inj"], mode="lines",
            line=dict(color=COLOR_TRUTH, width=3.4, dash="dash"),
            name=f"truth — inj ({inj_name})",
            hovertemplate="S_inj=%{x:.3f}<br>kr=%{y:.4f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=truth_kr["S_inj"], y=truth_kr["kr_disp"], mode="lines",
            line=dict(color=COLOR_TRUTH, width=3.4, dash="dashdot"),
            name=f"truth — disp ({disp_name})",
            hovertemplate="S_inj=%{x:.3f}<br>kr=%{y:.4f}<extra></extra>",
        ))
        y_candidates.append(float(np.nanmax(truth_kr["kr_inj"])))
        y_candidates.append(float(np.nanmax(truth_kr["kr_disp"])))

    y_max = max(max(y_candidates) if y_candidates else 1.0, 1e-3) * 1.05
    title = ("Fitted kr curves — all optimizers vs truth"
             if truth_kr is not None else
             "Fitted kr curves — all optimizers")
    _apply_layout(
        fig, "<b>S_inj  [-]</b>", "<b>kr  [-]</b>",
        (0.0, 1.0), (0.0, y_max), title,
        showlegend=True, height=450,
    )
    return fig


# -- 3. Parameter chart -----------------------------------------------------
def build_parameter_chart(runs, free_names, model_cls, truth_params=None):
    """
    One column per free parameter. Each run is a dot at its fitted value,
    normalised as (value - lo) / (hi - lo) using that parameter's own
    bounds, so every parameter shares a single 0-1 axis. Truth, when
    known, is a red cross.

    Dots stacked on top of each other mean the optimizers agree about
    that parameter; dots spread vertically mean they do not.
    """
    fig = go.Figure()
    x_positions = list(range(len(free_names)))

    def _norm(name, v):
        lo, hi = model_cls.parameter_bounds[name]
        return (float(v) - lo) / (hi - lo) if hi > lo else 0.5

    for r in runs:
        fp = r.get("fitted_params")
        if fp is None:
            continue
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=[_norm(n, fp[n]) for n in free_names],
            mode="markers",
            marker=dict(color=_color(r), size=13, opacity=0.85,
                        line=dict(color=COLOR_AXIS, width=0.6)),
            name=r["label"],
            text=[f"{r['label']}  {n}={fp[n]:.4g}" for n in free_names],
            hovertemplate="%{text}<extra></extra>",
        ))

    if truth_params is not None:
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=[_norm(n, truth_params[n]) for n in free_names],
            mode="markers",
            marker=dict(color=COLOR_TRUTH, size=17, symbol="x",
                        line=dict(width=2)),
            name="truth",
            text=[f"truth  {n}={truth_params[n]:.4g}" for n in free_names],
            hovertemplate="%{text}<extra></extra>",
        ))

    _apply_layout(
        fig, "<b>parameter</b>",
        "<b>position within bounds  [-]</b>",
        (-0.5, len(free_names) - 0.5), (-0.05, 1.05),
        "Where each optimizer landed (normalised to each parameter's bounds)",
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