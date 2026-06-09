"""
Plotly chart builders for the two-phase calculator.

Visual aesthetic mirrors PermCalc's existing inline charts:
    - white plot background on the dark page
    - Courier New typography throughout
    - teal  #2DD4BF for injected phase / primary data
    - orange #FB923C for displaced phase / secondary data
    - red   #DC2626 for reference / measured / target

Functions
---------
build_kr_chart              : static kr_inj / kr_disp vs S_inj.
build_pc_chart              : static Brooks-Corey Pc(S_inj), semi-log.
build_dp_time_chart         : animated ΔP vs time.
build_profile_animation     : animated S_inj(x), with slider.
build_history_match_chart   : measured ΔP (markers) + simulated ΔP (line).
render_chart_html           : Plotly figure -> HTML string (+ autoplay).
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


# ── Style constants ─────────────────────────────────────────────────────────
COLOR_INJ   = "#2DD4BF"
COLOR_DISP  = "#FB923C"
COLOR_REF   = "#DC2626"
COLOR_AXIS  = "#0B1014"
COLOR_GRID  = "#D1D5DB"
COLOR_BG    = "#FFFFFF"
COLOR_TITLE = "#111827"

AXIS_TITLE_FONT = dict(color=COLOR_AXIS, family="Courier New", size=15)
AXIS_TICK_FONT  = dict(color=COLOR_AXIS, family="Courier New", size=13)
TITLE_FONT      = dict(family="Courier New", color=COLOR_TITLE, size=14)
ANNOT_FONT      = dict(color=COLOR_REF, family="Courier New", size=12)


# ── Shared layout ───────────────────────────────────────────────────────────
def _apply_layout(fig, x_title, y_title, x_range, y_range, title,
                  showlegend=False, y_log=False, height=380):
    y_axis_range = ([np.log10(y_range[0]), np.log10(y_range[1])]
                    if y_log else list(y_range))
    fig.update_layout(
        title=dict(text=title, font=TITLE_FONT),
        xaxis=dict(
            title=dict(text=x_title, font=AXIS_TITLE_FONT),
            tickfont=AXIS_TICK_FONT,
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_AXIS,
            linecolor=COLOR_AXIS,
            range=list(x_range),
        ),
        yaxis=dict(
            title=dict(text=y_title, font=AXIS_TITLE_FONT),
            tickfont=AXIS_TICK_FONT,
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_AXIS,
            linecolor=COLOR_AXIS,
            range=y_axis_range,
            type="log" if y_log else "linear",
        ),
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
        font=dict(family="Courier New"),
        height=height,
        margin=dict(l=80, r=40, t=50, b=70),
        showlegend=showlegend,
        legend=dict(font=dict(family="Courier New", color=COLOR_AXIS, size=12),
                    bgcolor="rgba(255,255,255,0.7)",
                    x=0.02, y=0.98, xanchor="left", yanchor="top"),
    )


# ── kr chart (static) ───────────────────────────────────────────────────────
def build_kr_chart(kr_data, inj_name="Injected", disp_name="Displaced"):
    S = kr_data["S_inj"]
    fig = go.Figure([
        go.Scatter(
            x=S, y=kr_data["kr_inj"], mode="lines",
            line=dict(color=COLOR_INJ, width=2.8),
            name=f"kr_inj ({inj_name})",
            hovertemplate="S_inj=%{x:.3f}<br>kr=%{y:.4f}<extra></extra>",
        ),
        go.Scatter(
            x=S, y=kr_data["kr_disp"], mode="lines",
            line=dict(color=COLOR_DISP, width=2.8),
            name=f"kr_disp ({disp_name})",
            hovertemplate="S_inj=%{x:.3f}<br>kr=%{y:.4f}<extra></extra>",
        ),
    ])
    fig.add_vline(
        x=kr_data["S_inj_r"],
        line=dict(color=COLOR_AXIS, width=1, dash="dot"),
        annotation_text=f"S_r,inj={kr_data['S_inj_r']:.2f}",
        annotation_font=dict(color=COLOR_AXIS, family="Courier New", size=11),
        annotation_position="top",
    )
    fig.add_vline(
        x=1.0 - kr_data["S_disp_r"],
        line=dict(color=COLOR_AXIS, width=1, dash="dot"),
        annotation_text=f"1−S_r,disp={1 - kr_data['S_disp_r']:.2f}",
        annotation_font=dict(color=COLOR_AXIS, family="Courier New", size=11),
        annotation_position="top",
    )
    y_max = max(
        float(np.nanmax(kr_data["kr_inj"])),
        float(np.nanmax(kr_data["kr_disp"])),
        1e-3,
    ) * 1.05
    _apply_layout(
        fig,
        "<b>S_inj  [-]</b>", "<b>kr  [-]</b>",
        (0.0, 1.0), (0.0, y_max),
        "Relative permeability curves (Corey)",
        showlegend=True,
    )
    return fig


# ── Pc chart (static, semi-log) ─────────────────────────────────────────────
def build_pc_chart(pc_data, S_inj_r, S_disp_r):
    S  = pc_data["S_inj"]
    Pc = pc_data["Pc_mbar"]
    mask = (S >= S_inj_r) & (S <= 1.0 - S_disp_r)
    Sm = S[mask]
    Pm = Pc[mask]
    fig = go.Figure([
        go.Scatter(
            x=Sm, y=Pm, mode="lines",
            line=dict(color=COLOR_INJ, width=2.8),
            hovertemplate="S_inj=%{x:.3f}<br>Pc=%{y:.2f} mbar<extra></extra>",
        ),
    ])
    y_min = max(float(np.nanmin(Pm)) * 0.5, 1e-2)
    y_max = float(np.nanmax(Pm)) * 2.0
    _apply_layout(
        fig,
        "<b>S_inj  [-]</b>", "<b>Pc  [mbar]</b>",
        (0.0, 1.0), (y_min, y_max),
        "Capillary pressure (Brooks–Corey)",
        y_log=True,
    )
    return fig


# ── ΔP vs time (animated, PermCalc style) ───────────────────────────────────
def build_dp_time_chart(results):
    t  = results["t_min"]
    dp = results["dP_mbar"]
    n  = len(t)
    n_frames = 60
    step = max(1, n // n_frames)
    frame_indices = list(range(2, n, step))
    if not frame_indices or frame_indices[-1] != n:
        frame_indices.append(n)

    frames = [
        go.Frame(
            data=[go.Scatter(
                x=list(t[:i]), y=list(dp[:i]),
                mode="lines",
                line=dict(color=COLOR_INJ, width=2.8),
            )],
            name=str(i),
        )
        for i in frame_indices
    ]
    fig = go.Figure(
        data=[go.Scatter(
            x=[t[0]], y=[dp[0]], mode="lines",
            line=dict(color=COLOR_INJ, width=2.8),
            hovertemplate="t=%{x:.2f} min<br>ΔP=%{y:.2f} mbar<extra></extra>",
        )],
        frames=frames,
    )
    bt_time = results.get("BT_time_min")
    if bt_time is not None:
        fig.add_vline(
            x=bt_time,
            line=dict(color=COLOR_REF, width=2, dash="dash"),
            annotation_text=f"<b>BT at {bt_time:.2f} min</b>",
            annotation_position="top right",
            annotation_font=ANNOT_FONT,
        )
    y_max = float(np.nanmax(dp)) * 1.10
    _apply_layout(
        fig,
        "<b>Time [min]</b>", "<b>ΔP [mbar]</b>",
        (0.0, float(t[-1])), (0.0, y_max),
        "Pressure drop history",
    )
    return fig


# ── Saturation profile (animated, with slider) ─────────────────────────────
def build_profile_animation(results):
    x_cm   = results["x_cm"]
    S_prof = results["S_inj_profiles"]
    t_min  = results["t_min"]
    pvi    = results["PVI"]
    n      = len(t_min)

    n_frames = 50
    step = max(1, n // n_frames)
    frame_indices = list(range(0, n, step))
    if frame_indices[-1] != n - 1:
        frame_indices.append(n - 1)

    frames = []
    for i in frame_indices:
        frames.append(go.Frame(
            data=[go.Scatter(
                x=x_cm, y=S_prof[i], mode="lines",
                line=dict(color=COLOR_INJ, width=2.8),
            )],
            name=f"{i}",
            layout=go.Layout(title=dict(
                text=f"Saturation profile — t = {t_min[i]:.2f} min  "
                     f"(PVI = {pvi[i]:.3f})",
                font=TITLE_FONT,
            )),
        ))

    fig = go.Figure(
        data=[go.Scatter(
            x=x_cm, y=S_prof[0], mode="lines",
            line=dict(color=COLOR_INJ, width=2.8),
            hovertemplate="x=%{x:.2f} cm<br>S_inj=%{y:.3f}<extra></extra>",
        )],
        frames=frames,
    )
    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue=dict(
                prefix="t = ",
                font=dict(family="Courier New", color=COLOR_AXIS, size=12),
            ),
            pad=dict(t=40),
            steps=[
                dict(
                    method="animate",
                    args=[[f"{i}"],
                          dict(mode="immediate",
                               frame=dict(duration=0, redraw=True),
                               transition=dict(duration=0))],
                    label=f"{t_min[i]:.1f}",
                )
                for i in frame_indices
            ],
        )],
    )
    _apply_layout(
        fig,
        "<b>x [cm]</b>", "<b>S_inj  [-]</b>",
        (0.0, float(x_cm[-1])), (0.0, 1.0),
        f"Saturation profile — t = {t_min[0]:.2f} min  "
        f"(PVI = {pvi[0]:.3f})",
        height=440,
    )
    return fig


# ── History-match (measured markers + fitted line) ─────────────────────────
def build_history_match_chart(measured_t_min, measured_dp_mbar, sim_results):
    sim_t  = sim_results["t_min"]
    sim_dp = sim_results["dP_mbar"]
    measured_t  = np.asarray(measured_t_min,  dtype=float)
    measured_dp = np.asarray(measured_dp_mbar, dtype=float)

    fig = go.Figure([
        go.Scatter(
            x=measured_t, y=measured_dp,
            mode="markers",
            marker=dict(color=COLOR_REF, size=7,
                        line=dict(color=COLOR_AXIS, width=0.5)),
            name="measured",
            hovertemplate="t=%{x:.2f} min<br>ΔP=%{y:.2f} mbar<extra></extra>",
        ),
        go.Scatter(
            x=sim_t, y=sim_dp, mode="lines",
            line=dict(color=COLOR_INJ, width=2.8),
            name="fitted",
            hovertemplate="t=%{x:.2f} min<br>ΔP=%{y:.2f} mbar<extra></extra>",
        ),
    ])
    bt_time = sim_results.get("BT_time_min")
    if bt_time is not None:
        fig.add_vline(
            x=bt_time,
            line=dict(color=COLOR_AXIS, width=1, dash="dot"),
            annotation_text=f"BT at {bt_time:.2f} min",
            annotation_font=dict(color=COLOR_AXIS,
                                 family="Courier New", size=11),
            annotation_position="top right",
        )
    x_max = max(float(sim_t[-1]), float(np.max(measured_t)))
    y_max = max(float(np.nanmax(sim_dp)),
                float(np.nanmax(measured_dp))) * 1.10
    _apply_layout(
        fig,
        "<b>Time [min]</b>", "<b>ΔP [mbar]</b>",
        (0.0, x_max), (0.0, y_max),
        "History match — measured vs fitted",
        showlegend=True,
    )
    return fig


# ── HTML conversion + optional autoplay ────────────────────────────────────
def render_chart_html(fig, autoplay=False):
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