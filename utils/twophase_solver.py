"""
1D IMPES forward simulator for two-phase incompressible immiscible flow
in a horizontal core (gravity-free).

Pure NumPy. No Streamlit. Pyodide-compatible.

Discretization
--------------
- Cell-centered finite volume, N cells of width dx = L/N.
- Saturation advance: explicit upwind for f_inj * u_t (rightward flow),
  central differences for the capillary diffusion term.
- Adaptive time step from min(advective CFL, diffusive CFL); each
  step is also clipped to land on the next snapshot / t_total.

Boundary conditions
-------------------
- Inlet  (x=0): u_inj = u_t (100 % injected entering); no Pc flux.
- Outlet (x=L): zero-gradient outflow; no Pc flux.

Process-type awareness (Phase 2.4)
----------------------------------
The solver reads tp_inputs["process_type"] to determine the sign of
the capillary flux term applied to the injected phase.

Under drainage the injected phase is non-wetting. Starting from the
two-phase Darcy equations with Pc = P_nw - P_w and total velocity
u_t = u_w + u_nw, the non-wetting phase velocity is

    u_nw = f_nw * u_t - k * M * dPc/dx      with  M = lambda_w * lambda_nw / (lambda_w + lambda_nw)

The minus sign in front of the capillary term is the standard result
for the non-wetting phase: high Pc upstream (= high P_nw relative to
P_w) drives the non-wetting phase downstream, but the gradient of Pc
enters with a negative sign because the fractional-flow rearrangement
absorbs the wetting-phase pressure gradient.

Under imbibition the injected phase is wetting. The mirror derivation
gives

    u_w = f_w * u_t + k * M * dPc/dx

which is the same magnitude but the opposite sign in front of the
capillary term. Physically this represents the spontaneous
imbibition drive: the wetting phase is pulled forward by capillary
suction into its preferred pores rather than being resisted by
capillarity.

The solver therefore sets `pc_sign` = -1 for drainage, +1 for
imbibition, and applies `u_inj_face += pc_sign * k * M_face * dPc/dx`
uniformly. All other physics (advective flux, boundary conditions,
CFL time-step, saturation clipping) is process-type-agnostic.

Initial condition
-----------------
Both processes initialize S_inj := S_inj_r. The physical meaning
differs:

    drainage   -- initial S_inj_r is the connate injected-phase
                  saturation; typically 0 for a fully wetting-phase-
                  saturated core (S_w = 1). Under drainage the
                  invading phase then rises from S_inj_r toward
                  1 - S_disp_r.

    imbibition -- initial S_inj_r equals S_wr (residual water); the
                  core starts at maximum non-wetting saturation
                  (S_w = S_wr, i.e. S_inj = S_wr under the labelling
                  where injected = wetting) and imbibition increases
                  S_inj toward 1 - S_disp_r = 1 - S_or.

The same numerical initialization line therefore serves both
processes; the user sets S_inj_r and S_disp_r to values appropriate
to whichever process they've selected.

Units
-----
tp_inputs comes in cm / mD / cP / ml-min / mbar (Step 3 layout);
this module converts to SI internally and reports outputs back in
cm / min / mbar / fractional saturation.
"""

import numpy as np

from utils.twophase import make_kr_functions, make_pc_function


# -- Unit conversions -------------------------------------------------------
CM_TO_M        = 1e-2
CM2_TO_M2      = 1e-4
ML_MIN_TO_M3_S = 1e-6 / 60.0
MD_TO_M2       = 9.869233e-16
CP_TO_PA_S     = 1e-3
MBAR_TO_PA     = 1e2
PA_TO_MBAR     = 1e-2
S_TO_MIN       = 1.0 / 60.0


def run_forward(tp_inputs,
                n_snapshots=80,
                cfl_safety=0.4,
                bt_threshold=0.05,
                max_steps=500_000):
    """
    Run a 1D IMPES forward simulation of a core flood.

    Parameters
    ----------
    tp_inputs : dict
        From st.session_state["tp_inputs"] (Step 3 layout). Must
        include the key "process_type" ("drainage" or "imbibition");
        defaults to "drainage" if absent for backward compatibility.
    n_snapshots : int
        Number of (t, dP, profile) samples recorded.
    cfl_safety : float
        Multiplier (< 1) applied to the CFL time-step limit.
    bt_threshold : float
        f_inj at outlet at which breakthrough is flagged.
    max_steps : int
        Safety cap on inner time-steps.

    Returns
    -------
    dict with keys: x_cm, t_min, dP_mbar, PVI, S_inj_profiles,
    BT_PVI, BT_time_min, N, dx_cm, PV_ml, u_t_m_s, n_steps,
    process_type.
    """

    # -- 1. Unpack and SI-convert -------------------------------------------
    core = tp_inputs["core"]
    op   = tp_inputs["operating"]
    num  = tp_inputs["numerical"]
    process_type = tp_inputs.get("process_type", "drainage")

    # Sign of the capillary flux term applied to the injected phase.
    # See module docstring for the derivation.
    if process_type == "drainage":
        pc_sign = -1.0
    elif process_type == "imbibition":
        pc_sign = +1.0
    else:
        raise ValueError(
            f"Unknown process_type '{process_type}'. "
            f"Expected 'drainage' or 'imbibition'."
        )

    L   = core["L_cm"]  * CM_TO_M
    A   = core["A_cm2"] * CM2_TO_M2
    phi = core["phi"]
    k   = core["k_mD"]  * MD_TO_M2

    mu_inj  = tp_inputs["injected"]["viscosity_cP"]  * CP_TO_PA_S
    mu_disp = tp_inputs["displaced"]["viscosity_cP"] * CP_TO_PA_S

    q   = op["q_ml_min"] * ML_MIN_TO_M3_S     # [m^3/s]
    u_t = q / A                               # [m/s]
    t_total = op["t_total_min"] * 60.0        # [s]

    S_inj_r  = tp_inputs["S_inj_r"]
    S_disp_r = tp_inputs["S_disp_r"]

    N   = int(num["N_cells"])
    dx  = L / N
    x_centers_m = (np.arange(N) + 0.5) * dx
    PV  = phi * A * L                         # [m^3]

    # -- 2. kr / Pc closures ------------------------------------------------
    kr_inj_fn, kr_disp_fn = make_kr_functions(tp_inputs)
    pc_mbar_fn = make_pc_function(tp_inputs)

    def pc_fn(S):
        return pc_mbar_fn(S) * MBAR_TO_PA      # [Pa]
    pc_enabled = tp_inputs["pc"]["enabled"]

    def mobilities(S):
        l_i = kr_inj_fn(S)  / mu_inj
        l_d = kr_disp_fn(S) / mu_disp
        return l_i, l_d, l_i + l_d

    def frac_flow(S):
        l_i, _, l_t = mobilities(S)
        return np.where(l_t > 0.0, l_i / l_t, 0.0)

    # -- 3. CFL estimates over a fine S grid --------------------------------
    S_fine    = np.linspace(S_inj_r, 1.0 - S_disp_r, 800)
    f_fine    = frac_flow(S_fine)
    dfdS_max  = max(float(np.max(np.abs(np.gradient(f_fine, S_fine)))),
                    1e-12)

    if pc_enabled:
        li_f, ld_f, lt_f = mobilities(S_fine)
        M_fine     = np.where(lt_f > 0.0, li_f * ld_f / lt_f, 0.0)
        Pc_fine    = pc_fn(S_fine)
        dPcdS_fine = np.gradient(Pc_fine, S_fine)
        # M -> 0 at both saturation endpoints, so this self-regularizes.
        # abs() means the CFL bound is agnostic to the sign of Pc, which
        # matters for imbibition Skjaeveland where Pc can go negative.
        D_cap_max  = float(np.max(k * M_fine * np.abs(dPcdS_fine) / phi))
    else:
        D_cap_max = 0.0

    # -- 4. Initial condition -----------------------------------------------
    S = np.full(N, S_inj_r, dtype=float)

    snap_times = np.linspace(0.0, t_total, n_snapshots)
    rec_t, rec_dP, rec_PVI, rec_S = [], [], [], []

    def _dP_Pa(S_arr):
        _, _, l_t = mobilities(S_arr)
        return float(np.sum(u_t * dx / (k * np.maximum(l_t, 1e-30))))

    def _record(t_now, S_arr):
        rec_t.append(t_now)
        rec_dP.append(_dP_Pa(S_arr) * PA_TO_MBAR)
        rec_PVI.append(u_t * A * t_now / PV)
        rec_S.append(S_arr.copy())

    _record(0.0, S)
    next_snap = 1
    bt_PVI, bt_time = None, None

    # -- 5. Time loop -------------------------------------------------------
    t = 0.0
    step = 0
    for step in range(max_steps):
        if t >= t_total - 1e-15 or next_snap >= n_snapshots:
            break

        l_i, l_d, l_t = mobilities(S)
        f             = np.where(l_t > 0.0, l_i / l_t, 0.0)

        # Advective flux at faces (upwind, rightward flow)
        f_face       = np.empty(N + 1)
        f_face[0]    = 1.0                    # inlet ghost: pure injected
        f_face[1:]   = f                      # upwind = cell to the left
        u_inj_face   = u_t * f_face

        # Capillary flux at faces
        if pc_enabled:
            Pc_cells = pc_fn(S)
            M_cells  = np.where(l_t > 0.0, l_i * l_d / l_t, 0.0)

            M_face   = np.empty(N + 1)
            M_face[1:-1] = 0.5 * (M_cells[:-1] + M_cells[1:])
            M_face[0]    = 0.0                # no Pc flux at inlet
            M_face[-1]   = 0.0                # no Pc flux at outlet

            dPc_dx_face        = np.zeros(N + 1)
            dPc_dx_face[1:-1]  = (Pc_cells[1:] - Pc_cells[:-1]) / dx
            # pc_sign = -1 for drainage (injected is non-wetting),
            # pc_sign = +1 for imbibition (injected is wetting). See
            # module docstring for the derivation.
            u_inj_face        += pc_sign * k * M_face * dPc_dx_face

        # Time step from CFL
        dt_adv = cfl_safety * phi * dx / (u_t * dfdS_max)
        if D_cap_max > 0.0:
            dt_diff = cfl_safety * dx * dx / (2.0 * D_cap_max)
            dt = min(dt_adv, dt_diff)
        else:
            dt = dt_adv
        dt = min(dt, t_total - t)
        if next_snap < n_snapshots:
            dt = min(dt, snap_times[next_snap] - t)

        if dt <= 1e-18:
            break

        # Explicit saturation update
        S = S - (dt / (phi * dx)) * (u_inj_face[1:] - u_inj_face[:-1])
        np.clip(S, S_inj_r, 1.0 - S_disp_r, out=S)
        t += dt

        # Breakthrough detection at every time step (not tied to
        # snapshot times, so BT resolution equals dt rather than
        # t_total / n_snapshots). Without this, close-BT runs like
        # Brooks-Corey vs van Genuchten under the same conditions
        # get aliased into the same snapshot bin and appear to have
        # identical BT even when the underlying physics differs.
        if bt_PVI is None:
            f_out = frac_flow(S)[-1]
            if f_out > bt_threshold:
                bt_time = t
                bt_PVI  = u_t * A * t / PV

        # Snapshot recording (loop handles multiple crossings)
        while (next_snap < n_snapshots
               and t >= snap_times[next_snap] - 1e-12):
            _record(snap_times[next_snap], S)
            next_snap += 1  
    else:
        raise RuntimeError(
            f"IMPES forward sim hit max_steps={max_steps} before "
            f"reaching t_total. Try larger cfl_safety, fewer cells, "
            f"shorter t_total, or disable Pc."
        )

    return {
        "x_cm":           x_centers_m / CM_TO_M,
        "t_min":          np.array(rec_t) * S_TO_MIN,
        "dP_mbar":        np.array(rec_dP),
        "PVI":            np.array(rec_PVI),
        "S_inj_profiles": np.array(rec_S),
        "BT_PVI":         bt_PVI,
        "BT_time_min":    None if bt_time is None else bt_time * S_TO_MIN,
        "N":              N,
        "dx_cm":          dx / CM_TO_M,
        "PV_ml":          PV * 1e6,
        "u_t_m_s":        u_t,
        "n_steps":        step + 1,
        "process_type":   process_type,
    }