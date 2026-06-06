import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import scipy.optimize


def match_permeability(L_cm, A_cm2, phi, mu_cP, c_Pa, q_ml_min, P_out_bar,
                       total_time_mins, dt_mins, target_dp_mbar):

    # ── Unit Conversions ──────────────────────────────────────────────────────
    L = L_cm / 100.0
    A = A_cm2 / 10000.0
    mu = mu_cP * 1e-3
    q_inj = (q_ml_min * 1e-6) / 60.0
    P_out = P_out_bar * 1e5

    # ── Time Setup ────────────────────────────────────────────────────────────
    n_steps = int(total_time_mins / dt_mins)
    dt = dt_mins * 60.0

    # ── Grid Setup ────────────────────────────────────────────────────────────
    N = 20
    dx = L / N
    V_b = A * dx
    C = (V_b * phi * c_Pa) / dt

    # ── Dynamic Back-Pressure Reference (mbar) ───────────────────────────────
    p_out_mbar = P_out_bar * 1000.0

    iteration_counter = [0]

    def run_simulation(k_guess):
        k_m2 = k_guess * 9.869233e-16
        T = (k_m2 * A) / (mu * dx)
        T_out = (k_m2 * A) / (mu * (dx / 2.0))

        main_diag = np.zeros(N)
        lower_diag = np.zeros(N - 1)
        upper_diag = np.zeros(N - 1)

        for i in range(N):
            if i == 0:
                main_diag[i] = C + T
                upper_diag[i] = -T
            elif i == N - 1:
                main_diag[i] = C + T + T_out
                lower_diag[i - 1] = -T
            else:
                main_diag[i] = C + 2 * T
                lower_diag[i - 1] = -T
                upper_diag[i] = -T

        A_mat = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')
        P = np.ones(N) * P_out
        p_inj_results = []

        for step in range(n_steps):
            B = C * P.copy()
            B[0] += q_inj
            B[-1] += T_out * P_out
            P = spsolve(A_mat, B)
            inlet_pressure_mbar = P[0] / 100.0
            p_inj_results.append(inlet_pressure_mbar)

        steady_state_pressure = p_inj_results[-1]
        steady_state_delta = steady_state_pressure - p_out_mbar  # dynamic back-pressure
        return steady_state_delta

    def objective_function(k_array):
        k_guess = k_array[0]
        if k_guess <= 0:
            return 1e12
        simulated_dp = run_simulation(k_guess)
        error = abs(simulated_dp - target_dp_mbar)
        iteration_counter[0] += 1
        return error

    # ── Nelder-Mead Optimisation ──────────────────────────────────────────────
    x0 = np.array([1500.0])
    result = scipy.optimize.minimize(
        objective_function,
        x0,
        method='Nelder-Mead',
        options={
            'xatol': 1e-4,
            'fatol': 1e-4,
            'maxiter': 500,
            'disp': False,
        }
    )
    optimised_k = result.x[0]

    # ── Final Plotting Run ────────────────────────────────────────────────────
    k_m2_final = optimised_k * 9.869233e-16
    T_final = (k_m2_final * A) / (mu * dx)
    T_out_final = (k_m2_final * A) / (mu * (dx / 2.0))

    main_diag = np.zeros(N)
    lower_diag = np.zeros(N - 1)
    upper_diag = np.zeros(N - 1)

    for i in range(N):
        if i == 0:
            main_diag[i] = C + T_final
            upper_diag[i] = -T_final
        elif i == N - 1:
            main_diag[i] = C + T_final + T_out_final
            lower_diag[i - 1] = -T_final
        else:
            main_diag[i] = C + 2 * T_final
            lower_diag[i - 1] = -T_final
            upper_diag[i] = -T_final

    A_mat_final = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')
    P_final = np.ones(N) * P_out
    time_results = []
    delta_p_results = []

    for step in range(n_steps):
        B = C * P_final.copy()
        B[0] += q_inj
        B[-1] += T_out_final * P_out
        P_final = spsolve(A_mat_final, B)
        current_time_min = (step + 1) * dt_mins
        inlet_pressure_mbar = P_final[0] / 100.0
        time_results.append(current_time_min)
        delta_p_results.append(inlet_pressure_mbar - p_out_mbar)  # dynamic back-pressure

    time_plot = [0.0] + list(time_results)
    delta_p_plot = [0.0] + list(delta_p_results)

    # ── Streamlit-Compatible Plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_plot, delta_p_plot,
            label=f'Optimised (k = {optimised_k:.2f} mD)', color='blue', linewidth=2)
    ax.axhline(y=target_dp_mbar, color='red', linestyle='--', linewidth=2,
               label=f'Target ΔP ({target_dp_mbar} mbar)')
    ax.set_title(
        f'Differential Pressure Drop Across Core over Time\n'
        f'(History-Matched Result — k = {optimised_k:.2f} mD)',
        fontsize=14
    )
    ax.set_xlabel('Time [minutes]', fontsize=12)
    ax.set_ylabel('Differential Pressure (ΔP) [mbar]', fontsize=12)
    ax.set_xlim(0, total_time_mins)
    ax.set_ylim(0, max(60, max(delta_p_plot) * 1.1))
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='lower right', fontsize=11)
    fig.tight_layout()

    return optimised_k, fig