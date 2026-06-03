import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import scipy.optimize

#Parameters
L_cm = float(input("Length: "))                                     # Core Length (cm)
A_cm2 = float(input("Area: "))                                      # Area (cm^2)
phi = float(input("Porosity: "))                                    # Porosity 
mu_cP = float(input("Viscosity: "))                                 # Viscosity (cP) 
c_Pa = float(input("Compressibility: "))                            # Compressibility (1/Pa)
q_ml_min = float(input("Injection rate (in ml/min): "))             # Injection Rate (ml/min)
P_out_bar = float(input("Prosuction Back-Presure (in bar): "))      # Production Back-Pressure (bar)
 
# Conversion to SI 
L = L_cm / 100.0                        # m
A = A_cm2 / 10000.0                     # m^2
mu = mu_cP * 1e-3                       # Pa.s
q_inj = (q_ml_min * 1e-6) / 60.0        # m^3/s
P_out = P_out_bar * 1e5                 # Pa

# Time
total_time_mins = float(input("Enter duration of expeeriment: "))      # Total simulation time
dt_mins = float(input("Time interval: "))                              # Timestep
n_steps = int(total_time_mins / dt_mins)
dt = dt_mins * 60.0                                                    # Timestep in seconds

# Grid Setup 
N = 20                     # Number of grid blocks
dx = L / N                 # Length of each block [m]
V_b = A * dx               # Bulk volume of one block [m^3]

# Accumulation term: C = (V_b * phi * c) / dt
C = (V_b * phi * c_Pa) / dt

TARGET_DP = 50.0  # mbar
iteration_counter = [0]


def run_simulation(k_guess):
    """
    Run the 1D finite-difference core flooding simulation.
    Accepts permeability in mD, returns steady-state delta P in mbar.
    Core FD physics engine is preserved exactly as original.
    """
    k_m2 = k_guess * 9.869233e-16

    # Transmissibility between adjacent blocks
    T = (k_m2 * A) / (mu * dx)
    T_out = (k_m2 * A) / (mu * (dx / 2.0))

    # Constructing the Implicit Coefficient Matrix 
    main_diag = np.zeros(N)
    lower_diag = np.zeros(N-1)
    upper_diag = np.zeros(N-1)

    for i in range(N):
        if i == 0:
            main_diag[i] = C + T
            upper_diag[i] = -T
        elif i == N - 1:
            main_diag[i] = C + T + T_out
            lower_diag[i-1] = -T
        else:
            main_diag[i] = C + 2 * T
            lower_diag[i-1] = -T
            upper_diag[i] = -T

    # Create sparse matrix for efficient solving
    A_mat = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')

    # Core starts fully saturated at the outlet pressure
    P = np.ones(N) * P_out

    p_inj_results = []

    for step in range(n_steps):
        B = C * P.copy()
        
        B[0] += q_inj              # Inject fluid into block 0
        B[-1] += T_out * P_out     # Block N-1 feels the pressure from the outlet
        
        P = spsolve(A_mat, B)
        
        inlet_pressure_mbar = P[0] / 100.0
        p_inj_results.append(inlet_pressure_mbar)

    steady_state_pressure = p_inj_results[-1]
    steady_state_delta = steady_state_pressure - 1000.0  # subtract 1000 mbar back-pressure

    return steady_state_delta


def objective_function(k_array):
    """
    Objective function for the optimizer.
    Takes k_array (1-element array from Nelder-Mead), returns absolute error vs target DP.
    """
    k_guess = k_array[0]

    # Constraint: penalise non-physical permeability
    if k_guess <= 0:
        return 1e12

    simulated_dp = run_simulation(k_guess)
    error = abs(simulated_dp - TARGET_DP)

    iteration_counter[0] += 1
    print(f"  Iteration {iteration_counter[0]:>4d} | k = {k_guess:>12.4f} mD | "
          f"Simulated ΔP = {simulated_dp:>8.4f} mbar | Error = {error:>8.4f} mbar")

    return error


# ── Optimisation ──────────────────────────────────────────────────────────────
print("=" * 70)
print("  1D Core Flooding — Automated History Matching (Nelder-Mead)")
print(f"  Target ΔP : {TARGET_DP} mbar")
print("=" * 70)
print(f"\n  Starting optimisation with initial guess k₀ = 1500 mD ...\n")
print(f"  {'Iteration':>9} | {'k (mD)':>14} | {'Simulated ΔP':>20} | {'Error':>14}")
print("-" * 70)

x0 = np.array([1500.0])

result = scipy.optimize.minimize(
    objective_function,
    x0,
    method='Nelder-Mead',
    options={
        'xatol': 1e-4,   # tolerance on k
        'fatol': 1e-4,   # tolerance on error
        'maxiter': 500,
        'disp': False,
    }
)

optimised_k = result.x[0]
optimised_dp = run_simulation(optimised_k)

print("-" * 70)
print(f"\n  Optimisation complete after {iteration_counter[0]} iterations.")
print(f"  Optimised Permeability : {optimised_k:.4f} mD")
print(f"  Simulated ΔP           : {optimised_dp:.4f} mbar")
print(f"  Target ΔP              : {TARGET_DP:.4f} mbar")
print(f"  Absolute Error         : {abs(optimised_dp - TARGET_DP):.4f} mbar")
print("=" * 70)

# FINAL PLOT
k_m2_final = optimised_k * 9.869233e-16
T_final = (k_m2_final * A) / (mu * dx)
T_out_final = (k_m2_final * A) / (mu * (dx / 2.0))

main_diag = np.zeros(N)
lower_diag = np.zeros(N-1)
upper_diag = np.zeros(N-1)

for i in range(N):
    if i == 0:
        main_diag[i] = C + T_final
        upper_diag[i] = -T_final
    elif i == N - 1:
        main_diag[i] = C + T_final + T_out_final
        lower_diag[i-1] = -T_final
    else:
        main_diag[i] = C + 2 * T_final
        lower_diag[i-1] = -T_final
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
    delta_p_results.append(inlet_pressure_mbar - 1000.0)

time_plot = [0.0] + list(time_results)
delta_p_plot = [0.0] + list(delta_p_results)

plt.figure(figsize=(10, 6))
plt.plot(time_plot, delta_p_plot,
         label=f'Optimised (k = {optimised_k:.2f} mD)', color='blue', linewidth=2)
plt.axhline(y=TARGET_DP, color='red', linestyle='--', linewidth=2,
            label=f'Target $\\Delta P$ ({TARGET_DP} mbar)')
plt.title('Differential Pressure Drop Across Core over Time\n'
          f'(History-Matched Result — k = {optimised_k:.2f} mD)', fontsize=14)
plt.xlabel('Time [minutes]', fontsize=12)
plt.ylabel('Differential Pressure ($\\Delta P$) [mbar]', fontsize=12)
plt.xlim(0, 60)
min_y = 0
max_y = max(60, max(delta_p_plot) * 1.1)
plt.ylim(min_y, max_y)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.savefig(f"Delta_Pressure_Match_Optimised_{optimised_k:.2f}mD.png", dpi=150)
plt.show()