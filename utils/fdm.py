"""
Finite-difference helpers for the CoreFlood Lab solvers.

Currently exports:
    build_tridiagonal_1d_diffusion(N, T, T_out, C)
        → scipy.sparse CSR matrix for the implicit backward-Euler system
          used by the single-phase compressible flow solver.

Kept separate from the calculator modules so that Phase 2 (compressible
fluids, additional solvers) can reuse the same discretisation helpers.
"""

import numpy as np
from scipy.sparse import diags


def build_tridiagonal_1d_diffusion(N, T, T_out, C):
    """
    Assemble the tridiagonal coefficient matrix for a 1D slightly-
    compressible flow implicit backward-Euler scheme with:

        - N cells of equal width
        - inlet: Neumann (fixed rate)                → main[0]   = C + T
        - outlet: Dirichlet (fixed pressure), applied
          with a half-cell transmissibility T_out    → main[-1]  = C + T + T_out
        - interior cells                             → main[i]   = C + 2T
        - off-diagonals                              → -T

    Parameters
    ----------
    N : int
        Number of cells.
    T : float
        Interior transmissibility (k·A / (μ·dx)).
    T_out : float
        Outlet half-cell transmissibility (k·A / (μ·(dx/2))).
    C : float
        Accumulation coefficient (V_b · φ · c / dt).

    Returns
    -------
    scipy.sparse.csr_matrix
        The (N, N) tridiagonal coefficient matrix in CSR format.
    """
    main_diag  = np.zeros(N)
    lower_diag = np.zeros(N - 1)
    upper_diag = np.zeros(N - 1)

    for i in range(N):
        if i == 0:
            main_diag[i]  = C + T
            upper_diag[i] = -T
        elif i == N - 1:
            main_diag[i]      = C + T + T_out
            lower_diag[i - 1] = -T
        else:
            main_diag[i]      = C + 2.0 * T
            lower_diag[i - 1] = -T
            upper_diag[i]     = -T

    return diags(
        [lower_diag, main_diag, upper_diag], [-1, 0, 1], format="csr",
    )