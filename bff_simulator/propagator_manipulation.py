import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm


def generate_propagators_from_hamiltonian(uniformly_spaced_times_s: NDArray, hamiltonian_rad_per_s: NDArray) -> NDArray:
    return generate_propagators_from_evolution_matrix(uniformly_spaced_times_s, -1.0j * hamiltonian_rad_per_s)


def generate_propagators_from_evolution_matrix(
    uniformly_spaced_times_s: NDArray, evolution_matrix_rad_per_s: NDArray
) -> NDArray:
    initial_time_s = uniformly_spaced_times_s[0]
    time_step_s = uniformly_spaced_times_s[1] - uniformly_spaced_times_s[0]

    propagator = expm(evolution_matrix_rad_per_s * initial_time_s)
    propagators = [propagator]
    time_step_propagator = expm(evolution_matrix_rad_per_s * time_step_s)
    for _ in uniformly_spaced_times_s[1:]:
        propagator = time_step_propagator @ propagator
        propagators.append(propagator)

    return np.array(propagators)


def expand_dq_unitaries(
    mw_1_unitaries: ArrayLike,
    bare_unitaries: ArrayLike,
    mw_2_unitaries: ArrayLike,
    psi_initial: ArrayLike = np.array([0, 1, 0]),
) -> NDArray:
    """
    Generates the MxN output states following the evolution of U3 U2 U1 unitary 3x3 matrices on psi_initial
    of the double quantum pulse sequence.

    Args:
        mw_1_unitaries (ArrayLike): Mx3x3 array of 3x3 unitary matrices representing the M pulse times of the first MW pulse (U1)
        bare_unitaries (ArrayLike): Nx3x3 array of 3x3 unitary matrices representing evolution over N evolution time steps (U2)
        mw_2_unitaries (ArrayLike): Mx3x3 array of 3x3 unitary matrices representing the M pulse times of the first MW pulse (U3).
                                    May be different than U1 (eg phase shift) but must be same length
        psi_initial (ArrayLike, optional): Initial state all unitaries are acted on. Defaults to [0,1,0] (ms=0).

    Returns:
        final_states (NDArray): MxNx3 array of state amplitudes
    """
    mw_1_unitaries = np.expand_dims(mw_1_unitaries, axis=1)
    bare_unitaries = np.expand_dims(bare_unitaries, axis=0)
    mw_2_unitaries = np.expand_dims(mw_2_unitaries, axis=1)
    psi_initial = np.expand_dims(psi_initial, axis=(0, 1, 3))
    return np.squeeze(np.matmul(mw_2_unitaries, np.matmul(bare_unitaries, np.matmul(mw_1_unitaries, psi_initial))))
