from typing import Dict

import numpy as np
from numpy.typing import NDArray
from qutip import basis, sesolve

from bff_simulator.geometric_solver import GeometricSolver
from bff_simulator.geometric_solver_parameters import GeometricSolverParam
from bff_simulator.propagator_manipulation import expand_dq_unitaries


class GeometricUnitarySolver(GeometricSolver):
    """
    Unitary solver for getting the output states of a double quantum pulse sequence.
    Assumes no dephasing (unitary evolution).
    """

    SOLVER_KEY = 3.0

    def __init__(self) -> None:
        super().__init__()

    def solve(self, params: GeometricSolverParam) -> NDArray:  # type: ignore[override]
        initial_states = [basis(3, n) for n in range(3)]
        free_evolution_states = []
        mw_1_final_states = []
        mw_2_final_states = []

        bare_hamiltonian = GeometricUnitarySolver._internal_hamiltonian(
            params.efield_splitting_hz, params.bz_field_t, params.detuning_hz
        )
        mw_1_hamiltonian = bare_hamiltonian + GeometricUnitarySolver._control_hamiltonian(
            params.rabi_frequency_hz, 0, 0
        )
        mw_2_hamiltonian = bare_hamiltonian + GeometricUnitarySolver._control_hamiltonian(
            params.rabi_frequency_hz, params.second_pulse_phase_rad, params.second_pulse_geometric_angle_rad
        )

        for psi0 in initial_states:
            free_evolution_solution = sesolve(bare_hamiltonian, psi0, params.evolution_time_s)
            free_evolution_states.append([state.full() for state in free_evolution_solution.states])

            mw_1_solution = sesolve(mw_1_hamiltonian, psi0, params.mw_pulse_length_s)
            mw_1_final_states.append([state.full() for state in mw_1_solution.states])

            mw_2_solution = sesolve(mw_2_hamiltonian, psi0, params.mw_pulse_length_s)
            mw_2_final_states.append([state.full() for state in mw_2_solution.states])

        mw_2_unitary = np.swapaxes(np.squeeze(mw_2_final_states), 0, 1)
        free_evolution_unitary = np.swapaxes(np.squeeze(free_evolution_states), 0, 1)
        mw_1_unitary = np.swapaxes(np.squeeze(mw_1_final_states), 0, 1)
        psi_initial = [0, 1, 0]

        final_states = expand_dq_unitaries(mw_1_unitary, free_evolution_unitary, mw_2_unitary, psi_initial)

        return final_states

    def get_metadata_dict(self) -> Dict[str, float]:
        return {"Solver": GeometricUnitarySolver.SOLVER_KEY}
