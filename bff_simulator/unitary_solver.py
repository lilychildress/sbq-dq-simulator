from typing import Dict

import numpy as np
from numpy.typing import NDArray
from qutip import Qobj, basis, commutator, jmat, sesolve
from scipy import linalg

from bff_simulator.abstract_classes.abstract_ensemble import NVSpecies
from bff_simulator.abstract_classes.abstract_solver import Solver
from bff_simulator.constants import gammab
from bff_simulator.experiment_parameters import ExperimentParameters
from bff_simulator.propagator_manipulation import expand_dq_unitaries
from bff_simulator.solver_parameters import SolverParam
from bff_simulator.vector_manipulation import (
    perpendicular_projection,
)


class UnitarySolver(Solver):
    """
    Unitary solver for getting the output states of a double quantum pulse sequence.
    Assumes no dephasing (unitary evolution).
    """

    SOLVER_KEY = 1.0

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _internal_hamiltonian(efield_splitting_hz: float, bz_field_t: float, detuning_hz: float) -> Qobj:
        sx_matrix, sy_matrix, sz_matrix = jmat(1)
        sxy_matrix = sy_matrix * sy_matrix - sx_matrix * sx_matrix
        sz_squared_matrix = sz_matrix * sz_matrix
        return (
            2
            * np.pi
            * (efield_splitting_hz * sxy_matrix + gammab * bz_field_t * sz_matrix + detuning_hz * sz_squared_matrix)
        )

    @staticmethod
    def _control_hamiltonian(rabi_frequency_hz: float, phase_rad: float) -> Qobj:
        sx_matrix, _, sz_matrix = jmat(1)
        sy_prime_matrix = 1.0j * commutator(sz_matrix * sz_matrix, sx_matrix)
        return 2 * np.pi * rabi_frequency_hz * (np.cos(phase_rad) * sx_matrix - np.sin(phase_rad) * sy_prime_matrix)

    def solve(self, params: SolverParam) -> NDArray:
        initial_states = [basis(3, n) for n in range(3)]
        free_evolution_states = []
        mw_1_final_states = []
        mw_2_final_states = []

        bare_hamiltonian = UnitarySolver._internal_hamiltonian(
            params.efield_splitting_hz, params.bz_field_t, params.detuning_hz
        )
        mw_1_hamiltonian = bare_hamiltonian + UnitarySolver._control_hamiltonian(params.rabi_frequency_hz, 0)
        mw_2_hamiltonian = bare_hamiltonian + UnitarySolver._control_hamiltonian(
            params.rabi_frequency_hz, params.second_pulse_phase_rad
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

    @staticmethod
    def construct_solver_parameters(experiment_parameters: ExperimentParameters, nv_species: NVSpecies) -> SolverParam:  # type: ignore[override]
        nv_axis_unit_vector = np.array(nv_species.axis_vector) / linalg.norm(nv_species.axis_vector)

        bz_field_t = (
            np.dot(experiment_parameters.b_field_vector_t, nv_axis_unit_vector) + nv_species.residual_bz_field_t
        )
        bperp_field_t = perpendicular_projection(experiment_parameters.b_field_vector_t, nv_axis_unit_vector)

        rabi_frequency_hz = experiment_parameters.rabi_frequency_base_hz * nv_species.rabi_projection

        solver_param = SolverParam(
            experiment_parameters.mw_pulse_length_s,
            experiment_parameters.evolution_time_s,
            experiment_parameters.second_pulse_phase_rad,
            experiment_parameters.detuning_hz,
            rabi_frequency_hz,
            bz_field_t,
            bperp_field_t,
            nv_species.efield_splitting_hz,
            nv_species.t2_star_s,
        )

        return solver_param

    def get_metadata_dict(self) -> Dict[str, float]:
        return {"Solver": UnitarySolver.SOLVER_KEY}
