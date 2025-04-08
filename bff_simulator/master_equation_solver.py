from typing import Dict, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qutip import Qobj, mesolve, qutrit_basis, qutrit_ops
from scipy import linalg

from bff_simulator.abstract_classes.abstract_ensemble import NVSpecies
from bff_simulator.experiment_parameters import ExperimentParameters
from bff_simulator.solver_parameters import SolverParam
from bff_simulator.unitary_solver import UnitarySolver
from bff_simulator.vector_manipulation import (
    perpendicular_projection,
)


class MasterEquationSolver(UnitarySolver):
    """
    Unitary solver for getting the output states of a double quantum pulse sequence.
    ASSUMES DEPHASING (Lindblad master equation).
    """

    SOLVER_KEY = 2.0

    def __init__(self) -> None:
        super().__init__()

    def solve(self, params: SolverParam) -> NDArray:
        initial_density_state = qutrit_basis()[1]

        collapse_operators = MasterEquationSolver._t_2_decay_operator(t_2_star_s=params.t2star_s)
        bare_hamiltonian = MasterEquationSolver._internal_hamiltonian(
            params.efield_splitting_hz, params.bz_field_t, params.detuning_hz
        )
        pulse_1_hamiltonian = bare_hamiltonian + MasterEquationSolver._control_hamiltonian(params.rabi_frequency_hz, 0)
        pulse_2_hamiltonian = bare_hamiltonian + MasterEquationSolver._control_hamiltonian(
            params.rabi_frequency_hz, params.second_pulse_phase_rad
        )

        density_operators_after_first_pulse = MasterEquationSolver._apply_first_uw_pulse(
            pulse_1_hamiltonian, initial_density_state, params.mw_pulse_length_s, collapse_operators
        )
        density_operators_after_free_evolution = MasterEquationSolver._apply_free_evolution_pulse(
            bare_hamiltonian, density_operators_after_first_pulse, params.evolution_time_s, collapse_operators
        )
        density_operator_after_second_pulse = self._apply_second_uw_pulse(
            pulse_2_hamiltonian,
            density_operators_after_free_evolution,
            np.array(params.mw_pulse_length_s),
            collapse_operators,
        )
        return density_operator_after_second_pulse

    @staticmethod
    def _t_2_decay_operator(t_2_star_s: float) -> Sequence[Qobj]:
        gamma = np.sqrt(2 / t_2_star_s)
        decay_paths = qutrit_ops()[(0, 2),]
        return (gamma * decay_paths).tolist()

    @staticmethod
    def _apply_first_uw_pulse(
        hamiltonian: Qobj,
        initial_density_operator: Qobj,
        mw_pulse_length_s: ArrayLike,
        collapse_operators: Sequence[Qobj],
    ) -> Sequence[Qobj]:
        first_uw_pulse_solution = mesolve(
            hamiltonian,
            initial_density_operator,
            mw_pulse_length_s,
            c_ops=collapse_operators,
            e_ops=[],
        )
        density_operators_after_uw_pulse = first_uw_pulse_solution.states
        return density_operators_after_uw_pulse

    @staticmethod
    def _apply_free_evolution_pulse(
        hamiltonian: Qobj,
        density_operators_after_uw_pulses: Sequence[Qobj],
        evolution_time_s: ArrayLike,
        collapse_operators: Sequence[Qobj],
    ) -> Sequence[Sequence[Qobj]]:
        free_evolution_density_operators: list = []
        for density_operator in density_operators_after_uw_pulses:
            solution = mesolve(
                hamiltonian,
                density_operator,
                evolution_time_s,
                c_ops=collapse_operators,
                e_ops=[],
            )
            density_operator_after_free_evolution = solution.states
            free_evolution_density_operators = [
                *free_evolution_density_operators,
                density_operator_after_free_evolution,
            ]
        return free_evolution_density_operators

    def _apply_second_uw_pulse(
        self,
        hamiltonian: Qobj,
        density_operators_after_free_evolution: Sequence[Sequence[Qobj]],
        mw_pulse_length_s: NDArray,
        collapse_operators: Sequence[Qobj],
    ) -> NDArray:
        number_of_rabi_increments, number_of_ramsey_increments = np.array(density_operators_after_free_evolution).shape
        number_of_spin_states = len(density_operators_after_free_evolution[0][0].full())
        final_kets = np.zeros((number_of_rabi_increments, number_of_ramsey_increments, number_of_spin_states))
        for rabi_index in np.arange(number_of_rabi_increments):
            for ramsey_index in np.arange(number_of_ramsey_increments):
                if self._pulse_length_is_too_short(
                    pulse_duration=mw_pulse_length_s[rabi_index],
                    largest_frequency_order_of_magnitude=hamiltonian.norm(),
                ):
                    final_density_operator = density_operators_after_free_evolution[rabi_index][ramsey_index]
                else:
                    pulse_evolution_result = mesolve(
                        hamiltonian,
                        density_operators_after_free_evolution[rabi_index][ramsey_index],
                        mw_pulse_length_s[0 : rabi_index + 1],
                        c_ops=collapse_operators,
                        e_ops=[],
                    )
                    final_density_operator = pulse_evolution_result.states[-1]
                final_kets[rabi_index, ramsey_index] = self._get_ket_without_relative_phase(final_density_operator)
        return final_kets

    @staticmethod
    def _pulse_length_is_too_short(
        pulse_duration: float,
        largest_frequency_order_of_magnitude: float,
        smallest_increment_of_an_oscillation_considered: float = 1e-2,
    ) -> bool:
        period_order_of_magnitude = 1 / largest_frequency_order_of_magnitude
        return abs(pulse_duration) < abs(period_order_of_magnitude * smallest_increment_of_an_oscillation_considered)

    @staticmethod
    def _get_ket_without_relative_phase(state: Qobj) -> NDArray:
        return np.real(state.diag() ** 0.5)

    @staticmethod
    def construct_solver_parameters(experiment_parameters: ExperimentParameters, nv_species: NVSpecies) -> SolverParam:  # type:ignore [override]
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
        return {"Solver": MasterEquationSolver.SOLVER_KEY}
