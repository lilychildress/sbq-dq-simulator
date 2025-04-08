import numpy as np
from qutip import Qobj, commutator, jmat
from scipy import linalg

from bff_simulator.abstract_classes.abstract_ensemble import NVSpecies
from bff_simulator.abstract_classes.abstract_solver import Solver
from bff_simulator.constants import gammab
from bff_simulator.geometric_experiment_parameters import (
    GeometricExperimentParameters,
)
from bff_simulator.geometric_solver_parameters import (
    GeometricSolverParam,
)
from bff_simulator.vector_manipulation import perpendicular_projection


class GeometricSolver(Solver):
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
    def _control_hamiltonian(rabi_frequency_hz: float, phase_rad: float, geometric_angle_rad: float = 0) -> Qobj:
        sx_matrix, sy_matrix, sz_matrix = jmat(1)
        sy_prime_matrix = 1.0j * commutator(sz_matrix * sz_matrix, sx_matrix)
        sx_prime_matrix = 1.0j * commutator(sz_matrix * sz_matrix, sy_matrix)
        return (
            2
            * np.pi
            * rabi_frequency_hz
            * (
                np.cos(geometric_angle_rad) * (np.cos(phase_rad) * sx_matrix - np.sin(phase_rad) * sy_prime_matrix)
                + np.sin(geometric_angle_rad) * (np.cos(phase_rad) * sy_matrix - np.sin(phase_rad) * sx_prime_matrix)
            )
        )

    @staticmethod
    def construct_solver_parameters(  # type: ignore[override]
        experiment_parameters: GeometricExperimentParameters,
        nv_species: NVSpecies,
    ) -> GeometricSolverParam:
        nv_axis_unit_vector = np.array(nv_species.axis_vector) / linalg.norm(nv_species.axis_vector)

        bz_field_t = (
            np.dot(experiment_parameters.b_field_vector_t, nv_axis_unit_vector) + nv_species.residual_bz_field_t
        )
        bperp_field_t = perpendicular_projection(experiment_parameters.b_field_vector_t, nv_axis_unit_vector)

        rabi_frequency_hz = experiment_parameters.rabi_frequency_base_hz * nv_species.rabi_projection

        solver_param = GeometricSolverParam(
            experiment_parameters.mw_pulse_length_s,
            experiment_parameters.evolution_time_s,
            experiment_parameters.second_pulse_phase_rad,
            experiment_parameters.detuning_hz,
            rabi_frequency_hz,
            bz_field_t,
            bperp_field_t,
            nv_species.efield_splitting_hz,
            nv_species.t2_star_s,
            experiment_parameters.second_pulse_geometric_angle_rad,
        )

        return solver_param
