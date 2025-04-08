from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from bff_simulator.liouvillian_constructor import LiouvillianConstructor
from bff_simulator.offaxis_field_solver import OffAxisFieldSolver
from bff_simulator.offaxis_field_solver_parameters import OffAxisFieldSolverParam
from bff_simulator.propagator_manipulation import (
    expand_dq_unitaries,
    generate_propagators_from_evolution_matrix,
)
from bff_simulator.superoperator_to_matrix_functions import operator_density_operator_dagger_to_vector_multiplication


class LiouvillianSolver(OffAxisFieldSolver):
    SOLVER_KEY = 5.0

    def __init__(self) -> None:
        super().__init__()

    def solve(self, params: OffAxisFieldSolverParam) -> NDArray:  # type: ignore[override]
        # Find hamiltonians in the rotating frame and rotating wave approximation in the internal hamiltonian eigenbasis
        (
            ordered_eigenvector_matrix,
            h_rabi_eigenbasis_rwa_1,
            h_int_eigenbasis_rwa,
            h_rabi_eigenbasis_rwa_2,
        ) = self._eigenbasis_rwa_setup(params)

        # Construct liouvillians
        jump_operator_list = LiouvillianConstructor.spin_1_dephasing_jump_operators_eigenbasis(
            params.t2star_s, ordered_eigenvector_matrix
        )
        liouvillian_rabi_eigenbasis_1 = LiouvillianConstructor.construct_liouvillian(
            h_rabi_eigenbasis_rwa_1, jump_operator_list
        )
        liouvillian_ramsey_eigenbasis = LiouvillianConstructor.construct_liouvillian(
            h_int_eigenbasis_rwa, jump_operator_list
        )
        if LiouvillianSolver.identical_mw_phase(params):
            liouvillian_rabi_eigenbasis_2 = liouvillian_rabi_eigenbasis_1
        else:
            liouvillian_rabi_eigenbasis_2 = LiouvillianConstructor.construct_liouvillian(
                h_rabi_eigenbasis_rwa_2, jump_operator_list
            )

        # Compute propagators along each axis
        (
            rabi_propagators_eigenbasis_1,
            ramsey_propagators_eigenbasis,
            rabi_propagators_eigenbasis_2,
        ) = self._propagate_liouvillians(
            params, liouvillian_rabi_eigenbasis_1, liouvillian_ramsey_eigenbasis, liouvillian_rabi_eigenbasis_2
        )

        # Get initial density vector in the eigenbasis
        initial_density_vector_eigenbasis = LiouvillianConstructor.initial_density_vector_eigenbasis(
            ordered_eigenvector_matrix
        )

        # Tack on the conversion back to the bare basis on the second rabi pulse (more efficient than doing it after expand_dq_unitaries)
        # For more efficiency but less readability, this could be done while generating the second rabi propagators.
        conversion_matrix = operator_density_operator_dagger_to_vector_multiplication(ordered_eigenvector_matrix)
        rabi_propagators_2_convert_to_bare_basis = np.array(
            [conversion_matrix @ propagator for propagator in rabi_propagators_eigenbasis_2]
        )

        final_density_vectors_bare_basis = expand_dq_unitaries(
            rabi_propagators_eigenbasis_1,
            ramsey_propagators_eigenbasis,
            rabi_propagators_2_convert_to_bare_basis,
            initial_density_vector_eigenbasis,
        )

        # Convert density vectors to a 3-vector of sqrt(populations) for compatibility with Simulator
        return self._density_vectors_to_sqrt_populations(final_density_vectors_bare_basis)

    def _density_vectors_to_sqrt_populations(self, density_vectors: NDArray) -> NDArray:
        density_vector_dimension = density_vectors.shape[-1]
        density_matrix_dimension = np.sqrt(density_vector_dimension)
        n = round(density_matrix_dimension)
        if not np.isclose(density_matrix_dimension, n):
            raise ValueError("Density vectors should be n**2-component one-dimensional arrays")
        diagonal_elements = [i * (n + 1) for i in range(n)]
        return np.sqrt(density_vectors[..., diagonal_elements])

    def _propagate_liouvillians(
        self,
        params: OffAxisFieldSolverParam,
        liouvillian_rabi_1: NDArray,
        liouvillian_ramsey: NDArray,
        liouvillian_rabi_2: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        # generate the Rabi and Ramsey Liouvillian propagators in the eigenbasis
        rabi_propagators_1 = generate_propagators_from_evolution_matrix(
            np.array(params.mw_pulse_length_s), liouvillian_rabi_1
        )
        if LiouvillianSolver.identical_mw_phase(params):
            rabi_propagators_2 = rabi_propagators_1
        else:
            rabi_propagators_2 = generate_propagators_from_evolution_matrix(
                np.array(params.mw_pulse_length_s), liouvillian_rabi_2
            )
        ramsey_propagators = generate_propagators_from_evolution_matrix(
            np.array(params.evolution_time_s), liouvillian_ramsey
        )

        return rabi_propagators_1, ramsey_propagators, rabi_propagators_2

    def get_metadata_dict(self) -> Dict[str, float]:
        return {"Solver": LiouvillianSolver.SOLVER_KEY}
