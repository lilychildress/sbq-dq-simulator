from typing import List

import numpy as np
from numpy.typing import NDArray

from bff_simulator.offaxis_field_hamiltonian_constructor import OffAxisFieldHamiltonian
from bff_simulator.superoperator_to_matrix_functions import (
    left_matrix_multiplication_to_vector_multiplication,
    operator_density_operator_dagger_to_vector_multiplication,
    right_matrix_multiplication_to_vector_multiplication,
)


class LiouvillianConstructor:
    def __init__(self) -> None:
        pass

    @staticmethod
    def jump_term(jump_operator: NDArray) -> NDArray:
        return (
            operator_density_operator_dagger_to_vector_multiplication(jump_operator)
            - (1 / 2)
            * left_matrix_multiplication_to_vector_multiplication(np.conj(np.transpose(jump_operator)) @ jump_operator)
            - (1 / 2)
            * right_matrix_multiplication_to_vector_multiplication(np.conj(np.transpose(jump_operator)) @ jump_operator)
        )

    @staticmethod
    def coherent_terms(hamiltonian: NDArray) -> NDArray:
        return -1.0j * (
            left_matrix_multiplication_to_vector_multiplication(hamiltonian)
            - right_matrix_multiplication_to_vector_multiplication(hamiltonian)
        )

    @staticmethod
    def construct_liouvillian(hamiltonian: NDArray, list_of_jump_operators: List[NDArray]) -> NDArray:
        liouvillian = LiouvillianConstructor.coherent_terms(hamiltonian)
        for jump_operator in list_of_jump_operators:
            assert jump_operator.shape == hamiltonian.shape, (
                "Jump operators must have the same shape as the Hamiltonian"
            )
            liouvillian += LiouvillianConstructor.jump_term(jump_operator)
        return liouvillian

    @staticmethod
    def convert_operator_to_eigenbasis(operator: NDArray, ordered_eigenvector_column_matrix: NDArray) -> NDArray:
        return np.linalg.inv(ordered_eigenvector_column_matrix) @ operator @ ordered_eigenvector_column_matrix

    @staticmethod
    def spin_1_dephasing_jump_operators_eigenbasis(
        t2star_s: float, ordered_eigenvector_matrix: NDArray = np.identity(3)
    ) -> List[NDArray]:
        jump1_bare_basis = np.sqrt(2 / t2star_s) * np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        jump2_bare_basis = np.sqrt(2 / t2star_s) * np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        return [
            LiouvillianConstructor.convert_operator_to_eigenbasis(jump1_bare_basis, ordered_eigenvector_matrix),
            LiouvillianConstructor.convert_operator_to_eigenbasis(jump2_bare_basis, ordered_eigenvector_matrix),
        ]

    @staticmethod
    def initial_density_vector_eigenbasis(ordered_eigenvector_column_matrix: NDArray) -> NDArray:
        initial_state = OffAxisFieldHamiltonian.get_ms0_state_in_eigenbasis(ordered_eigenvector_column_matrix)
        initial_density_vector = np.kron(initial_state, np.conj(initial_state))
        return initial_density_vector
