import numpy as np
from numpy.typing import NDArray

# The following three functions implement superoperators on the density vector. Each returns
# a 9x9 matrix that, when multiplying the 9-component density vector, reproduces terms obtained from
# matrix multiplication of the operator and density matrix (as a matrix, not vector)
# If we define vec() as the action of flattening the nxn density matrix to a n**2-component vector,
# "tensor" as the tensor product, and identity(n) as the nxn identity matrix, the underlying math is:
# vec(A @ B) = (A tensor identity(n)) @ vec(B)
# vec(B @ A) = (identity(n) tensor transpose(A)) @ vec(B)
# vec(A @ B @ (transpose(conjugate(A)))) = (A tensor conjugate(A)) @ vec(B)


def left_matrix_multiplication_to_vector_multiplication(operator: NDArray) -> NDArray:
    matrix_dimension = len(operator)
    assert operator.shape == (matrix_dimension, matrix_dimension), "Operator must be an nxn matrix"
    return np.kron(operator, np.identity(matrix_dimension))


def right_matrix_multiplication_to_vector_multiplication(operator: NDArray) -> NDArray:
    matrix_dimension = len(operator)
    assert operator.shape == (matrix_dimension, matrix_dimension), "Operator must be an nxn matrix"
    return np.kron(np.identity(matrix_dimension), np.transpose(operator))


def operator_density_operator_dagger_to_vector_multiplication(operator: NDArray) -> NDArray:
    matrix_dimension = len(operator)
    assert operator.shape == (matrix_dimension, matrix_dimension), "Operator must be an nxn matrix"
    return np.kron(operator, np.conj(operator))
