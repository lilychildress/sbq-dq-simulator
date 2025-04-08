import numpy as np
from numpy.linalg import eig, inv
from numpy.typing import NDArray
from qutip import jmat

from bff_simulator.constants import gammab


class OffAxisFieldHamiltonian:
    def __init__(self) -> None:
        pass

    MS_PLUS_1_INDEX = int(np.where(np.isclose(jmat(1)[2].full(), 1))[0].item())
    MS0_INDEX = int(np.where(np.isclose(np.diag(jmat(1)[2].full()), 0))[0].item())
    MS_MINUS_1_INDEX = int(np.where(np.isclose(jmat(1)[2].full(), -1))[0].item())

    @staticmethod
    def get_spin_1_matrices() -> tuple:
        sx_qobj, sy_qobj, sz_qobj = jmat(1)
        return sx_qobj.full(), sy_qobj.full(), sz_qobj.full()

    @staticmethod
    def internal_hamiltonian_bare(
        e_field_vector_hz: NDArray, b_field_vector_t: NDArray, zero_field_splitting_hz: float
    ) -> NDArray:
        sx, sy, sz = OffAxisFieldHamiltonian.get_spin_1_matrices()
        bx, by, bz = gammab * b_field_vector_t
        ex, ey, ez = e_field_vector_hz
        return (
            2
            * np.pi
            * (
                (zero_field_splitting_hz + ez) * sz @ sz
                + bx * sx
                + by * sy
                + bz * sz
                + ex * (sx @ sx - sy @ sy)
                + ey * (sx @ sy + sy @ sx)
            )
        )

    @staticmethod
    def drive_hamiltonian_bare_no_cosine(mw_amplitude_vector_hz: NDArray) -> NDArray:
        sx, sy, sz = OffAxisFieldHamiltonian.get_spin_1_matrices()
        omega_x, omega_y, omega_z = mw_amplitude_vector_hz

        return 2 * np.pi * (omega_x * sx + omega_y * sy + omega_z * sz)

    @staticmethod
    def rotating_frame_and_rwa_multiplier(mw_phase_rad: float) -> NDArray:
        sx, _, sz = OffAxisFieldHamiltonian.get_spin_1_matrices()
        return (np.cos(mw_phase_rad) * sx - 1.0j * np.sin(mw_phase_rad) * (sz @ sz @ sx - sx @ sz @ sz)) / np.sqrt(2)

    @staticmethod
    def get_ordered_spin_1_eigensystem(hamiltonian: NDArray) -> tuple:
        evals, evects = eig(hamiltonian)
        # we want to return evects with columns in the order of eigenvectors that are closest to
        # ms = +1, ms = 0, ms = -1, corresponding to positions 0, 1, and 2.  Here, we figure out the order in which we want
        # the three eigenvectors to be returned. In the unlikely event that the
        # ms = +/-1 states are fully mixed, assign order randomly between them (this will not affect the algorithm).
        eig_vect_order: list[int] = []
        for eigenvector in np.transpose(evects):
            position: int = int(np.argmax(abs(eigenvector)))
            if position in eig_vect_order:
                print(
                    "Poorly-determined order for eigenvectors: ms = +/-1-like order set by numpy.eig output; ms=0-like state assumed recognizable."
                )
                position = int(
                    list(
                        set([OffAxisFieldHamiltonian.MS_MINUS_1_INDEX, OffAxisFieldHamiltonian.MS_PLUS_1_INDEX])
                        - set(eig_vect_order)
                    )[0]
                )
            eig_vect_order.append(position)
        # Convert the desired order into a list of indices by figuring out what is the index of the eigenvector that should come first,
        # then what is the index of the eigenvector that should come second, etc.
        indices = [
            eig_vect_order.index(x)
            for x in [
                OffAxisFieldHamiltonian.MS_PLUS_1_INDEX,
                OffAxisFieldHamiltonian.MS0_INDEX,
                OffAxisFieldHamiltonian.MS_MINUS_1_INDEX,
            ]
        ]
        return evals[indices], evects[:, indices]

    @staticmethod
    def drive_hamiltonian_eigenbasis_rwa(
        ordered_eigenvector_matrix: NDArray,
        drive_hamiltonian_bare_no_cosine: NDArray,
        rotating_frame_and_rwa_multiplier: NDArray,
    ) -> NDArray:
        drive_hamiltonian_eigenbasis_no_cosine = (
            inv(ordered_eigenvector_matrix) @ drive_hamiltonian_bare_no_cosine @ ordered_eigenvector_matrix
        )
        return np.multiply(rotating_frame_and_rwa_multiplier, drive_hamiltonian_eigenbasis_no_cosine)

    @staticmethod
    def internal_hamiltonian_eigenbasis_rwa(ordered_spin_1_eigenvalues: NDArray, mw_frequency_hz: float) -> NDArray:
        _, _, sz = OffAxisFieldHamiltonian.get_spin_1_matrices()
        return np.diag(ordered_spin_1_eigenvalues) - 2 * np.pi * mw_frequency_hz * sz @ sz

    @staticmethod
    def get_ms0_state_in_eigenbasis(ordered_eigenvector_matrix: NDArray) -> NDArray:
        return np.conjugate(ordered_eigenvector_matrix[OffAxisFieldHamiltonian.MS0_INDEX])
