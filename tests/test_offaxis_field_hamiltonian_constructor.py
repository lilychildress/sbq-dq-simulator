import unittest

import numpy as np

from bff_simulator.constants import gammab
from bff_simulator.offaxis_field_hamiltonian_constructor import OffAxisFieldHamiltonian


class TestOffAxisHamiltonianConstructor(unittest.TestCase):
    def test_internal_hamiltonian_bare(self) -> None:
        e_field_hz = np.array([20 / (2 * np.pi), 1 / (2 * np.pi), 0]) * 1e6
        b_field_t = np.array([50, 50, 10]) * 1e6 / gammab
        zfs_hz = 2.87e9
        test_hamiltonian = OffAxisFieldHamiltonian.internal_hamiltonian_bare(e_field_hz, b_field_t, zfs_hz)

        expected_hamiltonian = (
            np.array(
                [
                    [18095.57368467721, 222.1441469079183 - 222.1441469079183j, 20.0 - 1.0j],
                    [222.1441469079183 + 222.1441469079183j, 0.0, 222.1441469079183 - 222.1441469079183j],
                    [20.0 + 1.0j, 222.1441469079183 + 222.1441469079183j, 17969.90997853362],
                ]
            )
            * 1e6
        )

        for i in range(3):
            for j in range(3):
                with self.subTest():
                    self.assertAlmostEqual(test_hamiltonian[i, j], expected_hamiltonian[i, j], places=5)

    def test_drive_hamiltonian_bare_no_cosine(self) -> None:
        mw_vect_hz = np.array([2.5, 4.330127018922193, 10.0]) * 1e6
        test_hamiltonian = OffAxisFieldHamiltonian.drive_hamiltonian_bare_no_cosine(mw_vect_hz)
        expected_hamiltonian = (
            np.array(
                [
                    [62.83185307179586, 11.107207345395919 - 19.23824745242796j, 0.0],
                    [11.107207345395919 + 19.23824745242796j, 0.0, 11.107207345395919 - 19.23824745242796j],
                    [0.0, 11.107207345395919 + 19.23824745242796j, -62.83185307179586],
                ]
            )
            * 1e6
        )

        for i in range(3):
            for j in range(3):
                with self.subTest():
                    self.assertAlmostEqual(test_hamiltonian[i, j], expected_hamiltonian[i, j], places=5)

    def test_eigensystem(self) -> None:
        e_field_hz = np.array([20 / (2 * np.pi), 1 / (2 * np.pi), 0]) * 1e6
        b_field_t = np.array([50, 50, 10]) * 1e6 / gammab
        zfs_hz = 2.87e9
        int_hamiltonian = OffAxisFieldHamiltonian.internal_hamiltonian_bare(e_field_hz, b_field_t, zfs_hz)
        test_evals, test_evects = OffAxisFieldHamiltonian.get_ordered_spin_1_eigensystem(int_hamiltonian)

        expected_evals = np.array([18104.44633787797, -10.93921934930404, 17971.97654468214]) * 1e6

        expected_evects = np.array(
            [
                [
                    0.9868430547358201 + 0.0j,
                    -0.012250716044831717 + 0.012278000719463146j,
                    -0.15289565568505148 + 0.04962735583360206j,
                ],
                [
                    0.014589965838091665 + 0.010837797303455939j,
                    0.9996970148710742 + 0.0j,
                    0.009694812995191478 - 0.01347457254625017j,
                ],
                [
                    0.15289832037336026 + 0.049321029927737785j,
                    -0.012336433378209301 - 0.012363718052840702j,
                    0.9868558610664921 + 0.0j,
                ],
            ]
        )

        for i in range(3):
            with self.subTest():
                self.assertAlmostEqual(test_evals[i], expected_evals[i], places=4)
            for j in range(3):
                with self.subTest():
                    self.assertAlmostEqual(test_evects[i, j], expected_evects[i, j], places=7)

    def test_h_drive_eigenbasis_rwa(self) -> None:
        e_field_hz = np.array([20 / (2 * np.pi), 1 / (2 * np.pi), 0]) * 1e6
        b_field_t = np.array([50, 50, 10]) * 1e6 / gammab
        zfs_hz = 2.87e9
        mw_phase_rad = np.pi / 4
        int_hamiltonian = OffAxisFieldHamiltonian.internal_hamiltonian_bare(e_field_hz, b_field_t, zfs_hz)
        _, evects = OffAxisFieldHamiltonian.get_ordered_spin_1_eigensystem(int_hamiltonian)
        mw_vect_hz = np.array([2.5, 4.330127018922193, 10.0]) * 1e6
        drive_hamiltonian_bare = OffAxisFieldHamiltonian.drive_hamiltonian_bare_no_cosine(mw_vect_hz)
        rotating_frame_and_rwa_multiplier = OffAxisFieldHamiltonian.rotating_frame_and_rwa_multiplier(mw_phase_rad)

        test_hamiltonian = OffAxisFieldHamiltonian.drive_hamiltonian_eigenbasis_rwa(
            evects, drive_hamiltonian_bare, rotating_frame_and_rwa_multiplier
        )

        expected_hamiltonian = (
            np.array(
                [
                    [0.0, -0.9708838154675661 - 10.156482755869202j, 0.0],
                    [-0.970883815467567 + 10.156482755869206j, 0.0, 11.053952231586663 - 4.534878709967852j],
                    [0.0, 11.05395223158666 + 4.534878709967849j, 0.0],
                ]
            )
            * 1e6
        )

        for i in range(3):
            for j in range(3):
                with self.subTest():
                    self.assertAlmostEqual(test_hamiltonian[i, j], expected_hamiltonian[i, j], places=5)

        # Note: fully off-axis edge case is tricky because the phase and ordering of the eigenvectors is ambiguous. This test may fail without necessarily
        # breaking the algorithm if the order or phase of the ms = +/-1 - like eigenstates is changed.
        b_field_t = np.array([50, 50, 0]) * 1e6 / gammab
        int_hamiltonian = OffAxisFieldHamiltonian.internal_hamiltonian_bare(e_field_hz, b_field_t, zfs_hz)
        _, evects = OffAxisFieldHamiltonian.get_ordered_spin_1_eigensystem(int_hamiltonian)

        mw_vect_hz = np.array([2.5, 4.330127018922193, 10.0]) * 1e6
        drive_hamiltonian_bare = OffAxisFieldHamiltonian.drive_hamiltonian_bare_no_cosine(mw_vect_hz)
        rotating_frame_and_rwa_multiplier = OffAxisFieldHamiltonian.rotating_frame_and_rwa_multiplier(mw_phase_rad)

        test_hamiltonian = OffAxisFieldHamiltonian.drive_hamiltonian_eigenbasis_rwa(
            evects, drive_hamiltonian_bare, rotating_frame_and_rwa_multiplier
        )

        expected_hamiltonian = 1e6 * np.array(
            [
                [0.0, 6.17216792911417 - 7.707529910903584 * 1.0j, 0.0],
                [6.172167929114171 + 7.70752991090358 * 1.0j, 0.0, 7.68179277490455 - 9.500454055162871 * 1.0j],
                [0.0, 7.681792774904547 + 9.500454055162873 * 1.0j, 0.0],
            ]
        )
        for i in range(3):
            for j in range(3):
                with self.subTest():
                    self.assertAlmostEqual(test_hamiltonian[i, j], expected_hamiltonian[i, j], places=5)


if __name__ == "__main__":
    unittest.main()
