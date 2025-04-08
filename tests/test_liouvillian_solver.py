import unittest

import numpy as np
from numpy.linalg import norm

from bff_simulator.abstract_classes.abstract_ensemble import NV14HyperfineField, NVOrientation
from bff_simulator.constants import exy
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory
from bff_simulator.simulator import Simulation

MW_DIRECTION = np.array([0.97203398, 0.2071817, 0.11056978])
E_FIELD_VECTOR_V_PER_CM = np.array([1e5, 3e5, 0]) / exy
B_FIELD_VECTOR_T = [5e-5 * x for x in [4, 3, 1]]
RABI_TIMES = np.linspace(0, 1e-7, 301)
RAMSEY_TIMES = np.linspace(0, 1e-5, 301)
RABI_FREQ_BASE_HZ = 100e6
DETUNING_HZ = 5e6
SECOND_PULSE_PHASE = np.pi / 3
MW_PULSE_LENGTH_S = [1e-6 * x for x in [0.01, 0.02]]
EVOLUTION_TIME_S = [1e-6 * x for x in [1, 2]]
T2STAR_S = 1e16


class TestLiouvillianSolver(unittest.TestCase):
    # This performs the same test as for the OffAxisFieldSolver with a very large value of t2star (where the two should give the same results)
    def test_liouvillian_solver_no_dephasing(self) -> None:
        nv_ensemble = HomogeneousEnsemble()
        nv_ensemble.efield_splitting_hz = norm(E_FIELD_VECTOR_V_PER_CM) * exy
        nv_ensemble.t2_star_s = T2STAR_S
        nv_ensemble.add_nv_single_species(NVOrientation.A, NV14HyperfineField.N14_0)
        nv_ensemble.mw_direction = MW_DIRECTION

        exp_param_factory = OffAxisFieldExperimentParametersFactory()
        exp_param_factory.set_base_rabi_frequency(RABI_FREQ_BASE_HZ)
        exp_param_factory.set_mw_direction(MW_DIRECTION)
        exp_param_factory.set_e_field_v_per_m(E_FIELD_VECTOR_V_PER_CM)
        exp_param_factory.set_b_field_vector(B_FIELD_VECTOR_T)
        exp_param_factory.set_detuning(DETUNING_HZ)
        exp_param_factory.set_second_pulse_phase(SECOND_PULSE_PHASE)
        exp_param_factory.set_mw_pulse_lengths(MW_PULSE_LENGTH_S)
        exp_param_factory.set_evolution_times(EVOLUTION_TIME_S)
        off_axis_experiment_parameters = exp_param_factory.get_experiment_parameters()

        off_axis_solver = LiouvillianSolver()

        off_axis_simulation = Simulation(off_axis_experiment_parameters, nv_ensemble, off_axis_solver)
        expected_ms0_results = [
            [0.73848571, 0.29177136],
            [0.47807185, 0.28555257],
        ]  # From python output for OffAxisFieldSolver
        test_ms0_results = off_axis_simulation.ms0_results
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(expected_ms0_results[i][j], test_ms0_results[i][j])


if __name__ == "__main__":
    unittest.main()
