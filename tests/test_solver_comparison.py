import unittest

import numpy as np
from numpy.linalg import norm

from bff_simulator.abstract_classes.abstract_ensemble import NV14HyperfineField, NVOrientation
from bff_simulator.constants import exy
from bff_simulator.experiment_parameters import ExperimentParametersFactory
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.master_equation_solver import MasterEquationSolver
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory
from bff_simulator.offaxis_field_solver import OffAxisFieldSolver
from bff_simulator.simulator import Simulation
from bff_simulator.unitary_solver import UnitarySolver

MW_DIRECTION = np.array([0.97203398, 0.2071817, 0.11056978])
NVA_X_AXIS_MW_DIRECTION = np.array([0.78867513, -0.21132487, -0.57735027])

E_FIELD_VECTOR_V_PER_CM = np.array([1e5, 3e5, 0]) / exy
NVA_X_AXIS_E_FIELD_V_PER_CM = 10e6 * np.array([0.78867513, -0.21132487, -0.57735027]) / exy

OFF_AXIS_B_FIELD_VECTOR_T = [5e-5 * x for x in [-1, 1, 0]]
NVA_AXIAL_B_FIELD_VECTOR_T = [5e-5 * x for x in [1, 1, 1]]

RABI_FREQ_BASE_HZ = 100e6
DETUNING_HZ = 5e6
SECOND_PULSE_PHASE = np.pi / 3
T2STAR_S = 0.5e-6

# Usable by OffAxisFieldSolver and LiouvillianSolver
RABI_TIMES_S_NONZERO_START = [1e-6 * x for x in [0.01, 0.02, 0.03]]
EVOLUTION_TIME_S_NONZERO_START = [1e-6 * x for x in [1, 2]]
# Needed for UnitarySolver and MasterEquationSolver to behave correctly:
RABI_TIMES_S_ZERO_START = [1e-6 * x for x in [0, 0.01, 0.02, 0.03]]
RAMSEY_TIMES_S_ZERO_START = [1e-6 * x for x in [0, 1, 2]]


class TestSolverComparison(unittest.TestCase):
    def test_liouvillian_solver_vs_off_axis_solver(self) -> None:
        nv_ensemble = HomogeneousEnsemble()
        nv_ensemble.t2_star_s = 1e16  # Must be large for this test to pass
        nv_ensemble.add_nv_single_species(NVOrientation.A, NV14HyperfineField.N14_0)

        exp_param_factory = OffAxisFieldExperimentParametersFactory()
        exp_param_factory.set_base_rabi_frequency(RABI_FREQ_BASE_HZ)
        exp_param_factory.set_mw_direction(MW_DIRECTION)
        exp_param_factory.set_e_field_v_per_m(E_FIELD_VECTOR_V_PER_CM)
        exp_param_factory.set_b_field_vector(OFF_AXIS_B_FIELD_VECTOR_T)
        exp_param_factory.set_detuning(DETUNING_HZ)
        exp_param_factory.set_second_pulse_phase(SECOND_PULSE_PHASE)
        exp_param_factory.set_mw_pulse_lengths(RABI_TIMES_S_NONZERO_START)
        exp_param_factory.set_evolution_times(EVOLUTION_TIME_S_NONZERO_START)
        off_axis_experiment_parameters = exp_param_factory.get_experiment_parameters()

        liouvillian_solver = LiouvillianSolver()
        off_axis_solver = OffAxisFieldSolver()
        liouvillian_simulation = Simulation(off_axis_experiment_parameters, nv_ensemble, liouvillian_solver)
        test_ms0_results = liouvillian_simulation.ms0_results

        off_axis_simulation = Simulation(off_axis_experiment_parameters, nv_ensemble, off_axis_solver)
        expected_ms0_results = off_axis_simulation.ms0_results

        for i in range(len(RABI_TIMES_S_NONZERO_START)):
            for j in range(len(EVOLUTION_TIME_S_NONZERO_START)):
                self.assertAlmostEqual(expected_ms0_results[i][j], test_ms0_results[i][j])

    def test_master_equation_solver_vs_unitary_solver(self) -> None:
        nv_ensemble = HomogeneousEnsemble()
        nv_ensemble.efield_splitting_hz = (
            0  # NOTE: This test fails if there is a ~10kHz+ efield splitting and nonzero second pulse phase
        )
        nv_ensemble.t2_star_s = 1e16  # Must be large for this test to pass
        nv_ensemble.add_nv_single_species(NVOrientation.A, NV14HyperfineField.N14_0)
        nv_ensemble.mw_direction = MW_DIRECTION

        exp_param_factory = ExperimentParametersFactory()
        exp_param_factory.set_b_field_vector(NVA_AXIAL_B_FIELD_VECTOR_T)
        exp_param_factory.set_evolution_times(RAMSEY_TIMES_S_ZERO_START)
        exp_param_factory.set_mw_pulse_lengths(RABI_TIMES_S_ZERO_START)
        exp_param_factory.set_base_rabi_frequency(RABI_FREQ_BASE_HZ / 2)
        exp_param_factory.set_second_pulse_phase(
            SECOND_PULSE_PHASE
        )  # NOTE: This test fails if there is a nonzero efield and nonzero second pulse phase
        experiment_parameters = exp_param_factory.get_experiment_parameters()

        unitary_solver = UnitarySolver()
        unitary_simulation = Simulation(experiment_parameters, nv_ensemble, unitary_solver)
        unitary_ms0_results = unitary_simulation.ms0_results

        master_equation_solver = MasterEquationSolver()
        master_equation_simulation = Simulation(experiment_parameters, nv_ensemble, master_equation_solver)
        master_equation_ms0_results = master_equation_simulation.ms0_results

        for i in range(len(RABI_TIMES_S_ZERO_START)):
            for j in range(len(RAMSEY_TIMES_S_ZERO_START)):
                self.assertAlmostEqual(unitary_ms0_results[i][j], master_equation_ms0_results[i][j], places=3)

    def test_off_axis_solver_vs_unitary_solver(self) -> None:
        nv_ensemble = HomogeneousEnsemble()
        nv_ensemble.efield_splitting_hz = norm(NVA_X_AXIS_E_FIELD_V_PER_CM * exy)
        nv_ensemble.add_nv_single_species(NVOrientation.A, NV14HyperfineField.N14_0)
        nv_ensemble.mw_direction = NVA_X_AXIS_MW_DIRECTION

        # Parameters for unitary solver
        exp_param_factory = ExperimentParametersFactory()
        exp_param_factory.set_b_field_vector(NVA_AXIAL_B_FIELD_VECTOR_T)
        exp_param_factory.set_evolution_times(RAMSEY_TIMES_S_ZERO_START)
        exp_param_factory.set_mw_pulse_lengths(RABI_TIMES_S_ZERO_START)
        exp_param_factory.set_base_rabi_frequency(
            RABI_FREQ_BASE_HZ / 2
        )  # Accounts for factor of 2 error in Unitary Solver Rabi frequency
        exp_param_factory.set_second_pulse_phase(SECOND_PULSE_PHASE)
        experiment_parameters = exp_param_factory.get_experiment_parameters()

        unitary_solver = UnitarySolver()
        unitary_simulation = Simulation(experiment_parameters, nv_ensemble, unitary_solver)
        unitary_ms0_results = unitary_simulation.ms0_results

        # Parameters for off-axis solver
        exp_param_factory = OffAxisFieldExperimentParametersFactory()
        exp_param_factory.set_base_rabi_frequency(RABI_FREQ_BASE_HZ)
        exp_param_factory.set_mw_direction(
            NVA_X_AXIS_MW_DIRECTION
        )  # Need MW field to be along x in NV coordinates to match UnitarySolver
        exp_param_factory.set_e_field_v_per_m(
            NVA_X_AXIS_E_FIELD_V_PER_CM
        )  # Need E field to be along x in NV coordinates to match UnitarySolver
        exp_param_factory.set_b_field_vector(NVA_AXIAL_B_FIELD_VECTOR_T)
        exp_param_factory.set_evolution_times(RAMSEY_TIMES_S_ZERO_START)
        exp_param_factory.set_mw_pulse_lengths(RABI_TIMES_S_ZERO_START)
        exp_param_factory.set_second_pulse_phase(SECOND_PULSE_PHASE)
        off_axis_experiment_parameters = exp_param_factory.get_experiment_parameters()

        off_axis_solver = OffAxisFieldSolver()
        off_axis_simulation = Simulation(off_axis_experiment_parameters, nv_ensemble, off_axis_solver)
        off_axis_ms0_results = off_axis_simulation.ms0_results

        for i in range(len(RABI_TIMES_S_ZERO_START)):
            for j in range(len(RAMSEY_TIMES_S_ZERO_START)):
                self.assertAlmostEqual(unitary_ms0_results[i][j], off_axis_ms0_results[i][j], places=4)

    def test_liouvillian_solver_vs_unitary_solver(self) -> None:
        nv_ensemble = HomogeneousEnsemble()
        nv_ensemble.efield_splitting_hz = norm(NVA_X_AXIS_E_FIELD_V_PER_CM) * exy
        nv_ensemble.t2_star_s = 1e16  # Must be large for this test to pass
        nv_ensemble.add_nv_single_species(NVOrientation.A, NV14HyperfineField.N14_0)
        nv_ensemble.mw_direction = NVA_X_AXIS_MW_DIRECTION

        # Parameters for unitary solver
        exp_param_factory = ExperimentParametersFactory()
        exp_param_factory.set_b_field_vector(NVA_AXIAL_B_FIELD_VECTOR_T)
        exp_param_factory.set_evolution_times(RAMSEY_TIMES_S_ZERO_START)
        exp_param_factory.set_mw_pulse_lengths(RABI_TIMES_S_ZERO_START)
        exp_param_factory.set_base_rabi_frequency(
            RABI_FREQ_BASE_HZ / 2
        )  # Accounts for factor of 2 error in Unitary Solver Rabi frequency
        exp_param_factory.set_second_pulse_phase(SECOND_PULSE_PHASE)
        experiment_parameters = exp_param_factory.get_experiment_parameters()

        unitary_solver = UnitarySolver()
        unitary_simulation = Simulation(experiment_parameters, nv_ensemble, unitary_solver)
        unitary_ms0_results = unitary_simulation.ms0_results

        # Parameters for liouvillian solver
        exp_param_factory = OffAxisFieldExperimentParametersFactory()
        exp_param_factory.set_base_rabi_frequency(RABI_FREQ_BASE_HZ)
        exp_param_factory.set_mw_direction(NVA_X_AXIS_MW_DIRECTION)
        exp_param_factory.set_e_field_v_per_m(NVA_X_AXIS_E_FIELD_V_PER_CM)
        exp_param_factory.set_b_field_vector(NVA_AXIAL_B_FIELD_VECTOR_T)
        exp_param_factory.set_evolution_times(RAMSEY_TIMES_S_ZERO_START)
        exp_param_factory.set_mw_pulse_lengths(RABI_TIMES_S_ZERO_START)
        exp_param_factory.set_second_pulse_phase(SECOND_PULSE_PHASE)
        off_axis_experiment_parameters = exp_param_factory.get_experiment_parameters()

        liouvillian_solver = LiouvillianSolver()
        liouvillian_simulation = Simulation(off_axis_experiment_parameters, nv_ensemble, liouvillian_solver)
        liouvillian_ms0_results = liouvillian_simulation.ms0_results

        for i in range(len(RABI_TIMES_S_ZERO_START)):
            for j in range(len(RAMSEY_TIMES_S_ZERO_START)):
                self.assertAlmostEqual(unitary_ms0_results[i][j], liouvillian_ms0_results[i][j], places=4)

    def test_liouvillian_solver_vs_master_equation_solver(self) -> None:
        nv_ensemble = HomogeneousEnsemble()
        nv_ensemble.efield_splitting_hz = norm(NVA_X_AXIS_E_FIELD_V_PER_CM) * exy
        nv_ensemble.t2_star_s = T2STAR_S  # Must be large for this test to pass
        nv_ensemble.add_nv_single_species(NVOrientation.A, NV14HyperfineField.N14_0)
        nv_ensemble.mw_direction = NVA_X_AXIS_MW_DIRECTION

        exp_param_factory = ExperimentParametersFactory()
        exp_param_factory.set_b_field_vector(NVA_AXIAL_B_FIELD_VECTOR_T)
        exp_param_factory.set_evolution_times(RAMSEY_TIMES_S_ZERO_START)
        exp_param_factory.set_mw_pulse_lengths(RABI_TIMES_S_ZERO_START)
        exp_param_factory.set_base_rabi_frequency(
            RABI_FREQ_BASE_HZ / 2
        )  # Accounts for factor of 2 error in Unitary Solver Rabi frequency
        exp_param_factory.set_second_pulse_phase(0)  # Fails with nonzero phase
        experiment_parameters = exp_param_factory.get_experiment_parameters()

        master_equation_solver = MasterEquationSolver()
        master_equation_simulation = Simulation(experiment_parameters, nv_ensemble, master_equation_solver)
        master_equation_ms0_results = master_equation_simulation.ms0_results

        exp_param_factory = OffAxisFieldExperimentParametersFactory()
        exp_param_factory.set_base_rabi_frequency(RABI_FREQ_BASE_HZ)
        exp_param_factory.set_mw_direction(NVA_X_AXIS_MW_DIRECTION)
        exp_param_factory.set_e_field_v_per_m(NVA_X_AXIS_E_FIELD_V_PER_CM)
        exp_param_factory.set_b_field_vector(NVA_AXIAL_B_FIELD_VECTOR_T)
        exp_param_factory.set_evolution_times(RAMSEY_TIMES_S_ZERO_START)
        exp_param_factory.set_mw_pulse_lengths(RABI_TIMES_S_ZERO_START)
        exp_param_factory.set_second_pulse_phase(0)
        off_axis_experiment_parameters = exp_param_factory.get_experiment_parameters()

        liouvillian_solver = LiouvillianSolver()
        liouvillian_simulation = Simulation(off_axis_experiment_parameters, nv_ensemble, liouvillian_solver)
        liouvillian_ms0_results = liouvillian_simulation.ms0_results

        for i in range(len(RABI_TIMES_S_ZERO_START)):
            for j in range(len(RAMSEY_TIMES_S_ZERO_START)):
                self.assertAlmostEqual(master_equation_ms0_results[i][j], liouvillian_ms0_results[i][j], places=4)


if __name__ == "__main__":
    unittest.main()
