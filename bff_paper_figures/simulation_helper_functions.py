import numpy as np
from numpy.typing import NDArray

from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.offaxis_field_experiment_parameters import (
    OffAxisFieldExperimentParametersFactory,
)
from bff_simulator.simulator import Simulation


def sq_cancelled_signal_generator(
    exp_param_factory_set_except_phase: OffAxisFieldExperimentParametersFactory,
    nv_ensemble: HomogeneousEnsemble,
    off_axis_solver: LiouvillianSolver,
) -> NDArray:
    exp_param_factory_set_except_phase.set_second_pulse_phase(0)
    off_axis_experiment_parameters_0_phase = exp_param_factory_set_except_phase.get_experiment_parameters()
    off_axis_simulation_0_phase = Simulation(off_axis_experiment_parameters_0_phase, nv_ensemble, off_axis_solver)

    exp_param_factory_set_except_phase.set_second_pulse_phase(np.pi)
    off_axis_experiment_parameters_pi_phase = exp_param_factory_set_except_phase.get_experiment_parameters()
    off_axis_simulation_pi_phase = Simulation(off_axis_experiment_parameters_pi_phase, nv_ensemble, off_axis_solver)

    return off_axis_simulation_0_phase.ms0_results + off_axis_simulation_pi_phase.ms0_results


def angles_already_evaluated(test_theta, test_phi, theta_values, phi_values):
    theta_indices = np.where(np.isclose(theta_values, test_theta))
    phi_indices = np.where(np.isclose(phi_values, test_phi))
    return len(np.intersect1d(theta_indices, phi_indices)) > 0
