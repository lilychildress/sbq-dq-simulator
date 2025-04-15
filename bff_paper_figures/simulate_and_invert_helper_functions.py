import numpy as np
from numpy.typing import NDArray

from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.offaxis_field_experiment_parameters import (
    OffAxisFieldExperimentParametersFactory,
    OffAxisFieldExperimentParameters,
)
from bff_simulator.simulator import Simulation
from bff_simulator.abstract_classes.abstract_ensemble import NVOrientation
from bff_paper_figures.inner_product_functions import double_cosine_inner_product_vs_ramsey
from bff_paper_figures.fit_inner_product_vs_ramsey import (
    fit_constrained_hyperfine_peaks,
    fit_vs_eigenvalue_error_nT,
    extract_fit_centers,
)
from bff_paper_figures.extract_experiment_values import get_bare_rabi_frequencies, get_true_eigenvalues
from bff_paper_figures.inner_product_functions import InnerProductSettings


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


def double_cosine_inner_product_fit_inversion(
    sq_cancelled_signal: NDArray,
    off_axis_experiment_parameters: OffAxisFieldExperimentParameters,
    inner_product_settings: InnerProductSettings,
    ramsey_freq_range_init_guess_hz: NDArray,
    t2star_s: float,
    constrain_same_width: bool = True,
    allow_zero_peak:bool = True,
) -> NDArray:
    rabi_frequencies = get_bare_rabi_frequencies(off_axis_experiment_parameters)
    larmor_freqs_all_axes_hz, _ = get_true_eigenvalues(off_axis_experiment_parameters)

    errors_nT = []
    for i, orientation in enumerate(NVOrientation):
        cos_cos_inner_prod = double_cosine_inner_product_vs_ramsey(
            sq_cancelled_signal, rabi_frequencies[orientation], ramsey_freq_range_init_guess_hz, inner_product_settings
        )
        first_attempt_peaks = extract_fit_centers(
            fit_constrained_hyperfine_peaks(ramsey_freq_range_init_guess_hz, cos_cos_inner_prod, t2star_s,constrain_same_width=constrain_same_width, allow_zero_peak=allow_zero_peak)
        )

        expected_fwhm_hz = 2 / (np.pi * t2star_s)
        min_ramsey_freq_hz = max(expected_fwhm_hz, min(first_attempt_peaks) - expected_fwhm_hz)
        max_ramsey_freq_hz = max(first_attempt_peaks) + expected_fwhm_hz
        ramsey_freq_range_constrained_hz = np.linspace(
            min_ramsey_freq_hz, max_ramsey_freq_hz, len(ramsey_freq_range_init_guess_hz)
        )

        cos_cos_inner_prod = double_cosine_inner_product_vs_ramsey(
            sq_cancelled_signal, rabi_frequencies[orientation], ramsey_freq_range_constrained_hz, inner_product_settings
        )
        fit_result = fit_constrained_hyperfine_peaks(
            ramsey_freq_range_constrained_hz, cos_cos_inner_prod, t2star_s, constrain_same_width=constrain_same_width
        )

        errors_nT.append(fit_vs_eigenvalue_error_nT(fit_result, larmor_freqs_all_axes_hz[orientation]))

    return np.array(errors_nT)
