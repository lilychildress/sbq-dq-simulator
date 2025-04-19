import numpy as np
from lmfit.model import ModelResult
from numpy.typing import NDArray
from scipy.signal import find_peaks, windows

from bff_paper_figures.fitting_routines import (
    N_HYPERFINE,
    extract_fit_centers,
    fit_constrained_hyperfine_peaks,
    fit_three_cos_model,
)
from bff_paper_figures.inner_product_functions import (
    InnerProductSettings,
    double_cosine_inner_product_vs_ramsey,
    inner_product_sinusoid,
)
from bff_simulator.abstract_classes.abstract_ensemble import NVOrientation
from bff_simulator.constants import f_h
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


def double_cosine_inner_product_fit_inversion(
    sq_cancelled_signal: NDArray,
    rabi_frequencies: NDArray,
    inner_product_settings: InnerProductSettings,
    ramsey_freq_range_init_guess_hz: NDArray,
    t2star_s: float,
    constrain_same_width: bool = True,
    allow_zero_peak: bool = True,
    constrain_hyperfine_splittings: bool = True,
) -> list[type[ModelResult]]:
    fit_results = []
    for orientation in NVOrientation:
        cos_cos_inner_prod = double_cosine_inner_product_vs_ramsey(
            sq_cancelled_signal, rabi_frequencies[orientation], ramsey_freq_range_init_guess_hz, inner_product_settings
        )
        first_attempt_peaks = extract_fit_centers(
            fit_constrained_hyperfine_peaks(
                ramsey_freq_range_init_guess_hz,
                cos_cos_inner_prod,
                t2star_s,
                constrain_same_width=constrain_same_width,
                allow_zero_peak=allow_zero_peak,
                constrain_hyperfine_splittings=constrain_hyperfine_splittings,
            )
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
            ramsey_freq_range_constrained_hz,
            cos_cos_inner_prod,
            t2star_s,
            constrain_same_width=constrain_same_width,
            allow_zero_peak=allow_zero_peak,
            constrain_hyperfine_splittings=constrain_hyperfine_splittings,
        )
        fit_results.append(fit_result)

    return fit_results


def get_ramsey_freq_guesses_quickly(
    sq_cancelled_signal,
    rabi_frequencies,
    inner_product_settings,
    ramsey_freq_range_initial_guess_hz,
    height_factor=0.75,
    prominence_factor=0.75,
) -> NDArray:
    freq_guesses_all_orientations = []
    for orientation in NVOrientation:
        # Calculate the double inner product over a range of ramsey frequencies
        cos_cos_inner_prod = double_cosine_inner_product_vs_ramsey(
            sq_cancelled_signal,
            rabi_frequencies[orientation],
            ramsey_freq_range_initial_guess_hz,
            inner_product_settings,
        )

        # Find the features in the inner product. Note that they are normally dips, hence the inversion.
        min_inner_product = min(cos_cos_inner_prod)
        peaks = find_peaks(
            -cos_cos_inner_prod,
            height=(-min_inner_product) * height_factor,
            prominence=(-min_inner_product) * prominence_factor,
        )

        # In cases where two of the peaks overlap, the highest-frequency peak is always isolated, so use that to generate guesses
        rightmost_peak_location = ramsey_freq_range_initial_guess_hz[peaks[0][-1]]
        freq_guesses_all_orientations.append([rightmost_peak_location - 2 * f_h * i for i in range(N_HYPERFINE)])

    return np.array(freq_guesses_all_orientations)


def time_domain_fit_inversion(
    sq_cancelled_signal: NDArray,
    rabi_frequencies: NDArray,
    inner_product_settings: InnerProductSettings,
    ramsey_freq_guesses_all_orientations: NDArray,
    t2star_s: float,
    fix_phase_to_zero: bool = False,
    constrain_same_decay: bool = False,
    constrain_hyperfine_freqs: bool = False,
):
    # Automatically use "ideal" Rabi frequencies if they are not separately specified; otherwise use provided Rabi frequencies
    # if len(rabi_frequencies)==0:
    #    rabi_frequencies = get_ideal_rabi_frequencies(off_axis_experiment_parameters)

    # mw_pulse_length_s = off_axis_experiment_parameters.mw_pulse_length_s
    # evolution_time_s = off_axis_experiment_parameters.evolution_time_s
    mw_pulse_length_s = inner_product_settings.mw_pulse_durations_s
    evolution_time_s = inner_product_settings.free_evolution_times_s
    rabi_window = windows.get_window(inner_product_settings.rabi_window, len(mw_pulse_length_s))

    fit_results = []
    for orientation in NVOrientation:
        time_domain_ramsey_signal = inner_product_sinusoid(
            np.cos,
            rabi_frequencies[orientation],
            mw_pulse_length_s,
            rabi_window * np.transpose(sq_cancelled_signal),
            axis=1,
        )
        time_domain_result = fit_three_cos_model(
            evolution_time_s,
            time_domain_ramsey_signal,
            ramsey_freq_guesses_all_orientations[orientation],
            t2star_s,
            fix_phase_to_zero,
            constrain_same_decay,
            constrain_hyperfine_freqs,
        )
        fit_results.append(time_domain_result)

    return fit_results


def angles_already_evaluated(test_theta, test_phi, theta_values, phi_values):
    theta_indices = np.where(np.isclose(theta_values, test_theta))
    phi_indices = np.where(np.isclose(phi_values, test_phi))
    return len(np.intersect1d(theta_indices, phi_indices)) > 0
