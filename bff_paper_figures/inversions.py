import numpy as np
from lmfit.model import ModelResult
from numpy.typing import NDArray
from scipy.signal import windows

from bff_paper_figures.fitting_routines import (
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


def freq_domain_inversion(
    sq_cancelled_signal: NDArray,
    rabi_frequencies: NDArray,
    inner_product_settings: InnerProductSettings,
    ramsey_freq_range_hz: NDArray,
    t2star_s: float,
    constrain_same_width: bool = True,
    allow_zero_peak: bool = True,
    constrain_hyperfine_splittings: bool = True,
) -> list[type[ModelResult]]:
    """
    Takes the VPDR single-quantum-cancelled signal, calculates the double inner product for the
    specified Rabi frequencies and over a range of Ramsey frequencies, and fits each orientations'
    frequency-domain spectrum to three Lorentzians.

    :param sq_cancelled_signal: the single-quantum-cancelled VPDR signal sampled over a range of free evolution times
                                and mw pulse durations (specified in inner_product_settings)
    :param rabi_frequencies: an array of Rabi frequencies (corresponding to the Rabi frequencies for each orientation)
    :param inner_product_settings: specifies how the inner products should be done
    :param ramsey_freq_range_hz: the range of ramsey frequencies (free evolution time dimension) over which the inner
                                product should be calculated for the initial fit; a subsequent fit will restrict the
                                range to relevant frequencies but retain the same number of points.
    :param t2star_s: used for an initial guess for linewidth
    :param constrain_same_width: determines if all Lorentzians are constrained to have the same linewidth
    :param allow_zero_peak: if True, adds an additional Lorentzian at zero frequency
    :param constrain_hyperfine_splittings: determines if the three Lorentzian center frequencies will be constrained
                                to differ by the hyperfine splitting
    :return: a list of the ModelResult for each of the four orientations' fits.
    """
    fit_results = []
    for orientation in NVOrientation:
        cos_cos_inner_prod = double_cosine_inner_product_vs_ramsey(
            sq_cancelled_signal, rabi_frequencies[orientation], ramsey_freq_range_hz, inner_product_settings
        )
        first_attempt_peaks = extract_fit_centers(
            fit_constrained_hyperfine_peaks(
                ramsey_freq_range_hz,
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
            min_ramsey_freq_hz, max_ramsey_freq_hz, len(ramsey_freq_range_hz)
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


def time_domain_inversion(
    sq_cancelled_signal: NDArray,
    rabi_frequencies: NDArray,
    inner_product_settings: InnerProductSettings,
    ramsey_freq_guesses_all_orientations: NDArray,
    t2star_s: float,
    fix_phase_to_zero: bool = False,
    constrain_same_decay: bool = False,
    constrain_hyperfine_freqs: bool = False,
):
    """
    Takes the VPDR single-quantum-cancelled signal, calculates the inner product along the MW pulse duration dimension
    with cos(2 pi rabi_frequencies[i] t) for each of the specified rabi frequencies (leaving a function of free evolution time),
    and fits the result with a sum of three sinusoids.

    :param sq_cancelled_signal: the single-quantum-cancelled VPDR signal sampled over a range of free evolution times
                                and mw pulse durations (specified in inner_product_settings)
    :param rabi_frequencies: an array of 4 Rabi frequencies (corresponding to the Rabi frequencies for each orientation)
    :param inner_product_settings: specifies how the inner products should be done
    :param ramsey_freq_guesses_all_orientations: the initial guesses for the sinusoid frequencies. This first dimension of
                                this array must be of the same length as rabi_frequencies (the number of orientations),
                                and there should be three frequency guesses for each orientation, i.e. the shape is (4,3)
    :param t2star_s: used for an initial guess for linewidth
    :param fix_phase_to_zero: determines if all sinusoids are constrained to be cosines (0 phase)
    :param constrain_same_decay: if True, requires all sinusoids to have the same decay rate
    :param constrain_hyperfine_freqs: determines if the three sinusoid frequencies will be constrained
                                to differ by the hyperfine splitting
    :return: a list of the ModelResult for each of the four orientations' fits.
    """
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
