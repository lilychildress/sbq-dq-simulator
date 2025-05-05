import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.signal import find_peaks, windows

from bff_paper_figures.inner_product_functions import (
    InnerProductSettings,
    double_cosine_inner_product,
    inner_product_sinusoid,
)
from bff_simulator.abstract_classes.abstract_ensemble import NVOrientation
from bff_simulator.constants import NVaxes_100, exy, ez, f_h, gammab
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParameters
from bff_simulator.offaxis_field_hamiltonian_constructor import OffAxisFieldHamiltonian
from bff_simulator.vector_manipulation import perpendicular_projection, transform_from_crystal_to_nv_coords

SLIGHT_DECREASE_FACTOR = 0.8
Z_AXIS_INDEX = 2


# The rabi frequencies according to the "ground truth" parameters used for calculating the simulation
def get_ideal_rabi_frequencies(experiment_parameters: OffAxisFieldExperimentParameters) -> NDArray:
    base_rabi_hz = experiment_parameters.rabi_frequency_base_hz
    mw_direction = experiment_parameters.mw_direction
    return np.array([base_rabi_hz * perpendicular_projection(mw_direction, NVaxis) for NVaxis in NVaxes_100])


# Finds the ms = - 1 to +1 transition frequency of the free Hamiltonian used to calculate the simulation, for all four orientations
# and all three hyperfine states. Note that these values are extremely close to 2 * axial field * gammab/h, when there
# is no electric field, so the axial field is also provided as a reference.
def get_true_transition_frequencies(off_axis_experiment_parameters: OffAxisFieldExperimentParameters) -> tuple[NDArray]:
    dq_larmor_freqs_hz = []
    bz_values_nv_coords_t = []

    # For all orientations
    for nv_axis in NVaxes_100:
        bz_values_for_this_axis = []
        dq_larmor_freqs_for_this_axis = []

        # For all nuclear spin projections
        for m_hyperfine in [1, 0, -1]:
            nv_axis_unit_vector = nv_axis / np.linalg.norm(nv_axis)

            # Find the magnetic field in the nv coordinates, including effective field from the nuclear spin
            b_field_w_hyperfine_nv_coords_t = transform_from_crystal_to_nv_coords(
                np.array(off_axis_experiment_parameters.b_field_vector_t), nv_axis_unit_vector
            ) + np.array([0, 0, m_hyperfine * f_h / gammab])
            bz_values_for_this_axis.append(b_field_w_hyperfine_nv_coords_t[Z_AXIS_INDEX])

            # Calculate the internal Hamiltonian and find its eigenvalues
            e_field_vector_hz = np.diag(np.array([exy, exy, ez])) @ transform_from_crystal_to_nv_coords(
                off_axis_experiment_parameters.e_field_vector_v_per_m, nv_axis_unit_vector
            )
            int_hamiltonian = OffAxisFieldHamiltonian.internal_hamiltonian_bare(
                e_field_vector_hz,
                b_field_w_hyperfine_nv_coords_t,
                off_axis_experiment_parameters.zero_field_splitting_hz,
            )
            evals, _ = OffAxisFieldHamiltonian.get_ordered_spin_1_eigensystem(int_hamiltonian)

            # get_ordered_spin_1_eigensystem returns eigenvalues in the order of (ms +1-like, ms-0-like, ms--1-like)
            # so take the appropriate ms = + 1 to ms = -1 difference to find the double quantum frequency
            dq_larmor_freqs_for_this_axis.append(abs((evals[2] - evals[0]) / (2 * np.pi)))
        bz_values_nv_coords_t.append(bz_values_for_this_axis)
        dq_larmor_freqs_hz.append(dq_larmor_freqs_for_this_axis)
    return np.array(dq_larmor_freqs_hz), np.array(bz_values_nv_coords_t)


# Calculates \sum_i {S(t_i,\tau_j)W(t_i)\cos(2\pi freq_hz t_i)} where S is the sq-cancelled signal and W is
# the window function specified in inner_product_settings.
def summed_rabi_inner_product(
    freq_hz: float, sq_cancelled_signal: NDArray, inner_product_settings: InnerProductSettings
) -> float:
    mw_pulse_lengths_s = inner_product_settings.mw_pulse_durations_s
    window = windows.get_window(inner_product_settings.rabi_window, len(mw_pulse_lengths_s))
    return np.sum(
        inner_product_sinusoid(np.cos, freq_hz, mw_pulse_lengths_s, window * np.transpose(sq_cancelled_signal), axis=1)
    )


# A function to be used in the minimization routine to find the Rabi frequency of each orientation.
def summed_rabi_inner_product_to_minimize(
    rabi_freq: list, sq_cancelled_signal: NDArray, inner_product_settings: InnerProductSettings
) -> float:
    return -summed_rabi_inner_product(rabi_freq[0], sq_cancelled_signal, inner_product_settings)


# Extracts the approximate rabi frequencies for each NV orientation from the sq-cancelled VPDR signal S(t_i,\tau_j).
# Calculates f(\tau_j, \nu) = \sum_i {S(t_i,\tau_j)W(t_i)\cos(2\pi \nu t_i)}, which exhibits peaks when \nu is equal to a Rabi
# frequency for one of the orientations. Peak-finding is used to determine initial guesses, and then scipy.optimize.minimize is
# used to extract the final estimate for the Rabi frequency. The initial peak-finding searches over rabi_freq_range_to_probe_hz, hence
# this range must be at least large enough to catch all of the Rabi frequencies.
#
# Default values are appropriate for MW strength and orientation near the nominal value, MW_DIRECTION = [0.2054, 0.11882, 0.97144].
# If the angle is very different, the default ordering will be incorrect and Rabi frequencies will be assigned to the wrong orientation.
# If there is significant variation in the Rabi frequency magnitude, such that you are searching over a large range of rabi frequencies,
# use_time_domain_summation_for_initial_guesses = False will more robustly find the correct peaks as initial guesses. In that case,
# to sum over the free evolution dimension in the frequency domain, you also need to supply the ramsey frequencies you wish to sum over
# (ramsey_freqs_to_sum_hz). Otherwise, ramsey_freqs_to_sum_hz is ignored.
# However, regardless of the value of use_time_domain_summation_for_initial_guesses, the final minimization always uses the time domain
# summation f(\tau_j, \nu) to find the Rabi frequencies.
def find_rabi_frequencies_from_signal(
    sq_cancelled_signal: NDArray,
    inner_product_settings: InnerProductSettings,
    rabi_freq_range_to_probe_hz: NDArray = np.linspace(50e6, 100e6, 101),
    height_factor: float = 0.75,
    prominence_factor: float = 0.75,
    ordering: list = [0, 1, 3, 2],
    use_time_domain_summation_for_initial_guesses: bool = True,
    ramsey_freqs_to_sum_hz: NDArray = np.linspace(0, 8.5e6, 11),
) -> NDArray:
    # Calculate the inner product summed over the free evolution (ramsey) dimension, either summing over evolution time (faster)
    # or performing a 2D inner product and then summing over free-evolution frequency (more robust)
    if use_time_domain_summation_for_initial_guesses:
        ramsey_summed = np.array(
            [
                summed_rabi_inner_product(rabi_freq, sq_cancelled_signal, inner_product_settings)
                for rabi_freq in rabi_freq_range_to_probe_hz
            ]
        )
    else:
        inner_product_2d = np.array(
            [
                [
                    double_cosine_inner_product(sq_cancelled_signal, rabi_hz, ramsey_hz, inner_product_settings)
                    for ramsey_hz in ramsey_freqs_to_sum_hz
                ]
                for rabi_hz in rabi_freq_range_to_probe_hz
            ]
        )
        ramsey_summed = -np.sum(inner_product_2d, axis=1)

    # Find the approximate locations of the peaks in the time-domain-ramsey-summed rabi inner product
    max_rs = max(ramsey_summed)
    peak_indices = find_peaks(
        ramsey_summed,
        height=(max_rs) * height_factor,
        prominence=(-max_rs) * prominence_factor,
    )
    # Check for the correct number of peaks, keep searching if necessary
    while len(peak_indices[0]) < len(NVOrientation):
        print(f"Did not find all peaks! Found only {len(peak_indices[0])}. Trying again...")
        height_factor = height_factor * SLIGHT_DECREASE_FACTOR
        prominence_factor = prominence_factor * SLIGHT_DECREASE_FACTOR
        peak_indices = find_peaks(
            ramsey_summed,
            height=(max_rs) * height_factor,
            prominence=(-max_rs) * prominence_factor,
        )
    if len(peak_indices[0]) != len(NVOrientation):
        raise ValueError("Could not find 4 rabi frequencies")

    # Perform a more careful minimization to get a better value for the Rabi frequencies, using the peak-finding
    # results as an initial guess
    rabi_frequencies = []
    for rabi_guess in rabi_freq_range_to_probe_hz[peak_indices[0]]:
        peak = minimize(
            summed_rabi_inner_product_to_minimize,
            [rabi_guess],
            (sq_cancelled_signal, inner_product_settings),
            method="Powell",
        )
        rabi_frequencies.append(peak.x[0])

    return np.array(rabi_frequencies)[ordering]
