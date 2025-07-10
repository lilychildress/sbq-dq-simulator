import numpy as np
from scipy.optimize import fsolve, minimize
from numpy.random import Generator

from matplotlib import pyplot as plt
from scipy.signal import windows
from lmfit import Model
from bff_paper_figures.inner_product_functions import (
    inner_product_sinusoid,
)

from bff_paper_figures.extract_experiment_values import  get_true_transition_frequencies, get_ideal_rabi_frequencies
from bff_paper_figures.simulation_helper_functions import sq_cancelled_signal_generator
from bff_paper_figures.fitting_routines import decaying_cosine, offset
from bff_simulator.abstract_classes.abstract_ensemble import NVOrientation, NV14HyperfineField
from bff_simulator.constants import NVaxes_100, gammab
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.vector_manipulation import perpendicular_projection
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory, OffAxisFieldExperimentParameters
from bff_paper_figures.shared_parameters import  MW_DIRECTION, E_FIELD_VECTOR_V_PER_CM, RABI_FREQ_BASE_HZ, DETUNING_HZ, T2STAR_S, B_PHI_FIG4, B_THETA_FIG4

SIG_STD_DEV = 1e-4
N_SAMPLES = 1000

# Functions to determine the optimal free evolution time for measuring a DQ signal
def optimal_time(tau_s, larmor_actual_hz, t2star_s):
    w = 2*np.pi*larmor_actual_hz/2
    t = tau_s
    t2s = t2star_s
    return 2 * t2s * t * w * np.cos(2 * t * w) + (t2s - 2* t)*np.sin(2 * t * w)

def get_optimal_evolution_time_s(larmor_actual_hz, t2star_s):
   return fsolve(optimal_time, [t2star_s/1.99], (larmor_actual_hz, t2star_s))[0]

def slope(tau_s, larmor_actual_hz, t2star_s):
    w = 2*np.pi*larmor_actual_hz/2
    t = tau_s
    t2s = t2star_s
    return 2*np.exp(-2*t/t2s)*t*np.sin(2*t*w)

# These include a hyperfine triplet in finding the optimal evolution time
def slope_triplet(tau_s, larmor_actual_hz, a_hf_hz, t2star_s):
    return slope(tau_s, larmor_actual_hz, t2star_s) + slope(tau_s, larmor_actual_hz + 2*a_hf_hz, t2star_s) +  + slope(tau_s, larmor_actual_hz - 2*a_hf_hz, t2star_s) 

def minimize_for_opt_evololution_time_hf(evolution_time, larmor_freq_mi0, f_h, t2star_s):
    return -np.abs(slope_triplet(evolution_time, larmor_freq_mi0, f_h, t2star_s))

def find_optimal_hf_revival_time(larmor_freq_mi0, f_h, t2star_s):
    hyperfine_revival_times = np.arange(0, 2*t2star_s, 1/(2*f_h))
    slopes_at_revivals = []
    for evolution_time in hyperfine_revival_times:
        slopes_at_revivals.append(slope_triplet(evolution_time, larmor_freq_mi0, f_h, t2star_s ))

    max_slope_idx = np.where(np.isclose(slopes_at_revivals, max(slopes_at_revivals)))[0][0]
    optimal_evolution_time_guess_s = hyperfine_revival_times[max_slope_idx]

    result = minimize(minimize_for_opt_evololution_time_hf, [optimal_evolution_time_guess_s], (larmor_freq_mi0, f_h, t2star_s), method="Nelder-Mead")

    optimal_evolution_time_s = result.x[0]
    return optimal_evolution_time_s


def fit_decaying_cosine(evolution_times_s, time_domain_ramsey_signal, larmor_freq):
    model = Model(offset) + Model(decaying_cosine)
    params = model.make_params()
    params["amplitude"].value = max(time_domain_ramsey_signal) - min(time_domain_ramsey_signal)
    params["offset_value"].value = np.mean(time_domain_ramsey_signal)
    params["decay_time"].value = T2STAR_S/2
    params["phase"].value = np.pi
    params["freq"].value = larmor_freq
    time_domain_result = model.fit(time_domain_ramsey_signal, params, x=evolution_times_s)
    return time_domain_result

# This function assumes that exp_param_factory is set up and ready to go aside from specifying magnetic field. It calculates the change in signal
# at the optimal evolution time both for the inner-producted VPDR signal and a DQ signal (i.e. with MW pi pulses) for two different magnetic field conditions
# and returns the "slope" - i.e. the change in axial magnetic field divided by the change in signal for both possibilities. mi_index does not matter. 
def get_signal_slopes(exp_param_factory:OffAxisFieldExperimentParametersFactory, nv_ensemble, off_axis_solver, rabi_window_name, b_vector_t, b_plus_db_vector_t, opt_evolution_idx, opt_mw_pulse_idx, orientation=NVOrientation.A, mi_index=0):
    # Calculate the SQ cancelled signal at the magnetic field of interest
    exp_param_factory.set_b_field_vector(b_vector_t)
    _, bz_values_all_axes_t = get_true_transition_frequencies(
        exp_param_factory.get_experiment_parameters()
    )
    sq_cancelled_signal = sq_cancelled_signal_generator(exp_param_factory, nv_ensemble, off_axis_solver)

    # Now do the same thing at a slightly different magnetic field so that we can calculate the "slope" with
    # respect to magnetic field at the optimal evolution time for measurement
    exp_param_factory.set_b_field_vector(b_plus_db_vector_t)
    sq_cancelled_signal_delta_b = sq_cancelled_signal_generator(exp_param_factory, nv_ensemble, off_axis_solver)
    _, bz_values_all_axes_delta_b_t = get_true_transition_frequencies(
        exp_param_factory.get_experiment_parameters()
    )

    # Pick out the inner-product signals at the expected magnetic field and the slightly shifted one
    mw_pulse_length_s = exp_param_factory.get_experiment_parameters().mw_pulse_length_s
    rabi_window = windows.get_window(rabi_window_name, len(mw_pulse_length_s))
    rabi_frequency_hz = get_ideal_rabi_frequencies(exp_param_factory.get_experiment_parameters())[orientation]
    time_domain_ramsey_signal = inner_product_sinusoid(np.cos, rabi_frequency_hz, mw_pulse_length_s, rabi_window * np.transpose(sq_cancelled_signal), axis=1)
    time_domain_ramsey_signal_delta_b = inner_product_sinusoid(np.cos, rabi_frequency_hz, mw_pulse_length_s, rabi_window * np.transpose(sq_cancelled_signal_delta_b), axis=1)

    # Calculate how much the axial field changes per unit change in the inner-producted signal (the "slope" with respect to axial field)
    delta_b_t = (bz_values_all_axes_delta_b_t[orientation] - bz_values_all_axes_t[orientation])[mi_index]
    slope_dbz_dsignal_inner_product = delta_b_t/((time_domain_ramsey_signal_delta_b - time_domain_ramsey_signal)[opt_evolution_idx])

    # Calculate how much the axial field changes per unit change in the single-point signal (the "slope" with respect to axial field)
    slope_dbz_dsignal_dq = delta_b_t/((sq_cancelled_signal - sq_cancelled_signal_delta_b)[opt_mw_pulse_idx][opt_evolution_idx])

    # Reset the magnetic field
    exp_param_factory.set_b_field_vector(b_vector_t)

    return slope_dbz_dsignal_dq, slope_dbz_dsignal_inner_product, sq_cancelled_signal, time_domain_ramsey_signal

# Monte Carlo simulation of sensitivity ratio for VPDR vs DQ. Calculates the standard
# deviation of the magnetic field readings in the presence of Gaussian readout noise.
# Compares fluctuations in the average value of a DQ signal [averaged over as many samples (at the optimal MW time)
# as there are pulse durations] to fluctuations in a VPDR signal analyzed with an inner product on 
# the Rabi dimension. Both signals are evaluated for a single NV orientation, single hyperfine line
# at the optimal free evolution time for sensitivity. 
def compare_single_point_sensitivity(slope_dbz_dsignal_dq, slope_dbz_dsignal_inner_product, sq_cancelled_signal, time_domain_ramsey_signal, rabi_window_name, exp_params: OffAxisFieldExperimentParameters, rng: Generator, opt_evolution_idx, sig_std_dev = SIG_STD_DEV, n_samples= N_SAMPLES,orientation=NVOrientation.A):

    # find the slope in signal at the optimal time to measure
    #slope_dbz_dsignal_dq, slope_dbz_dsignal_inner_product, sq_cancelled_signal, time_domain_ramsey_signal = get_signal_slopes(exp_param_factory, nv_ensemble, off_axis_solver, rabi_window_name, B_VECTOR_T, B_PLUS_DB_VECTOR_T, EVOLUTION_STEPS_UNTIL_OPTIMAL, int(MW_RABI_PERIOD_DIVISION/2), orientation)
    
    mw_pulse_length_s = exp_params.mw_pulse_length_s
    rabi_frequency_hz = get_ideal_rabi_frequencies(exp_params)[orientation]
    rabi_window = windows.get_window(rabi_window_name, len(mw_pulse_length_s))

    # Now add in measurement noise, and see how much that changes the signals
    noisy_delta_b_inner_product_t = []
    noisy_delta_b_dq_t = []
    for _ in range(n_samples):
        # Calculate the noisy inner-producted signal
        noisy_sq_cancelled_signal = sq_cancelled_signal + rng.normal(0, sig_std_dev, size=sq_cancelled_signal.shape)
        noisy_time_domain_ramsey_signal = inner_product_sinusoid(np.cos, rabi_frequency_hz, mw_pulse_length_s, rabi_window * np.transpose(noisy_sq_cancelled_signal), axis=1)
        
        # And convert it to a noisy field excursion using the slope
        noisy_delta_b_inner_product_t.append(((noisy_time_domain_ramsey_signal - time_domain_ramsey_signal)[opt_evolution_idx])*slope_dbz_dsignal_inner_product)

        # do the same thing for the standard dq signal (averaging over the same number of measurements as there are steps in mw_pulse_length_s)
        noisy_delta_b_dq_t.append(np.mean(rng.normal(0, sig_std_dev, size=len(mw_pulse_length_s)) ) * slope_dbz_dsignal_dq)
    
    # Calculate the ratio of the noise in the magnetic field, which should be (approximately) the ratio of the sensitivities, assuming that the time is 
    # dominated by measurement time.
    return np.std(np.array(noisy_delta_b_inner_product_t))/np.std(np.array(noisy_delta_b_dq_t))