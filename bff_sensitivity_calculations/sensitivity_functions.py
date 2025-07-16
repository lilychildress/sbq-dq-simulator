import numpy as np
from scipy.optimize import fsolve, minimize
from numpy.random import Generator

from matplotlib import pyplot as plt
from scipy.signal import windows
from lmfit import Model
from bff_paper_figures.inner_product_functions import (
    inner_product_sinusoid,
)


from bff_review_response.sensitivity_helper_functions import get_optimal_evolution_time_no_hf_s, get_optimal_hf_revival_time, slope, slope_triplet, fit_decaying_cosine
from bff_paper_figures.extract_experiment_values import  get_true_transition_frequencies, get_ideal_rabi_frequencies
from bff_paper_figures.simulation_helper_functions import sq_cancelled_signal_generator
from bff_paper_figures.fitting_routines import decaying_cosine, offset, fit_three_cos_model
from bff_simulator.abstract_classes.abstract_ensemble import NVOrientation, NV14HyperfineField
from bff_simulator.constants import NVaxes_100, gammab, f_h
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.vector_manipulation import perpendicular_projection
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory, OffAxisFieldExperimentParameters
from bff_paper_figures.shared_parameters import  MW_DIRECTION, E_FIELD_VECTOR_V_PER_CM, RABI_FREQ_BASE_HZ, DETUNING_HZ, T2STAR_S, B_PHI_FIG4, B_THETA_FIG4

EVOLUTION_TIME_S = np.arange(0, 3e-6, 20e-9)  # p.linspace(0, 15e-6, 801)
INDEX_FOR_MI0 = 1
S_TO_NS = 1e9

######################## Functions pertaining to analytic theory #################################
# Functions to determine the optimal free evolution time for measuring a DQ signal
def optimal_time(tau_s, double_larmor_actual_hz, t2star_s):
    w = 2*np.pi*double_larmor_actual_hz/2
    t = tau_s
    t2s = t2star_s
    return 2 * t2s * t * w * np.cos(2 * t * w) + (t2s - 2* t)*np.sin(2 * t * w)

def get_optimal_evolution_time_no_hf_s(double_larmor_actual_hz, t2star_s):
    result = minimize(minimize_for_opt_evolution_time_no_hf, [t2star_s/1.99], (double_larmor_actual_hz, t2star_s), method="Nelder-Mead")
    return fsolve(optimal_time, result.x[0], (double_larmor_actual_hz, t2star_s))[0]  

def slope(tau_s, double_larmor_actual_hz, t2star_s):
    w = 2*np.pi*double_larmor_actual_hz/2
    t = tau_s
    t2s = t2star_s
    return 2*np.exp(-2*t/t2s)*t*np.sin(2*t*w)

# These include a hyperfine triplet in finding the optimal evolution time
def slope_triplet(tau_s, double_larmor_actual_hz, a_hf_hz, t2star_s):
    return slope(tau_s, double_larmor_actual_hz, t2star_s) + slope(tau_s, double_larmor_actual_hz + 2*a_hf_hz, t2star_s) +  + slope(tau_s, double_larmor_actual_hz - 2*a_hf_hz, t2star_s) 

def minimize_for_opt_evolution_time_no_hf(evolution_time, double_larmor_freq_hz, t2star_s):
    return -np.abs(slope(evolution_time, double_larmor_freq_hz, t2star_s))

def minimize_for_opt_evolution_time_hf(evolution_time, double_larmor_freq_mi0, f_h, t2star_s):
    return -np.abs(slope_triplet(evolution_time, double_larmor_freq_mi0, f_h, t2star_s))

def get_optimal_hf_revival_time(double_larmor_freq_mi0, f_h, t2star_s):
    hyperfine_revival_times = np.arange(0, 2*t2star_s, 1/(2*f_h))
    slopes_at_revivals= slope_triplet(hyperfine_revival_times, double_larmor_freq_mi0,f_h, t2star_s)

    max_slope_idx = np.where(np.isclose(np.abs(slopes_at_revivals), max(np.abs(slopes_at_revivals))))[0][0]
    optimal_evolution_time_guess_s = hyperfine_revival_times[max_slope_idx]

    result = minimize(minimize_for_opt_evolution_time_hf, [optimal_evolution_time_guess_s], (double_larmor_freq_mi0, f_h, t2star_s), method="Nelder-Mead")
    optimal_evolution_time_s = result.x[0]
    return optimal_evolution_time_s

def get_optimal_evolution_time_hf_agnostic_s(double_larmor_mi0_hz, t2star_s, use_hyperfine=False, a_hf_hz = f_h, plot_evolution_time_s=EVOLUTION_TIME_S, do_plot=True):
    if use_hyperfine:
        optimal_evolution_time_s = get_optimal_hf_revival_time(double_larmor_mi0_hz, a_hf_hz, t2star_s)
        abs_slopes_vs_tau = np.abs(slope_triplet(plot_evolution_time_s, double_larmor_mi0_hz, a_hf_hz, t2star_s))
    else:
        optimal_evolution_time_s = get_optimal_evolution_time_no_hf_s(double_larmor_mi0_hz, t2star_s)
        abs_slopes_vs_tau = np.abs(slope(plot_evolution_time_s, double_larmor_mi0_hz, t2star_s))

    # Verify that we did indeed find the optimal evolution time
    if do_plot:
        print(f"Optimal evolution time: {optimal_evolution_time_s*S_TO_NS} ns")
        plt.plot(plot_evolution_time_s, abs_slopes_vs_tau)
        plt.vlines(optimal_evolution_time_s, 0, max(abs_slopes_vs_tau), color="red")
        plt.show()   
    return optimal_evolution_time_s

def avg_to_optimal_slope_ratio(evolution_time_s, optimal_evolution_time_s, double_larmor_mi0_hz, f_h, t2star_s, use_hyperfine):
    if use_hyperfine:
        abs_triplet_slopes = np.abs(slope_triplet(evolution_time_s, double_larmor_mi0_hz, f_h, t2star_s))
        theory_slope_ratio = np.abs(slope_triplet(optimal_evolution_time_s, double_larmor_mi0_hz, f_h, t2star_s))/np.mean(abs_triplet_slopes)
    else:
        abs_slopes = np.abs(slope(evolution_time_s, double_larmor_mi0_hz, t2star_s))
        theory_slope_ratio = np.abs(slope(optimal_evolution_time_s, double_larmor_mi0_hz, t2star_s))/np.mean(abs_slopes)
    return theory_slope_ratio


############################ Functions relating to fitting #################################
def fit_decaying_cosine(evolution_times_s, time_domain_ramsey_signal, double_larmor_freq_hz, t2star_s = T2STAR_S):
    model = Model(offset) + Model(decaying_cosine)
    params = model.make_params()
    params["amplitude"].value = max(time_domain_ramsey_signal) - min(time_domain_ramsey_signal)
    params["offset_value"].value = np.mean(time_domain_ramsey_signal)
    params["decay_time"].value = t2star_s/2
    params["phase"].value = np.pi
    params["freq"].value = double_larmor_freq_hz
    time_domain_result = model.fit(time_domain_ramsey_signal, params, x=evolution_times_s)
    return time_domain_result

def fit_time_domain_signal_for_bz(evolution_time_s, noisy_signal, double_larmor_mi0_hz, t2star_s, use_hyperfine):
    if use_hyperfine:
        freq_guesses_hz = np.sort(np.abs(np.array([double_larmor_mi0_hz-2*f_h, double_larmor_mi0_hz, double_larmor_mi0_hz+2*f_h])))[::-1]
        fit = fit_three_cos_model(evolution_time_s, noisy_signal,freq_guesses_hz, t2star_s, constrain_hyperfine_freqs=True, constrain_same_decay=True)
        bz = fit.params["p0_freq"].value/(2*gammab)
    else:
        fit = fit_decaying_cosine(evolution_time_s, noisy_signal, double_larmor_mi0_hz, t2star_s)
        bz = fit.params["freq"].value/(2*gammab)
    return bz

############################### Functions for extracting time domain Ramsey signals either with inner product or mw pi pulse #######################
def get_pi_pulse_ramsey_signal(exp_param_factory:OffAxisFieldExperimentParametersFactory, nv_ensemble, off_axis_solver, t_pi_s):
    # retrieve the old mw pulse durations
    old_mw_pulse_durations_s = exp_param_factory.get_experiment_parameters().mw_pulse_length_s

    # we want to calculate the signal at the pi pulse time, but we need at least two MW durations
    exp_param_factory.set_mw_pulse_lengths([0, t_pi_s])
    sq_cancelled_signal = sq_cancelled_signal_generator(exp_param_factory, nv_ensemble, off_axis_solver)
    ramsey_signal = sq_cancelled_signal[-1]

    # reset exp_param_factory
    exp_param_factory.set_mw_pulse_lengths(old_mw_pulse_durations_s)

    return ramsey_signal

def extract_time_domain_ramsey_signal(sq_cancelled_signal, orientation, exp_params:OffAxisFieldExperimentParameters, rabi_window_name):
    mw_pulse_length_s = exp_params.mw_pulse_length_s
    rabi_window = windows.get_window(rabi_window_name, len(mw_pulse_length_s))
    rabi_freq_hz = get_ideal_rabi_frequencies(exp_params)[orientation]
    time_domain_ramsey_signal =  inner_product_sinusoid(np.cos, rabi_freq_hz, mw_pulse_length_s, rabi_window * np.transpose(sq_cancelled_signal), axis=1)
    return time_domain_ramsey_signal

###################################### Functions for extracting the slope dBz/dSignal needed for determining sensitivity ################################

def get_vpdr_slope(exp_param_factory:OffAxisFieldExperimentParametersFactory, nv_ensemble, off_axis_solver, rabi_window_name, b_vector_t, b_plus_db_vector_t,  orientation=NVOrientation.B, mi0_index=INDEX_FOR_MI0):
    # Calculate the SQ cancelled signal at the magnetic field of interest and extract time domain ramsey signal
    exp_param_factory.set_b_field_vector(b_vector_t)
    _, bz_values_all_axes_t = get_true_transition_frequencies(exp_param_factory.get_experiment_parameters())
    sq_cancelled_signal = sq_cancelled_signal_generator(exp_param_factory, nv_ensemble, off_axis_solver)
    time_domain_ramsey_signal = extract_time_domain_ramsey_signal(sq_cancelled_signal, orientation, exp_param_factory.get_experiment_parameters(), rabi_window_name)

    # Now do the same thing at a slightly different magnetic field so that we can calculate the "slope" with
    # respect to magnetic field at the optimal evolution time for measurement
    exp_param_factory.set_b_field_vector(b_plus_db_vector_t)
    _, bz_values_all_axes_delta_b_t = get_true_transition_frequencies(exp_param_factory.get_experiment_parameters())
    sq_cancelled_signal_delta_b = sq_cancelled_signal_generator(exp_param_factory, nv_ensemble, off_axis_solver)
    time_domain_ramsey_signal_delta_b = extract_time_domain_ramsey_signal(sq_cancelled_signal_delta_b, orientation, exp_param_factory.get_experiment_parameters(), rabi_window_name)

    # Calculate how much the axial field changes per unit change in the inner-producted signal (the "slope" with respect to axial field)
    delta_b_t = (bz_values_all_axes_delta_b_t[orientation] - bz_values_all_axes_t[orientation])[mi0_index]
    slope_dbz_dsignal_inner_product = delta_b_t/((time_domain_ramsey_signal_delta_b - time_domain_ramsey_signal))
    # Reset the magnetic field
    exp_param_factory.set_b_field_vector(b_vector_t)

    return slope_dbz_dsignal_inner_product


def get_ramsey_slope(exp_param_factory:OffAxisFieldExperimentParametersFactory, nv_ensemble, off_axis_solver, t_pi_s,  b_vector_t, b_plus_db_vector_t,  orientation=NVOrientation.B, mi0_index=INDEX_FOR_MI0):

    # Calculate the signal at the initial field
    exp_param_factory.set_b_field_vector(b_vector_t)
    _, bz_values_all_axes_t = get_true_transition_frequencies(exp_param_factory.get_experiment_parameters())
    ramsey_signal = get_pi_pulse_ramsey_signal(exp_param_factory, nv_ensemble, off_axis_solver, t_pi_s)

    # and the offset field
    exp_param_factory.set_b_field_vector(b_plus_db_vector_t)
    _, bz_values_all_axes_delta_b_t = get_true_transition_frequencies(exp_param_factory.get_experiment_parameters())
    ramsey_signal_delta_b = get_pi_pulse_ramsey_signal(exp_param_factory, nv_ensemble, off_axis_solver, t_pi_s)

    # Calculate how much the axial field changes per unit change in the inner-producted signal (the "slope" with respect to axial field)
    delta_b_t = (bz_values_all_axes_delta_b_t[orientation] - bz_values_all_axes_t[orientation])[mi0_index]
    slope_dbz_dsignal_ramsey = delta_b_t/(ramsey_signal_delta_b-ramsey_signal)

    return slope_dbz_dsignal_ramsey

def get_vpdr_slope_at_tau_opt(tau_opt_s, exp_param_factory:OffAxisFieldExperimentParametersFactory, nv_ensemble, off_axis_solver, rabi_window_name, b_vector_t, b_plus_db_vector_t,  orientation=NVOrientation.B, mi0_index=INDEX_FOR_MI0):

    old_evolution_times = exp_param_factory.get_experiment_parameters().evolution_time_s
    
    # we need at least two evolution times or we will have an error
    exp_param_factory.set_evolution_times([0,tau_opt_s])

    vdpr_slope_at_tau_opt = get_vpdr_slope(exp_param_factory, nv_ensemble, off_axis_solver, rabi_window_name, b_vector_t, b_plus_db_vector_t, orientation, mi0_index)[-1]
    
    # reset the evolution times
    exp_param_factory.set_evolution_times(old_evolution_times)
    return vdpr_slope_at_tau_opt

def get_ramsey_slope_at_tau_opt(tau_opt_s, exp_param_factory:OffAxisFieldExperimentParametersFactory, nv_ensemble, off_axis_solver, t_pi_s, b_vector_t, b_plus_db_vector_t, orientation=NVOrientation.B, mi0_index = INDEX_FOR_MI0):
    
    old_evolution_times = exp_param_factory.get_experiment_parameters().evolution_time_s

    # we need at least two evolution times or we will have an error
    exp_param_factory.set_evolution_times([0,tau_opt_s])
    ramsey_slope_at_tau_opt = get_ramsey_slope(exp_param_factory, nv_ensemble, off_axis_solver, t_pi_s, b_vector_t, b_plus_db_vector_t, orientation, mi0_index)[-1]
    
    # reset the evolution times
    exp_param_factory.set_evolution_times(old_evolution_times)
    return ramsey_slope_at_tau_opt

def get_vdpr_and_ramsey_min_slopes(exp_param_factory:OffAxisFieldExperimentParametersFactory, nv_ensemble, off_axis_solver, rabi_window_name, optimal_evolution_time_s, t_pi_s, b_vector_t, b_plus_db_vector_t, orientation, index_mi0, do_plots=False):
    dbz_dsignal_vpdr = get_vpdr_slope(exp_param_factory, nv_ensemble, off_axis_solver, rabi_window_name, b_vector_t, b_plus_db_vector_t, orientation, index_mi0)
    dbz_dsignal_vpdr_tau_opt = get_vpdr_slope_at_tau_opt(optimal_evolution_time_s, exp_param_factory, nv_ensemble, off_axis_solver, rabi_window_name, b_vector_t,  b_plus_db_vector_t, orientation,index_mi0)
    
    evolution_time_s = exp_param_factory.get_experiment_parameters().evolution_time_s
    
    if min(np.abs(dbz_dsignal_vpdr))< np.abs(dbz_dsignal_vpdr_tau_opt):
        if do_plots: print("Didn't find best slope at optimal evolution time for VPDR")
        min_dbz_dsignal_vpdr =  min(np.abs(dbz_dsignal_vpdr))
        max_idx = np.argmax(np.abs(1/dbz_dsignal_vpdr))
        tau_opt_vpdr_s = evolution_time_s[max_idx]
    else:
        min_dbz_dsignal_vpdr = np.abs(dbz_dsignal_vpdr_tau_opt)
        tau_opt_vpdr_s = optimal_evolution_time_s

    dbz_dsignal_ramsey = get_ramsey_slope(exp_param_factory, nv_ensemble, off_axis_solver, t_pi_s, b_vector_t, b_plus_db_vector_t, orientation, index_mi0)
    dbz_dsignal_ramsey_tau_opt = get_ramsey_slope_at_tau_opt(optimal_evolution_time_s, exp_param_factory, nv_ensemble, off_axis_solver, t_pi_s, b_vector_t,  b_plus_db_vector_t, orientation, index_mi0)
    if min(np.abs(dbz_dsignal_ramsey))< np.abs(dbz_dsignal_ramsey_tau_opt):
        if do_plots: print("Didn't find best slope at optimal evolution time for Ramsey")
        min_dbz_dsignal_ramsey =  min(np.abs(dbz_dsignal_ramsey))
        max_idx = np.argmax(np.abs(1/dbz_dsignal_ramsey))
        tau_opt_ramsey_s = evolution_time_s[max_idx]
        
    else:
        min_dbz_dsignal_ramsey = np.abs(dbz_dsignal_ramsey_tau_opt)
        tau_opt_ramsey_s = optimal_evolution_time_s
    
    if do_plots:
        evolution_time_s = exp_param_factory.get_experiment_parameters().evolution_time_s
        plt.plot(evolution_time_s, np.abs( 1/dbz_dsignal_vpdr))
        plt.scatter(optimal_evolution_time_s, np.abs(1/dbz_dsignal_vpdr_tau_opt))
        plt.scatter(tau_opt_vpdr_s, 1/min_dbz_dsignal_vpdr)
        plt.show()

        plt.plot(evolution_time_s, np.abs(1/dbz_dsignal_ramsey))
        plt.scatter(optimal_evolution_time_s, np.abs(1/dbz_dsignal_ramsey_tau_opt))
        plt.scatter(tau_opt_ramsey_s, 1/min_dbz_dsignal_ramsey)
        plt.show()
    
    return min_dbz_dsignal_vpdr, min_dbz_dsignal_ramsey, tau_opt_vpdr_s, tau_opt_ramsey_s