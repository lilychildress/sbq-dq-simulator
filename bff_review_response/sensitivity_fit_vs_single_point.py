
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import windows
from bff_paper_figures.inner_product_functions import (
    inner_product_sinusoid,
)

from bff_paper_figures.extract_experiment_values import  get_true_transition_frequencies
from bff_review_response.sensitivity_helper_functions import get_optimal_evolution_time_s, slope, fit_decaying_cosine, find_optimal_hf_revival_time, slope_triplet, get_signal_slopes
from bff_paper_figures.fitting_routines import fit_three_cos_model
from bff_simulator.abstract_classes.abstract_ensemble import NVOrientation, NV14HyperfineField
from bff_simulator.constants import NVaxes_100, gammab, f_h
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.vector_manipulation import perpendicular_projection
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory
from bff_paper_figures.shared_parameters import  MW_DIRECTION, E_FIELD_VECTOR_V_PER_CM, RABI_FREQ_BASE_HZ, DETUNING_HZ, T2STAR_S, B_PHI_FIG4, B_THETA_FIG4

B_MAGNITUDE_T = 50e-6
DELTA_B_T = 10e-8

B_VECTOR_T = B_MAGNITUDE_T * np.array(
    [np.sin(B_THETA_FIG4) * np.cos(B_PHI_FIG4), np.sin(B_THETA_FIG4) * np.sin(B_PHI_FIG4), np.cos(B_THETA_FIG4)]
)

B_PLUS_DB_VECTOR_T = (B_MAGNITUDE_T + DELTA_B_T) * np.array(
    [np.sin(B_THETA_FIG4) * np.cos(B_PHI_FIG4), np.sin(B_THETA_FIG4) * np.sin(B_PHI_FIG4), np.cos(B_THETA_FIG4)]
)

IDEAL_RABI_FREQUENCIES= np.array([RABI_FREQ_BASE_HZ * perpendicular_projection(MW_DIRECTION, NVaxis) for NVaxis in NVaxes_100])
MW_RABI_PERIOD_DIVISION = 4
EVOLUTION_STEPS_UNTIL_OPTIMAL = 50
EVOLUTION_SAMPLE_SPACING = 11

INDEX_FOR_MI0 = 1

MAX_MW_PULSE_S= 800e-9

SEED = 29481311

USE_HYPERFINE = False
N_SAMPLES = 10**4
SUB_SAMPLING = 100
SIG_STD_DEV = 1e-3

def compare_fit_to_single_point_sensitivity(use_hyperfine, orientation, hyperfine_line=NV14HyperfineField.N14_plus, index_for_hf = 0, rabi_window_name="blackman"):

    rabi_freq_hz = IDEAL_RABI_FREQUENCIES[orientation]

    mw_step_s = 1/(MW_RABI_PERIOD_DIVISION*IDEAL_RABI_FREQUENCIES[orientation])
    mw_pulse_length_s =np.arange(0, MAX_MW_PULSE_S, mw_step_s) 

    off_axis_solver = LiouvillianSolver()

    exp_param_factory = OffAxisFieldExperimentParametersFactory()
    exp_param_factory.set_base_rabi_frequency(RABI_FREQ_BASE_HZ)
    exp_param_factory.set_mw_direction(MW_DIRECTION)
    exp_param_factory.set_e_field_v_per_m(E_FIELD_VECTOR_V_PER_CM)
    exp_param_factory.set_detuning(DETUNING_HZ)
    exp_param_factory.set_mw_pulse_lengths(mw_pulse_length_s)

    # Extract the ground truth for the expected magnetic field
    exp_param_factory.set_b_field_vector(B_VECTOR_T)
    larmor_freqs_all_axes_hz, _ = get_true_transition_frequencies(
        exp_param_factory.get_experiment_parameters()
    )

    nv_ensemble = HomogeneousEnsemble()
    nv_ensemble.t2_star_s = T2STAR_S

    if use_hyperfine:
        nv_ensemble.add_n14_triplet(orientation)

        # Determine the optimal time for measurement (in theory) and make sure it appears in our evolution times
        larmor_mi0_hz = larmor_freqs_all_axes_hz[orientation][INDEX_FOR_MI0] # double quantum larmor frequency
        larmor_actual_hz = larmor_mi0_hz
        optimal_evolution_time_s = find_optimal_hf_revival_time(larmor_mi0_hz, f_h, T2STAR_S)
        evolution_times_s = np.arange(0, 2*T2STAR_S, optimal_evolution_time_s/(EVOLUTION_STEPS_UNTIL_OPTIMAL))
        exp_param_factory.set_evolution_times(evolution_times_s)

        # Check that we actually got the optimal time
        evolution_times_fine_s = np.arange(0, 2*T2STAR_S, optimal_evolution_time_s/(EVOLUTION_STEPS_UNTIL_OPTIMAL*10))
        plt.plot(evolution_times_fine_s, np.abs(slope_triplet(evolution_times_fine_s, larmor_mi0_hz, f_h, T2STAR_S)))
        plt.vlines(optimal_evolution_time_s, 0, max(slope_triplet(evolution_times_fine_s, larmor_mi0_hz, f_h, T2STAR_S)), color="red")
        plt.show()
    else:
        nv_ensemble.add_nv_single_species(orientation, hyperfine_line)

        # Determine the optimal time for measurement (in theory) and make sure it appears in our evolution times
        larmor_actual_hz = larmor_freqs_all_axes_hz[orientation][index_for_hf]

        optimal_evolution_time_s = get_optimal_evolution_time_s(larmor_actual_hz, T2STAR_S)
        evolution_times_s = np.arange(0, 2*T2STAR_S, optimal_evolution_time_s/EVOLUTION_STEPS_UNTIL_OPTIMAL)
        exp_param_factory.set_evolution_times(evolution_times_s)

        # Check that we got the right optimal time
        evolution_times_fine_s = np.arange(0, 2*T2STAR_S, optimal_evolution_time_s/(EVOLUTION_STEPS_UNTIL_OPTIMAL*10))
        plt.plot(evolution_times_fine_s, np.abs(slope(evolution_times_fine_s, larmor_actual_hz, T2STAR_S)))
        plt.vlines(optimal_evolution_time_s, 0, max(slope(evolution_times_fine_s, larmor_actual_hz, T2STAR_S)), color="red")
        plt.show()

    print(f"tau_opt: {optimal_evolution_time_s}, evolution_time_step: {evolution_times_s[1]}, evolution_time_max: {evolution_times_s[-1]}")
    print(f"pi pulse: {mw_pulse_length_s[1]*int(MW_RABI_PERIOD_DIVISION/2)}, mw_pulse_step: {mw_pulse_length_s[1]}, max_mw_pulse: {mw_pulse_length_s[-1]} ")
    # Find the single-point optimal-time measurement slopes (dbz/dsignal)
    slope_dbz_dsignal_dq, slope_dbz_dsignal_inner_product, sq_cancelled_signal, time_domain_ramsey_signal = get_signal_slopes(exp_param_factory, nv_ensemble, off_axis_solver, rabi_window_name, B_VECTOR_T, B_PLUS_DB_VECTOR_T, EVOLUTION_STEPS_UNTIL_OPTIMAL, int(MW_RABI_PERIOD_DIVISION/2), orientation)  
        
    # Now add in measurement noise, and see how much that changes the extracted fields
    rng = np.random.default_rng(SEED)

    # Select a range of maximum evolution times to consider for fitting; drop the very short-time values.
    max_evolution_times_indices = np.arange(0, len(evolution_times_s)+1, EVOLUTION_SAMPLE_SPACING)[2:]

    # Get the windowing function for the inner product
    rabi_window = windows.get_window(rabi_window_name, len(mw_pulse_length_s))

    noisy_delta_b_inner_product_t = []
    noisy_delta_b_dq_t = []
    noisy_fit_bz_t = []
    noisy_fit_bz_dq_t = []
    for n in range(N_SAMPLES):
        if n% SUB_SAMPLING == 0:
            print(f"{n} of {N_SAMPLES} done")
        # Calculate the noisy inner-producted signal
        noisy_sq_cancelled_signal = sq_cancelled_signal + rng.normal(0, SIG_STD_DEV, size=sq_cancelled_signal.shape)
        noisy_time_domain_ramsey_signal = inner_product_sinusoid(np.cos,rabi_freq_hz,mw_pulse_length_s,rabi_window * np.transpose(noisy_sq_cancelled_signal),axis=1)
        
        # And convert it to a noisy field excursion using optimal-evolution-time measurements and the slope
        noisy_delta_b_inner_product_t.append((noisy_time_domain_ramsey_signal[EVOLUTION_STEPS_UNTIL_OPTIMAL] - time_domain_ramsey_signal[EVOLUTION_STEPS_UNTIL_OPTIMAL])*slope_dbz_dsignal_inner_product)

        # calculate the noisy DQ signal, recognizing that we need to reduce the noise by a factor of sqrt(len(mw_pulse_durations_s)) to account for 
        # averaging the same number of times.
        pi_pulse_idx = int(MW_RABI_PERIOD_DIVISION/2)
        noisy_dq_signal = sq_cancelled_signal[pi_pulse_idx] + rng.normal(0, SIG_STD_DEV/np.sqrt(len(mw_pulse_length_s)), size=len(evolution_times_s))

        # find the field excursion in the standard single-point dq signal (averaging over the same number of measurements as there are steps in mw_pulse_length_s)
        noisy_delta_b_dq_t.append(np.mean(rng.normal(0, SIG_STD_DEV, size=len(mw_pulse_length_s)) ) * slope_dbz_dsignal_dq)

        # As a function of maximum evolution time to include in the fit, use the same data sets to extract the magnetic field by fitting to decaying cosines.
        bz_fit_vs_idx = []
        bz_fit_vs_idx_dq = []

        # Don't do as many fits because they are really really slow!
        if n % SUB_SAMPLING == 0:
            for max_idx in max_evolution_times_indices:

                if use_hyperfine:

                    freq_guesses_hz = np.sort(np.abs(np.array([larmor_mi0_hz-2*f_h, larmor_mi0_hz, larmor_mi0_hz+2*f_h])))[::-1]

                    time_domain_fit = fit_three_cos_model(evolution_times_s[:int(max_idx)], noisy_time_domain_ramsey_signal[:int(max_idx)],freq_guesses_hz, T2STAR_S, constrain_hyperfine_freqs=True, constrain_same_decay=True)
                    time_domain_fit_dq = fit_three_cos_model(evolution_times_s[:int(max_idx)], noisy_dq_signal[:int(max_idx)],freq_guesses_hz, T2STAR_S, constrain_hyperfine_freqs=True, constrain_same_decay=True)
                    
                    bz_fit_vs_idx.append(time_domain_fit.params["p0_freq"].value/(2*gammab))
                    bz_fit_vs_idx_dq.append(time_domain_fit_dq.params["p0_freq"].value/(2*gammab))
                else:
                    # Fit the inner-producted VPDR signal
                    time_domain_fit = fit_decaying_cosine(evolution_times_s[:int(max_idx)], noisy_time_domain_ramsey_signal[:int(max_idx)], larmor_freqs_all_axes_hz[orientation][index_for_hf])

                    # And also the noisy DQ signal

                    time_domain_fit_dq = fit_decaying_cosine(evolution_times_s[:int(max_idx)], noisy_dq_signal[:int(max_idx)], larmor_freqs_all_axes_hz[orientation][index_for_hf])
                
                    bz_fit_vs_idx.append(time_domain_fit.params["freq"].value/(2*gammab))
                    bz_fit_vs_idx_dq.append(time_domain_fit_dq.params["freq"].value/(2*gammab))

            noisy_fit_bz_t.append(bz_fit_vs_idx)
            noisy_fit_bz_dq_t.append(bz_fit_vs_idx_dq)


    # Calculate the ratio of the noise in the magnetic field, which should be (approximately) the ratio of the sensitivities, assuming that the time is 
    # dominated by measurement time.
    vpdr_uncertainty = np.std(np.array(noisy_delta_b_inner_product_t))
    dq_uncertainty = np.std(np.array(noisy_delta_b_dq_t))
    print(vpdr_uncertainty, dq_uncertainty, vpdr_uncertainty/dq_uncertainty)
    # Note that the fit uses more time, so we should multipy by sqrt[number of evolution times] to account for that
    fit_sensitivity_ratio_vs_max_evolution_time=np.std(np.array(noisy_fit_bz_t), axis=0)*np.sqrt(max_evolution_times_indices)/vpdr_uncertainty
    fit_sensitivity_ratio_vs_max_evolution_time_dq=np.std(np.array(noisy_fit_bz_dq_t), axis=0)*np.sqrt(max_evolution_times_indices)/dq_uncertainty

    return fit_sensitivity_ratio_vs_max_evolution_time, fit_sensitivity_ratio_vs_max_evolution_time_dq, evolution_times_s[max_evolution_times_indices-1], optimal_evolution_time_s, larmor_actual_hz

fit_sensitivity_ratio_vs_max_evolution_time, fit_sensitivity_ratio_vs_max_evolution_time_dq, sampled_max_evolution_times_s, optimal_evolution_time_s, larmor_hz = compare_fit_to_single_point_sensitivity(False, NVOrientation.B, NV14HyperfineField.N14_0, index_for_hf=1)

fit_sensitivity_ratio_vs_max_evolution_time_hf, fit_sensitivity_ratio_vs_max_evolution_time_dq_hf, sampled_max_evolution_times_hf_s, optimal_evolution_time_hf_s, larmor_mi0_hz = compare_fit_to_single_point_sensitivity(True, NVOrientation.B)

# Determine the expected sensitivity ratios from the average absolute value of the slope of the Ramsey signal
# compared to its maximum value.
opt_slope = slope(optimal_evolution_time_s, larmor_hz, T2STAR_S)
opt_slope_triplet = np.abs(slope_triplet(optimal_evolution_time_hf_s, larmor_mi0_hz, f_h, T2STAR_S))

evolution_times_fine_s = np.arange(0, 2*T2STAR_S, optimal_evolution_time_s/(EVOLUTION_STEPS_UNTIL_OPTIMAL))
evolution_times_fine_hf_s = np.arange(0, 2*T2STAR_S, optimal_evolution_time_hf_s/(EVOLUTION_STEPS_UNTIL_OPTIMAL))
abs_triplet_slopes = np.abs(slope_triplet(evolution_times_fine_hf_s, larmor_mi0_hz, f_h, T2STAR_S))
abs_slopes = np.abs(slope(evolution_times_fine_s, larmor_hz, T2STAR_S))

avg_abs_slopes = []
for max_idx, max_time in enumerate(evolution_times_fine_s):
    avg_abs_slopes.append(np.mean(abs_slopes[:max_idx + 1]))

avg_abs_slopes_triplet = []    
for max_idx, max_time in enumerate(evolution_times_fine_hf_s):
    avg_abs_slopes_triplet.append(np.mean(abs_triplet_slopes[:max_idx + 1]))

np.savetxt("evolution_times_s.txt", sampled_max_evolution_times_s)
np.savetxt("fit_sens_ratio_vs_evolution.txt", fit_sensitivity_ratio_vs_max_evolution_time)
np.savetxt("fit_sens_ratio_vs_evolution_dq.txt", fit_sensitivity_ratio_vs_max_evolution_time_dq)

np.savetxt("evolution_times_hf_s.txt", sampled_max_evolution_times_hf_s)
np.savetxt("fit_sens_ratio_vs_evolution_hf.txt", fit_sensitivity_ratio_vs_max_evolution_time_hf)
np.savetxt("fit_sens_ratio_vs_evolution_dq_hf.txt", fit_sensitivity_ratio_vs_max_evolution_time_dq_hf)

s_to_us = 1e6
s_to_ns = 1e9
plt.figure(0, figsize=(2.9, 1.75))
plt.rcParams["font.size"] = 9
plt.rcParams["font.family"] = "arial"
plt.plot(s_to_us*sampled_max_evolution_times_s, fit_sensitivity_ratio_vs_max_evolution_time, marker = ".", linestyle="", label="VPDR", color="blue")
plt.plot(s_to_us*sampled_max_evolution_times_s, fit_sensitivity_ratio_vs_max_evolution_time_dq, marker = "*", linestyle="", label="DQ", color="blue")
plt.plot(s_to_us*sampled_max_evolution_times_hf_s, fit_sensitivity_ratio_vs_max_evolution_time_hf, marker = ".", linestyle="", markerfacecolor="none",label="VPDR HF", color="green")
plt.plot(s_to_us*sampled_max_evolution_times_hf_s, fit_sensitivity_ratio_vs_max_evolution_time_dq_hf, marker = "*", linestyle="",  markerfacecolor="none", label="DQ HF", color="green")

plt.plot(s_to_us*evolution_times_fine_s, opt_slope/np.array(avg_abs_slopes), color="blue")
plt.plot(s_to_us*evolution_times_fine_hf_s, opt_slope_triplet/np.array(avg_abs_slopes_triplet), color="green")

plt.xlabel(r"Maximum free evolution time ($\mu$s)")
plt.ylabel("Fit sensitivity vs \noptimal-time sensitivity")
plt.legend()
plt.vlines(optimal_evolution_time_s*s_to_us, 0,50, linestyle="dashed", color="blue")#, label="Optimal evolution time")
plt.vlines(optimal_evolution_time_hf_s*s_to_us, 0,50, linestyle="dotted", color="green")#, label="Optimal evolution time")
plt.legend(loc="upper right", bbox_to_anchor=(.7,1))
plt.gca().yaxis.set_ticks_position("both")
plt.gca().xaxis.set_ticks_position("both")
plt.ylim((1, 40))
plt.xlim(0, 4)
plt.yscale("log")
plt.text(4.3, 1,  f"{MAX_MW_PULSE_S*s_to_ns} ns max \npulse duration")
plt.gca().minorticks_on()
plt.gca().tick_params(direction="in", which="both", width=1.5)
plt.gca().tick_params(direction="in", which="minor", length=2.5)
plt.gca().tick_params(direction="in", which="major", length=5)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.25)
plt.savefig("fit_sensitivity.svg")
plt.show()