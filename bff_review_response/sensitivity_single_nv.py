
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from scipy.signal import windows
from lmfit import Model
from bff_paper_figures.inner_product_functions import (
    inner_product_sinusoid,
)

from bff_paper_figures.extract_experiment_values import  get_true_transition_frequencies
from bff_paper_figures.simulation_helper_functions import sq_cancelled_signal_generator
from bff_review_response.sensitivity_helper_functions import get_optimal_evolution_time_s, slope, fit_decaying_cosine, find_optimal_hf_revival_time, slope_triplet, get_signal_slopes, SIG_STD_DEV, N_SAMPLES
from bff_paper_figures.fitting_routines import decaying_cosine, offset, fit_three_cos_model
from bff_simulator.abstract_classes.abstract_ensemble import NVOrientation, NV14HyperfineField
from bff_simulator.constants import NVaxes_100, gammab, f_h
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.vector_manipulation import perpendicular_projection
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory
from bff_paper_figures.shared_parameters import  MW_DIRECTION, E_FIELD_VECTOR_V_PER_CM, RABI_FREQ_BASE_HZ, DETUNING_HZ, T2STAR_S, B_PHI_FIG4, B_THETA_FIG4

B_MAGNITUDE_T = 50e-6
DELTA_B_T = 1e-8

B_VECTOR_T = B_MAGNITUDE_T * np.array(
    [np.sin(B_THETA_FIG4) * np.cos(B_PHI_FIG4), np.sin(B_THETA_FIG4) * np.sin(B_PHI_FIG4), np.cos(B_THETA_FIG4)]
)

B_PLUS_DB_VECTOR_T = (B_MAGNITUDE_T + DELTA_B_T) * np.array(
    [np.sin(B_THETA_FIG4) * np.cos(B_PHI_FIG4), np.sin(B_THETA_FIG4) * np.sin(B_PHI_FIG4), np.cos(B_THETA_FIG4)]
)

IDEAL_RABI_FREQUENCIES= np.array([RABI_FREQ_BASE_HZ * perpendicular_projection(MW_DIRECTION, NVaxis) for NVaxis in NVaxes_100])
MW_RABI_PERIOD_DIVISION = 4
EVOLUTION_STEPS_UNTIL_OPTIMAL = 100

INDEX_FOR_MI0 = 1

MAX_MW_PULSE_S= 800e-9

SEED = 294813022

USE_HYPERFINE = False

rabi_window_name = "blackman"
orientation = NVOrientation.A
hyperfine_line = NV14HyperfineField.N14_plus
index_for_hf = 0
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
larmor_freqs_all_axes_hz, bz_values_all_axes_t = get_true_transition_frequencies(
    exp_param_factory.get_experiment_parameters()
)

nv_ensemble = HomogeneousEnsemble()
nv_ensemble.t2_star_s = T2STAR_S

if USE_HYPERFINE:
    nv_ensemble.add_n14_triplet(orientation)

    # Determine the optimal time for measurement (in theory) and make sure it appears in our evolution times
    larmor_mi0_hz = larmor_freqs_all_axes_hz[orientation][INDEX_FOR_MI0] # double quantum larmor frequency
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

# Find the single-point optimal-time measurement slopes (dbz/dsignal)
slope_dbz_dsignal_dq, slope_dbz_dsignal_inner_product, sq_cancelled_signal, time_domain_ramsey_signal = get_signal_slopes(exp_param_factory, nv_ensemble, off_axis_solver, rabi_window_name, B_VECTOR_T, B_PLUS_DB_VECTOR_T, EVOLUTION_STEPS_UNTIL_OPTIMAL, int(MW_RABI_PERIOD_DIVISION/2), orientation)
      
# Now add in measurement noise, and see how much that changes the signals
rng = np.random.default_rng(SEED)

# Select a range of maximum evolution times to consider for fitting
max_evolution_times_indices = np.arange(0, len(evolution_times_s)+1, 22)[1:]

rabi_window = windows.get_window(rabi_window_name, len(mw_pulse_length_s))

noisy_delta_b_inner_product_t = []
noisy_delta_b_dq_t = []
noisy_fit_bz_t = []
noisy_fit_bz_dq_t = []
for n in range(N_SAMPLES):
    if n%(N_SAMPLES/10) == 0:
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

    # As a function of maximum evolution time to include in a fit...
    bz_fit_vs_idx = []
    bz_fit_vs_idx_dq = []
    for max_idx in max_evolution_times_indices:

        if USE_HYPERFINE:

            freq_guesses_hz = np.sort(np.abs(np.array([larmor_mi0_hz-2*f_h, larmor_mi0_hz, larmor_mi0_hz+2*f_h])))[::-1]

            time_domain_fit = fit_three_cos_model(evolution_times_s[:int(max_idx)], noisy_time_domain_ramsey_signal[:int(max_idx)],freq_guesses_hz, T2STAR_S, constrain_hyperfine_freqs=True, constrain_same_decay=True)
            time_domain_fit_dq = fit_three_cos_model(evolution_times_s[:int(max_idx)], noisy_dq_signal[:int(max_idx)],freq_guesses_hz, T2STAR_S, constrain_hyperfine_freqs=True, constrain_same_decay=True)
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
print(np.std(np.array(noisy_delta_b_inner_product_t))/np.std(np.array(noisy_delta_b_dq_t)))
# Note that the fit uses more time, so we should multipy by sqrt[number of evolution times] to account for that
fit_sensitivity_ratio_vs_max_evolution_time=np.std(np.array(noisy_fit_bz_t), axis=0)*np.sqrt(max_evolution_times_indices)/np.std(np.array(noisy_delta_b_dq_t))
fit_sensitivity_ratio_vs_max_evolution_time_dq=np.std(np.array(noisy_fit_bz_dq_t), axis=0)*np.sqrt(max_evolution_times_indices)/np.std(np.array(noisy_delta_b_dq_t))


np.savetxt("evolution_times_s.txt", evolution_times_s)
np.savetxt("fit_sens_ratio_vs_evolution_800ns_mw.txt", fit_sensitivity_ratio_vs_max_evolution_time)

s_to_us = 1e6
s_to_ns = 1e9
plt.figure(0, figsize=(3.4, 1.5))
plt.rcParams["font.size"] = 9
plt.rcParams["font.family"] = "arial"
plt.plot(s_to_us*evolution_times_s[max_evolution_times_indices-1], fit_sensitivity_ratio_vs_max_evolution_time, marker = ".", linestyle="")
plt.plot(s_to_us*evolution_times_s[max_evolution_times_indices-1], fit_sensitivity_ratio_vs_max_evolution_time_dq, marker = ".", linestyle="")
plt.xlabel(r"Maximum free evolution time ($\mu$s)")
plt.ylabel("VPDR fit vs DQ opt. time \nsensitivity ratio")
plt.vlines(optimal_evolution_time_s*s_to_us, 10, 40, linestyle="dashed", label="Optimal evolution time")
plt.legend(loc="upper right")
plt.gca().yaxis.set_ticks_position("both")
plt.gca().xaxis.set_ticks_position("both")
plt.ylim((10, 50))
plt.text(2.8, 30,  f"{MAX_MW_PULSE_S*s_to_ns} ns max \npulse duration")
plt.gca().minorticks_on()
plt.gca().tick_params(direction="in", which="both", width=1.5)
plt.gca().tick_params(direction="in", which="minor", length=2.5)
plt.gca().tick_params(direction="in", which="major", length=5)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.25)
plt.savefig("fit_sensitivity.svg")
plt.show()