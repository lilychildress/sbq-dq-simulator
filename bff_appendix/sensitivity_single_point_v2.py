import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import windows
from lmfit import Model

from bff_paper_figures.extract_experiment_values import  get_true_transition_frequencies, get_ideal_rabi_frequencies
from bff_review_response.sensitivity_helper_functions import get_optimal_evolution_time_s, slope, get_signal_slopes, compare_single_point_sensitivity, find_optimal_hf_revival_time, slope_triplet
from bff_simulator.abstract_classes.abstract_ensemble import NVOrientation, NV14HyperfineField
from bff_simulator.constants import NVaxes_100, gammab, f_h
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.vector_manipulation import perpendicular_projection
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory, OffAxisFieldExperimentParameters
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
MW_RABI_PERIOD_DIVISION = 10
INDEX_FOR_MI0=1
EVOLUTION_STEPS_UNTIL_OPTIMAL = 10

# MW_PULSE_LENGTH_S =np.arange(0, 800e-9, MW_STEP_S)  #np.arange(0, 800e-9, 2.5e-9)  # np.linspace(0, 0.5e-6, 1001)
# EVOLUTION_TIME_S = np.arange(0, 3e-6, 20e-9)  # p.linspace(0, 15e-6, 801)

MAX_MW_DURATION_RANGE_S = np.arange(50e-9, 800e-9, 25e-9)

SEED = 294813022

S_TO_NS = 1e9  

def singlepoint_sensitivity_ratios_vs_mw_pulse_max(use_hyperfine=False, orientation=NVOrientation.A, hyperfine=NV14HyperfineField.N14_plus, INDEX_FOR_MI=0):

    off_axis_solver = LiouvillianSolver()

    exp_param_factory = OffAxisFieldExperimentParametersFactory()
    exp_param_factory.set_base_rabi_frequency(RABI_FREQ_BASE_HZ)
    exp_param_factory.set_mw_direction(MW_DIRECTION)
    exp_param_factory.set_e_field_v_per_m(E_FIELD_VECTOR_V_PER_CM)
    exp_param_factory.set_detuning(DETUNING_HZ)

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
        optimal_evolution_time_s = find_optimal_hf_revival_time(larmor_mi0_hz, f_h, T2STAR_S)

        evolution_times_s = np.arange(0, 2*T2STAR_S, optimal_evolution_time_s/(EVOLUTION_STEPS_UNTIL_OPTIMAL))
        exp_param_factory.set_evolution_times(evolution_times_s)

        # Check that we actually got the optimal time
        evolution_times_fine_s = np.arange(0, 2*T2STAR_S, optimal_evolution_time_s/(EVOLUTION_STEPS_UNTIL_OPTIMAL*10))
        plt.plot(evolution_times_fine_s, np.abs(slope_triplet(evolution_times_fine_s, larmor_mi0_hz, f_h, T2STAR_S)))
        plt.vlines(optimal_evolution_time_s, 0, max(slope_triplet(evolution_times_fine_s, larmor_mi0_hz, f_h, T2STAR_S)), color="red")
        plt.show()
    else:
        nv_ensemble.add_nv_single_species(orientation, hyperfine)

        # Determine the optimal time for measurement (in theory) and make sure it appears in our evolution times
        larmor_actual_hz = larmor_freqs_all_axes_hz[orientation][INDEX_FOR_MI] # double quantum larmor frequency
        optimal_evolution_time_s = get_optimal_evolution_time_s(larmor_actual_hz, T2STAR_S)

        evolution_times_s = np.arange(0, 2*T2STAR_S, optimal_evolution_time_s/(EVOLUTION_STEPS_UNTIL_OPTIMAL))
        exp_param_factory.set_evolution_times(evolution_times_s)

        # Check that we actually got the optimal time
        evolution_times_fine_s = np.arange(0, 2*T2STAR_S, optimal_evolution_time_s/(EVOLUTION_STEPS_UNTIL_OPTIMAL*10))
        plt.plot(evolution_times_fine_s, np.abs(slope(evolution_times_fine_s, larmor_actual_hz, T2STAR_S)))
        plt.vlines(optimal_evolution_time_s, 0, max(slope(evolution_times_fine_s, larmor_actual_hz, T2STAR_S)), color="red")
        plt.show()

    max_mw_duration_range = MAX_MW_DURATION_RANGE_S
    mw_step_s = 1/(MW_RABI_PERIOD_DIVISION*IDEAL_RABI_FREQUENCIES[orientation])

    blackman_sensitivity_ratios = []
    boxcar_sensitivity_ratios = []
    rng = np.random.default_rng(SEED)
    for max_mw_duration in max_mw_duration_range: 
        print(f"Maximum MW duration: {max_mw_duration*S_TO_NS} ns")
        mw_pulse_length_s =np.arange(0, max_mw_duration, mw_step_s) 
        exp_param_factory.set_mw_pulse_lengths(mw_pulse_length_s)

        # find the slope in signal at the optimal time to measure, and use it to determine the ratio of sensitivities
        # for Rabi-inner-producted VPDR vs DQ signals both evaluated at the optimal evolution time.
        rabi_window_name = "blackman"
        slope_dbz_dsignal_dq, slope_dbz_dsignal_inner_product, sq_cancelled_signal, time_domain_ramsey_signal = get_signal_slopes(exp_param_factory, nv_ensemble, off_axis_solver, rabi_window_name, B_VECTOR_T, B_PLUS_DB_VECTOR_T, EVOLUTION_STEPS_UNTIL_OPTIMAL, int(MW_RABI_PERIOD_DIVISION/2), orientation)
        blackman_sensitivity_ratios.append(compare_single_point_sensitivity(slope_dbz_dsignal_dq, slope_dbz_dsignal_inner_product, sq_cancelled_signal, time_domain_ramsey_signal, rabi_window_name, exp_param_factory.get_experiment_parameters(), rng,EVOLUTION_STEPS_UNTIL_OPTIMAL))
        
        rabi_window_name = "boxcar"
        slope_dbz_dsignal_dq, slope_dbz_dsignal_inner_product, sq_cancelled_signal, time_domain_ramsey_signal = get_signal_slopes(exp_param_factory, nv_ensemble, off_axis_solver, rabi_window_name, B_VECTOR_T, B_PLUS_DB_VECTOR_T, EVOLUTION_STEPS_UNTIL_OPTIMAL, int(MW_RABI_PERIOD_DIVISION/2), orientation)
        boxcar_sensitivity_ratios.append(compare_single_point_sensitivity(slope_dbz_dsignal_dq, slope_dbz_dsignal_inner_product, sq_cancelled_signal, time_domain_ramsey_signal, rabi_window_name, exp_param_factory.get_experiment_parameters(), rng, EVOLUTION_STEPS_UNTIL_OPTIMAL))

    return max_mw_duration_range, blackman_sensitivity_ratios, boxcar_sensitivity_ratios

orientation = NVOrientation.B
hyperfine_line = NV14HyperfineField.N14_0
hyperfine_line_index = 1
max_mw_duration_range, blackman_sensitivity_ratios, boxcar_sensitivity_ratios = singlepoint_sensitivity_ratios_vs_mw_pulse_max(False, orientation, hyperfine_line, hyperfine_line_index)
max_mw_duration_range_hf, blackman_sensitivity_ratios_hf, boxcar_sensitivity_ratios_hf = singlepoint_sensitivity_ratios_vs_mw_pulse_max(True, orientation)

np.savetxt("max_mw_duration_range.txt", max_mw_duration_range)
np.savetxt("blackman_sensitivity_ratios.txt", blackman_sensitivity_ratios)
np.savetxt("boxcar_sensitivity_ratios.txt", boxcar_sensitivity_ratios)
np.savetxt("max_mw_duration_range_hf.txt", max_mw_duration_range_hf)
np.savetxt("blackman_sensitivity_ratios_hf.txt", blackman_sensitivity_ratios_hf)
np.savetxt("boxcar_sensitivity_ratios_hf.txt", boxcar_sensitivity_ratios_hf)

rabi_window = windows.get_window("blackman", 1000)
np.mean(rabi_window)
blackman_factor = np.sqrt(np.mean(rabi_window**2))/np.mean(rabi_window)

plt.figure(0, figsize=(2.9, 1.75))
plt.rcParams["font.size"] = 9
plt.rcParams["font.family"] = "arial"
plt.plot(max_mw_duration_range *S_TO_NS, blackman_sensitivity_ratios, marker="o", linestyle="", label= "Blackman", color="blue")
plt.plot(max_mw_duration_range *S_TO_NS, blackman_sensitivity_ratios_hf, marker="x", linestyle="", label= "Blackman HF", color="blue")
plt.hlines(blackman_factor*2*np.sqrt(2), min(max_mw_duration_range*S_TO_NS), max(max_mw_duration_range*S_TO_NS), linestyle="dashed", label=r"2$\sqrt{2}W_{\text{rms}}/\bar{W}$")
plt.plot(max_mw_duration_range *S_TO_NS, boxcar_sensitivity_ratios, marker="o", linestyle="", color="red", label = "Boxcar")
plt.plot(max_mw_duration_range *S_TO_NS, boxcar_sensitivity_ratios_hf, marker="x", linestyle="", color="red", label = "Boxcar HF")
plt.hlines(2*np.sqrt(2), min(max_mw_duration_range*S_TO_NS), max(max_mw_duration_range*S_TO_NS), color="red", linestyle="dotted", label = r"2$\sqrt{2}$")
plt.legend(loc="lower center", ncol=1, bbox_to_anchor=(1.5,.5))
plt.xlabel("Maximum pulse duration (ns)")
plt.ylabel("VPDR vs DQ optimal sensitivity")
plt.ylim((2,6))
plt.gca().yaxis.set_ticks_position("both")
plt.gca().xaxis.set_ticks_position("both")
plt.gca().minorticks_on()
plt.gca().tick_params(direction="in", which="both", width=1.5)
plt.gca().tick_params(direction="in", which="minor", length=2.5)
plt.gca().tick_params(direction="in", which="major", length=5)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.25)
plt.savefig("single_pt_sensitivity.svg")
plt.show()
    