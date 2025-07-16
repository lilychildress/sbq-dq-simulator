
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import windows

from bff_paper_figures.extract_experiment_values import  get_true_transition_frequencies
from bff_sensitivity_calculations.sensitivity_functions import extract_time_domain_ramsey_signal, get_optimal_evolution_time_hf_agnostic_s, get_vdpr_and_ramsey_min_slopes
from bff_simulator.abstract_classes.abstract_ensemble import NVOrientation, NV14HyperfineField
from bff_simulator.constants import NVaxes_100, f_h
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.vector_manipulation import perpendicular_projection
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory
from bff_paper_figures.shared_parameters import MW_DIRECTION, E_FIELD_VECTOR_V_PER_CM, RABI_FREQ_BASE_HZ, DETUNING_HZ, T2STAR_S, B_PHI_FIG4, B_THETA_FIG4

B_MAGNITUDE_T = 20e-6
DELTA_B_T = 1e-9
SEED = 291
SIG_STD_DEV = 1e-4
N_SAMPLES = 1000

B_VECTOR_T = B_MAGNITUDE_T * np.array(
    [np.sin(B_THETA_FIG4) * np.cos(B_PHI_FIG4), np.sin(B_THETA_FIG4) * np.sin(B_PHI_FIG4), np.cos(B_THETA_FIG4)]
)

DELTA_B_VECTOR_T = (DELTA_B_T) * np.array(
    [np.sin(B_THETA_FIG4) * np.cos(B_PHI_FIG4), np.sin(B_THETA_FIG4) * np.sin(B_PHI_FIG4), np.cos(B_THETA_FIG4)]
)

IDEAL_RABI_FREQUENCIES= np.array([RABI_FREQ_BASE_HZ * perpendicular_projection(MW_DIRECTION, NVaxis) for NVaxis in NVaxes_100])
INDEX_FOR_MI0 = 1
MW_PULSE_LENGTH_S = np.arange(0, 800e-9, 2.5e-9)  # np.linspace(0, 0.5e-6, 1001)
EVOLUTION_TIME_S = np.arange(0, 3e-6, 10e-9)  # p.linspace(0, 15e-6, 801)
S_TO_NS = 1e9

RABI_FREQ_BASE_HZ = 100e6
MAX_PULSE_DURATIONS = np.arange(50e-9, 800e-9, 50e-9)

def tau_opt_sensitivity_ratios_vs_mw_pulse_max(use_hyperfine=False, orientation=NVOrientation.A, rabi_window_name="boxcar", seed=SEED):

    rabi_freq_hz = IDEAL_RABI_FREQUENCIES[orientation]
    exp_param_factory = OffAxisFieldExperimentParametersFactory()
    exp_param_factory.set_base_rabi_frequency(RABI_FREQ_BASE_HZ)
    exp_param_factory.set_mw_direction(MW_DIRECTION)
    exp_param_factory.set_e_field_v_per_m(E_FIELD_VECTOR_V_PER_CM)
    exp_param_factory.set_detuning(DETUNING_HZ)
    exp_param_factory.set_mw_pulse_lengths(MW_PULSE_LENGTH_S)
    exp_param_factory.set_evolution_times(EVOLUTION_TIME_S)

    # Extract the ground truth for the expected magnetic field
    exp_param_factory.set_b_field_vector(B_VECTOR_T)
    larmor_freqs_all_axes_hz, _ = get_true_transition_frequencies(
        exp_param_factory.get_experiment_parameters()
    )

    nv_ensemble = HomogeneousEnsemble()
    nv_ensemble.t2_star_s = T2STAR_S
    if use_hyperfine:
        nv_ensemble.add_n14_triplet(orientation)
    else:
        nv_ensemble.add_nv_single_species(orientation, NV14HyperfineField.N14_0)

    off_axis_solver = LiouvillianSolver()

    # Calculate the pi pulse time
    t_pi_s = 1/(2*rabi_freq_hz)
    print(f"Pi pulse duration: {t_pi_s*S_TO_NS} ns")

    # Extract the true mi = 0 larmor precession frequency and use it to find the optimal free precession time
    double_larmor_mi0_hz = larmor_freqs_all_axes_hz[orientation][INDEX_FOR_MI0] # double quantum larmor frequency
    optimal_evolution_time_s = get_optimal_evolution_time_hf_agnostic_s(double_larmor_mi0_hz, T2STAR_S, use_hyperfine, f_h, EVOLUTION_TIME_S, do_plot=False)

    rng = np.random.default_rng(seed)

    sensitivity_ratios = []
    for max_pulse_duration in MAX_PULSE_DURATIONS:
        mw_pulse_length_s = np.arange(0, max_pulse_duration, MW_PULSE_LENGTH_S[1]-MW_PULSE_LENGTH_S[0])
        exp_param_factory.set_mw_pulse_lengths(mw_pulse_length_s)    

        # Find the signal response to magnetic field at the best time (which may not be quite the theoretical optimum due to precession during the MW pulses)
        min_dbz_dsignal_vpdr, min_dbz_dsignal_ramsey, tau_opt_vpdr_s, tau_opt_ramsey_s = get_vdpr_and_ramsey_min_slopes(exp_param_factory, nv_ensemble, off_axis_solver, rabi_window_name, optimal_evolution_time_s, t_pi_s, B_VECTOR_T, B_VECTOR_T + DELTA_B_VECTOR_T, orientation, INDEX_FOR_MI0, do_plots = False)

        # Find the noise in the vpdr and ramsey signals at their optimal taus due to injection of Gaussian noise
        n_mw_pulse_lengths = len(mw_pulse_length_s)
        vpdr_noise = []
        for _ in range(N_SAMPLES):
            noise_shape = (n_mw_pulse_lengths, 1)
            noise_instance = extract_time_domain_ramsey_signal(rng.normal(0, SIG_STD_DEV, size=noise_shape), orientation, exp_param_factory.get_experiment_parameters(), rabi_window_name)[0]
            vpdr_noise.append(noise_instance)
        vpdr_sensitivity = np.std(vpdr_noise) * min_dbz_dsignal_vpdr
        ramsey_sensitivity = SIG_STD_DEV/np.sqrt(n_mw_pulse_lengths)* min_dbz_dsignal_ramsey
        sensitivity_ratios.append(vpdr_sensitivity/ramsey_sensitivity)
    return sensitivity_ratios

orientation = NVOrientation.B
max_mw_duration_range = MAX_PULSE_DURATIONS
print("Working on blackman...")
blackman_sensitivity_ratios = tau_opt_sensitivity_ratios_vs_mw_pulse_max(False, orientation, "blackman", SEED)
print("Working on blackman with hyperfine...")
blackman_sensitivity_ratios_hf = tau_opt_sensitivity_ratios_vs_mw_pulse_max(True, orientation, "blackman", SEED+1)
print("Working on boxcar...")
boxcar_sensitivity_ratios = tau_opt_sensitivity_ratios_vs_mw_pulse_max(False, orientation, "boxcar", SEED+2)
print("Working on boxcar with hyperfine..")
boxcar_sensitivity_ratios_hf = tau_opt_sensitivity_ratios_vs_mw_pulse_max(True, orientation, "boxcar", SEED+3)

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