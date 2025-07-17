import numpy as np
from matplotlib import pyplot as plt

from bff_paper_figures.extract_experiment_values import  get_true_transition_frequencies
from bff_sensitivity_calculations.sensitivity_functions import get_optimal_evolution_time_hf_agnostic_s, get_vpdr_slope_dbz_dsignal_at_tau_opt, get_ramsey_slope_dbz_dsignal_at_tau_opt, optimal_to_avg_slope_ratio_dsignal_dlarmor,  vpdr_and_ramsey_sensitivity_fitting, vpdr_and_ramsey_sensitivity_at_tau_opt
from bff_simulator.abstract_classes.abstract_ensemble import NVOrientation, NV14HyperfineField
from bff_simulator.constants import NVaxes_100, f_h
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.vector_manipulation import perpendicular_projection
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory
from bff_paper_figures.shared_parameters import MW_DIRECTION, E_FIELD_VECTOR_V_PER_CM, RABI_FREQ_BASE_HZ, DETUNING_HZ, T2STAR_S, B_PHI_FIG4, B_THETA_FIG4

B_MAGNITUDE_T = 50e-6
DELTA_B_T = 1e-9
SEED = 29
SIG_STD_DEV = 1e-4
N_SAMPLES = 1000
N_SAMPLES_FIT = 100
S_TO_US = 1e6
S_TO_NS = 1e9

B_VECTOR_T = B_MAGNITUDE_T * np.array(
    [np.sin(B_THETA_FIG4) * np.cos(B_PHI_FIG4), np.sin(B_THETA_FIG4) * np.sin(B_PHI_FIG4), np.cos(B_THETA_FIG4)]
)

DELTA_B_VECTOR_T = (DELTA_B_T) * np.array(
    [np.sin(B_THETA_FIG4) * np.cos(B_PHI_FIG4), np.sin(B_THETA_FIG4) * np.sin(B_PHI_FIG4), np.cos(B_THETA_FIG4)]
)

IDEAL_RABI_FREQUENCIES= np.array([RABI_FREQ_BASE_HZ * perpendicular_projection(MW_DIRECTION, NVaxis) for NVaxis in NVaxes_100])
INDEX_FOR_MI0 = 1
MW_PULSE_LENGTH_S = np.arange(0, 800e-9, 2.5e-9) 
EVOLUTION_TIME_S = np.arange(0, 3e-6, 20e-9)  
S_TO_NS = 1e9

RABI_FREQ_BASE_HZ = 100e6
MAX_EVOLUTION_TIMES_S = np.arange(400e-9, 4e-6, 200e-9)

use_hyperfine = True
seed = 29
orientation = NVOrientation.B
rabi_window_name = "blackman"

def fit_vs_tau_opt_sensitivity_vs_max_evolution_time(use_hyperfine=False, orientation=NVOrientation.A, rabi_window_name="blackman", seed=SEED):

    rabi_freq_hz = IDEAL_RABI_FREQUENCIES[orientation]

    # Set up the experiment
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
    print(f"Larmor frequency: {double_larmor_mi0_hz/2 * 1e-6:.02} MHz")
    optimal_evolution_time_s = get_optimal_evolution_time_hf_agnostic_s(double_larmor_mi0_hz, T2STAR_S, use_hyperfine, f_h, EVOLUTION_TIME_S, do_plot=False)

    rng = np.random.default_rng(SEED)

    ##################### Determine the optimal-time sensitivities #############################
    # Find the signal response to magnetic field at the best time 
    # (commented out: use the numerical maximum if it's better; it doesn't make a noticeable difference)
    # min_dbz_dsignal_vpdr, min_dbz_dsignal_ramsey, tau_opt_vpdr_s, tau_opt_ramsey_s = get_vdpr_and_ramsey_min_slopes_dbz_dsignal(exp_param_factory, nv_ensemble, off_axis_solver, rabi_window_name, optimal_evolution_time_s, t_pi_s, B_VECTOR_T, B_VECTOR_T + DELTA_B_VECTOR_T, orientation, INDEX_FOR_MI0, do_plots = False)
    min_dbz_dsignal_vpdr = np.abs(get_vpdr_slope_dbz_dsignal_at_tau_opt(optimal_evolution_time_s, exp_param_factory, nv_ensemble, off_axis_solver, rabi_window_name, B_VECTOR_T, B_VECTOR_T + DELTA_B_VECTOR_T, orientation, INDEX_FOR_MI0))
    min_dbz_dsignal_ramsey = np.abs(get_ramsey_slope_dbz_dsignal_at_tau_opt(optimal_evolution_time_s, exp_param_factory, nv_ensemble, off_axis_solver, t_pi_s,  B_VECTOR_T, B_VECTOR_T + DELTA_B_VECTOR_T,orientation, INDEX_FOR_MI0))
    
    # Find the noise in the vpdr and ramsey signals at their optimal taus due to injection of Gaussian noise
    vpdr_sensitivity, ramsey_sensitivity = vpdr_and_ramsey_sensitivity_at_tau_opt(exp_param_factory.get_experiment_parameters(), orientation, rabi_window_name, min_dbz_dsignal_vpdr, min_dbz_dsignal_ramsey, rng, N_SAMPLES, SIG_STD_DEV)

    vpdr_fit_vs_tau_opt = []
    ramsey_fit_vs_tau_opt = []
    for max_evolution_time_s in MAX_EVOLUTION_TIMES_S:
        print(f"Max evolution time: {max_evolution_time_s*S_TO_NS:.0f} ns")
        evolution_time_s = np.arange(0, max_evolution_time_s, EVOLUTION_TIME_S[1] - EVOLUTION_TIME_S[0])
        exp_param_factory.set_evolution_times(evolution_time_s)
        
        # Determine the sensitivity from fitting 
        vpdr_fit_sensitivity, ramsey_fit_sensitivity = vpdr_and_ramsey_sensitivity_fitting(exp_param_factory, nv_ensemble, off_axis_solver, t_pi_s, orientation, rabi_window_name, use_hyperfine, double_larmor_mi0_hz, rng, N_SAMPLES_FIT, SIG_STD_DEV)

        vpdr_fit_vs_tau_opt.append(vpdr_fit_sensitivity/vpdr_sensitivity)
        ramsey_fit_vs_tau_opt.append(ramsey_fit_sensitivity/ramsey_sensitivity)

    return vpdr_fit_vs_tau_opt, ramsey_fit_vs_tau_opt, slope_ratios, optimal_evolution_time_s

vpdr_fit_vs_tau_opt, ramsey_fit_vs_tau_opt, slope_ratios, optimal_evolution_time_s = fit_vs_tau_opt_sensitivity_vs_max_evolution_time(False, NVOrientation.B, "blackman", SEED)
vpdr_fit_vs_tau_opt_hf, ramsey_fit_vs_tau_opt_hf, slope_ratios_hf, optimal_evolution_time_hf_s = fit_vs_tau_opt_sensitivity_vs_max_evolution_time(True, NVOrientation.B, "blackman", SEED)

plt.figure(0, figsize=(2.9, 1.75))
plt.rcParams["font.size"] = 9
plt.rcParams["font.family"] = "arial"
plt.plot(S_TO_US*MAX_EVOLUTION_TIMES_S, vpdr_fit_vs_tau_opt, marker = ".", linestyle="", label="VPDR", color="blue")
plt.plot(S_TO_US*MAX_EVOLUTION_TIMES_S, ramsey_fit_vs_tau_opt, marker = "*", linestyle="", label="DQ", color="blue")
plt.plot(S_TO_US*MAX_EVOLUTION_TIMES_S, vpdr_fit_vs_tau_opt_hf, marker = ".", linestyle="", markerfacecolor="none",label="VPDR HF", color="green")
plt.plot(S_TO_US*MAX_EVOLUTION_TIMES_S, ramsey_fit_vs_tau_opt_hf, marker = "*", linestyle="",  markerfacecolor="none", label="DQ HF", color="green")
plt.xlabel(r"Maximum free evolution time ($\mu$s)")
plt.ylabel("Fit sensitivity vs \noptimal-time sensitivity")
plt.legend()
plt.vlines(optimal_evolution_time_s*S_TO_US, 0,50, linestyle="dashed", color="blue")
plt.vlines(optimal_evolution_time_hf_s*S_TO_US, 0,50, linestyle="dotted", color="green")
plt.legend(loc="upper right", bbox_to_anchor=(.7,1))
plt.gca().yaxis.set_ticks_position("both")
plt.gca().xaxis.set_ticks_position("both")
plt.ylim((1, 40))
plt.xlim(0, 4)
plt.yscale("log")
plt.gca().minorticks_on()
plt.gca().tick_params(direction="in", which="both", width=1.5)
plt.gca().tick_params(direction="in", which="minor", length=2.5)
plt.gca().tick_params(direction="in", which="major", length=5)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.25)
plt.savefig("fit_sensitivity.svg")
plt.show()
