
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import windows

from bff_paper_figures.extract_experiment_values import  get_true_transition_frequencies
from bff_sensitivity_calculations.sensitivity_functions import get_vpdr_slope_dbz_dsignal_at_tau_opt, get_ramsey_slope_dbz_dsignal_at_tau_opt, get_optimal_evolution_time_hf_agnostic_s, vpdr_and_ramsey_sensitivity_at_tau_opt
from bff_simulator.abstract_classes.abstract_ensemble import NVOrientation, NV14HyperfineField
from bff_simulator.constants import NVaxes_100, f_h
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.vector_manipulation import perpendicular_projection
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory
from bff_paper_figures.shared_parameters import MW_DIRECTION, E_FIELD_VECTOR_V_PER_CM, RABI_FREQ_BASE_HZ, DETUNING_HZ, T2STAR_S, B_PHI_FIG4, B_THETA_FIG4

B_MAGNITUDE_T = 50e-6
DELTA_B_T = 1e-9
SEED = 31
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
EVOLUTION_TIME_S = np.arange(0, 3e-6, 20e-9)  # p.linspace(0, 15e-6, 801)
S_TO_NS = 1e9

RABI_FREQ_BASE_HZ = 100e6
MAX_PULSE_DURATIONS = np.arange(50e-9, 800e-9, 25e-9)

def tau_opt_sensitivity_ratios_vs_mw_pulse_max(use_hyperfine=False, orientation=NVOrientation.A, rabi_window_name="boxcar", seed=SEED):
    
    rabi_freq_hz = IDEAL_RABI_FREQUENCIES[orientation]

    # Set up the experiment
    exp_param_factory = OffAxisFieldExperimentParametersFactory()
    exp_param_factory.set_base_rabi_frequency(RABI_FREQ_BASE_HZ)
    exp_param_factory.set_mw_direction(MW_DIRECTION)
    exp_param_factory.set_e_field_v_per_m(E_FIELD_VECTOR_V_PER_CM)
    exp_param_factory.set_detuning(DETUNING_HZ)
    exp_param_factory.set_mw_pulse_lengths(MW_PULSE_LENGTH_S)
    exp_param_factory.set_evolution_times(EVOLUTION_TIME_S)

    # Extract the ground truth larmor frequencies for the expected magnetic field
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

    # Extract the true mi = 0 larmor precession frequency and use it to find the theoretical optimal free precession time
    double_larmor_mi0_hz = larmor_freqs_all_axes_hz[orientation][INDEX_FOR_MI0] # double quantum larmor frequency
    optimal_evolution_time_s = get_optimal_evolution_time_hf_agnostic_s(double_larmor_mi0_hz, T2STAR_S, use_hyperfine, f_h, EVOLUTION_TIME_S, do_plot=False)
    print(f"Nominal optimal evolution time: {S_TO_NS*optimal_evolution_time_s:.02f} ns")
    rng = np.random.default_rng(seed)

    sensitivity_ratios = []
    for max_pulse_duration in MAX_PULSE_DURATIONS:

        # Set the new MW pulse lengths to be employed
        mw_pulse_length_s = np.arange(0, max_pulse_duration, MW_PULSE_LENGTH_S[1]-MW_PULSE_LENGTH_S[0])
        exp_param_factory.set_mw_pulse_lengths(mw_pulse_length_s)    

        # Find the signal response to magnetic field at the theoretically optimal time, extracting 1/(df/dBz) where Bz is the axial field
        # (commented out: picking a slightly better time if it exists; this doesn't make a noticeable difference)
        # min_dbz_dsignal_vpdr, min_dbz_dsignal_ramsey, tau_opt_vpdr_s, tau_opt_ramsey_s = get_vdpr_and_ramsey_min_slopes_dbz_dsignal(exp_param_factory, nv_ensemble, off_axis_solver, rabi_window_name, optimal_evolution_time_s, t_pi_s, B_VECTOR_T, B_VECTOR_T + DELTA_B_VECTOR_T, orientation, INDEX_FOR_MI0, do_plots = False)
        min_dbz_dsignal_vpdr = np.abs(get_vpdr_slope_dbz_dsignal_at_tau_opt(optimal_evolution_time_s, exp_param_factory, nv_ensemble, off_axis_solver, rabi_window_name, B_VECTOR_T, B_VECTOR_T + DELTA_B_VECTOR_T, orientation, INDEX_FOR_MI0))
        min_dbz_dsignal_ramsey = np.abs(get_ramsey_slope_dbz_dsignal_at_tau_opt(optimal_evolution_time_s, exp_param_factory, nv_ensemble, off_axis_solver, t_pi_s,  B_VECTOR_T, B_VECTOR_T + DELTA_B_VECTOR_T,orientation, INDEX_FOR_MI0))
    
        # Find the noise in the vpdr and ramsey signals at their optimal taus due to injection of Gaussian noise
        vpdr_sensitivity, ramsey_sensitivity = vpdr_and_ramsey_sensitivity_at_tau_opt(exp_param_factory.get_experiment_parameters(), orientation, rabi_window_name, min_dbz_dsignal_vpdr, min_dbz_dsignal_ramsey,rng, N_SAMPLES, SIG_STD_DEV)
        sensitivity_ratios.append(vpdr_sensitivity/ramsey_sensitivity)
    return sensitivity_ratios

orientation = NVOrientation.B
print("Working on blackman...")
blackman_sensitivity_ratios = tau_opt_sensitivity_ratios_vs_mw_pulse_max(False, orientation, "blackman", SEED)
print("Working on blackman with hyperfine...")
blackman_sensitivity_ratios_hf = tau_opt_sensitivity_ratios_vs_mw_pulse_max(True, orientation, "blackman", SEED+1)
print("Working on boxcar...")
boxcar_sensitivity_ratios = tau_opt_sensitivity_ratios_vs_mw_pulse_max(False, orientation, "boxcar", SEED+2)
print("Working on boxcar with hyperfine..")
boxcar_sensitivity_ratios_hf = tau_opt_sensitivity_ratios_vs_mw_pulse_max(True, orientation, "boxcar", SEED+3)

# Calculate factor in the expected hard-pulse limit associated with a Blackman window
rabi_window = windows.get_window("blackman", 1000)
np.mean(rabi_window)
blackman_factor = np.sqrt(np.mean(rabi_window**2))/np.mean(rabi_window)

# Generate the plot
plt.figure(0, figsize=(2.9, 1.75))
plt.rcParams["font.size"] = 9
plt.rcParams["font.family"] = "arial"
plt.plot(MAX_PULSE_DURATIONS *S_TO_NS, blackman_sensitivity_ratios, marker=".", linestyle="", label= "Blackman", color="blue")
plt.plot(MAX_PULSE_DURATIONS *S_TO_NS, blackman_sensitivity_ratios_hf, marker="x", linestyle="", label= "Blackman HF", color="blue")
plt.hlines(blackman_factor*2*np.sqrt(2), min(MAX_PULSE_DURATIONS*S_TO_NS), max(MAX_PULSE_DURATIONS*S_TO_NS), linestyle="dashed", label=r"2$\sqrt{2}W_{\text{rms}}/\bar{W}$")
plt.plot(MAX_PULSE_DURATIONS *S_TO_NS, boxcar_sensitivity_ratios, marker=".", linestyle="", color="red", label = "Boxcar")
plt.plot(MAX_PULSE_DURATIONS *S_TO_NS, boxcar_sensitivity_ratios_hf, marker="x", linestyle="", color="red", label = "Boxcar HF")
plt.hlines(2*np.sqrt(2), min(MAX_PULSE_DURATIONS*S_TO_NS), max(MAX_PULSE_DURATIONS*S_TO_NS), color="red", linestyle="dotted", label = r"2$\sqrt{2}$")
plt.legend(loc="lower center", ncol=1, bbox_to_anchor=(1.5,.5))
plt.xlabel("Maximum pulse duration (ns)")
plt.ylabel("VPDR vs DQ optimal sensitivity")
#plt.ylim((2,6))
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