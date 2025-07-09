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
from bff_paper_figures.fitting_routines import decaying_cosine, offset
from bff_simulator.abstract_classes.abstract_ensemble import NVOrientation, NV14HyperfineField
from bff_simulator.constants import NVaxes_100, gammab
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
MW_RABI_PERIOD_DIVISION = 10
EVOLUTION_STEPS_UNTIL_OPTIMAL = 10
INDEX_FOR_MI_1 = 0
MW_STEP_S = 1/(MW_RABI_PERIOD_DIVISION*IDEAL_RABI_FREQUENCIES[NVOrientation.A])
MW_PULSE_LENGTH_S =np.arange(0, 800e-9, MW_STEP_S)  #np.arange(0, 800e-9, 2.5e-9)  # np.linspace(0, 0.5e-6, 1001)
# EVOLUTION_TIME_S = np.arange(0, 3e-6, 20e-9)  # p.linspace(0, 15e-6, 801)

SEED = 294813022
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

# Do everything that won't change if we change the MW pulse durations
def setup_simulation():
    nv_ensemble = HomogeneousEnsemble()
    nv_ensemble.t2_star_s = T2STAR_S
    nv_ensemble.add_nv_single_species(NVOrientation.A, NV14HyperfineField.N14_plus)

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

    # Determine the optimal time for measurement (in theory) and make sure it appears in our evolution times
    larmor_actual_hz = larmor_freqs_all_axes_hz[NVOrientation.A][INDEX_FOR_MI_1] # double quantum larmor frequency
    optimal_evolution_time_s = get_optimal_evolution_time_s(larmor_actual_hz, T2STAR_S)
    evolution_times_s = np.arange(0, 2*T2STAR_S, optimal_evolution_time_s/(EVOLUTION_STEPS_UNTIL_OPTIMAL))
    evolution_times_fine_s = np.arange(0, 2*T2STAR_S, optimal_evolution_time_s/(EVOLUTION_STEPS_UNTIL_OPTIMAL*10))
    plt.plot(evolution_times_fine_s, np.abs(slope(evolution_times_fine_s, larmor_actual_hz, T2STAR_S)))
    plt.vlines(optimal_evolution_time_s, 0, max(slope(evolution_times_fine_s, larmor_actual_hz, T2STAR_S)), color="red")
    plt.show()
    exp_param_factory.set_evolution_times(evolution_times_s)

    return nv_ensemble, exp_param_factory, off_axis_solver

def get_signal_slopes(exp_param_factory:OffAxisFieldExperimentParametersFactory, nv_ensemble, off_axis_solver, rabi_window_name, b_vector_t=B_VECTOR_T, b_plus_db_vector_t=B_PLUS_DB_VECTOR_T):
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
    orientation = NVOrientation.A
    mw_pulse_length_s = exp_param_factory.get_experiment_parameters().mw_pulse_length_s
    rabi_window = windows.get_window(rabi_window_name, len(mw_pulse_length_s))
    time_domain_ramsey_signal = inner_product_sinusoid(
        np.cos,
        IDEAL_RABI_FREQUENCIES[orientation],
        mw_pulse_length_s,
        rabi_window * np.transpose(sq_cancelled_signal),
        axis=1,
    )
    time_domain_ramsey_signal_delta_b = inner_product_sinusoid(
        np.cos,
        IDEAL_RABI_FREQUENCIES[orientation],
        mw_pulse_length_s,
        rabi_window * np.transpose(sq_cancelled_signal_delta_b),
        axis=1,
    )

    # Calculate how much the axial field changes per unit change in the inner-producted signal (the "slope" with respect to axial field)
    slope_dbz_dsignal_inner_product = (bz_values_all_axes_delta_b_t[NVOrientation.A][INDEX_FOR_MI_1]/2 - bz_values_all_axes_t[NVOrientation.A][INDEX_FOR_MI_1]/2)/(
    (time_domain_ramsey_signal_delta_b[EVOLUTION_STEPS_UNTIL_OPTIMAL] - time_domain_ramsey_signal[EVOLUTION_STEPS_UNTIL_OPTIMAL]))

    # Calculate how much the axial field changes per unit change in the single-point signal (the "slope" with respect to axial field)
    slope_dbz_dsignal_dq = (bz_values_all_axes_delta_b_t[NVOrientation.A][INDEX_FOR_MI_1]/2 - bz_values_all_axes_t[NVOrientation.A][INDEX_FOR_MI_1]/2)/(
    (sq_cancelled_signal[int(MW_RABI_PERIOD_DIVISION/2)][EVOLUTION_STEPS_UNTIL_OPTIMAL] - sq_cancelled_signal_delta_b[int(MW_RABI_PERIOD_DIVISION/2)][EVOLUTION_STEPS_UNTIL_OPTIMAL]))

    return slope_dbz_dsignal_dq, slope_dbz_dsignal_inner_product, sq_cancelled_signal, time_domain_ramsey_signal


# Monte Carlo simulation of sensitivity ratio for VPDR vs DQ. Calculates the standard
# deviation of the magnetic field readings in the presence of Gaussian readout noise.
# Compares fluctuations in the average value of a DQ signal [averaged over as many samples (at the optimal MW time)
# as there are pulse durations] to fluctuations in a VPDR signal analyzed with an inner product on 
# the Rabi dimension. Both signals are evaluated for a single NV orientation, single hyperfine line
# at the optimal free evolution time for sensitivity. 
def compare_single_point_sensitivity(rabi_window_name, exp_param_factory:OffAxisFieldExperimentParametersFactory, off_axis_solver, nv_ensemble, rng, orientation=NVOrientation.A):

    # find the slope in signal at the optimal time to measure
    slope_dbz_dsignal_dq, slope_dbz_dsignal_inner_product, sq_cancelled_signal, time_domain_ramsey_signal = get_signal_slopes(exp_param_factory, nv_ensemble, off_axis_solver, rabi_window_name, B_VECTOR_T, B_PLUS_DB_VECTOR_T)
    
    mw_pulse_length_s = exp_param_factory.get_experiment_parameters().mw_pulse_length_s
    rabi_window = windows.get_window(rabi_window_name, len(mw_pulse_length_s))

    # Now add in measurement noise, and see how much that changes the signals
    noisy_delta_b_inner_product_t = []
    noisy_delta_b_dq_t = []
    for _ in range(N_SAMPLES):
        # Calculate the noisy inner-producted signal
        noisy_sq_cancelled_signal = sq_cancelled_signal + rng.normal(0, SIG_STD_DEV, size=sq_cancelled_signal.shape)
        noisy_time_domain_ramsey_signal = inner_product_sinusoid(
            np.cos,
            IDEAL_RABI_FREQUENCIES[orientation],
            mw_pulse_length_s,
            rabi_window * np.transpose(noisy_sq_cancelled_signal),
            axis=1,
        )
        # And convert it to a noisy field excursion using the slope
        noisy_delta_b_inner_product_t.append((noisy_time_domain_ramsey_signal[EVOLUTION_STEPS_UNTIL_OPTIMAL] - time_domain_ramsey_signal[EVOLUTION_STEPS_UNTIL_OPTIMAL])*slope_dbz_dsignal_inner_product)

        # do the same thing for the standard dq signal (averaging over the same number of measurements as there are steps in mw_pulse_length_s)
        noisy_delta_b_dq_t.append(np.mean(rng.normal(0, SIG_STD_DEV, size=len(mw_pulse_length_s)) ) * slope_dbz_dsignal_dq)
    
    # Calculate the ratio of the noise in the magnetic field, which should be (approximately) the ratio of the sensitivities, assuming that the time is 
    # dominated by measurement time.
    return np.std(np.array(noisy_delta_b_inner_product_t))/np.std(np.array(noisy_delta_b_dq_t))

s_to_ns = 1e9  
nv_ensemble, exp_param_factory, off_axis_solver = setup_simulation()
  
max_mw_duration_range = np.arange(50e-9, 800e-9, 25e-9)
blackman_sensitivity_ratios = []
boxcar_sensitivity_ratios = []
rng = np.random.default_rng(SEED)
for max_mw_duration in max_mw_duration_range: 
    print(f"Maximum MW duration: {max_mw_duration*s_to_ns} ns")
    mw_pulse_length_s =np.arange(0, max_mw_duration, MW_STEP_S) 
    exp_param_factory.set_mw_pulse_lengths(mw_pulse_length_s)

    blackman_sensitivity_ratios.append(compare_single_point_sensitivity( "blackman", exp_param_factory, off_axis_solver, nv_ensemble,rng))
    boxcar_sensitivity_ratios.append(compare_single_point_sensitivity( "boxcar", exp_param_factory, off_axis_solver, nv_ensemble, rng))

np.savetxt("max_mw_duration_range.txt", max_mw_duration_range)
np.savetxt("blackman_sensitivity_ratios.txt", blackman_sensitivity_ratios)
np.savetxt("boxcar_sensitivity_ratios.txt", boxcar_sensitivity_ratios)

rabi_window = windows.get_window("blackman", 1000)
np.mean(rabi_window)
blackman_factor = np.sqrt(np.mean(rabi_window**2))/np.mean(rabi_window)

plt.figure(0, figsize=(3.4, 2.5))
plt.rcParams["font.size"] = 9
plt.rcParams["font.family"] = "arial"
plt.plot(max_mw_duration_range *s_to_ns, blackman_sensitivity_ratios, marker="o", linestyle="", label= "Blackman")
plt.hlines(blackman_factor*2*np.sqrt(2), min(max_mw_duration_range*s_to_ns), max(max_mw_duration_range*s_to_ns), linestyle="dashed", label=r"2$\sqrt{2}W_{\text{rms}}/\bar{W}$")
plt.plot(max_mw_duration_range *s_to_ns, boxcar_sensitivity_ratios, marker="x", linestyle="", color="red", label = "Boxcar")

plt.hlines(2*np.sqrt(2), min(max_mw_duration_range*s_to_ns), max(max_mw_duration_range*s_to_ns), color="red", linestyle="dotted", label = r"2$\sqrt{2}$")
plt.legend(loc="lower center", ncol=2)
plt.xlabel("Maximum pulse duration (ns)")
plt.ylabel("VPDR vs DQ optimal sensitivity")
plt.ylim((1.5,6))
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
    