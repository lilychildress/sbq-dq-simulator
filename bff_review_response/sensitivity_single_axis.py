from math import floor

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from scipy.signal import windows
from bff_paper_figures.inner_product_functions import (
    InnerProductSettings,
    inner_product_sinusoid,
)

from bff_paper_figures.extract_experiment_values import get_ideal_rabi_frequencies, get_true_transition_frequencies
from bff_paper_figures.simulation_helper_functions import angles_already_evaluated, sq_cancelled_signal_generator
from bff_simulator.abstract_classes.abstract_ensemble import NVOrientation, NV14HyperfineField
from bff_simulator.constants import NVaxes_100, exy
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.vector_manipulation import perpendicular_projection, transform_from_crystal_to_nv_coords
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory
from bff_paper_figures.shared_parameters import T_TO_UT, MW_DIRECTION, E_FIELD_VECTOR_V_PER_CM, RABI_FREQ_BASE_HZ, DETUNING_HZ, RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ, T2STAR_S, PEAK_INDEX, B_PHI_FIG4, B_THETA_FIG4

B_MAGNITUDE_T = 50e-6
DELTA_B_T = 1e-8

B_VECTOR_T = B_MAGNITUDE_T * np.array(
    [np.sin(B_THETA_FIG4) * np.cos(B_PHI_FIG4), np.sin(B_THETA_FIG4) * np.sin(B_PHI_FIG4), np.cos(B_THETA_FIG4)]
)

B_PLUS_DB_VECTOR_T = (B_MAGNITUDE_T + DELTA_B_T) * np.array(
    [np.sin(B_THETA_FIG4) * np.cos(B_PHI_FIG4), np.sin(B_THETA_FIG4) * np.sin(B_PHI_FIG4), np.cos(B_THETA_FIG4)]
)

IDEAL_RABI_FREQUENCIES= np.array([RABI_FREQ_BASE_HZ * perpendicular_projection(MW_DIRECTION, NVaxis) for NVaxis in NVaxes_100])
MW_RABI_PERIOD_DIVISION = 20
EVOLUTION_STEPS_UNTIL_OPTIMAL = 10
INDEX_FOR_MI_0 = 1
MW_STEP_S = 1/(MW_RABI_PERIOD_DIVISION*IDEAL_RABI_FREQUENCIES[NVOrientation.A])
MW_PULSE_LENGTH_S =np.arange(0, 800e-9, MW_STEP_S)  #np.arange(0, 800e-9, 2.5e-9)  # np.linspace(0, 0.5e-6, 1001)
# EVOLUTION_TIME_S = np.arange(0, 3e-6, 20e-9)  # p.linspace(0, 15e-6, 801)

SEED = 294813022
SIG_STD_DEV = 1e-4
N_SAMPLES = 1000


def optimal_time(tau_s, larmor_actual_hz, t2star_s):
    w = 2*np.pi*larmor_actual_hz/2
    t = tau_s
    t2s = t2star_s
    return 2 * t2s * t * w * np.cos(2 * t * w) + (t2s - 2* t)*np.sin(2 * t * w)

def get_optimal_evolution_time_s(larmor_actual_hz, t2star_s):
    return fsolve(optimal_time, [t2star_s/2], (larmor_actual_hz, t2star_s))[0]

def compare_single_point_sensitivity(max_mw_pulse_s, rabi_window_name):
    mw_pulse_length_s =np.arange(0, max_mw_pulse_s, MW_STEP_S) 
    nv_ensemble = HomogeneousEnsemble()
    nv_ensemble.t2_star_s = T2STAR_S
    nv_ensemble.add_nv_single_species(NVOrientation.A, NV14HyperfineField.N14_0)

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

    # Determine the optimal time for measurement (in theory) and make sure it appears in our evolution times
    optimal_evolution_time_s = get_optimal_evolution_time_s(larmor_freqs_all_axes_hz[NVOrientation.A][INDEX_FOR_MI_0], T2STAR_S)
    evolution_times_s = np.arange(0, 3e-6, optimal_evolution_time_s/EVOLUTION_STEPS_UNTIL_OPTIMAL)
    exp_param_factory.set_evolution_times(evolution_times_s)


    # Calculate the SQ cancelled signal at the magnetic field of interest
    exp_param_factory.set_b_field_vector(B_VECTOR_T)
    sq_cancelled_signal = sq_cancelled_signal_generator(exp_param_factory, nv_ensemble, off_axis_solver)

    # Now do the same thing at a slightly different magnetic field so that we can calculate the "slope" with
    # respect to magnetic field at the optimal evolution time for measurement
    exp_param_factory.set_b_field_vector(B_PLUS_DB_VECTOR_T)
    sq_cancelled_signal_delta_b = sq_cancelled_signal_generator(exp_param_factory, nv_ensemble, off_axis_solver)
    _, bz_values_all_axes_delta_b_t = get_true_transition_frequencies(
        exp_param_factory.get_experiment_parameters()
    )

    # Pick out the inner-product signals at the expected magnetic field and the slightly shifted one
    orientation = NVOrientation.A
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
    slope_dbz_dsignal_inner_product = (bz_values_all_axes_delta_b_t[NVOrientation.A][INDEX_FOR_MI_0]/2 - bz_values_all_axes_t[NVOrientation.A][INDEX_FOR_MI_0]/2)/(
    (time_domain_ramsey_signal_delta_b[EVOLUTION_STEPS_UNTIL_OPTIMAL] - time_domain_ramsey_signal[EVOLUTION_STEPS_UNTIL_OPTIMAL]))
    # Calculate how much the axial field changes per unit change in the single-point signal (the "slope" with respect to axial field)
    slope_dbz_dsignal_dq = (bz_values_all_axes_delta_b_t[NVOrientation.A][INDEX_FOR_MI_0]/2 - bz_values_all_axes_t[NVOrientation.A][INDEX_FOR_MI_0]/2)/(
    (sq_cancelled_signal[int(MW_RABI_PERIOD_DIVISION/2)][EVOLUTION_STEPS_UNTIL_OPTIMAL] - sq_cancelled_signal_delta_b[int(MW_RABI_PERIOD_DIVISION/2)][EVOLUTION_STEPS_UNTIL_OPTIMAL]))

    # Now add in measurement noise, and see how much that changes the signals
    rng = np.random.default_rng(SEED)

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

max_mw_duration_range = np.arange(50e-9, 800e-9, 25e-9)
blackman_sensitivity_ratios = []
boxcar_sensitivity_ratios = []
for max_mw_duration in max_mw_duration_range: 
    blackman_sensitivity_ratios.append(compare_single_point_sensitivity(max_mw_duration, "blackman"))
    boxcar_sensitivity_ratios.append(compare_single_point_sensitivity(max_mw_duration, "boxcar"))

np.savetxt("max_mw_duration_range.txt", max_mw_duration_range)
np.savetxt("blackman_sensitivity_ratios.txt", blackman_sensitivity_ratios)
np.savetxt("boxcar_sensitivity_ratios.txt", boxcar_sensitivity_ratios)

rabi_window = windows.get_window("blackman", 1000)
np.mean(rabi_window)
blackman_factor = np.sqrt(np.mean(rabi_window**2))/np.mean(rabi_window)

s_to_ns = 1e9    
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
plt.show()
    