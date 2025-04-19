import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.optimize import fsolve
from scipy.signal import windows
from bff_simulator.constants import f_h, gammab, exy, NVaxes_100
from bff_paper_figures.imshow_extensions import imshow_with_extents_and_crop
from bff_paper_figures.inner_product_functions import double_cosine_inner_product, inner_product_sinusoid
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.abstract_classes.abstract_ensemble import NVOrientation
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory
from bff_paper_figures.simulate_and_invert_helper_functions import sq_cancelled_signal_generator
from bff_paper_figures.extract_experiment_values import get_ideal_rabi_frequencies, get_true_eigenvalues
from bff_paper_figures.fitting_routines import fit_vs_eigenvalue_error_nT
from bff_paper_figures.imshow_extensions import imshow_with_extents_and_crop
from bff_paper_figures.inner_product_functions import InnerProductSettings, double_cosine_inner_product_vs_ramsey
from bff_paper_figures.fitting_routines import (
    fit_constrained_hyperfine_peaks,
    extract_fit_centers,
    set_up_three_cos_model,
    fit_three_cos_model
)


T_TO_UT = 1e6
HZ_TO_MHZ = 1e-6

# Generate the example data set at 50 uT and an arbitrary field angle

MW_DIRECTION = np.array([0.20539827217056314, 0.11882075246379901, 0.9714387158093318])  # Lily's new "magic angle"
E_FIELD_VECTOR_V_PER_CM = 0 * np.array([1e5, 3e5, 0]) / exy

B_MAGNITUDE_T = 50e-6
B_THETA =  3*np.pi/8
B_PHI = 13*np.pi/16

RABI_FREQ_BASE_HZ = 100e6
DETUNING_HZ = 0e6
SECOND_PULSE_PHASE = 0
MW_PULSE_LENGTH_S = np.arange(0, 400e-9, 2.5e-9)  
EVOLUTION_TIME_S = np.arange(0, 3e-6, 10e-9)  
T2STAR_S = 2e-6
N_RAMSEY_POINTS = 251
RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ = np.linspace(0, 8.5e6, N_RAMSEY_POINTS)

RAD_TO_DEGREE = 360/(2*np.pi)
PHI_RANGE_HALF_HYPERFINE = np.linspace(1.4607, 3.25133, 301)

EXAMPLE_ORIENTATION = 3 # Orientation index that we will analyze in detail

BASE_PATH = "/Users/lilianchildress/Documents/GitHub/sbq-dq-simulator/bff_paper_figures/data/"

# Set up the simulation
nv_ensemble = HomogeneousEnsemble()
nv_ensemble.efield_splitting_hz = np.linalg.norm(E_FIELD_VECTOR_V_PER_CM) * exy
nv_ensemble.t2_star_s = T2STAR_S
nv_ensemble.add_full_diamond_populations()
nv_ensemble.mw_direction = MW_DIRECTION

off_axis_solver = LiouvillianSolver()

exp_param_factory = OffAxisFieldExperimentParametersFactory()
exp_param_factory.set_base_rabi_frequency(RABI_FREQ_BASE_HZ)
exp_param_factory.set_mw_direction(MW_DIRECTION)
exp_param_factory.set_e_field_v_per_m(E_FIELD_VECTOR_V_PER_CM)
exp_param_factory.set_detuning(DETUNING_HZ)
exp_param_factory.set_mw_pulse_lengths(MW_PULSE_LENGTH_S)
exp_param_factory.set_evolution_times(EVOLUTION_TIME_S)
 
b_field_vector_t = B_MAGNITUDE_T*np.array([np.sin(B_THETA)*np.cos(B_PHI), np.sin(B_THETA)*np.sin(B_PHI), np.cos(B_THETA)])
exp_param_factory.set_b_field_vector(b_field_vector_t)
sq_cancelled_signal = sq_cancelled_signal_generator(exp_param_factory, nv_ensemble, off_axis_solver)

# Extract the rabi frequency for each orientation for the MW_DIRECTION set in the experiment parameters
rabi_frequencies = get_ideal_rabi_frequencies(exp_param_factory.get_experiment_parameters())

###################################################################
# Plot the double inner product (discrete time fourier transform) over rabifreqs, ramseyfreqs range
rabifreqs = np.linspace(60e6, 100e6, 201)
ramseyfreqs = np.linspace(0, 8.5e6, 251)

inner_product_settings = InnerProductSettings(
    MW_PULSE_LENGTH_S,
    EVOLUTION_TIME_S,
    rabi_window="boxcar",
    ramsey_window="boxcar",
    subtract_mean=True,
    use_effective_rabi_frequency=False,
)

dtft = np.array([[double_cosine_inner_product(sq_cancelled_signal, rabi_hz, ramsey_hz, inner_product_settings) for ramsey_hz in ramseyfreqs] for rabi_hz in rabifreqs])
imshow_with_extents_and_crop(HZ_TO_MHZ*ramseyfreqs,HZ_TO_MHZ*rabifreqs, -dtft/min(dtft.flatten()),ymin=60, ymax=100, xmin=0, xmax=8.5)
plt.colorbar(orientation="horizontal",location="top", shrink=0.6, label="2D inner product amplitude (a.u.)")
plt.plot(HZ_TO_MHZ*ramseyfreqs, np.sqrt((HZ_TO_MHZ*rabi_frequencies[EXAMPLE_ORIENTATION])**2 + (HZ_TO_MHZ*ramseyfreqs)**2), color="white")
plt.gca().xaxis.set_ticks_position("both")
plt.gca().yaxis.set_ticks_position("both")
plt.gca().minorticks_on()
plt.gca().tick_params(direction="in", which = "both", width=1.5)
plt.gca().tick_params(direction="in", which = "minor", length=2.5)
plt.gca().tick_params(direction="in", which = "major", length=5)
plt.savefig(BASE_PATH+"double_inner_product.svg")
plt.show()

#############################################################
# Plot the associated filter function for boxcar and blackman
ideal_signal = np.cos(2*np.pi*np.sqrt(rabi_frequencies[3]**2 +max(ramseyfreqs)**2)*MW_PULSE_LENGTH_S)
blackman_window = windows.get_window("blackman", len(MW_PULSE_LENGTH_S))

boxcar_filter_function=np.array([inner_product_sinusoid(np.cos, rabi_hz, MW_PULSE_LENGTH_S, ideal_signal) for rabi_hz in rabifreqs])
blackman_filter_function=np.array([inner_product_sinusoid(np.cos, rabi_hz, MW_PULSE_LENGTH_S, ideal_signal*blackman_window) for rabi_hz in rabifreqs])

fig = plt.figure(0,(1,5))
plt.plot(boxcar_filter_function,HZ_TO_MHZ*rabifreqs, label="Boxcar")
plt.ylim((60, 100))
plt.plot(blackman_filter_function,HZ_TO_MHZ*rabifreqs, label="Blackman")
plt.gca().yaxis.set_ticks_position("both")
plt.gca().xaxis.set_ticks_position("both")
plt.gca().minorticks_on()
plt.gca().tick_params(direction="in", which = "both", width=1.5)
plt.gca().tick_params(direction="in", which = "minor", length=2.5)
plt.gca().tick_params(direction="in", which = "major", length=5)
plt.legend(loc="lower left")
plt.savefig(BASE_PATH+"filter_functions.svg")
plt.show()

##########################################################
# Calculate and plot the frequency domain fit

# Do an initial fit (first fit is to set the fitting range)
inner_product_settings.use_effective_rabi_frequency=True
inner_product_settings.rabi_window="blackman"

cos_cos_inner_prod_init = double_cosine_inner_product_vs_ramsey(
    sq_cancelled_signal, rabi_frequencies[EXAMPLE_ORIENTATION], RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ, inner_product_settings
)
first_attempt_peaks = extract_fit_centers(
    fit_constrained_hyperfine_peaks(RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ, cos_cos_inner_prod_init, T2STAR_S)
)

# Set the smaller fitting range (yields a better fit by avoiding zero frequency and frequencies with little information)
min_ramsey_freq_hz = max(2 / (np.pi * T2STAR_S), min(first_attempt_peaks) - 2 / (np.pi * T2STAR_S))
max_ramsey_freq_hz = max(first_attempt_peaks) + 2 / (np.pi * T2STAR_S)
ramsey_freq_range_constrained_hz = np.linspace(min_ramsey_freq_hz, max_ramsey_freq_hz, N_RAMSEY_POINTS)
cos_cos_inner_prod = double_cosine_inner_product_vs_ramsey(
    sq_cancelled_signal, rabi_frequencies[EXAMPLE_ORIENTATION], ramsey_freq_range_constrained_hz, inner_product_settings
)
peakfit_result = fit_constrained_hyperfine_peaks(
    ramsey_freq_range_constrained_hz, cos_cos_inner_prod, T2STAR_S, constrain_same_width=True,allow_zero_peak=True
)

# Plot the frequency domain signal and fit
fig = plt.figure(0,(5,1))
renormalization =np.abs(min(cos_cos_inner_prod_init))
plt.plot(HZ_TO_MHZ * RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ, cos_cos_inner_prod_init/renormalization, label="inner product")
plt.plot(HZ_TO_MHZ * ramsey_freq_range_constrained_hz, peakfit_result.best_fit/renormalization, label="fit")
plt.xlabel("Inner product Ramsey frequency (MHz)")
plt.ylabel("Double cosine inner product (a.u.)")
plt.legend()

plt.title(
    f"B field: {np.array2string(T_TO_UT * np.array(b_field_vector_t), precision=1)} uT, Axis: {np.array2string(np.sqrt(3) * NVaxes_100[EXAMPLE_ORIENTATION], precision=0)}, Rabi: {rabi_frequencies[EXAMPLE_ORIENTATION] * 1e-6:.1f} MHz" #\nErrors: {np.array2string(errors_nT, precision=2)} nT"
)
plt.xlim(0, 8.5)
plt.gca().xaxis.set_ticks_position("both")
plt.gca().yaxis.set_ticks_position("both")
plt.gca().minorticks_on()
plt.gca().tick_params(direction="in", which = "both", width=1.5)
plt.gca().tick_params(direction="in", which = "minor", length=2.5)
plt.gca().tick_params(direction="in", which = "major", length=5)
plt.gca().xaxis.set_major_locator(MultipleLocator(2))
plt.savefig(BASE_PATH+"example_fit.svg")
plt.show()

larmor_freqs_all_axes_hz, _ = get_true_eigenvalues(exp_param_factory.get_experiment_parameters())
errors_nT = fit_vs_eigenvalue_error_nT(peakfit_result, larmor_freqs_all_axes_hz[EXAMPLE_ORIENTATION])
print(f"Peak fit errors: {np.array2string(errors_nT, precision=2)}")

########################################################
# Calculate and plot the time domain inversion for the same data set

freq_guesses = extract_fit_centers(peakfit_result)
off_axis_experiment_parameters=exp_param_factory.get_experiment_parameters()
mw_pulse_length_s = off_axis_experiment_parameters.mw_pulse_length_s
evolution_time_s = off_axis_experiment_parameters.evolution_time_s
rabi_window = windows.get_window(inner_product_settings.rabi_window, len(mw_pulse_length_s))

time_domain_ramsey_signal = inner_product_sinusoid(np.cos, rabi_frequencies[EXAMPLE_ORIENTATION], mw_pulse_length_s, rabi_window*np.transpose(sq_cancelled_signal),axis=1)
plt.figure(0,(5,2.5))
plt.plot(1e6*evolution_time_s, time_domain_ramsey_signal, marker="o", linestyle="", label="simulation")

time_domain_result = fit_three_cos_model(evolution_time_s, time_domain_ramsey_signal, freq_guesses, T2STAR_S, False, True,True)
plt.plot(1e6*evolution_time_s, time_domain_result.best_fit, label="fit")
plt.gca().yaxis.set_ticks_position("both")
plt.gca().xaxis.set_ticks_position("both")
plt.gca().minorticks_on()
plt.gca().tick_params(direction="in", which = "both", width=1.5)
plt.gca().tick_params(direction="in", which = "minor", length=2.5)
plt.gca().tick_params(direction="in", which = "major", length=5)
plt.legend(loc="lower right")
plt.xlabel("Free evolution time (us)")
plt.ylabel("Inner-product-selected Ramsey signal (a.u.)")
plt.ylim((0.02, 0.07))
plt.savefig(BASE_PATH + "time_domain_fit.svg")
plt.show()

errors_nT = fit_vs_eigenvalue_error_nT(time_domain_result, larmor_freqs_all_axes_hz[EXAMPLE_ORIENTATION])
print(f"Time domain fit errors: {np.array2string(errors_nT, precision=2)}")
