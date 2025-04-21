import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import windows

from bff_paper_figures.extract_experiment_values import get_ideal_rabi_frequencies, get_true_transition_frequencies
from bff_paper_figures.fitting_routines import (
    extract_fit_centers,
    fit_constrained_hyperfine_peaks,
    fit_three_cos_model,
    fit_vs_eigenvalue_error_nt,
)
from bff_paper_figures.imshow_extensions import imshow_with_extents_and_crop
from bff_paper_figures.inner_product_functions import (
    InnerProductSettings,
    double_cosine_inner_product,
    double_cosine_inner_product_vs_ramsey,
    inner_product_sinusoid,
)
from bff_paper_figures.shared_parameters import (
    B_PHI_FIG4,
    B_THETA_FIG4,
    BASE_PATH,
    DETUNING_HZ,
    E_FIELD_VECTOR_V_PER_CM,
    HZ_TO_MHZ,
    MW_DIRECTION,
    RABI_FREQ_BASE_HZ,
    S_TO_US,
    T2STAR_S,
    T_TO_UT,
)
from bff_paper_figures.simulation_helper_functions import sq_cancelled_signal_generator
from bff_simulator.constants import NVaxes_100, exy
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory

#################################################################################################################
# This script generates the components of Figure 4a and 4b. It does not do the final layout of the figure components.
##################################################################################################################

# Parameters describing the example scenario that will be analyzed in Fig. 4a and 4b
B_MAGNITUDE_T = 50e-6
B_FIELD_VECTOR_T = B_MAGNITUDE_T * np.array(
    [np.sin(B_THETA_FIG4) * np.cos(B_PHI_FIG4), np.sin(B_THETA_FIG4) * np.sin(B_PHI_FIG4), np.cos(B_THETA_FIG4)]
)

MW_PULSE_LENGTH_S = np.arange(0, 400e-9, 2.5e-9)
EVOLUTION_TIME_S = np.arange(0, 3e-6, 10e-9)

EXAMPLE_ORIENTATION = 3  # Orientation index that we will analyze in detail

# Frequency range over which the inner products will be plotted
N_RAMSEY_POINTS = 251
RAMSEY_FREQ_RANGE_HZ = np.linspace(0, 8.5e6, N_RAMSEY_POINTS)
RABI_FREQ_RANGE_HZ = np.linspace(60e6, 100e6, 201)

###############################################################################################################
# Generate the VPDR data set at 50 uT and an example field angle; this will be used for all of Figure 4a and 4b

# Set up the simulation and calculate the single-quantum-cancelled VPDR signal
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

exp_param_factory.set_b_field_vector(B_FIELD_VECTOR_T)
sq_cancelled_signal = sq_cancelled_signal_generator(exp_param_factory, nv_ensemble, off_axis_solver)

# Extract the rabi frequency for each orientation for the MW_DIRECTION set in the experiment parameters
rabi_frequencies = get_ideal_rabi_frequencies(exp_param_factory.get_experiment_parameters())

##################################################################################################
# Plot the double inner product (discrete time fourier transform) over rabifreqs, ramseyfreqs range

# Define the settings for the inner product calculation. Note that the boolean use_effective_rabi_frequency
# defines whether the inner product is taken at I(\nu, \omega) (if False) or at I(\sqrt{\nu^2 + \omega^2}, \omega) (if True).
# The boolean subtract_mean determines whether or not the mean value of f(\tau_j, \nu) = \sum_i S(t_i,\tau_j) W(t_i) \cos{2\pi \nu t_i}
# is subtracted before taking the inner product with \cos{\omega \tau}.
inner_product_settings = InnerProductSettings(
    MW_PULSE_LENGTH_S,  # pulse durations sampled
    EVOLUTION_TIME_S,  # evolution times sampled
    rabi_window="boxcar",
    ramsey_window="boxcar",
    subtract_mean=True,
    use_effective_rabi_frequency=False,
)

# Calculate the double inner product for a range of Rabi and Ramsey frequencies (corresponding to the
# frequency domain for the pulse duration and free evolution times respectively)
double_inner_product = np.array(
    [
        [
            double_cosine_inner_product(sq_cancelled_signal, rabi_hz, ramsey_hz, inner_product_settings)
            for ramsey_hz in RAMSEY_FREQ_RANGE_HZ
        ]
        for rabi_hz in RABI_FREQ_RANGE_HZ
    ]
)

# Normalize and plot the double inner product.
imshow_with_extents_and_crop(
    HZ_TO_MHZ * RAMSEY_FREQ_RANGE_HZ,
    HZ_TO_MHZ * RABI_FREQ_RANGE_HZ,
    double_inner_product / abs(min(double_inner_product.flatten())),
    ymin=60,
    ymax=100,
    xmin=0,
    xmax=8.5,
)
plt.colorbar(orientation="horizontal", location="top", shrink=0.6, label="2D inner product amplitude (a.u.)")
plt.plot(
    HZ_TO_MHZ * RAMSEY_FREQ_RANGE_HZ,
    np.sqrt((HZ_TO_MHZ * rabi_frequencies[EXAMPLE_ORIENTATION]) ** 2 + (HZ_TO_MHZ * RAMSEY_FREQ_RANGE_HZ) ** 2),
    color="white",
)
plt.gca().xaxis.set_ticks_position("both")
plt.gca().yaxis.set_ticks_position("both")
plt.gca().minorticks_on()
plt.gca().tick_params(direction="in", which="both", width=1.5)
plt.gca().tick_params(direction="in", which="minor", length=2.5)
plt.gca().tick_params(direction="in", which="major", length=5)
plt.savefig(BASE_PATH + "double_inner_product.svg")
plt.show()

#############################################################
# Plot the associated filter function for boxcar and blackman windows
ideal_cosine_signal = np.cos(
    2 * np.pi * np.sqrt(rabi_frequencies[EXAMPLE_ORIENTATION] ** 2 + max(RAMSEY_FREQ_RANGE_HZ) ** 2) * MW_PULSE_LENGTH_S
)
blackman_window = windows.get_window("blackman", len(MW_PULSE_LENGTH_S))

boxcar_filter_function = np.array(
    [inner_product_sinusoid(np.cos, rabi_hz, MW_PULSE_LENGTH_S, ideal_cosine_signal) for rabi_hz in RABI_FREQ_RANGE_HZ]
)
blackman_filter_function = np.array(
    [
        inner_product_sinusoid(np.cos, rabi_hz, MW_PULSE_LENGTH_S, ideal_cosine_signal * blackman_window)
        for rabi_hz in RABI_FREQ_RANGE_HZ
    ]
)

fig = plt.figure(0, (1, 5))
plt.plot(boxcar_filter_function, HZ_TO_MHZ * RABI_FREQ_RANGE_HZ, label="Boxcar")
plt.ylim((60, 100))
plt.plot(blackman_filter_function, HZ_TO_MHZ * RABI_FREQ_RANGE_HZ, label="Blackman")
plt.gca().yaxis.set_ticks_position("both")
plt.gca().xaxis.set_ticks_position("both")
plt.gca().minorticks_on()
plt.gca().tick_params(direction="in", which="both", width=1.5)
plt.gca().tick_params(direction="in", which="minor", length=2.5)
plt.gca().tick_params(direction="in", which="major", length=5)
plt.legend(loc="lower left")
plt.savefig(BASE_PATH + "filter_functions.svg")
plt.show()

##########################################################
# Calculate and plot the frequency domain fit

# Do an initial fit to the double inner product evaluated at I(\sqrt{\Omega^2 + \omega^2}, \omega) for
# \omega over the full RAMSEY_FREQ_RANGE_HZ and \Omega given by the EXAMPLE_ORIENTATION Rabi frequency,
# evaluated with a Blackman window
inner_product_settings.use_effective_rabi_frequency = True
inner_product_settings.rabi_window = "blackman"

cos_cos_inner_prod_init = double_cosine_inner_product_vs_ramsey(
    sq_cancelled_signal,
    rabi_frequencies[EXAMPLE_ORIENTATION],
    RAMSEY_FREQ_RANGE_HZ,
    inner_product_settings,
)
first_attempt_peaks = extract_fit_centers(
    fit_constrained_hyperfine_peaks(RAMSEY_FREQ_RANGE_HZ, cos_cos_inner_prod_init, T2STAR_S)
)

# Set a smaller range of data to fit based on results of the first fit
# (this yields a better fit by avoiding zero frequency and frequencies with little information)
min_ramsey_freq_hz = max(2 / (np.pi * T2STAR_S), min(first_attempt_peaks) - 2 / (np.pi * T2STAR_S))
max_ramsey_freq_hz = max(first_attempt_peaks) + 2 / (np.pi * T2STAR_S)
ramsey_freq_range_constrained_hz = np.linspace(min_ramsey_freq_hz, max_ramsey_freq_hz, N_RAMSEY_POINTS)

# Fit the double inner product again using the smaller range of data
cos_cos_inner_prod = double_cosine_inner_product_vs_ramsey(
    sq_cancelled_signal, rabi_frequencies[EXAMPLE_ORIENTATION], ramsey_freq_range_constrained_hz, inner_product_settings
)
peakfit_result = fit_constrained_hyperfine_peaks(
    ramsey_freq_range_constrained_hz, cos_cos_inner_prod, T2STAR_S, constrain_same_width=True, allow_zero_peak=True
)

# Plot the frequency domain signal and fit
fig = plt.figure(0, (5, 1))
renormalization = np.abs(min(cos_cos_inner_prod_init))
plt.plot(HZ_TO_MHZ * RAMSEY_FREQ_RANGE_HZ, cos_cos_inner_prod_init / renormalization, label="inner product")
plt.plot(HZ_TO_MHZ * ramsey_freq_range_constrained_hz, peakfit_result.best_fit / renormalization, label="fit")
plt.xlabel("Inner product Ramsey frequency (MHz)")
plt.ylabel("Double cosine inner product (a.u.)")
plt.legend()

plt.title(
    f"B field: {np.array2string(T_TO_UT * np.array(B_FIELD_VECTOR_T), precision=1)} uT, Axis: {np.array2string(np.sqrt(3) * NVaxes_100[EXAMPLE_ORIENTATION], precision=0)}, Rabi: {rabi_frequencies[EXAMPLE_ORIENTATION] * 1e-6:.1f} MHz"  # \nErrors: {np.array2string(errors_nT, precision=2)} nT"
)
plt.xlim(0, 8.5)
plt.gca().xaxis.set_ticks_position("both")
plt.gca().yaxis.set_ticks_position("both")
plt.gca().minorticks_on()
plt.gca().tick_params(direction="in", which="both", width=1.5)
plt.gca().tick_params(direction="in", which="minor", length=2.5)
plt.gca().tick_params(direction="in", which="major", length=5)
plt.gca().xaxis.set_major_locator(MultipleLocator(2))
plt.savefig(BASE_PATH + "example_fit.svg")
plt.show()

# Calculate and print the errors in the transition frequency, expressed as equivalent axial magnetic field
larmor_freqs_all_axes_hz, _ = get_true_transition_frequencies(exp_param_factory.get_experiment_parameters())
errors_nt = fit_vs_eigenvalue_error_nt(peakfit_result, larmor_freqs_all_axes_hz[EXAMPLE_ORIENTATION])
print(f"Peak fit errors: {np.array2string(errors_nt, precision=2)} nT")

######################################################################################
# Calculate and plot the time domain inversion for the same set of simulated VPDR data

# Use the outcome of the frequency domain fit as initial guesses for the time domain
freq_guesses = extract_fit_centers(peakfit_result)

# Use a Blackman window to take the inner product just along the pulse duration dimension
mw_pulse_lengths_s = inner_product_settings.mw_pulse_durations_s
evolution_times_s = inner_product_settings.free_evolution_times_s
rabi_window = windows.get_window(inner_product_settings.rabi_window, len(mw_pulse_lengths_s))

time_domain_ramsey_signal = inner_product_sinusoid(
    np.cos,
    rabi_frequencies[EXAMPLE_ORIENTATION],
    mw_pulse_lengths_s,
    rabi_window * np.transpose(sq_cancelled_signal),
    axis=1,
)

# Fit the time-domain ramsey signal to the three-cosine model
time_domain_result = fit_three_cos_model(
    evolution_times_s, time_domain_ramsey_signal, freq_guesses, T2STAR_S, False, True, True
)

# Plot the time-domain ramsey signal and fit
plt.figure(0, (5, 2.5))
plt.plot(S_TO_US * evolution_times_s, time_domain_ramsey_signal, marker="o", linestyle="", label="simulation")
plt.plot(S_TO_US * evolution_times_s, time_domain_result.best_fit, label="fit")
plt.gca().yaxis.set_ticks_position("both")
plt.gca().xaxis.set_ticks_position("both")
plt.gca().minorticks_on()
plt.gca().tick_params(direction="in", which="both", width=1.5)
plt.gca().tick_params(direction="in", which="minor", length=2.5)
plt.gca().tick_params(direction="in", which="major", length=5)
plt.legend(loc="lower right")
plt.xlabel("Free evolution time (us)")
plt.ylabel("Inner-product-selected Ramsey signal (a.u.)")
plt.ylim((0.02, 0.07))
plt.savefig(BASE_PATH + "time_domain_fit.svg")
plt.show()

# Calculate and print the errors associated with the fit.
errors_nt = fit_vs_eigenvalue_error_nt(time_domain_result, larmor_freqs_all_axes_hz[EXAMPLE_ORIENTATION])
print(f"Time domain fit errors: {np.array2string(errors_nt, precision=2)} nT")
