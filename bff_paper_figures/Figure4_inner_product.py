import numpy as np
from matplotlib import pyplot as plt
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
from bff_paper_figures.extract_experiment_values import get_bare_rabi_frequencies, get_true_eigenvalues
from bff_paper_figures.fitting_routines import fit_vs_eigenvalue_error_nT
from bff_paper_figures.imshow_extensions import imshow_with_extents_and_crop
from bff_paper_figures.inner_product_functions import InnerProductSettings, double_cosine_inner_product_vs_ramsey
from bff_paper_figures.fitting_routines import (
    fit_constrained_hyperfine_peaks,
    extract_fit_centers,
)


T_TO_UT = 1e6
HZ_TO_MHZ = 1e-6

# Generate the example data set at 50 uT and a fairly random field angle

MW_DIRECTION = np.array([0.20539827217056314, 0.11882075246379901, 0.9714387158093318])  # Lily's new "magic angle"
E_FIELD_VECTOR_V_PER_CM = 0 * np.array([1e5, 3e5, 0]) / exy

B_MAGNITUDE_T = 50e-6
B_THETA =  3*np.pi/8
B_PHI = 13*np.pi/16

RABI_FREQ_BASE_HZ = 100e6
DETUNING_HZ = 0e6
SECOND_PULSE_PHASE = 0
MW_PULSE_LENGTH_S = np.arange(0, 400e-9, 2.5e-9)  # np.linspace(0, 0.5e-6, 1001)
EVOLUTION_TIME_S = np.arange(0, 5e-6, 20e-9)  # p.linspace(0, 15e-6, 801)
T2STAR_S = 2e-6
N_RAMSEY_POINTS = 251
RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ = np.linspace(0, 8.5e6, N_RAMSEY_POINTS)

RAD_TO_DEGREE = 360/(2*np.pi)
PHI_RANGE_HALF_HYPERFINE = np.linspace(1.4607, 3.25133, 301)

EXAMPLE_ORIENTATION = 3

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

rabi_frequencies = get_bare_rabi_frequencies(exp_param_factory.get_experiment_parameters())

# Plot the double inner product

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
imshow_with_extents_and_crop(1e-6*ramseyfreqs,1e-6*rabifreqs, -dtft/min(dtft.flatten()),ymin=60, ymax=100, xmin=0, xmax=8.5)
plt.colorbar(orientation="horizontal",location="top", shrink=0.6, label="2D inner product amplitude (a.u.)")
plt.plot(1e-6*ramseyfreqs, np.sqrt((1e-6*rabi_frequencies[3])**2 + (1e-6*ramseyfreqs)**2), color="white")
plt.savefig("double_inner_product.svg")
plt.show()

# Plot the associated filter function for boxcar and blackman

fake_signal = np.cos(2*np.pi*np.sqrt(rabi_frequencies[3]**2 +max(ramseyfreqs)**2)*MW_PULSE_LENGTH_S)
blackman_window = windows.get_window("blackman", len(MW_PULSE_LENGTH_S))

boxcar_filter_function=np.array([inner_product_sinusoid(np.cos, rabi_hz, MW_PULSE_LENGTH_S, fake_signal) for rabi_hz in rabifreqs])
blackman_filter_function=np.array([inner_product_sinusoid(np.cos, rabi_hz, MW_PULSE_LENGTH_S, fake_signal*blackman_window) for rabi_hz in rabifreqs])

fig = plt.figure(0,(1,5))
plt.plot(boxcar_filter_function,1e-6*rabifreqs, label="Boxcar")
plt.plot(blackman_filter_function,1e-6*rabifreqs, label="Blackman")
plt.gca().yaxis.set_ticks_position("right")
plt.legend(loc="lower left")
plt.savefig("filter_functions.svg")
plt.show()

# Do the fit (first fit is to set the fitting range)
inner_product_settings.use_effective_rabi_frequency=True
inner_product_settings.rabi_window="blackman"

cos_cos_inner_prod_init = double_cosine_inner_product_vs_ramsey(
    sq_cancelled_signal, rabi_frequencies[EXAMPLE_ORIENTATION], RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ, inner_product_settings
)
first_attempt_peaks = extract_fit_centers(
    fit_constrained_hyperfine_peaks(RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ, cos_cos_inner_prod_init, T2STAR_S)
)

min_ramsey_freq_hz = max(2 / (np.pi * T2STAR_S), min(first_attempt_peaks) - 2 / (np.pi * T2STAR_S))
max_ramsey_freq_hz = max(first_attempt_peaks) + 2 / (np.pi * T2STAR_S)
ramsey_freq_range_constrained_hz = np.linspace(min_ramsey_freq_hz, max_ramsey_freq_hz, N_RAMSEY_POINTS)
cos_cos_inner_prod = double_cosine_inner_product_vs_ramsey(
    sq_cancelled_signal, rabi_frequencies[EXAMPLE_ORIENTATION], ramsey_freq_range_constrained_hz, inner_product_settings
)
fit_result = fit_constrained_hyperfine_peaks(
    ramsey_freq_range_constrained_hz, cos_cos_inner_prod, T2STAR_S, constrain_same_width=True,allow_zero_peak=True
)

fig = plt.figure(0,(5,3))

renormalization =np.abs(min(cos_cos_inner_prod_init))

plt.plot(HZ_TO_MHZ * RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ, cos_cos_inner_prod_init/renormalization, label="inner product")
plt.plot(HZ_TO_MHZ * ramsey_freq_range_constrained_hz, fit_result.best_fit/renormalization, label="fit")
plt.xlabel("Inner product Ramsey frequency (MHz)")
plt.ylabel("Double cosine inner product (a.u.)")
plt.legend()

plt.title(
    f"B field: {np.array2string(T_TO_UT * np.array(b_field_vector_t), precision=1)} uT, Axis: {np.array2string(np.sqrt(3) * NVaxes_100[EXAMPLE_ORIENTATION], precision=0)}, Rabi: {rabi_frequencies[EXAMPLE_ORIENTATION] * 1e-6:.1f} MHz" #\nErrors: {np.array2string(errors_nT, precision=2)} nT"
)
plt.tight_layout()
plt.savefig("example_fit.svg")
plt.show()

larmor_freqs_all_axes_hz, _ = get_true_eigenvalues(exp_param_factory.get_experiment_parameters())
errors_nT = fit_vs_eigenvalue_error_nT(fit_result, larmor_freqs_all_axes_hz[EXAMPLE_ORIENTATION])
print(f"Fit errors: {np.array2string(errors_nT, precision=2)}")


# Plot pre-generated data on accuracy at 50 uT. Generated using simulate_and_invert_vs_b_angle.py" with the following settings:
# MW_DIRECTION = np.array([0.20539827217056314, 0.11882075246379901, 0.9714387158093318]) 
# E_FIELD_VECTOR_V_PER_CM = 0 * np.array([1e5, 3e5, 0]) / exy
# B_MAGNITUDE_T = 50e-6
# RABI_FREQ_BASE_HZ = 100e6
# DETUNING_HZ = 0e6
# MW_PULSE_LENGTH_S = np.arange(0, 400e-9, 2.5e-9)  
# EVOLUTION_TIME_S = np.arange(0, 5e-6, 20e-9) 
# T2STAR_S = 2e-6
# N_RAMSEY_POINTS = 251
# RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ = np.linspace(0, 10e6, N_RAMSEY_POINTS)
# In InnerProductSettings:  rabi_window="blackman", ramsey_window="boxcar", subtract_mean=True, use_effective_rabi_frequency=True
# In double_cosine_inner_product_inversion: constrain_same_width=True, allow_zero_peak=True

def half_hyperfine(theta, phi):
    return (-np.cos(theta)-np.cos(phi)*np.sin(theta) + np.sin(theta)*np.sin(phi))/np.sqrt(3) - f_h/(2*B_MAGNITUDE_T*gammab)

theta_hh_rad = np.array([fsolve(half_hyperfine, [2.5], (phi))[0] for phi in PHI_RANGE_HALF_HYPERFINE])
theta_hh_rad_low = np.array([fsolve(half_hyperfine, [1.5], (phi))[0] for phi in PHI_RANGE_HALF_HYPERFINE])

errors_vs_b_nT=np.loadtxt("errors_vs_b_nt.txt")
phi_values=np.loadtxt("phi_values.txt")
theta_values=np.loadtxt("theta_values.txt")

errors_vs_b_nT_fine=np.loadtxt("errors_nt_b_50_ut_t2s_2.000000_us.txt")
phi_values_fine=np.loadtxt("phi_values_b_50_ut_t2s_2.000000_us.txt")
theta_values_fine=np.loadtxt("theta_values_b_50_ut_t2s_2.000000_us.txt")

errors_vs_b_nT_fine2=np.loadtxt("errors_nt_b_50_ut_t2s_2.000000_us_interleaved.txt")
phi_values_fine2=np.loadtxt("phi_values_b_50_ut_t2s_2.000000_us_interleaved.txt")
theta_values_fine2=np.loadtxt("theta_values_b_50_ut_t2s_2.000000_us_interleaved.txt")

errors_vs_b_nT_total=np.concatenate((errors_vs_b_nT, errors_vs_b_nT_fine,errors_vs_b_nT_fine2))
phi_values_total = np.concatenate((phi_values, phi_values_fine,phi_values_fine2))
theta_values_total = np.concatenate((theta_values, theta_values_fine,theta_values_fine2))

for i,EXAMPLE_ORIENTATION in enumerate(NVOrientation):
    plt.subplot(2,2,i+1)
    plt.tripcolor(np.array(phi_values_total)*RAD_TO_DEGREE, np.array(theta_values_total)*RAD_TO_DEGREE, np.abs(np.array(errors_vs_b_nT_total)[:,EXAMPLE_ORIENTATION]),cmap="inferno", norm="log", vmin=.01, vmax=300)
    plt.colorbar()
    plt.xlabel("Azimuthal angle (deg)")
    plt.ylabel("Polar angle (deg)")
    plt.title(
        f"Axis: {np.array2string(np.sqrt(3) * NVaxes_100[EXAMPLE_ORIENTATION], precision=0)}, Rabi: {rabi_frequencies[EXAMPLE_ORIENTATION] * 1e-6:.1f} MHz", fontsize=10
    )
    if i==0:
        plt.vlines([155, 290], [90], [180])
        plt.hlines([90, 180],[155], [290])
    if i==1:
        phi_range_full = np.linspace(0, 2*np.pi, 101)
        theta_no_projection = np.atan2(1, np.sin(phi_range_full) - np.cos(phi_range_full))
        plt.plot(phi_range_full*RAD_TO_DEGREE, theta_no_projection * RAD_TO_DEGREE, color="white", linestyle="dashdot")

        plt.plot(PHI_RANGE_HALF_HYPERFINE*RAD_TO_DEGREE, theta_hh_rad*RAD_TO_DEGREE, color="white", linestyle="dashed")
        plt.plot(PHI_RANGE_HALF_HYPERFINE*RAD_TO_DEGREE, theta_hh_rad_low*RAD_TO_DEGREE, color="white", linestyle="dashed")

plt.suptitle("Inversion error (nT) for |B| = 50 uT")
plt.tight_layout()
plt.savefig("accuracy50uT.svg")
plt.show()