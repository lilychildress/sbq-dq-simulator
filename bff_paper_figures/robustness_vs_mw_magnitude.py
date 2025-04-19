import numpy as np

from bff_paper_figures.extract_experiment_values import (
    find_rabi_frequencies_from_signal,
    get_ideal_rabi_frequencies,
    get_true_eigenvalues,
)
from bff_paper_figures.fitting_routines import (
    extract_fit_centers_all_orientations,
    fit_vs_eigenvalue_error_all_orientations_nt,
)
from bff_paper_figures.inner_product_functions import InnerProductSettings
from bff_paper_figures.simulate_and_invert_helper_functions import (
    double_cosine_inner_product_fit_inversion,
    sq_cancelled_signal_generator,
    time_domain_fit_inversion,
)
from bff_simulator.constants import exy
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory

T_TO_UT = 1e6
HZ_TO_MHZ = 1e-6

MW_DIRECTION = np.array([0.97203398, 0.2071817, 0.11056978])  # Vincent's old "magic angle"
MW_DIRECTION = np.array([0.20539827217056314, 0.11882075246379901, 0.9714387158093318])  # Lily's new "magic angle"
E_FIELD_VECTOR_V_PER_CM = 0 * np.array([1e5, 3e5, 0]) / exy

B_MAGNITUDE_T = 50e-6
B_THETA = 3 * np.pi / 8
B_PHI = 13 * np.pi / 16
B_FIELD_VECTOR_T = B_MAGNITUDE_T * np.array(
    [np.sin(B_THETA) * np.cos(B_PHI), np.sin(B_THETA) * np.sin(B_PHI), np.cos(B_THETA)]
)

NOMINAL_RABI_FREQ_BASE_HZ = 100e6
RABI_MAX_RANGE = np.linspace(50e6, 150e6, 201)
DETUNING_HZ = 0e6
SECOND_PULSE_PHASE = 0
MW_PULSE_LENGTH_S = np.arange(0, 800e-9, 2.5e-9)  # np.linspace(0, 0.5e-6, 1001)
EVOLUTION_TIME_S = np.arange(0, 3e-6, 20e-9)  # p.linspace(0, 15e-6, 801)
T2STAR_S = 2e-6
N_RAMSEY_POINTS = 251
RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ = np.linspace(0, 10e6, N_RAMSEY_POINTS)
RABI_FREQ_RANGE_TO_PROBE_HZ = np.linspace(20e6, 150e6, 201)
HEIGHT_FACTOR = 0.3
PROMINENCE_FACTOR = 0.3

PEAK_INDEX = 0  # Which frequency we will be comparing to expected value; 0 is highest-frequency peak

RUN_LABEL = f"rabi_{min(RABI_MAX_RANGE) * 1e-6:.02f}_to_{max(RABI_MAX_RANGE) * 1e-6:.02f}_mhz"

# Set up simulation for everything except maximum Rabi frequency
nv_ensemble = HomogeneousEnsemble()
nv_ensemble.efield_splitting_hz = np.linalg.norm(E_FIELD_VECTOR_V_PER_CM) * exy
nv_ensemble.t2_star_s = T2STAR_S
nv_ensemble.add_full_diamond_populations()
nv_ensemble.mw_direction = MW_DIRECTION

off_axis_solver = LiouvillianSolver()

exp_param_factory = OffAxisFieldExperimentParametersFactory()
exp_param_factory.set_base_rabi_frequency(NOMINAL_RABI_FREQ_BASE_HZ)
exp_param_factory.set_mw_direction(MW_DIRECTION)
exp_param_factory.set_e_field_v_per_m(E_FIELD_VECTOR_V_PER_CM)
exp_param_factory.set_detuning(DETUNING_HZ)
exp_param_factory.set_mw_pulse_lengths(MW_PULSE_LENGTH_S)
exp_param_factory.set_evolution_times(EVOLUTION_TIME_S)
exp_param_factory.set_b_field_vector(B_FIELD_VECTOR_T)

inner_product_settings = InnerProductSettings(
    MW_PULSE_LENGTH_S,
    EVOLUTION_TIME_S,
    rabi_window="blackman",
    ramsey_window="boxcar",
    subtract_mean=True,
    use_effective_rabi_frequency=True,
)

rabi_maxes = []
errors_freq_nt = []
errors_time_nt = []

# Optionally load a partially completed simulation
# errors_freq_nt = list(np.loadtxt(f"errors_nt_freq_{RUN_LABEL}.txt"))
# errors_time_nt = list(np.loadtxt(f"errors_nt_time_{RUN_LABEL}.txt", ))
# rabi_maxes = list(np.loadtxt(f"rabi_values_{RUN_LABEL}.txt"))

for rabi_max in RABI_MAX_RANGE:
    if rabi_max not in rabi_maxes:
        # Run the experiment at a different Rabi frequency (corresponding to e.g. amplifier drift) and calculate the signal
        exp_param_factory.set_base_rabi_frequency(rabi_max)
        sq_cancelled_signal = sq_cancelled_signal_generator(exp_param_factory, nv_ensemble, off_axis_solver)

        # Use the "ground truth" hamiltonian to calculate the transition frequencies we wish to extract, as well as
        # the "ground truth" rabi frequencies for each
        dq_larmor_freqs_all_axes_hz, bz_values_all_axes_t = get_true_eigenvalues(
            exp_param_factory.get_experiment_parameters()
        )
        ideal_rabi_frequencies = get_ideal_rabi_frequencies(exp_param_factory.get_experiment_parameters())
        print(f"Ideal Rabi frequencies: {np.array2string(ideal_rabi_frequencies * 1e-6, precision=2)}")

        try:
            # Run the analysis using the rabi frequency extracted from the signal
            rabi_frequencies = find_rabi_frequencies_from_signal(
                sq_cancelled_signal,
                inner_product_settings,
                rabi_freq_range_to_probe_hz=RABI_FREQ_RANGE_TO_PROBE_HZ,
                height_factor=HEIGHT_FACTOR,
                prominence_factor=PROMINENCE_FACTOR,
                use_time_domain_summation_for_initial_guesses=False,
            )
            print(f"Extracted Rabi frequencies: {np.array2string(rabi_frequencies * 1e-6, precision=2)}")

            # frequency domain inversion
            peakfit_results = double_cosine_inner_product_fit_inversion(
                sq_cancelled_signal,
                rabi_frequencies,
                inner_product_settings,
                RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ,
                T2STAR_S,
                constrain_same_width=True,
                allow_zero_peak=True,
            )
            errors_fd_nt = fit_vs_eigenvalue_error_all_orientations_nt(peakfit_results, dq_larmor_freqs_all_axes_hz)
            print(
                f"Rabi_max = {1e-6 * rabi_max:.2f} MHz, freq domain errors = {np.array2string(errors_fd_nt[:, PEAK_INDEX], precision=2)} nT"
            )

            # time domain inversion using results from frequency domain inversion as an initial guess
            freq_guesses_all_orientations = extract_fit_centers_all_orientations(peakfit_results)
            time_domain_fit_results = time_domain_fit_inversion(
                sq_cancelled_signal,
                rabi_frequencies,
                inner_product_settings,
                freq_guesses_all_orientations,
                T2STAR_S,
                fix_phase_to_zero=False,
                constrain_same_decay=True,
                constrain_hyperfine_freqs=True,
            )

            # Compare extracted transition frequencies to the ideal values
            errors_td_nt = fit_vs_eigenvalue_error_all_orientations_nt(
                time_domain_fit_results, dq_larmor_freqs_all_axes_hz
            )
            print(f"\t \t \t time domain fit errors = {np.array2string(errors_td_nt[:, PEAK_INDEX], precision=3)} nT")

            rabi_maxes.append(rabi_max)
            errors_freq_nt.append(errors_fd_nt[:, PEAK_INDEX])
            errors_time_nt.append(errors_td_nt[:, PEAK_INDEX])

        except ValueError:
            print(f"Rabi_max = {1e-6 * rabi_max:.2f} MHz generated ValueError ")

        # np.savetxt(f"errors_nt_freq_{RUN_LABEL}.txt", errors_freq_nt)
        # np.savetxt(f"errors_nt_time_{RUN_LABEL}.txt", errors_time_nt)
        # np.savetxt(f"rabi_values_{RUN_LABEL}.txt", rabi_maxes)
