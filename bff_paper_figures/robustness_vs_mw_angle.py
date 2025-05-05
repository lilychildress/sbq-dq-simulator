import numpy as np

from bff_paper_figures.extract_experiment_values import (
    find_rabi_frequencies_from_signal,
    get_ideal_rabi_frequencies,
    get_true_transition_frequencies,
)
from bff_paper_figures.fitting_routines import (
    extract_fit_centers_all_orientations,
    fit_vs_eigenvalue_error_all_orientations_nt,
)
from bff_paper_figures.inner_product_functions import InnerProductSettings
from bff_paper_figures.inversions import (
    freq_domain_inversion,
    time_domain_inversion,
)
from bff_paper_figures.shared_parameters import (
    B_PHI_FIG4,
    B_THETA_FIG4,
    DETUNING_HZ,
    E_FIELD_VECTOR_V_PER_CM,
    MW_PHI,
    MW_THETA,
    PEAK_INDEX,
    RABI_FREQ_BASE_HZ,
    RAD_TO_DEGREE,
    RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ,
    T2STAR_S,
)
from bff_paper_figures.simulation_helper_functions import angles_already_evaluated, sq_cancelled_signal_generator
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory

MW_THETA_EXCURSION = 10 / RAD_TO_DEGREE
MW_PHI_EXCURSION = 10 / RAD_TO_DEGREE
MW_THETA_RANGE = np.linspace(MW_THETA - MW_THETA_EXCURSION, MW_THETA + MW_THETA_EXCURSION, 15)
MW_PHI_RANGE = np.linspace(MW_PHI - MW_PHI_EXCURSION, MW_PHI + MW_PHI_EXCURSION, 21)

B_MAGNITUDE_T = 50e-6
B_THETA = B_THETA_FIG4
B_PHI = B_PHI_FIG4

MW_PULSE_LENGTH_S = np.arange(0, 800e-9, 2.5e-9)
EVOLUTION_TIME_S = np.arange(0, 3e-6, 20e-9)

RUN_LABEL = f"mw_angle_variation_freq_and_time_fit_rabi_{max(MW_PULSE_LENGTH_S) * 1e9:.0f}_ns"

# Define the experiment parameters except for MW direction
nv_ensemble = HomogeneousEnsemble()
nv_ensemble.t2_star_s = T2STAR_S
nv_ensemble.add_full_diamond_populations()

off_axis_solver = LiouvillianSolver()

exp_param_factory = OffAxisFieldExperimentParametersFactory()
exp_param_factory.set_base_rabi_frequency(RABI_FREQ_BASE_HZ)
exp_param_factory.set_e_field_v_per_m(E_FIELD_VECTOR_V_PER_CM)
exp_param_factory.set_detuning(DETUNING_HZ)
exp_param_factory.set_mw_pulse_lengths(MW_PULSE_LENGTH_S)
exp_param_factory.set_evolution_times(EVOLUTION_TIME_S)
exp_param_factory.set_b_field_vector(
    B_MAGNITUDE_T * np.array([np.sin(B_THETA) * np.cos(B_PHI), np.sin(B_THETA) * np.sin(B_PHI), np.cos(B_THETA)])
)

# Define the inner product settings for the inversion
inner_product_settings = InnerProductSettings(
    MW_PULSE_LENGTH_S,
    EVOLUTION_TIME_S,
    rabi_window="blackman",
    ramsey_window="boxcar",
    subtract_mean=True,
    use_effective_rabi_frequency=True,
)

# Lists in which we will store the angles at which we run the simulations along with the errors in
# extracted ms = +/-1 transition frequency for both frequency- and time-domain inversions
mw_theta_values = []
mw_phi_values = []
errors_freq_domain_nt = []
errors_time_domain_nt = []
problem_theta_values = []
problem_phi_values = []

# Optionally load partially completed simulation
# errors_freq_vs_angle_nt = list(np.loadtxt(f"errors_nt_freq_{RUN_LABEL}.txt"))
# errors_time_vs_angle_nt = list(np.loadtxt(f"errors_nt_time_{RUN_LABEL}.txt"))
# mw_phi_values = list(np.loadtxt(f"phi_values_{RUN_LABEL}.txt"))
# mw_theta_values = list(np.loadtxt(f"theta_values_{RUN_LABEL}.txt"))
# problem_phi_values = list(np.loadtxt(f"problem_phi_values_{RUN_LABEL}.txt"))
# problem_theta_values = list(np.loadtxt(f"problem_theta_values_{RUN_LABEL}.txt"))

for theta in MW_THETA_RANGE:
    for phi in MW_PHI_RANGE:
        # Check if we have already run the simulation for this angle - useful when loading partially completed
        # simultions
        if not angles_already_evaluated(theta, phi, mw_theta_values, mw_phi_values):
            # Find the new MW direction
            mw_direction = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

            # Run the numerical experiment at this new MW direction
            exp_param_factory.set_mw_direction(mw_direction)
            sq_cancelled_signal = sq_cancelled_signal_generator(exp_param_factory, nv_ensemble, off_axis_solver)

            # Extract the true transition frequencies and true Rabi frequencies
            larmor_freqs_all_axes_hz, _ = get_true_transition_frequencies(exp_param_factory.get_experiment_parameters())
            ideal_rabi_frequencies = get_ideal_rabi_frequencies(exp_param_factory.get_experiment_parameters())
            print(f"Ideal Rabi frequencies: {np.array2string(ideal_rabi_frequencies * 1e-6, precision=2)}")

            try:
                # Extract Rabi frequencies from the data
                rabi_frequencies = find_rabi_frequencies_from_signal(sq_cancelled_signal, inner_product_settings)
                print(f"Extracted Rabi frequencies: {np.array2string(rabi_frequencies * 1e-6, precision=2)}")

                # frequency domain inversion
                peakfit_results = freq_domain_inversion(
                    sq_cancelled_signal,
                    rabi_frequencies,
                    inner_product_settings,
                    RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ,
                    T2STAR_S,
                    constrain_same_width=True,
                    allow_zero_peak=True,
                )
                errors_freq_nt = fit_vs_eigenvalue_error_all_orientations_nt(peakfit_results, larmor_freqs_all_axes_hz)
                print(
                    f"theta = {RAD_TO_DEGREE * theta:.2f} deg, phi = {RAD_TO_DEGREE * phi:.2f} deg, freq domain errors = {np.array2string(errors_freq_nt[:, PEAK_INDEX], precision=2)} nT"
                )

                # time domain inversion
                freq_guesses_all_orientations = extract_fit_centers_all_orientations(peakfit_results)
                time_domain_fit_results = time_domain_inversion(
                    sq_cancelled_signal,
                    rabi_frequencies,
                    inner_product_settings,
                    freq_guesses_all_orientations,
                    T2STAR_S,
                    fix_phase_to_zero=False,
                    constrain_same_decay=True,
                    constrain_hyperfine_freqs=True,
                )
                errors_time_nt = fit_vs_eigenvalue_error_all_orientations_nt(
                    time_domain_fit_results, larmor_freqs_all_axes_hz
                )
                print(
                    f"\t \t \t time domain fit errors = {np.array2string(errors_time_nt[:, PEAK_INDEX], precision=3)} nT"
                )

                mw_theta_values.append(theta)
                mw_phi_values.append(phi)
                errors_freq_domain_nt.append(errors_freq_nt[:, PEAK_INDEX])
                errors_time_domain_nt.append(errors_time_nt[:, PEAK_INDEX])

            except ValueError:
                print(f"theta = {RAD_TO_DEGREE * theta:.2f}, phi = {RAD_TO_DEGREE * phi:.2f} MHz generated ValueError")
                problem_theta_values.append(theta)
                problem_phi_values.append(phi)

        np.savetxt(f"errors_nt_freq_{RUN_LABEL}.txt", errors_freq_domain_nt)
        np.savetxt(f"errors_nt_time_{RUN_LABEL}.txt", errors_time_domain_nt)
        np.savetxt(f"phi_values_{RUN_LABEL}.txt", mw_phi_values)
        np.savetxt(f"theta_values_{RUN_LABEL}.txt", mw_theta_values)
        np.savetxt(f"problem_phi_values_{RUN_LABEL}.txt", problem_phi_values)
        np.savetxt(f"problem_theta_values_{RUN_LABEL}.txt", problem_theta_values)
