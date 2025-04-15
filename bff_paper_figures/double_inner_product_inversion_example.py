import numpy as np
from matplotlib import pyplot as plt

from bff_simulator.abstract_classes.abstract_ensemble import NVOrientation
from bff_simulator.constants import exy, NVaxes_100, gammab
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory
from bff_simulator.simulator import Simulation
from bff_paper_figures.inner_product_functions import InnerProductSettings, double_cosine_inner_product_vs_ramsey
from bff_paper_figures.fit_inner_product_vs_ramsey import (
    fit_constrained_hyperfine_peaks,
    plot_fit_vs_inner_product,
    fit_vs_eigenvalue_error_nT,
    extract_fit_centers,
)
from bff_paper_figures.extract_experiment_values import get_bare_rabi_frequencies, get_true_eigenvalues

T_TO_UT = 1e6

MW_DIRECTION = np.array([0.97203398, 0.2071817, 0.11056978])  # Vincent's old "magic angle"
MW_DIRECTION = np.array([0.20539827217056314, 0.11882075246379901, 0.9714387158093318])  # Lily's new "magic angle"
E_FIELD_VECTOR_V_PER_CM = 0 * np.array([1e5, 3e5, 0]) / exy
B_FIELD_VECTOR_T = [20e-6 / np.sqrt(21) * x for x in [4, 2, 1]]
RABI_FREQ_BASE_HZ = 100e6
DETUNING_HZ = 0e6
SECOND_PULSE_PHASE = 0
MW_PULSE_LENGTH_S = np.arange(0, 500e-9, 2.5e-9)  # np.linspace(0, 0.5e-6, 1001)
EVOLUTION_TIME_S = np.arange(0, 5e-6, 20e-9)  # p.linspace(0, 15e-6, 801)
T2STAR_S = 2e-6
N_RAMSEY_POINTS = 201
RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ = np.linspace(0, 10e6, N_RAMSEY_POINTS)

nv_ensemble = HomogeneousEnsemble()
nv_ensemble.efield_splitting_hz = np.linalg.norm(E_FIELD_VECTOR_V_PER_CM) * exy
nv_ensemble.t2_star_s = T2STAR_S
# nv_ensemble.add_n14_triplet(NVOrientation.A)
# nv_ensemble.add_n14_triplet(NVOrientation.C)
# nv_ensemble.add_n14_triplet(NVOrientation.D)
# nv_ensemble.add_nv_single_species(NVOrientation.B, NV14HyperfineField.N14_plus)
# nv_ensemble.add_nv_single_species(NVOrientation.B, NV14HyperfineField.N14_0)
nv_ensemble.add_full_diamond_populations()
nv_ensemble.mw_direction = MW_DIRECTION

off_axis_solver = LiouvillianSolver()

exp_param_factory = OffAxisFieldExperimentParametersFactory()
exp_param_factory.set_base_rabi_frequency(RABI_FREQ_BASE_HZ)
exp_param_factory.set_mw_direction(MW_DIRECTION)
exp_param_factory.set_e_field_v_per_m(E_FIELD_VECTOR_V_PER_CM)
exp_param_factory.set_b_field_vector(B_FIELD_VECTOR_T)
exp_param_factory.set_detuning(DETUNING_HZ)
exp_param_factory.set_mw_pulse_lengths(MW_PULSE_LENGTH_S)
exp_param_factory.set_evolution_times(EVOLUTION_TIME_S)

exp_param_factory.set_second_pulse_phase(0)
off_axis_experiment_parameters = exp_param_factory.get_experiment_parameters()
off_axis_simulation_0_phase = Simulation(off_axis_experiment_parameters, nv_ensemble, off_axis_solver)

exp_param_factory.set_second_pulse_phase(np.pi)
off_axis_experiment_parameters = exp_param_factory.get_experiment_parameters()
off_axis_simulation_pi_phase = Simulation(off_axis_experiment_parameters, nv_ensemble, off_axis_solver)

sq_cancelled_signal = off_axis_simulation_0_phase.ms0_results + off_axis_simulation_pi_phase.ms0_results

rabi_frequencies = get_bare_rabi_frequencies(off_axis_experiment_parameters)
larmor_freqs_all_axes_hz, bz_all_axes_t = get_true_eigenvalues(off_axis_experiment_parameters)

inner_product_settings = InnerProductSettings(
    MW_PULSE_LENGTH_S,
    EVOLUTION_TIME_S,
    rabi_window="blackman",
    ramsey_window="boxcar",
    subtract_mean=True,
    use_effective_rabi_frequency=True,
)

for i, orientation in enumerate(NVOrientation):
    plt.subplot(2, 2, i + 1)

    cos_cos_inner_prod = double_cosine_inner_product_vs_ramsey(
        sq_cancelled_signal, rabi_frequencies[orientation], RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ, inner_product_settings
    )
    first_attempt_peaks = extract_fit_centers(
        fit_constrained_hyperfine_peaks(RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ, cos_cos_inner_prod, T2STAR_S)
    )

    min_ramsey_freq_hz = max(2 / (np.pi * T2STAR_S), min(first_attempt_peaks) - 2 / (np.pi * T2STAR_S))
    max_ramsey_freq_hz = max(first_attempt_peaks) + 2 / (np.pi * T2STAR_S)
    ramsey_freq_range_constrained_hz = np.linspace(min_ramsey_freq_hz, max_ramsey_freq_hz, N_RAMSEY_POINTS)
    cos_cos_inner_prod = double_cosine_inner_product_vs_ramsey(
        sq_cancelled_signal, rabi_frequencies[orientation], ramsey_freq_range_constrained_hz, inner_product_settings
    )
    fit_result = fit_constrained_hyperfine_peaks(
        ramsey_freq_range_constrained_hz, cos_cos_inner_prod, T2STAR_S, constrain_same_width=True,allow_zero_peak=True
    )
    print(fit_result.fit_report())
    plot_fit_vs_inner_product(ramsey_freq_range_constrained_hz, cos_cos_inner_prod, fit_result)
    errors_nT = fit_vs_eigenvalue_error_nT(fit_result, larmor_freqs_all_axes_hz[orientation])
    plt.title(
        f"Axis: {np.array2string(np.sqrt(3) * NVaxes_100[orientation], precision=0)}, Rabi: {rabi_frequencies[orientation] * 1e-6:.1f} MHz\nErrors: {np.array2string(errors_nT, precision=2)} nT"
    )
plt.tight_layout()

plt.suptitle(
    f"B field: {np.array2string(T_TO_UT * np.array(B_FIELD_VECTOR_T), precision=1)} uT, Rabi window: {inner_product_settings.rabi_window}, Ramsey window: {inner_product_settings.ramsey_window}, {'mean subtracted' if inner_product_settings.subtract_mean else ''}, {'using effective Rabi' if inner_product_settings.use_effective_rabi_frequency else ''}"
)
plt.show()

print(larmor_freqs_all_axes_hz)
print(2*bz_all_axes_t*gammab)
