import numpy as np
from matplotlib import pyplot as plt

from bff_simulator.constants import exy
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.abstract_classes.abstract_ensemble import NVOrientation
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory
from bff_paper_figures.simulate_and_invert_helper_functions import (
    sq_cancelled_signal_generator,
    double_cosine_inner_product_fit_inversion,
)
from bff_paper_figures.inner_product_functions import InnerProductSettings
from bff_paper_figures.extract_experiment_values import get_bare_rabi_frequencies

T_TO_UT = 1e6
HZ_TO_MHZ = 1e-6

MW_DIRECTION = np.array([0.97203398, 0.2071817, 0.11056978])  # Vincent's old "magic angle"
MW_DIRECTION = np.array([0.20539827217056314, 0.11882075246379901, 0.9714387158093318])  # Lily's new "magic angle"
E_FIELD_VECTOR_V_PER_CM = 0 * np.array([1e5, 3e5, 0]) / exy

B_FIELD_VECTOR_DIRECTION = np.array([4, 2, 1])/np.sqrt(21)
B_MAGNITUDES_T = np.linspace(20e-6, 60e-6, 41) 

RABI_FREQ_BASE_HZ = 100e6
DETUNING_HZ = 0e6
SECOND_PULSE_PHASE = 0
MW_PULSE_LENGTH_S = np.arange(0, 400e-9, 2.5e-9)  # np.linspace(0, 0.5e-6, 1001)
EVOLUTION_TIME_S = np.arange(0, 5e-6, 20e-9)  # p.linspace(0, 15e-6, 801)
T2STAR_S = 2e-6
N_RAMSEY_POINTS = 251
RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ = np.linspace(0, 10e6, N_RAMSEY_POINTS)

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

inner_product_settings = InnerProductSettings(
    MW_PULSE_LENGTH_S,
    EVOLUTION_TIME_S,
    rabi_window="blackman",
    ramsey_window="boxcar",
    subtract_mean=True,
    use_effective_rabi_frequency=True,
)
 
b_field_vector_values_t = [mag*B_FIELD_VECTOR_DIRECTION for mag in B_MAGNITUDES_T]
rabi_frequencies = get_bare_rabi_frequencies(exp_param_factory.get_experiment_parameters())

errors_vs_b_nT = []
for b_field_vector_t in b_field_vector_values_t:
    exp_param_factory.set_b_field_vector(b_field_vector_t)
    sq_cancelled_signal = sq_cancelled_signal_generator(exp_param_factory, nv_ensemble, off_axis_solver)

    errors_nT = double_cosine_inner_product_fit_inversion(
        sq_cancelled_signal,
        exp_param_factory.get_experiment_parameters(),
        inner_product_settings,
        RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ,
        T2STAR_S,
        constrain_same_width=True,
        allow_zero_peak=True,
    )

    print(f"B = {np.array2string(T_TO_UT*b_field_vector_t, precision=1)} uT, largest eigenvalue errors = {np.array2string(errors_nT[:, 2], precision=2)} nT")
    errors_vs_b_nT.append(errors_nT[:, 2])

for orientation in NVOrientation:
    plt.plot(T_TO_UT*B_MAGNITUDES_T, np.array(errors_vs_b_nT)[:,orientation], label=f"Rabi: {HZ_TO_MHZ*rabi_frequencies[orientation]:.1f} MHz")
plt.xlabel("Magnetic field magnitude (uT) along <4,2,1>")
plt.ylabel("Inversion error (nT) from largest eigenvalue")
plt.legend()
plt.title(
    f"T2* = {T2STAR_S*1e6} us, Rabi window: {inner_product_settings.rabi_window}, Ramsey window: {inner_product_settings.ramsey_window},\n{'mean subtracted' if inner_product_settings.subtract_mean else ''}, {'using effective Rabi' if inner_product_settings.use_effective_rabi_frequency else ''}"
)
plt.show()
