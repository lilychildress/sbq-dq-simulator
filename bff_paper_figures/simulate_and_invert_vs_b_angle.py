import numpy as np
from math import floor
from matplotlib import pyplot as plt

from bff_simulator.constants import exy, NVaxes_100
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

B_MAGNITUDE_T = 50e-6
B_THETA_START = 0
B_THETA_STOP = np.pi
B_THETA_N = 42

B_PHI_START = 0
B_PHI_STOP = 2*np.pi
B_PHI_N_MAX = 84

RABI_FREQ_BASE_HZ = 100e6
DETUNING_HZ = 0e6
SECOND_PULSE_PHASE = 0
MW_PULSE_LENGTH_S = np.arange(0, 400e-9, 2.5e-9)  # np.linspace(0, 0.5e-6, 1001)
EVOLUTION_TIME_S = np.arange(0, 5e-6, 20e-9)  # p.linspace(0, 15e-6, 801)
T2STAR_S = 2e-6
N_RAMSEY_POINTS = 251
RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ = np.linspace(0, 10e6, N_RAMSEY_POINTS)

RUN_LABEL = f"b_{B_MAGNITUDE_T*T_TO_UT:.0f}_ut_t2s_{T2STAR_S*1e6:0f}_us"

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
 
rabi_frequencies = get_bare_rabi_frequencies(exp_param_factory.get_experiment_parameters())

theta_values = []
phi_values = []
errors_vs_b_nT = []
for theta in np.linspace(B_THETA_START, B_THETA_STOP, B_THETA_N):
    b_phi_n = max(5, floor(np.sin(theta)*B_PHI_N_MAX))
    for phi in np.linspace(B_PHI_START, B_PHI_STOP, b_phi_n):
        theta_values.append(theta)
        phi_values.append(phi)
        b_field_vector_t = B_MAGNITUDE_T*np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
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
        print(f"theta = {theta:.2f}, phi = {phi:.2f}, largest eigenvalue errors = {np.array2string(errors_nT[:, 2], precision=2)} nT")
        errors_vs_b_nT.append(errors_nT[:, 2])


np.savetxt(f"errors_nt_{RUN_LABEL}.txt", errors_vs_b_nT)
np.savetxt(f"phi_values_{RUN_LABEL}.txt", phi_values)
np.savetxt(f"theta_values_{RUN_LABEL}.txt", theta_values)

for i,orientation in enumerate(NVOrientation):
    plt.subplot(2,2,i+1)
    plt.tripcolor(np.array(phi_values)*360/(2*np.pi), np.array(theta_values)*360/(2*np.pi), np.abs(np.array(errors_vs_b_nT)[:,orientation]),  cmap="inferno", vmax=3)
    plt.colorbar()
    plt.xlabel("Azimuthal angle (deg)")
    plt.ylabel("Polar angle (deg)")
    plt.title(
        f"Axis: {np.array2string(np.sqrt(3) * NVaxes_100[orientation], precision=0)}, Rabi: {rabi_frequencies[orientation] * 1e-6:.1f} MHz", fontsize=10
    )
plt.suptitle("Inversion error (nT) for |B| = 50 uT")
plt.tight_layout()
plt.show()