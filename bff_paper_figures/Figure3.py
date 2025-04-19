from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import numpy as np
from bff_paper_figures.imshow_extensions import imshow_with_extents_and_crop
from bff_paper_figures.simulate_and_invert_helper_functions import sq_cancelled_signal_generator
from bff_paper_figures.extract_experiment_values import get_ideal_rabi_frequencies

from bff_simulator.abstract_classes.abstract_ensemble import NV14HyperfineField, NVOrientation
from bff_simulator.constants import exy, NVaxes_100, gammab
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble, NVOrientation
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory
from bff_simulator.simulator import Simulation
from bff_simulator.vector_manipulation import perpendicular_projection

MW_DIRECTION = np.array([0.97203398, 0.2071817, 0.11056978])  # Vincent's old "magic angle"
MW_DIRECTION = np.array([0.20539827217056314, 0.11882075246379901, 0.9714387158093318])  # Lily's new "magic angle"
E_FIELD_VECTOR_V_PER_CM = 0 * np.array([1e5, 3e5, 0]) / exy
B_FIELD_VECTOR_T = [9e6 / (gammab * np.sqrt(3)) * x for x in [1, 1.1, 1.2]]
RABI_FREQ_BASE_HZ = 100e6
DETUNING_HZ = 0e6
SECOND_PULSE_PHASE = 0
MW_PULSE_LENGTH_S = np.linspace(0, 0.5e-6, 1001)
EVOLUTION_TIME_S = np.linspace(0, 15e-6, 1001)
T2STAR_S = 2e-6

def main()-> None:

    nv_ensemble = HomogeneousEnsemble()
    nv_ensemble.efield_splitting_hz = np.linalg.norm(E_FIELD_VECTOR_V_PER_CM) * exy
    nv_ensemble.t2_star_s = T2STAR_S
    nv_ensemble.add_full_diamond_populations()
    nv_ensemble.mw_direction = MW_DIRECTION

    exp_param_factory = OffAxisFieldExperimentParametersFactory()
    exp_param_factory.set_base_rabi_frequency(RABI_FREQ_BASE_HZ)
    exp_param_factory.set_mw_direction(MW_DIRECTION)
    exp_param_factory.set_e_field_v_per_m(E_FIELD_VECTOR_V_PER_CM)
    exp_param_factory.set_b_field_vector(B_FIELD_VECTOR_T)
    exp_param_factory.set_detuning(DETUNING_HZ)
    exp_param_factory.set_second_pulse_phase(0)
    exp_param_factory.set_mw_pulse_lengths(MW_PULSE_LENGTH_S)
    exp_param_factory.set_evolution_times(EVOLUTION_TIME_S)

    off_axis_solver = LiouvillianSolver()

    sq_cancelled_signal = sq_cancelled_signal_generator(exp_param_factory, nv_ensemble, off_axis_solver)

    # Calculate the frequency axes for the 2D FFT
    rabifreqs = np.sort(np.fft.fftfreq(len(MW_PULSE_LENGTH_S), MW_PULSE_LENGTH_S[1] - MW_PULSE_LENGTH_S[0]))
    ramseyfreqs = np.sort(np.fft.fftfreq(len(EVOLUTION_TIME_S), EVOLUTION_TIME_S[1] - EVOLUTION_TIME_S[0]))
    
    # Set the display plotting limits
    ramsey_max = 30e6 
    rabi_max = RABI_FREQ_BASE_HZ * 2

    # Calculate the FFT of the single-quantum-cancelled signal
    fourier_transform_sq_subtracted = np.fft.fft2(sq_cancelled_signal)
    fourier_transform_sq_subtracted = np.fft.fftshift(fourier_transform_sq_subtracted)
    imshow_with_extents_and_crop(
        ramseyfreqs,
        rabifreqs,
        abs(fourier_transform_sq_subtracted),
        ymin=0,
        ymax=rabi_max,
        xmin=0,
        xmax=ramsey_max,
        vmax=500,
    )
    plt.colorbar()

    # Plot the rabi frequencies for each orientation on top of the FFT
    rabi_frequencies = get_ideal_rabi_frequencies(exp_param_factory.get_experiment_parameters())
    colors = list(TABLEAU_COLORS.keys())[:len(NVOrientation)]
    for i in range(len(NVOrientation)):
        for j in [0.5, 1, 1.5, 2]:
            plt.plot(
                [12e6 if i != 0 else 25e6, ramsey_max], [j * rabi_frequencies[i], j * rabi_frequencies[i]], color=colors[i]
            )

    plt.savefig("figure3.svg", format="svg")
    plt.show()

if __name__ == "__main__":
    main()