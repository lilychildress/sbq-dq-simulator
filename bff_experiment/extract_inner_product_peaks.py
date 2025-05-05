from os import listdir
from pandas import DataFrame
from numpy.typing import NDArray
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from bff_simulator.constants import f_h, gammab
from bff_paper_figures.inner_product_functions import InnerProductSettings
from bff_paper_figures.imshow_extensions import imshow_with_extents_and_crop
from bff_experiment.inner_product_exp_functions import double_inner_product_exponential, inner_product_exp_to_minimize
from bff_experiment.data_processing_functions import sum_over_column, get_truncated_signal, COLUMN_TO_SUM
from bff_paper_figures.shared_parameters import HZ_TO_MHZ

#path_to_directory = Path("/Users/lilianchildress/Documents/GitHub/sbq-dq-simulator/bff_experiment/data/sorted_by_voltage/0.0v")
#example_file = Path('0_2867000.csv')


RAMSEY_FREQ_RANGE_HZ = np.linspace(0, 13e6, 151)
RABI_FREQ_RANGE_HZ = np.linspace(0, 50e6, 151)

EVOLUTION_TIME_CROP_S = 3e-6
PULSE_DURATION_CROP_S = 2e-6
RABI_WINDOW = "cosine"

RAMSEY_INDEX = 0
RABI_INDEX=1

# Initial guesses for the location of peaks in the double inner product in 
# terms of their free evolution frequency and pulse duration frequency,
# with one guess for each orientation and for each coil voltage.
INITIAL_GUESSES = [
    [[4.8e6,40e6], [4.6e6,22e6], [4.7e6,15e6], [5e6,23.5e6]],
    [[5.e6,20e6],[5e6,23e6],[4.6e6,15e6],[4.3e6,25e6]],
    [[6.2e6,30e6],[6e6,23e6],[5e6,20e6], [4.5e6,24e6]],
    [[6.2e6,30e6],[6e6,23e6],[5e6,14e6],[4.8e6,24e6]],
    [[7.5e6,42e6],[7.2e6,47e6],[5e6,15e6],[5e6,25e6]],
    [[8.2e6,42e6],[8e6,47e6],[5e6,15e6],[5.8e6,48e6]],
    [[9.e6,42e6],[8e6,47e6],[5.5e6,15e6], [6e6,25e6]],
    [[10.e6,22e6],[9.5e6,47e6],[5.5e6,15e6],[6e6,24e6]],
    [[10.7e6,23e6],[9.5e6,45e6],[5.7e6,15e6],[6.6e6,48e6]], 
    ]
# An approximate value for the bare Rabi freqiuency for each orientation, used
# to determine which harmonic we are looking at in the peak-finding operation.
NOMINAL_RABI_FREQS = [19.5e6, 23e6, 9e6, 24e6]
COLORS = ["green", "red", "blue", "white"]

HARMONICS = [0.5, 1, 1.5, 2]
M_I_VALUES = [-1, 0, 1]
M_I_GUESS = 1

overall_directory = Path("/Users/lilianchildress/Documents/GitHub/sbq-dq-simulator/bff_experiment/data/sorted_by_voltage")
sorted_voltage_folders = sorted(listdir(overall_directory))
rabi_frequencies_hz = []
ramsey_frequencies_hz = []
field_projections_t = []
voltages = []

for voltage_index, voltage_folder in enumerate(sorted_voltage_folders):
    data = sum_over_column(overall_directory / voltage_folder, COLUMN_TO_SUM)
    coil_voltage = data["Coil voltage"][0]
    voltages.append(coil_voltage)

    truncated_signal, truncated_mw_pulse_durations, truncated_evolution_times = get_truncated_signal(data, EVOLUTION_TIME_CROP_S, PULSE_DURATION_CROP_S)

    inner_product_settings = InnerProductSettings(
        truncated_mw_pulse_durations,  # pulse durations sampled
        truncated_evolution_times,  # evolution times sampled
        rabi_window=RABI_WINDOW,
        ramsey_window="boxcar",
        subtract_mean=True,
        use_effective_rabi_frequency=False,
    )

    double_inner_product_exp = np.array(
        [
            [
                double_inner_product_exponential(truncated_signal, rabi_hz, ramsey_hz, inner_product_settings)
                for ramsey_hz in RAMSEY_FREQ_RANGE_HZ
            ]
            for rabi_hz in RABI_FREQ_RANGE_HZ
        ]
    )

    imshow_with_extents_and_crop(HZ_TO_MHZ * RAMSEY_FREQ_RANGE_HZ, HZ_TO_MHZ * RABI_FREQ_RANGE_HZ, np.abs(double_inner_product_exp),
        ymin=0,
        ymax=50,
        xmin=0,
        xmax=13,
        vmax=15
    )
    plt.colorbar()

    current_field_projections_t = []
    current_rabi_freqs_hz = []
    current_ramsey_freqs_hz = []
    for axis_index, initial_guess in enumerate(INITIAL_GUESSES[voltage_index]):
        fmin = minimize(inner_product_exp_to_minimize, initial_guess, args=(truncated_signal, inner_product_settings), method="Nelder-Mead")
        harmonic = round(2*fmin.x[RABI_INDEX]/NOMINAL_RABI_FREQS[axis_index])/2
        true_rabi_hz=np.sqrt(fmin.x[RABI_INDEX]**2 - harmonic**2 * fmin.x[RAMSEY_INDEX]**2)/harmonic
        current_rabi_freqs_hz.append(true_rabi_hz)
        current_ramsey_freqs_hz.append(fmin.x[RAMSEY_INDEX])
        print(f"{coil_voltage:.1f} V: Rabi {HZ_TO_MHZ*true_rabi_hz:.2f} MHz, Double Ramsey mI = 1 {HZ_TO_MHZ*(fmin.x[RAMSEY_INDEX]):.2f} MHz")
        current_field_projections_t.append((fmin.x[RAMSEY_INDEX] - 2*f_h*M_I_GUESS)/(2*gammab))

        for rabi_factor in HARMONICS:
            for m_i in M_I_VALUES:
                delta_m_i = m_i - M_I_GUESS
                ramsey = abs(fmin.x[RAMSEY_INDEX] + 2 * delta_m_i * f_h)
                rabi_eff =  rabi_factor*np.sqrt(true_rabi_hz**2 + ramsey**2)
                if rabi_factor == harmonic and delta_m_i  == 0:
                    marker = "x"
                else:
                    marker = "."
                plt.scatter(HZ_TO_MHZ*ramsey, HZ_TO_MHZ*rabi_eff, marker = marker, color = COLORS[axis_index])
    print(f"Field projections: {np.array2string(1e6*np.array(current_field_projections_t), precision=6)} uT")
    plt.title(f"Coil voltage: {coil_voltage:.1f} V")   
    rabi_frequencies_hz.append(current_rabi_freqs_hz)
    field_projections_t.append(current_field_projections_t)
    ramsey_frequencies_hz.append(current_ramsey_freqs_hz)     
    plt.savefig(f"voltage_{coil_voltage:.1f}v.png", dpi=1000)  
    plt.show()

np.savetxt("rabi_frequencies_hz.txt", np.array(rabi_frequencies_hz))
np.savetxt("ramsey_frequencies_hz.txt", np.array(ramsey_frequencies_hz))
np.savetxt("field_projections_t.txt", np.array(field_projections_t))
np.savetxt("voltages.txt", np.array(voltages))





