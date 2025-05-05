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

RAMSEY_FREQ_RANGE_HZ = np.linspace(0, 13e6, 151)
RABI_FREQ_RANGE_HZ = np.linspace(0, 50e6, 151)

EVOLUTION_TIME_CROP_S = 3e-6
PULSE_DURATION_CROP_S = 2e-6
RABI_WINDOW = "cosine"

RAMSEY_INDEX = 0
RABI_INDEX=1

INITIAL_GUESSES = [
    [[[4.8e6,40e6],[4.8e6,20e6]], [[4.6e6,22e6]], [[4.7e6,15e6]], [[5e6,23.5e6]]],
    [[[5.e6,20e6],[5.e6,40e6]],[[5e6,23e6],[5e6,45e6]],[[4.6e6,15e6],[4.6e6,20e6]],[[4.3e6,25e6],[4.6e6,47e6]]],
    [[[6.2e6,40e6],[6.2e6,30e6],[6.2e6,20e6]],[[6e6,23e6],[5.9e6,46e6]],[[5e6,20e6]], [[4.5e6,24e6]]],
    [[[6.2e6,30e6],[6.2e6,41e6]],[[6e6,23e6]],[[5e6,14e6],[4.8e6,20e6]],[[4.8e6,24e6]]],
    [[[7.5e6,42e6], [7.5e6,31e6]],[[7.2e6,47e6],[7.2e6,34e6]],[[5e6,15e6],[4.8e6,5e6]],[[5e6,25e6]]],
    [[[8.2e6,42e6],[8.2e6,20e6]],[[8e6,47e6],[8.2e6,34e6]],[[5e6,15e6],[5e6,20e6]],[[5.8e6,48e6]]],
    [[[9.e6,42e6],[9.e6,32e6],[9.e6,21e6]],[[8e6,47e6],[8e6,36e6],[8e6,24e6]],[[5.5e6,15e6],[5.3e6,5e6],[5.3e6,20e6],[5.3e6,10e6]], [[6e6,25e6]]],
    [[[10.e6,22e6],[9.8e6,44e6],[9.8e6,33e6]],[[9.5e6,47e6],[9.5e6,35e6],[9.5e6,23e6]],[[5.5e6,15e6],[5.5e6,5e6]],[[6e6,24e6],[6e6,12e6]]],
    [[[10.57e6,20e6],[10.57e6,32e6],[10.57e6,43.5e6]],[[9.5e6,45e6],[9.5e6,35e6],[9.5e6,24e6]],[[5.7e6,15e6],[5.5e6,5e6],[5.45e6, 10e6]],[[6.3e6,49e6]]], 
]
NOMINAL_RABI_FREQS = [19.5e6, 23e6, 9e6, 24e6]
COLORS = ["green", "red", "blue", "yellow"]

HARMONICS = [0.5, 1, 1.5, 2]
M_I_VALUES = [-1, 0, 1]
M_I_GUESS = 1

overall_directory = Path("/Users/lilianchildress/Documents/GitHub/sbq-dq-simulator/bff_experiment/data/sorted_by_voltage")
sorted_voltage_folders = sorted(listdir(overall_directory))

for voltage_index, voltage_folder in enumerate(sorted_voltage_folders):
    data = sum_over_column(overall_directory / voltage_folder, COLUMN_TO_SUM)
    coil_voltage = data["Coil voltage"][0]

    truncated_signal, truncated_mw_pulse_durations, truncated_evolution_times = get_truncated_signal(data, EVOLUTION_TIME_CROP_S, PULSE_DURATION_CROP_S)

    inner_product_settings = InnerProductSettings(
        truncated_mw_pulse_durations,  # pulse durations sampled
        truncated_evolution_times,  # evolution times sampled
        rabi_window=RABI_WINDOW,
        ramsey_window="boxcar",
        subtract_mean=True,
        use_effective_rabi_frequency=False,
    )

    for axis_index, initial_guesses in enumerate(INITIAL_GUESSES[voltage_index]):

        for initial_guess in initial_guesses:
            fmin = minimize(inner_product_exp_to_minimize, initial_guess, args=(truncated_signal, inner_product_settings), method="Nelder-Mead")
            harmonic = round(2*fmin.x[RABI_INDEX]/NOMINAL_RABI_FREQS[axis_index])/2
            true_rabi_hz=np.sqrt(fmin.x[RABI_INDEX]**2 - harmonic**2 * fmin.x[RAMSEY_INDEX]**2)/harmonic
            ms0_ramsey_hz = (fmin.x[RAMSEY_INDEX] - 2*f_h*M_I_GUESS)/(2*gammab)
            plt.scatter(coil_voltage, ms0_ramsey_hz, color=COLORS[axis_index], label=f"Rabi: {fmin.x[RABI_INDEX]/1e6:.2f} MHz, Ramsey: {ms0_ramsey_hz/1e6:.2f} MHz")
      
plt.show()



