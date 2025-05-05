from pandas import read_csv
from os import listdir
from pathlib import Path
from pandas import DataFrame
import numpy as np

COLUMN_TO_SUM = "Photoluminescence (Arb. Units)"
MW_PULSE_DURATION_LABEL = "Pulse duration (ns)"
EVOLUTION_TIME_LABEL = "Interpulse duration (ns)"
SIGNAL_LABEL = "Photoluminescence (Arb. Units)"
NS_TO_S = 1e-9

def sum_over_column(path_to_directory: Path, column_to_sum: str):
    paths = listdir(path_to_directory)
    for index, data_folder_path in enumerate(paths):
        if index == 0:
            data = read_csv(path_to_directory / data_folder_path)
        else:
            new_data = read_csv(path_to_directory / data_folder_path)
            data[column_to_sum] = data[column_to_sum] + new_data[column_to_sum]
        
    return data


def get_truncated_signal(data:DataFrame, evolution_time_crop_s:float, pulse_duration_crop_s:float):
    mw_pulse_durations_s = NS_TO_S*data[MW_PULSE_DURATION_LABEL].to_numpy()
    evolution_times_s = NS_TO_S*data[EVOLUTION_TIME_LABEL].to_numpy()
    dq_signal = data[SIGNAL_LABEL].to_numpy()

    unique_mw_pulse_durations = np.unique(mw_pulse_durations_s)
    unique_evolution_times = np.unique(evolution_times_s)
    dq_signal_2darray = np.reshape(dq_signal, (len(unique_mw_pulse_durations), len(unique_evolution_times)))

    ramsey_mask = unique_evolution_times < evolution_time_crop_s
    rabi_mask = unique_mw_pulse_durations < pulse_duration_crop_s
    truncated_signal = dq_signal_2darray[rabi_mask][:,ramsey_mask]
    truncated_mw_pulse_durations = unique_mw_pulse_durations[rabi_mask]
    truncated_evolution_times = unique_evolution_times[ramsey_mask]

    return truncated_signal, truncated_mw_pulse_durations, truncated_evolution_times
