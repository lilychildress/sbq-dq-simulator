from os import listdir
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from pandas import read_csv
from bff_paper_figures.shared_parameters import HZ_TO_MHZ
from bff_paper_figures.imshow_extensions import imshow_with_extents_and_crop

KHZ_TO_HZ = 1e3
NS_TO_S = 1e-9

FREQUENCY_LABEL = 'Frequency (kHz)'
MW_PULSE_LABEL = 'Pulse duration (ns)'
DATA_LABEL = 'Photoluminescence'
FMIN = 2.8680e+09   # Minimum frequency to include in Rabi data
FMAX = 2.8680e+09   # Maximum frequency to include in Rabi data
NORMALIZATION = .5e5   # Normalization factor for the FFT
ANGLE_OFFSET = 55   # Offset to add to the angle for plotting
CHOSEN_ANGLE = 18 + ANGLE_OFFSET 

def get_angle(path:Path):
    current_angle = int(path.split('degrees.csv')[0])
    if current_angle > 135:
        current_angle -= 360
    return current_angle + ANGLE_OFFSET

path_to_directory = Path("/Users/lilianchildress/Documents/GitHub/sbq-dq-simulator/bff_experiment/data/rabi_chevrons/")
paths = listdir(path_to_directory)
paths.sort(key = get_angle)

angles = []
ft_vs_angle = []
for path in paths: 
    current_angle = get_angle(path)
    angles.append(current_angle)

    data = read_csv(path_to_directory / path)
    frequencies_hz = KHZ_TO_HZ*data[FREQUENCY_LABEL].to_numpy()
    mw_pulse_s = NS_TO_S*data[MW_PULSE_LABEL].to_numpy()
    rabi_signal = data[DATA_LABEL].to_numpy()

    unique_freqs_hz = np.unique(frequencies_hz)
    unique_mw_pulse_s = np.unique(mw_pulse_s)
    rabi_signal_reshaped = rabi_signal.reshape(len(unique_freqs_hz), len(unique_mw_pulse_s))

    mw_pulse_freqs_hz = np.fft.rfftfreq(len(unique_mw_pulse_s), unique_mw_pulse_s[1] - unique_mw_pulse_s[0])
    ft = np.fft.rfft(rabi_signal_reshaped, axis=1)
    #if current_angle == 170:
    #    mask = np.all(np.array([(2.8675e+09 <= unique_freqs_hz), unique_freqs_hz <= 2.8680e+09]), axis=0)
    #else:
    mask = np.all(np.array([(FMIN <= unique_freqs_hz), unique_freqs_hz <= FMAX]), axis=0)
    mw_mask = np.all(np.array([(7e6 <= mw_pulse_freqs_hz), mw_pulse_freqs_hz <= 7.5e6]), axis=0)
    normalization = np.abs(np.mean(ft[mask][:, mw_mask]))
    ft_vs_angle.append(np.mean(ft[mask], axis=0))

plt.rcParams["font.size"] = 9
plt.rcParams["font.family"] = "arial"
plt.figure(figsize=(3.5, 2.5))
imshow_with_extents_and_crop(np.array(angles), 4*HZ_TO_MHZ* mw_pulse_freqs_hz, np.transpose(np.abs(np.array(ft_vs_angle)))/NORMALIZATION, vmax=4, ymax=30, aspect_ratio=0.6, cmap="viridis")
plt.colorbar(label='Rabi oscillation FFT (a.u.)', shrink = 0.6)
plt.xlabel('Rotation offset (degrees)')
plt.ylabel('Rabi frequency (MHz)')
plt.vlines(CHOSEN_ANGLE, -1.5, 31.5, color='white', linestyle='dotted', lw=1.5)
plt.gca().set_xlim(-10, 200)
plt.gca().set_ylim(-1.5, 31.5)
plt.gca().xaxis.set_ticks_position("both")
plt.gca().yaxis.set_ticks_position("both")
plt.gca().minorticks_on()
plt.gca().tick_params(direction="in", which="both", width=1.5)
plt.gca().tick_params(direction="in", which="minor", length=2.5)
plt.gca().tick_params(direction="in", which="major", length=4)
for spine in plt.gca().spines.values():
        spine.set_linewidth(1.25)
plt.tight_layout()
plt.savefig("figure7b.svg", bbox_inches='tight')
plt.show()