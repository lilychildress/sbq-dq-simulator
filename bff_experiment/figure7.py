from os import listdir
from scipy.optimize import minimize
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from bff_simulator.abstract_classes.abstract_ensemble import NVOrientation
from scipy.optimize import minimize
from bff_simulator.constants import f_h, gammab, NVaxes_100
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
T_TO_UT = 1e6

RAMSEY_INDEX = 0    # Index of the free evolution frequency in the initial guesses
RABI_INDEX=1        # Index of the pulse duration frequency in the initial guesses

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
NOMINAL_RABI_FREQS = [19.5e6, 23e6, 9e6, 24e6]
COLORS = list(TABLEAU_COLORS.keys())[: len(NVOrientation)]

HARMONICS = [0.5, 1, 1.5, 2]
M_I_VALUES = [-1, 0, 1]
M_I_GUESS = 1               # Initial guesses are all for mI = 1 peaks in the data
VOLTAGE_INDEX_TO_PLOT = 8   # This picks out the data at the 4.0 V coil voltage; others will work.

DO_INNER_PRODUCT_PLOT = True   # Provided because the inner product calculation is slow

overall_directory = Path("/Users/lilianchildress/Documents/GitHub/sbq-dq-simulator/bff_experiment/data/sorted_by_voltage")
sorted_voltage_folders = sorted(listdir(overall_directory))
plt.rcParams["font.size"] = 9
plt.rcParams["font.family"] = "arial"

# Plots the example inner product with orientation identifications
if DO_INNER_PRODUCT_PLOT:
    data = sum_over_column(overall_directory / sorted_voltage_folders[VOLTAGE_INDEX_TO_PLOT], COLUMN_TO_SUM)
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

    double_inner_product_exp = np.array(
        [
            [
                double_inner_product_exponential(truncated_signal, rabi_hz, ramsey_hz, inner_product_settings)
                for ramsey_hz in RAMSEY_FREQ_RANGE_HZ
            ]
            for rabi_hz in RABI_FREQ_RANGE_HZ
        ]
    )

    plt.figure(figsize=(3.37, 3.5))
    imshow_with_extents_and_crop(HZ_TO_MHZ * RAMSEY_FREQ_RANGE_HZ, HZ_TO_MHZ * RABI_FREQ_RANGE_HZ, np.abs(double_inner_product_exp),
        ymin=0,
        ymax=50,
        xmin=0,
        xmax=13,
        vmax=15
    )
    plt.colorbar(orientation="horizontal", location="top", shrink=0.6, label="2D inner product magnitude (a.u.)")

    for axis_index, initial_guess in enumerate(INITIAL_GUESSES[VOLTAGE_INDEX_TO_PLOT]):
        fmin = minimize(inner_product_exp_to_minimize, initial_guess, args=(truncated_signal, inner_product_settings), method="Nelder-Mead")
        harmonic = round(2*fmin.x[RABI_INDEX]/NOMINAL_RABI_FREQS[axis_index])/2
        true_rabi_hz=np.sqrt(fmin.x[RABI_INDEX]**2 - harmonic**2 * fmin.x[RAMSEY_INDEX]**2)/harmonic
        print(f"{coil_voltage:.1f} V: Rabi {HZ_TO_MHZ*true_rabi_hz:.2f} MHz, Double Ramsey mI = 1 {HZ_TO_MHZ*(fmin.x[RAMSEY_INDEX]):.2f} MHz")

        for rabi_factor in HARMONICS:
            for m_i in M_I_VALUES:
                delta_m_i = m_i - M_I_GUESS
                ramsey = abs(fmin.x[RAMSEY_INDEX] + 2 * delta_m_i * f_h)
                rabi_eff =  rabi_factor*np.sqrt(true_rabi_hz**2 + ramsey**2)
                if rabi_factor == harmonic and delta_m_i  == 0:
                    marker = "x"
                else:
                    marker = "."
                plt.scatter(HZ_TO_MHZ*ramsey, HZ_TO_MHZ*rabi_eff, marker = marker, color = COLORS[axis_index], s=15)
    plt.xlabel("Free evolution frequency (MHz)")
    plt.ylabel("Pulse duration frequency (MHz)")
    #plt.title(f"Coil voltage: {coil_voltage:.1f} V")
    plt.gca().set_xlim(-0.5, 13.5)
    plt.gca().set_ylim(-1.5, 51.5)
    plt.gca().xaxis.set_ticks_position("both")
    plt.gca().yaxis.set_ticks_position("both")
    plt.gca().minorticks_on()
    plt.gca().tick_params(direction="in", which="both", width=1.5)
    plt.gca().tick_params(direction="in", which="minor", length=2.5)
    plt.gca().tick_params(direction="in", which="major", length=4)
    plt.gca().text(8, 3, "Coil 4.0 V", color="white", fontsize=9)
    for spine in plt.gca().spines.values():
            spine.set_linewidth(1.25)
    plt.tight_layout()
    plt.savefig(f"voltage_{coil_voltage:.1f}v.svg")  
    plt.show()

##########################################################################
# Load data to be used for the next three plots. This data is generated by
# the bff_experiment/extract_inner_product.py script, which is run on the
# data in the bff_experiment/data/sorted_by_voltage directory.
########################################################################
base_path="/Users/lilianchildress/Documents/GitHub/sbq-dq-simulator/bff_experiment/data/"
rabi_frequencies_hz = np.loadtxt(base_path+"rabi_frequencies_hz.txt")
ramsey_frequencies_hz = np.loadtxt("ramsey_frequencies_hz.txt")
field_projections_t = np.loadtxt("field_projections_t.txt")
voltages = np.loadtxt("voltages.txt")
#########################################################################


######################################################################
# Plot the Rabi frequencies as a function of coil voltage
plt.figure(figsize=(1, 2))
for i, rabi_frequency_fixed_orientation in enumerate(np.transpose(rabi_frequencies_hz)):
    plt.scatter(voltages, HZ_TO_MHZ*rabi_frequency_fixed_orientation, color=COLORS[i])
#plt.gca().set_aspect(1)
plt.gca().yaxis.set_ticks_position("both")
plt.gca().xaxis.set_ticks_position("both")
plt.gca().minorticks_on()
plt.gca().tick_params(direction="in", which="both", width=1.5)
plt.gca().tick_params(direction="in", which="minor", length=2.5)
plt.gca().tick_params(direction="in", which="major", length=4)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.25)
plt.gca().set_xticks([0, 2, 4])
plt.gca().set_xlim(-0.5, 4.5)
plt.xlabel("Coil voltage (V)")
plt.ylabel("Rabi frequency (MHz)")
plt.savefig("rabi_frequency_vs_voltage.svg")
plt.show()

######################################################################
# Plot the field projection on each axis as a function of coil voltage as well as 
# the predictions from a fit to a linear field mode. 


# We first find a linearly-increasing magnetic field that is consistent with our data. The
# field in crystal coordinates is given by (bx0, by0, bz0) + (x_slope, y_slope, z_slope) * coil
# We fit this to our observed projections to extract bx0, by0, bz0, x_slope, y_slope, z_slope
# and compare the predicted field projections to the observed ones. Note that this is not a
# unique inversion (we could use other permutations of the order of the NV axes), but it does 
# confirm that our data is consistent with a linearly increasing field.

def minimize_linear_field_discrepancy_function(crystal_coordinate_field_offsets_slopes, field_projections_all_voltages_ut, nv_orientations, coil_voltages):
    quantity_to_minimize = 0.0
    bx0, by0, bz0, x_slope, y_slope, z_slope = crystal_coordinate_field_offsets_slopes
    for coil_voltage_index, coil_voltage in enumerate(coil_voltages):
        crystal_coordinate_field = np.array([bx0+x_slope*coil_voltage, by0+y_slope*coil_voltage, bz0+z_slope*coil_voltage])
        field_projections_ut = field_projections_all_voltages_ut[coil_voltage_index]
        for i, orientation in enumerate(nv_orientations):
            nv_bz = np.dot(np.array(crystal_coordinate_field), np.array(orientation))
            quantity_to_minimize += (nv_bz**2 - field_projections_ut[i]**2)**2
    return quantity_to_minimize


result = minimize(minimize_linear_field_discrepancy_function, [2, 1, 2, 4, 12, 20], (T_TO_UT*field_projections_t, NVaxes_100, voltages))
bx0, by0, bz0, x_slope, y_slope, z_slope = result.x

# Calculated predicted field projections on each NV axis over a fine range of coil voltages
coil_voltages_fine = np.linspace(0,4,201)
crystal_coordinate_field_vs_voltage_fine = []
predicted_field_projections_fine_ut = []

for coil_voltage in coil_voltages_fine:
    crystal_coordinate_field = np.array([bx0+x_slope*coil_voltage, by0+y_slope*coil_voltage, bz0+z_slope*coil_voltage])
    predicted_field_projections_fine_ut.append(np.abs(NVaxes_100 @ crystal_coordinate_field))
    crystal_coordinate_field_vs_voltage_fine.append(crystal_coordinate_field)

# Plot the measured and predicted field projections
plt.figure(figsize=(3.37, 2))
for i, field_projection_fixed_orientation in enumerate(np.transpose(field_projections_t)):
    plt.scatter(voltages, T_TO_UT*field_projection_fixed_orientation, color=COLORS[i], marker = "x")
    plt.plot(coil_voltages_fine, np.array(predicted_field_projections_fine_ut)[:, i], marker="", color=COLORS[i])
#plt.gca().set_aspect(1)
plt.xlabel("Coil voltage (V)")
plt.ylabel("Field projection magnitude (uT)")
plt.gca().yaxis.set_ticks_position("both")
plt.gca().xaxis.set_ticks_position("both")
plt.gca().minorticks_on()
plt.gca().tick_params(direction="in", which="both", width=1.5)
plt.gca().tick_params(direction="in", which="minor", length=2.5)
plt.gca().tick_params(direction="in", which="major", length=4)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.25)
plt.savefig("field_projection_vs_voltage.svg")
plt.show()

########################################################################
# Plot the magnetic field in crystal coordinates as a function of coil voltage
plt.figure(figsize=(1.5, 2))
field_colors = ["purple", "gray", "blue"]
labels = [r"$B_x$", r"$B_y$", r"$B_z$"]
for i in range(3):
    plt.plot(coil_voltages_fine, np.array(crystal_coordinate_field_vs_voltage_fine)[:,i], marker="", color=field_colors[i], label = labels[i]) 
plt.xlabel("Coil voltage (V)")
plt.ylabel("Possible field in crystal coordinates (uT)")
plt.legend(loc="upper left", fontsize=9)
plt.gca().yaxis.set_ticks_position("both")
plt.gca().xaxis.set_ticks_position("both")
plt.gca().minorticks_on()
plt.gca().tick_params(direction="in", which="both", width=1.5)
plt.gca().tick_params(direction="in", which="minor", length=2.5)
plt.gca().tick_params(direction="in", which="major", length=4)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.25)
plt.savefig("field_in_crystal_coordinates.svg")
plt.show()
