import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve

from bff_paper_figures.shared_parameters import BASE_PATH, RAD_TO_DEGREE
from bff_simulator.constants import NVaxes_100
from bff_simulator.vector_manipulation import perpendicular_projection
from bff_paper_figures.Figure5_inversion import get_rms_error_within_range

NOMINAL_MW_THETA = 0.23957582149174544
NOMINAL_MW_PHI = 0.5244528089360776

##############################################################################
# This script loads data and generates Fig. 6a and 6b from it.
# Relevant data is produced by "robustness_vs_mw_magnitude.py" (for Fig. 6a)
# and "robustness_vs_mw_angle.py" (for Fig. 6b)
##############################################################################


# This function allows us to calculate the curves where the 3/2 rabi harmonic of the <111> direction intersects the other
# orientation's rabi frequencies. For a given angle (theta, phi), it returns the difference between the
# rabi frequency of the target orientation and the rabi harmonic of the crosstalk orientation.
def rabi_difference_to_crosstalk_harmonic(
    theta: float, phi: float, crosstalk_orientation: int, harmonic: float, target_orientation: int
):
    mw_direction = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
    return harmonic * perpendicular_projection(
        NVaxes_100[crosstalk_orientation], mw_direction
    ) - perpendicular_projection(NVaxes_100[target_orientation], mw_direction)


##############################################################################
# Plot Fig. 6a, inversion error vs Rabi frequency

# Load relevant data
RUN_LABEL = "rabi_50.00_to_150.00_mhz"

errors_freq_nt = list(np.loadtxt(BASE_PATH + f"errors_nt_freq_{RUN_LABEL}.txt"))
errors_time_nt = list(
    np.loadtxt(
        BASE_PATH + f"errors_nt_time_{RUN_LABEL}.txt",
    )
)
rabi_maxes = list(np.loadtxt(BASE_PATH + f"rabi_values_{RUN_LABEL}.txt"))

# Plot the data
plt.figure(0, figsize=(3.4, 2))
plt.rcParams["font.size"] = 9
plt.rcParams["font.family"] = "arial"

labels = [
    r"$\langle111\rangle$",
    r"$\langle1\bar{1}1\rangle$",
    r"$\langle11\bar{1}\rangle$",
    r"$\langle\bar{1}11\rangle$",
]
plt.plot(1e-6 * np.array(rabi_maxes), errors_time_nt, label=labels)
plt.legend(loc="lower right", ncol=2)
plt.xlabel(r"$\Omega_\text{max}/2\pi$ (MHz)")
plt.ylabel("Inversion error (nT)")
plt.gca().yaxis.set_ticks_position("both")
plt.gca().xaxis.set_ticks_position("both")
plt.gca().minorticks_on()
plt.gca().tick_params(direction="in", which="both", width=1.5)
plt.gca().tick_params(direction="in", which="minor", length=2.5)
plt.gca().tick_params(direction="in", which="major", length=5)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.25)
plt.ylim((-2, 2))
plt.tight_layout()
plt.savefig(BASE_PATH + "robustness_vs_mw_mag.svg")
plt.show()

##########################################################################
# Plot Fig. 6b, inversion error vs MW direction

# Load relevant data
RUN_LABEL = "mw_angle_variation_freq_and_time_fit_rabi_798_ns"

errors_freq_vs_angle_nt = list(np.loadtxt(BASE_PATH + f"errors_nt_freq_{RUN_LABEL}.txt"))
errors_time_vs_angle_nt = list(np.loadtxt(BASE_PATH + f"errors_nt_time_{RUN_LABEL}.txt"))
mw_phi_values = list(np.loadtxt(BASE_PATH + f"phi_values_{RUN_LABEL}.txt"))
mw_theta_values = list(np.loadtxt(BASE_PATH + f"theta_values_{RUN_LABEL}.txt"))
problem_phi_values = list(np.loadtxt(BASE_PATH + f"problem_phi_values_{RUN_LABEL}.txt"))
problem_theta_values = list(np.loadtxt(BASE_PATH + f"problem_theta_values_{RUN_LABEL}.txt"))

# values of phi that will be used to calculate theta values where we expect large crosstalk
phi_crosstalk_rad = np.linspace(min(mw_phi_values), max(mw_phi_values), 51)

# define z axis of plot
plotstyle = "log"
vmin = 0.01
vmax = 10

# define labels and the positions at which they will be written
labels = [
    r"$\langle111\rangle$",
    r"$\langle1\bar{1}1\rangle$",
    r"$\langle11\bar{1}\rangle$",
    r"$\langle\bar{1}11\rangle$",
]
positions = [[21, 4.5], [35, 4.5], [21, 4.5], [35, 4.5]]


# Define the ranges over which we will calculate the RMS error
theta_ranges = [[8, 15], [8, 15], [8, 15], [8, 15]]
phi_ranges = [[21, 33], [21, 33], [21, 33], [21, 33]]

# Calculate the rms error and define how and where it should be plotted
errors = [
    get_rms_error_within_range(
        theta_ranges[i],
        phi_ranges[i],
        np.array(mw_phi_values),
        np.array(mw_theta_values),
        np.array(errors_time_vs_angle_nt),
        i,
    )
    for i in range(4)
]
error_labels = [f"{errors[i]:.2f}\nnT-rms" for i in range(4)]
error_positions = [[22, 9], [22, 9], [22, 9], [22, 9]]

# plot all orientations
plt.rcParams["font.size"] = 9
plt.rcParams["font.family"] = "arial"
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
fig.set_figheight(3)
fig.set_figwidth(3.4)
for i, ax in enumerate(axes.flat):
    im = ax.tripcolor(
        np.array(mw_phi_values) * RAD_TO_DEGREE,
        np.array(mw_theta_values) * RAD_TO_DEGREE,
        np.abs(np.array(errors_time_vs_angle_nt)[:, i]),
        cmap="inferno",
        norm=plotstyle,
        vmin=vmin,
        vmax=vmax,
    )

    # Add axis labels
    if i == 2 or i == 3:
        ax.set_xlabel("Azimuthal angle (deg.)", fontsize=9)
    if i == 0 or i == 2:
        ax.set_ylabel("Polar angle (deg.)", fontsize=9)

    # Plot the angle at which Fig. 6a is calculated
    ax.scatter(NOMINAL_MW_PHI * RAD_TO_DEGREE, NOMINAL_MW_THETA * RAD_TO_DEGREE, marker="x", color="cyan")

    # Add the orientation labels
    ax.text(positions[i][0], positions[i][1], labels[i], color="black", fontsize=8)

    # Beautify the ticks
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")
    ax.minorticks_on()
    ax.tick_params(direction="in", which="both", width=1.25)
    ax.tick_params(direction="in", which="minor", length=2)
    ax.tick_params(direction="in", which="major", length=3.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)

    # Add in lines and rms errors
    ax.text(error_positions[i][0], error_positions[i][1], error_labels[i], color="white", fontsize=8)
    ax.vlines(phi_ranges[i], [theta_ranges[i][0]], [theta_ranges[i][1]], linestyle="dotted", color="white")
    ax.hlines(theta_ranges[i], [phi_ranges[i][0]], [phi_ranges[i][1]], linestyle="dotted", color="white")

    # Plot crosstalk
    if i > 0:
        # Find the angles at which the 3/2 harmonic of the <111> orientation intersects the rabi frequency of this orientation
        theta_crosstalk_rad = np.array(
            [fsolve(rabi_difference_to_crosstalk_harmonic, [0.5], (phi, 0, 1.5, i))[0] for phi in phi_crosstalk_rad]
        )

        # Only plot values that will actually fit in the plot
        mask = theta_crosstalk_rad < max(mw_theta_values)
        ax.plot(
            np.array(phi_crosstalk_rad)[mask] * RAD_TO_DEGREE, theta_crosstalk_rad[mask] * RAD_TO_DEGREE, color="red"
        )

# Adjust layout
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.subplots_adjust(top=0.8, right=0.99, left=0.14, bottom=0.12)

# Add a shared colorbar
cbar_ax = fig.add_axes([0.25, 0.82, 0.65, 0.03])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Inversion error (nT)", fontsize=9)
cbar.ax.xaxis.set_label_position("top")
cbar.ax.xaxis.set_ticks_position("top")

plt.savefig(BASE_PATH + "robustness_vs_mw_angle.png", dpi=2000)
plt.show()
