from matplotlib import pyplot as plt
from scipy.optimize import fsolve
import numpy as np

from bff_simulator.constants import NVaxes_100
from bff_simulator.vector_manipulation import perpendicular_projection


NOMINAL_MW_THETA = 0.23957582149174544
NOMINAL_MW_PHI = 0.5244528089360776
RAD_TO_DEGREE = 360/(2*np.pi)
BASE_PATH = "/Users/lilianchildress/Documents/GitHub/sbq-dq-simulator/bff_paper_figures/data/"

##############################################################################
# Plot inversion error vs Rabi frequency (using rabi frequencies extracted from the data
# for the inversion)
RABI_MAX_RANGE = np.linspace(50e6, 150e6, 201)
RUN_LABEL = f"rabi_{min(RABI_MAX_RANGE)*1e-6:.02f}_to_{max(RABI_MAX_RANGE)*1e-6:.02f}_mhz"

errors_freq_nT = list(np.loadtxt(BASE_PATH+f"errors_nt_freq_{RUN_LABEL}.txt"))
errors_time_nT = list(np.loadtxt(BASE_PATH+f"errors_nt_time_{RUN_LABEL}.txt", ))
rabi_maxes = list(np.loadtxt(BASE_PATH+f"rabi_values_{RUN_LABEL}.txt"))

plt.figure(0, figsize=(3.4, 2))
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'arial'

labels = [r"$\langle111\rangle$", r"$\langle1\bar{1}1\rangle$", r"$\langle11\bar{1}\rangle$", r"$\langle\bar{1}11\rangle$"]
plt.plot(1e-6*np.array(rabi_maxes), errors_time_nT, label = labels)
plt.legend(loc="lower right", ncol=2)
plt.xlabel(r"$\Omega_\text{max}/2\pi$ (MHz)")
plt.ylabel("Inversion error (nT)")
plt.gca().yaxis.set_ticks_position("both")
plt.gca().xaxis.set_ticks_position("both")
plt.gca().minorticks_on()
plt.gca().tick_params(direction="in", which = "both", width=1.5)
plt.gca().tick_params(direction="in", which = "minor", length=2.5)
plt.gca().tick_params(direction="in", which = "major", length=5)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.25)
plt.ylim((-2, 2))
plt.tight_layout()
plt.savefig(BASE_PATH+"robustness_vs_mw_mag.svg")
plt.show()

##########################################################################
# Plot inversion error vs MW direction (using rabi frequencies extracted from data)
RUN_LABEL = f"mw_angle_variation_freq_and_time_fit_rabi_798_ns"

errors_freq_vs_angle_nT = list(np.loadtxt(BASE_PATH+f"errors_nt_freq_{RUN_LABEL}.txt"))
errors_time_vs_angle_nT = list(np.loadtxt(BASE_PATH+f"errors_nt_time_{RUN_LABEL}.txt"))
mw_phi_values = list(np.loadtxt(BASE_PATH+f"phi_values_{RUN_LABEL}.txt"))
mw_theta_values = list(np.loadtxt(BASE_PATH+f"theta_values_{RUN_LABEL}.txt") )
problem_phi_values= list(np.loadtxt(BASE_PATH+f"problem_phi_values_{RUN_LABEL}.txt"))
problem_theta_values=list(np.loadtxt(BASE_PATH+f"problem_theta_values_{RUN_LABEL}.txt"))

# This function allows us to calculate the curves where the 3/2 rabi harmonic of the <111> direction intersects the other
# orientation's rabi frequencies.
def rabi_difference_to_crosstalk_harmonic(theta:float, phi:float, crosstalk_orientation: int, harmonic: float, target_orientation: int):
    mw_direction = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    return harmonic*perpendicular_projection(NVaxes_100[crosstalk_orientation], mw_direction) - perpendicular_projection(NVaxes_100[target_orientation], mw_direction)
phi_crosstalk_rad = np.linspace(min(mw_phi_values), max(mw_phi_values), 51)

plotstyle = "log" 
vmin= 0.01
vmax = 10

labels = [r"$\langle111\rangle$", r"$\langle1\bar{1}1\rangle$", r"$\langle11\bar{1}\rangle$", r"$\langle\bar{1}11\rangle$"]
positions = [[21, 4.5],[35, 4.5],[21, 4.5], [35, 4.5]]

plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'arial'
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
fig.set_figheight(3)
fig.set_figwidth(3.4)
for i, ax in enumerate(axes.flat):
    
    im=ax.tripcolor(np.array(mw_phi_values)*RAD_TO_DEGREE, np.array(mw_theta_values)*RAD_TO_DEGREE, np.abs(np.array(errors_time_vs_angle_nT)[:,i]),cmap="inferno", norm=plotstyle, vmin=vmin, vmax=vmax)
    if i == 2 or i == 3:
        ax.set_xlabel("Azimuthal angle (deg.)", fontsize=9)
    if i==0 or i ==2:
        ax.set_ylabel("Polar angle (deg.)", fontsize=9)
    ax.scatter(NOMINAL_MW_PHI*RAD_TO_DEGREE, NOMINAL_MW_THETA*RAD_TO_DEGREE, marker="x", color = "cyan")
    ax.text(positions[i][0], positions[i][1], labels[i], color="black", fontsize=8)
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")
    ax.minorticks_on()
    ax.tick_params(direction="in", which = "both", width=1.25)
    ax.tick_params(direction="in", which = "minor", length=2)
    ax.tick_params(direction="in", which = "major", length=3.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
    if i>0:
        # Find the angles at which the 3/2 harmonic of the <111> orientation intersects the rabi frequency of this orientation
        theta_crosstalk_rad = np.array([fsolve(rabi_difference_to_crosstalk_harmonic, [.5], (phi, 0, 1.5, i))[0] for phi in phi_crosstalk_rad])
        mask = theta_crosstalk_rad < max(mw_theta_values)
        ax.plot(np.array(phi_crosstalk_rad)[mask]*RAD_TO_DEGREE, theta_crosstalk_rad[mask]*RAD_TO_DEGREE, color="red")
        
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.subplots_adjust(top=0.8, right=.99, left=0.14, bottom=0.12)

cbar_ax = fig.add_axes([.25, 0.82, 0.65, 0.03])
cbar =fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Inversion error (nT)", fontsize=9)
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks_position('top')
plt.savefig(BASE_PATH+"robustness_vs_mw_angle.png", dpi=2000)
plt.show()