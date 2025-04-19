import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colorbar
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.optimize import fsolve

from bff_simulator.constants import f_h, gammab, exy, NVaxes_100

from bff_simulator.abstract_classes.abstract_ensemble import NVOrientation


RAD_TO_DEGREE = 360/(2*np.pi)
T_TO_UT = 1e6
PHI_RANGE_HALF_HYPERFINE = np.linspace(1.4607, 3.25133, 301)
B_MAGNITUDE_T = 50e-6
RABI_FREQUENCIES = [np.float64(66364423.69292802), np.float64(79174946.48739058), np.float64(92756072.06279232), np.float64(85965509.82093404)]

B_THETA_FIG4 =  3*np.pi/8
B_PHI_FIG4 = 13*np.pi/16

BASE_PATH = "/Users/lilianchildress/Documents/GitHub/sbq-dq-simulator/bff_paper_figures/data/"
RUN_LABEL = f"b_{1e6*B_MAGNITUDE_T:.0f}_ut_t2s_2_us_fine_3us_ramsey_800ns_rabi"

def half_hyperfine(theta, phi):
    return (-np.cos(theta)-np.cos(phi)*np.sin(theta) + np.sin(theta)*np.sin(phi))/np.sqrt(3) - f_h/(2*B_MAGNITUDE_T*gammab)

def get_rms_error_within_range( theta_range,phi_range, phi_values_total, theta_values_total, errors_vs_b_nT_total, orientation):
    masks = np.array([theta_values_total*360/(2*np.pi) < theta_range[1], theta_values_total*360/(2*np.pi) > theta_range[0], phi_values_total*360/(2*np.pi) < phi_range[1],phi_values_total*360/(2*np.pi) > phi_range[0]])
    mask = np.all(masks, axis=0)
    errors_vs_b_nT_masked = errors_vs_b_nT_total[mask]
    return np.sqrt(np.mean(errors_vs_b_nT_masked[:,orientation]**2))

def plot_errors_all_orientations(theta_values_total, phi_values_total, errors_vs_b_nT_total, plot_half_hyperfine=False, log_plot=True, vmin= 0.01, vmax=300):
    plotstyle = "log" if log_plot else "linear"

    theta_hh_rad = np.array([fsolve(half_hyperfine, [2.5], (phi))[0] for phi in PHI_RANGE_HALF_HYPERFINE])
    theta_hh_rad_low = np.array([fsolve(half_hyperfine, [1.5], (phi))[0] for phi in PHI_RANGE_HALF_HYPERFINE])
    
    theta_ranges = [[90,180],[90,180],[0,90],[0,90]]
    phi_ranges = [[155, 295], [155-90, 295-90],[155, 295], [155-90, 295-90]]
    labels = [r"$\langle111\rangle$", r"$\langle1\bar{1}1\rangle$", r"$\langle11\bar{1}\rangle$", r"$\langle\bar{1}11\rangle$"]
    rabis = [f"{1e-6*RABI_FREQUENCIES[i]:.0f} MHz" for i in range(4)]
    positions = [[2,4],[2,4],[2,155], [2, 155]]
    rabi_positions= [[245,4],[245,4],[245,155], [245, 155]]
    errors = [get_rms_error_within_range(theta_ranges[i], phi_ranges[i], np.array(phi_values_total), np.array(theta_values_total), np.array(errors_vs_b_nT_total), i) for i in range(4)]
    error_labels = [f"{errors[i]:.2f}\nnT-rms" for i in range(4)]
    error_positions=[[185, 120], [85, 120], [185, 15], [85, 15]]

    plt.rcParams['font.size'] = 9
    plt.rcParams['font.family'] = 'arial'
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    fig.set_figheight(3)
    fig.set_figwidth(3.4)
    for i, ax in enumerate(axes.flat):
        im=ax.tripcolor(np.array(phi_values_total)*RAD_TO_DEGREE, np.array(theta_values_total)*RAD_TO_DEGREE, np.abs(np.array(errors_vs_b_nT_total)[:,i]),cmap="inferno", norm=plotstyle, vmin=vmin, vmax=vmax)
        if i == 3:
            ax.scatter(B_PHI_FIG4*RAD_TO_DEGREE, B_THETA_FIG4*RAD_TO_DEGREE, marker="x", color="cyan")
        if i == 2 or i == 3:
            ax.set_xlabel("Azimuthal angle (deg.)", fontsize=9)
        if i==0 or i ==2:
            ax.set_ylabel("Polar angle (deg.)", fontsize=9)
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_ticks_position("both")
        ax.minorticks_on()
        ax.tick_params(direction="in", which = "both", width=1.25)
        ax.tick_params(direction="in", which = "minor", length=2)
        ax.tick_params(direction="in", which = "major", length=3.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.25)
        ax.text(positions[i][0], positions[i][1], labels[i], color="white", fontsize=8)
        ax.text(rabi_positions[i][0], rabi_positions[i][1], rabis[i], color="white", fontsize=8)
        ax.text(error_positions[i][0], error_positions[i][1], error_labels[i], color="white", fontsize=8)
        rms_error = get_rms_error_within_range(theta_ranges[i], phi_ranges[i], np.array(phi_values_total), np.array(theta_values_total), np.array(errors_vs_b_nT_total), i)
        ax.vlines(phi_ranges[i], [theta_ranges[i][0]],[theta_ranges[i][1]], linestyle="dotted", color="white")
        ax.hlines(theta_ranges[i], [phi_ranges[i][0]], [phi_ranges[i][1]], linestyle="dotted", color="white")

        if i==1:
            phi_range_full = np.linspace(0, 2*np.pi, 101)
            theta_no_projection = np.atan2(1, np.sin(phi_range_full) - np.cos(phi_range_full))
            ax.plot(phi_range_full*RAD_TO_DEGREE, theta_no_projection * RAD_TO_DEGREE, color="white", linestyle="dashdot")

            if plot_half_hyperfine:
                ax.plot(PHI_RANGE_HALF_HYPERFINE*RAD_TO_DEGREE, theta_hh_rad*RAD_TO_DEGREE, color="white", linestyle="dashed")
                ax.plot(PHI_RANGE_HALF_HYPERFINE*RAD_TO_DEGREE, theta_hh_rad_low*RAD_TO_DEGREE, color="white", linestyle="dashed")
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.subplots_adjust(top=0.8, right=.99, left=0.14, bottom=0.12)
    cbar_ax = fig.add_axes([.25, 0.82, 0.65, 0.03])
    cbar =fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(f"Inversion error (nT) at {T_TO_UT*B_MAGNITUDE_T:.0f} uT", fontsize=9)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_ticks_position('top')


errors_vs_b_freq_domain_nT = list(np.loadtxt(BASE_PATH+f"errors_nt_freq_{RUN_LABEL}.txt"))
errors_vs_b_time_domain_nT = list(np.loadtxt(BASE_PATH+f"errors_nt_time_{RUN_LABEL}.txt", ))
phi_values = list(np.loadtxt(BASE_PATH+f"phi_values_{RUN_LABEL}.txt"))
theta_values = list(np.loadtxt(BASE_PATH+f"theta_values_{RUN_LABEL}.txt"))
#plot_errors_all_orientations(theta_values, phi_values, errors_vs_b_freq_domain_nT, True)
#plt.show()
plot_errors_all_orientations(theta_values, phi_values, errors_vs_b_time_domain_nT, False, True, vmin=0.01, vmax=10)
plt.savefig(BASE_PATH+"time_domain_inversion.png", dpi=2000)
plt.show()