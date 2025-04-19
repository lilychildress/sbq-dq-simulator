import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.signal import windows

from bff_simulator.abstract_classes.abstract_ensemble import NV14HyperfineField, NVOrientation
from bff_simulator.constants import exy, ez, NVaxes_100, gammab, f_h
from bff_simulator.homogeneous_ensemble import HomogeneousEnsemble
from bff_simulator.liouvillian_solver import LiouvillianSolver
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParametersFactory
from bff_simulator.simulator import Simulation
from bff_simulator.vector_manipulation import transform_from_crystal_to_nv_coords, perpendicular_projection
from bff_simulator.offaxis_field_hamiltonian_constructor import OffAxisFieldHamiltonian

MW_DIRECTION = np.array([0.97203398, 0.2071817, 0.11056978])  # Vincent's old "magic angle"
MW_DIRECTION = np.array([0.20539827217056314, 0.11882075246379901, 0.9714387158093318])  # Lily's new "magic angle"
E_FIELD_VECTOR_V_PER_CM = 0 * np.array([1e5, 3e5, 0]) / exy
B_FIELD_VECTOR_T = [1e6 / (gammab * np.sqrt(3)) * x for x in [1, 1.1, 1.2]]
RABI_FREQ_BASE_HZ = 100e6
DETUNING_HZ = 0e6
SECOND_PULSE_PHASE = 0
MW_PULSE_LENGTH_S = np.linspace(0, 0.5e-6, 1001)
EVOLUTION_TIME_S = np.linspace(0, 15e-6, 801)
T2STAR_S = 2e-6

# func should be either np.cos or np.sin
def inner_product_func_rabi_axis(func, rabi_frequency_hz, mw_pulse_length_s, dq_signal, window=1):
    numerator = np.sum(
        func(2 * np.pi * (rabi_frequency_hz) * mw_pulse_length_s) * window * np.transpose(dq_signal), axis=1
    )
    denominator = np.sum(
        func(2 * np.pi * (rabi_frequency_hz) * mw_pulse_length_s)
        * func(2 * np.pi * (rabi_frequency_hz) * mw_pulse_length_s)
    )
    return numerator / denominator


def inner_product_func_ramsey_axis(func, ramsey_freq_hz, evolution_time_s, ramsey_signal, window=1):
    numerator = np.sum(func(2 * np.pi * (ramsey_freq_hz) * evolution_time_s) * window * ramsey_signal)
    denominator = np.sum(
        func(2 * np.pi * (ramsey_freq_hz) * evolution_time_s) * func(2 * np.pi * (ramsey_freq_hz) * evolution_time_s)
    )
    return numerator / denominator


def double_inner_product(
    rabi_frequency_hz, ramsey_freq_hz, mw_pulse_length_s, evolution_time_s, dq_signal, rabi_window=1, ramsey_window=1
):
    rabi_inner_products = [
        inner_product_func_rabi_axis(func, rabi_frequency_hz, mw_pulse_length_s, dq_signal, window=rabi_window)
        for func in [np.cos, np.sin]
    ]

    all_double_inner_products = np.array(
        [
            [
                inner_product_func_ramsey_axis(
                    func, ramsey_freq_hz, evolution_time_s, rabi_inner_product, window=ramsey_window
                )
                for func in [np.cos, np.sin]
            ]
            for rabi_inner_product in rabi_inner_products
        ]
    ).flatten()

    return np.sqrt(np.sum(all_double_inner_products**2))


def double_inner_product_first_quadrant(
    rabi_frequency_hz, ramsey_freq_hz, mw_pulse_length_s, evolution_time_s, dq_signal, rabi_window=1, ramsey_window=1
):
    rabi_inner_product_cos, rabi_inner_product_sin = [
        inner_product_func_rabi_axis(func, rabi_frequency_hz, mw_pulse_length_s, dq_signal, window=rabi_window)
        for func in [np.cos, np.sin]
    ]

    double_inner_product_cos_cos, double_inner_product_cos_sin = [
        inner_product_func_ramsey_axis(
            func, ramsey_freq_hz, evolution_time_s, rabi_inner_product_cos, window=ramsey_window
        )
        for func in [np.cos, np.sin]
    ]
    double_inner_product_sin_cos, double_inner_product_sin_sin = [
        inner_product_func_ramsey_axis(
            func, ramsey_freq_hz, evolution_time_s, rabi_inner_product_sin, window=ramsey_window
        )
        for func in [np.cos, np.sin]
    ]

    return np.sqrt(
        (double_inner_product_cos_cos - double_inner_product_sin_sin) ** 2
        + (double_inner_product_cos_sin + double_inner_product_sin_cos) ** 2
    )


def double_inner_product_all_terms(
    rabi_frequency_hz, ramsey_freq_hz, mw_pulse_length_s, evolution_time_s, dq_signal, rabi_window=1, ramsey_window=1
):
    rabi_inner_product_cos, rabi_inner_product_sin = [
        inner_product_func_rabi_axis(func, rabi_frequency_hz, mw_pulse_length_s, dq_signal, window=rabi_window)
        for func in [np.cos, np.sin]
    ]

    double_inner_product_cos_cos, double_inner_product_cos_sin = [
        inner_product_func_ramsey_axis(
            func, ramsey_freq_hz, evolution_time_s, rabi_inner_product_cos, window=ramsey_window
        )
        for func in [np.cos, np.sin]
    ]
    double_inner_product_sin_cos, double_inner_product_sin_sin = [
        inner_product_func_ramsey_axis(
            func, ramsey_freq_hz, evolution_time_s, rabi_inner_product_sin, window=ramsey_window
        )
        for func in [np.cos, np.sin]
    ]

    return (
        double_inner_product_cos_cos,
        double_inner_product_sin_sin,
        double_inner_product_cos_sin,
        double_inner_product_sin_cos,
    )


def double_inner_product_mean_subtracted(
    rabi_frequency_hz, ramsey_freq_hz, mw_pulse_length_s, evolution_time_s, dq_signal, rabi_window=1, ramsey_window=1
):
    rabi_inner_products = [
        inner_product_func_rabi_axis(func, rabi_frequency_hz, mw_pulse_length_s, dq_signal, window=rabi_window)
        for func in [np.cos, np.sin]
    ]

    rabi_inner_products = [
        rabi_inner_products[i] - np.mean(rabi_inner_products[i]) for i in range(len(rabi_inner_products))
    ]
    all_double_inner_products = np.array(
        [
            [
                inner_product_func_ramsey_axis(
                    func, ramsey_freq_hz, evolution_time_s, rabi_inner_product, window=ramsey_window
                )
                for func in [np.cos, np.sin]
            ]
            for rabi_inner_product in rabi_inner_products
        ]
    ).flatten()

    return np.sqrt(np.sum(all_double_inner_products**2))


def inner_product_to_minimize(
    ramsey_freq_hz, rabi_frequency_hz, evolution_time_s, mw_pulse_length_s, dq_signal, rabi_window, ramsey_window
):
    return 1 / double_inner_product(
        rabi_frequency_hz,
        ramsey_freq_hz,
        mw_pulse_length_s,
        evolution_time_s,
        dq_signal,
        rabi_window=rabi_window,
        ramsey_window=ramsey_window,
    )


nv_ensemble = HomogeneousEnsemble()
nv_ensemble.efield_splitting_hz = np.linalg.norm(E_FIELD_VECTOR_V_PER_CM) * exy
nv_ensemble.t2_star_s = T2STAR_S
# nv_ensemble.add_nv_single_species(NVOrientation.A, NV14HyperfineField.N14_plus)
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
off_axis_experiment_parameters = exp_param_factory.get_experiment_parameters()

off_axis_solver = LiouvillianSolver()

off_axis_simulation_0_phase = Simulation(off_axis_experiment_parameters, nv_ensemble, off_axis_solver)

exp_param_factory.set_second_pulse_phase(np.pi)
off_axis_experiment_parameters = exp_param_factory.get_experiment_parameters()
off_axis_simulation_pi_phase = Simulation(off_axis_experiment_parameters, nv_ensemble, off_axis_solver)

sq_cancelled_signal = off_axis_simulation_0_phase.ms0_results + off_axis_simulation_pi_phase.ms0_results

# Calculate the eigenvalue difference that we'd like to extract
larmor_freqs_hz = []
bz_values_nv_coords_t = []
for NVaxis in NVaxes_100:
    nv_axis_unit_vector = NVaxis / np.linalg.norm(NVaxis)

    b_field_vector_t_nv_coords = transform_from_crystal_to_nv_coords(
        np.array(B_FIELD_VECTOR_T), nv_axis_unit_vector
    ) + np.array([0, 0, f_h / gammab])

    bz_values_nv_coords_t.append(b_field_vector_t_nv_coords[2])

    e_field_vector_hz = np.diag(np.array([exy, exy, ez])) @ transform_from_crystal_to_nv_coords(
        E_FIELD_VECTOR_V_PER_CM, nv_axis_unit_vector
    )

    int_hamiltonian = OffAxisFieldHamiltonian.internal_hamiltonian_bare(
        e_field_vector_hz, b_field_vector_t_nv_coords, off_axis_experiment_parameters.zero_field_splitting_hz
    )
    evals, evects = OffAxisFieldHamiltonian.get_ordered_spin_1_eigensystem(int_hamiltonian)
    h_int_eigenbasis_rwa = OffAxisFieldHamiltonian.internal_hamiltonian_eigenbasis_rwa(
        evals, off_axis_experiment_parameters.zero_field_splitting_hz + off_axis_experiment_parameters.detuning_hz
    )
    larmor_freqs_hz.append(abs(np.real_if_close((evals[2] - evals[0]) / (2 * np.pi), tol=1e10)))

# Find the Rabi frequencies
rabi_frequencies = [RABI_FREQ_BASE_HZ * perpendicular_projection(MW_DIRECTION, NVaxis) for NVaxis in NVaxes_100]

rabi_window = 1  # windows.blackman(len(MW_PULSE_LENGTH_S))
ramsey_window = np.exp(-EVOLUTION_TIME_S / T2STAR_S)

optimal_value = minimize(
    inner_product_to_minimize,
    [2 * (gammab * bz_values_nv_coords_t[0] + f_h)],
    (rabi_frequencies[0], EVOLUTION_TIME_S, MW_PULSE_LENGTH_S, sq_cancelled_signal, rabi_window, ramsey_window),
    method="Nelder-Mead",
)

ramsey_freqs_fine = np.linspace(0 * larmor_freqs_hz[0], 1.25 * larmor_freqs_hz[0], 501)
total_inner_product_1 = [
    double_inner_product(
        rabi_frequencies[0],
        ramsey_freq,
        MW_PULSE_LENGTH_S,
        EVOLUTION_TIME_S,
        sq_cancelled_signal,
        ramsey_window=ramsey_window,
        rabi_window=rabi_window,
    )
    for ramsey_freq in ramsey_freqs_fine
]
total_inner_product_2 = [
    double_inner_product_first_quadrant(
        rabi_frequencies[0],
        ramsey_freq,
        MW_PULSE_LENGTH_S,
        EVOLUTION_TIME_S,
        sq_cancelled_signal,
        ramsey_window=ramsey_window,
        rabi_window=rabi_window,
    )
    for ramsey_freq in ramsey_freqs_fine
]
plt.plot(ramsey_freqs_fine, total_inner_product_1)
plt.plot(ramsey_freqs_fine, total_inner_product_2)
plt.scatter(
    [larmor_freqs_hz[0]],
    [
        double_inner_product(
            rabi_frequencies[0],
            larmor_freqs_hz[0],
            MW_PULSE_LENGTH_S,
            EVOLUTION_TIME_S,
            sq_cancelled_signal,
            ramsey_window=ramsey_window,
            rabi_window=rabi_window,
        )
    ],
)
plt.scatter(
    optimal_value.x,
    [
        double_inner_product(
            rabi_frequencies[0],
            (optimal_value.x)[0],
            MW_PULSE_LENGTH_S,
            EVOLUTION_TIME_S,
            sq_cancelled_signal,
            ramsey_window=ramsey_window,
            rabi_window=rabi_window,
        )
    ],
)
print(1e9 * (optimal_value.x[0] - larmor_freqs_hz[0]) / (2 * gammab))
print(1e9 * (bz_values_nv_coords_t[0] - larmor_freqs_hz[0] / (2 * gammab)))
plt.show()
