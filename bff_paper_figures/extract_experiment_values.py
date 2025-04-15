import numpy as np
from bff_simulator.constants import NVaxes_100, f_h, gammab, exy, ez
from bff_simulator.vector_manipulation import transform_from_crystal_to_nv_coords, perpendicular_projection
from bff_simulator.offaxis_field_hamiltonian_constructor import OffAxisFieldHamiltonian
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParameters


def get_bare_rabi_frequencies(experiment_parameters: OffAxisFieldExperimentParameters):
    base_rabi_hz = experiment_parameters.rabi_frequency_base_hz
    mw_direction = experiment_parameters.mw_direction
    return [base_rabi_hz * perpendicular_projection(mw_direction, NVaxis) for NVaxis in NVaxes_100]


def get_true_eigenvalues(off_axis_experiment_parameters: OffAxisFieldExperimentParameters):
    larmor_freqs_hz = []
    bz_values_nv_coords_t = []
    for NVaxis in NVaxes_100:
        bz_values_for_this_axis = []
        larmor_freqs_for_this_axis = []
        for m_hyperfine in [-1, 0, 1]:
            nv_axis_unit_vector = NVaxis / np.linalg.norm(NVaxis)

            b_field_vector_t_nv_coords = transform_from_crystal_to_nv_coords(
                np.array(off_axis_experiment_parameters.b_field_vector_t), nv_axis_unit_vector
            ) + np.array([0, 0, m_hyperfine * f_h / gammab])

            bz_values_for_this_axis.append(b_field_vector_t_nv_coords[2])

            e_field_vector_hz = np.diag(np.array([exy, exy, ez])) @ transform_from_crystal_to_nv_coords(
                off_axis_experiment_parameters.e_field_vector_v_per_m, nv_axis_unit_vector
            )

            int_hamiltonian = OffAxisFieldHamiltonian.internal_hamiltonian_bare(
                e_field_vector_hz, b_field_vector_t_nv_coords, off_axis_experiment_parameters.zero_field_splitting_hz
            )
            evals, _ = OffAxisFieldHamiltonian.get_ordered_spin_1_eigensystem(int_hamiltonian)
            larmor_freqs_for_this_axis.append(abs((evals[2] - evals[0]) / (2 * np.pi)))
        bz_values_nv_coords_t.append(bz_values_for_this_axis)
        larmor_freqs_hz.append(larmor_freqs_for_this_axis)
    return larmor_freqs_hz, bz_values_nv_coords_t
