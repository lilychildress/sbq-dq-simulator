from enum import IntEnum
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from bff_simulator.abstract_classes.abstract_ensemble import NVSpecies
from bff_simulator.abstract_classes.abstract_solver import Solver
from bff_simulator.constants import exy, ez
from bff_simulator.offaxis_field_experiment_parameters import OffAxisFieldExperimentParameters
from bff_simulator.offaxis_field_hamiltonian_constructor import OffAxisFieldHamiltonian
from bff_simulator.offaxis_field_solver_parameters import OffAxisFieldSolverParam
from bff_simulator.propagator_manipulation import expand_dq_unitaries, generate_propagators_from_hamiltonian
from bff_simulator.vector_manipulation import (
    get_vector_from_vpar_vperp_and_angle,
    transform_from_crystal_to_nv_coords,
)


class _Indices(IntEnum):
    x = 0
    y = 1
    z = 2


class OffAxisFieldSolver(Solver):
    SOLVER_KEY = 4.0

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def uniformly_spaced(test_array: NDArray, relative_tolerance: float = 1e-6) -> bool:
        if len(test_array) < 2:
            raise ValueError("Array must contain more than 1 element")
        first_diff = test_array[1] - test_array[0]
        uniform_spacings = np.isclose(
            np.diff(test_array), first_diff, rtol=relative_tolerance, atol=relative_tolerance * first_diff
        )
        return bool(uniform_spacings.all())

    @staticmethod
    def norm_in_xy_plane(vector: NDArray) -> float:
        assert len(vector) >= 2, "Need at least a two-dimensional vector to take norm in xy plane"
        return np.sqrt(vector[_Indices.x] ** 2 + vector[_Indices.y] ** 2)

    @staticmethod
    def validate_uniform_spacing(array_1d: NDArray, array_name: str = "array") -> None:
        if len(array_1d) > 2:
            if not OffAxisFieldSolver.uniformly_spaced(np.array(array_1d)):
                raise ValueError(f"Off Axis Field Solver requires uniform spacing of {np.array}")

    @staticmethod
    def construct_solver_parameters(
        experiment_parameters: OffAxisFieldExperimentParameters,  # type: ignore[override]
        nv_species: NVSpecies,
    ) -> OffAxisFieldSolverParam:
        nv_axis_unit_vector = np.array(nv_species.axis_vector) / np.linalg.norm(nv_species.axis_vector)

        rabi_vector_hz_xtal_coords = (
            experiment_parameters.rabi_frequency_base_hz
            * experiment_parameters.mw_direction
            / np.linalg.norm(experiment_parameters.mw_direction)
        )
        rabi_vector_hz_nv_coords = transform_from_crystal_to_nv_coords(rabi_vector_hz_xtal_coords, nv_axis_unit_vector)
        rabi_xyplane_angle_from_x_rad = np.arctan2(
            rabi_vector_hz_nv_coords[_Indices.y], rabi_vector_hz_nv_coords[_Indices.x]
        )

        b_field_vector_t_nv_coords = transform_from_crystal_to_nv_coords(
            np.array(experiment_parameters.b_field_vector_t), nv_axis_unit_vector
        )
        b_xyplane_angle_from_x_rad = np.arctan2(
            b_field_vector_t_nv_coords[_Indices.y], b_field_vector_t_nv_coords[_Indices.x]
        )

        e_field_vector_hz = np.diag(np.array([exy, exy, ez])) @ transform_from_crystal_to_nv_coords(
            experiment_parameters.e_field_vector_v_per_m, nv_axis_unit_vector
        )
        e_xyplane_angle_from_x_rad = np.arctan2(e_field_vector_hz[_Indices.y], e_field_vector_hz[_Indices.x])

        OffAxisFieldSolver.validate_uniform_spacing(
            np.array(experiment_parameters.mw_pulse_length_s), "MW pulse durations"
        )
        OffAxisFieldSolver.validate_uniform_spacing(
            np.array(experiment_parameters.evolution_time_s), "free evolution times"
        )

        off_axis_field_solver_params = OffAxisFieldSolverParam(
            experiment_parameters.mw_pulse_length_s,
            experiment_parameters.evolution_time_s,
            experiment_parameters.second_pulse_phase_rad,
            experiment_parameters.detuning_hz,
            rabi_frequency_hz=OffAxisFieldSolver.norm_in_xy_plane(rabi_vector_hz_nv_coords),
            bz_field_t=b_field_vector_t_nv_coords[_Indices.z] + nv_species.residual_bz_field_t,
            bperp_field_t=OffAxisFieldSolver.norm_in_xy_plane(b_field_vector_t_nv_coords),
            efield_splitting_hz=OffAxisFieldSolver.norm_in_xy_plane(e_field_vector_hz),
            t2star_s=nv_species.t2_star_s,
            zero_field_splitting_hz=experiment_parameters.zero_field_splitting_hz,
            b_inplane_angle_from_x_rad=b_xyplane_angle_from_x_rad,
            e_inplane_angle_from_x_rad=e_xyplane_angle_from_x_rad,
            e_field_z_component_hz=e_field_vector_hz[_Indices.z],
            rabi_inplane_angle_from_x_rad=rabi_xyplane_angle_from_x_rad,
            rabi_z_component_hz=rabi_vector_hz_nv_coords[_Indices.z],
        )

        return off_axis_field_solver_params

    @staticmethod
    def identical_mw_phase(params: OffAxisFieldSolverParam, absolute_tolerance: float = 1e-12) -> bool:
        return bool(np.isclose(params.second_pulse_phase_rad, 0, atol=absolute_tolerance))

    def solve(self, params: OffAxisFieldSolverParam) -> NDArray:  # type: ignore[override]
        # Set up rotating wave approximation
        evects, h_rabi_eigenbasis_rwa_1, h_int_eigenbasis_rwa, h_rabi_eigenbasis_rwa_2 = self._eigenbasis_rwa_setup(
            params
        )
        # Compute propagators along each axis
        (
            rabi_propagators_eigenbasis_1,
            ramsey_propagators_eigenbasis,
            rabi_propagators_eigenbasis_2,
        ) = self._propagate_eigenbasis_hamiltonians(
            params, evects, h_rabi_eigenbasis_rwa_1, h_int_eigenbasis_rwa, h_rabi_eigenbasis_rwa_2
        )
        # Get initial eigenbasis
        initial_state_eigenbasis = OffAxisFieldHamiltonian.get_ms0_state_in_eigenbasis(evects)

        # Tack on the conversion back to the bare basis on the second rabi pulse (more efficient than doing it after einsum)
        # For more efficiency but less readability, this could be done while generating the second rabi propagators.
        rabi_propagators_2_convert_to_bare_basis = np.array(
            [evects @ propagator for propagator in rabi_propagators_eigenbasis_2]
        )

        final_states_bare_basis = expand_dq_unitaries(
            rabi_propagators_eigenbasis_1,
            ramsey_propagators_eigenbasis,
            rabi_propagators_2_convert_to_bare_basis,
            initial_state_eigenbasis,
        )

        return final_states_bare_basis

    def _eigenbasis_rwa_setup(self, params: OffAxisFieldSolverParam) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        e_perp_field_vector_hz = get_vector_from_vpar_vperp_and_angle(
            params.e_field_z_component_hz, params.efield_splitting_hz, params.e_inplane_angle_from_x_rad
        )
        b_field_vector_t = get_vector_from_vpar_vperp_and_angle(
            params.bz_field_t, params.bperp_field_t, params.b_inplane_angle_from_x_rad
        )
        mw_vector_hz = get_vector_from_vpar_vperp_and_angle(
            params.rabi_z_component_hz, params.rabi_frequency_hz, params.rabi_inplane_angle_from_x_rad
        )

        # generate the hamiltonian in the eigenbasis and RWA
        int_hamiltonian = OffAxisFieldHamiltonian.internal_hamiltonian_bare(
            e_perp_field_vector_hz, b_field_vector_t, params.zero_field_splitting_hz
        )
        drive_hamiltonian_bare_no_cos = OffAxisFieldHamiltonian.drive_hamiltonian_bare_no_cosine(mw_vector_hz)
        evals, evects = OffAxisFieldHamiltonian.get_ordered_spin_1_eigensystem(int_hamiltonian)
        h_int_eigenbasis_rwa = OffAxisFieldHamiltonian.internal_hamiltonian_eigenbasis_rwa(
            evals, params.zero_field_splitting_hz + params.detuning_hz
        )
        h_rabi_eigenbasis_rwa_1 = (
            OffAxisFieldHamiltonian.drive_hamiltonian_eigenbasis_rwa(
                evects, drive_hamiltonian_bare_no_cos, OffAxisFieldHamiltonian.rotating_frame_and_rwa_multiplier(0)
            )
            + h_int_eigenbasis_rwa
        )
        if OffAxisFieldSolver.identical_mw_phase(params):
            h_rabi_eigenbasis_rwa_2 = h_rabi_eigenbasis_rwa_1
        else:
            h_rabi_eigenbasis_rwa_2 = (
                OffAxisFieldHamiltonian.drive_hamiltonian_eigenbasis_rwa(
                    evects,
                    drive_hamiltonian_bare_no_cos,
                    OffAxisFieldHamiltonian.rotating_frame_and_rwa_multiplier(params.second_pulse_phase_rad),
                )
                + h_int_eigenbasis_rwa
            )

        return evects, h_rabi_eigenbasis_rwa_1, h_int_eigenbasis_rwa, h_rabi_eigenbasis_rwa_2

    def _propagate_eigenbasis_hamiltonians(
        self,
        params: OffAxisFieldSolverParam,
        evects: NDArray,
        h_rabi_eigenbasis_rwa_1: NDArray,
        h_int_eigenbasis_rwa: NDArray,
        h_rabi_eigenbasis_rwa_2: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        # generate the Rabi and Ramsey propagators in the eigenbasis
        rabi_propagators_eigenbasis_1 = generate_propagators_from_hamiltonian(
            np.array(params.mw_pulse_length_s), h_rabi_eigenbasis_rwa_1
        )
        if OffAxisFieldSolver.identical_mw_phase(params):
            rabi_propagators_eigenbasis_2 = rabi_propagators_eigenbasis_1
        else:
            rabi_propagators_eigenbasis_2 = generate_propagators_from_hamiltonian(
                np.array(params.mw_pulse_length_s), h_rabi_eigenbasis_rwa_2
            )
        ramsey_propagators_eigenbasis = generate_propagators_from_hamiltonian(
            np.array(params.evolution_time_s), h_int_eigenbasis_rwa
        )

        return rabi_propagators_eigenbasis_1, ramsey_propagators_eigenbasis, rabi_propagators_eigenbasis_2

    def get_metadata_dict(self) -> Dict[str, float]:
        return {"Solver": OffAxisFieldSolver.SOLVER_KEY}
