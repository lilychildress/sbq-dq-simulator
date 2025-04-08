from dataclasses import dataclass

from bff_simulator.solver_parameters import SolverParam


@dataclass
class OffAxisFieldSolverParam(SolverParam):
    zero_field_splitting_hz: float
    b_inplane_angle_from_x_rad: float
    e_inplane_angle_from_x_rad: float
    e_field_z_component_hz: float
    rabi_inplane_angle_from_x_rad: float
    rabi_z_component_hz: float
