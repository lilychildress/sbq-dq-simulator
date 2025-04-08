from dataclasses import dataclass

from bff_simulator.solver_parameters import SolverParam


@dataclass
class GeometricSolverParam(SolverParam):
    second_pulse_geometric_angle_rad: float
