from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

import numpy as np

from bff_simulator.experiment_parameters import (
    ExperimentParameters,
    ExperimentParametersFactory,
)


@dataclass
class GeometricExperimentParameters(ExperimentParameters):
    second_pulse_geometric_angle_rad: float

    def get_metadata_dict(self) -> Dict[str, float]:
        metadata = super().get_metadata_dict()
        metadata["Geometric Phase of Second Pulse (rad)"] = self.second_pulse_geometric_angle_rad
        return metadata


class GeometricExperimentParametersFactory(ExperimentParametersFactory):
    def __init__(self):
        self._experiment_parameters = GeometricExperimentParameters(
            np.linspace(0, 1e-7, 101),
            np.linspace(0, 1e-6, 101),
            0.0,
            np.array([0, 0, 0]),
            0.0,
            1e8,
            0.0,
        )

    def set_second_pulse_geometric_angle(self, second_pulse_geometric_angle_rad: float) -> None:
        self._experiment_parameters.second_pulse_geometric_angle_rad = second_pulse_geometric_angle_rad

    def get_experiment_parameters(self) -> GeometricExperimentParameters:
        return deepcopy(self._experiment_parameters)
