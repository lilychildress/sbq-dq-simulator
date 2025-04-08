from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray

from bff_simulator.constants import D, exy
from bff_simulator.experiment_parameters import (
    ExperimentParameters,
    ExperimentParametersFactory,
)

MW_POLAR_ANGLE_RAD = 1.46
MW_AZIMUTHAL_ANGLE_RAD = 0.21

DEFAULT_MW_DIRECTION = np.array(
    [
        np.cos(MW_AZIMUTHAL_ANGLE_RAD) * np.sin(MW_POLAR_ANGLE_RAD),
        np.sin(MW_AZIMUTHAL_ANGLE_RAD) * np.sin(MW_POLAR_ANGLE_RAD),
        np.cos(MW_POLAR_ANGLE_RAD),
    ]
)

DEFAULT_E_FIELD_V_PER_M = np.array([0, 0, 0]) / exy


@dataclass
class OffAxisFieldExperimentParameters(ExperimentParameters):
    mw_direction: NDArray
    e_field_vector_v_per_m: NDArray
    zero_field_splitting_hz: float

    def get_metadata_dict(self) -> Dict[str, float]:
        metadata = super().get_metadata_dict()
        metadata["ZFS"] = self.zero_field_splitting_hz
        metadata.update(
            {f"E field {axis} (V/m)": self.e_field_vector_v_per_m[i] for i, axis in enumerate(["X", "Y", "Z"])}
        )
        metadata.update(
            {f"Experiment MW Direction {axis}": self.mw_direction[i] for i, axis in enumerate(["X", "Y", "Z"])}
        )
        return metadata


class OffAxisFieldExperimentParametersFactory(ExperimentParametersFactory):
    def __init__(self):
        # Set some default initial experiment parameters
        self._experiment_parameters = OffAxisFieldExperimentParameters(
            np.linspace(0, 1e-7, 101),
            np.linspace(0, 1e-6, 101),
            0.0,
            np.array([0, 0, 0]),
            0.0,
            1e8,
            DEFAULT_MW_DIRECTION / norm(DEFAULT_MW_DIRECTION),
            DEFAULT_E_FIELD_V_PER_M,
            D,
        )
        print(
            "Warning: OffAxisFieldSolver only uses MW direction and electric field information from OffAxisFieldExperimentParametersFactory, not from HomogeneousEnsemble."
        )

    def set_mw_direction(self, mw_direction: NDArray) -> None:
        self._experiment_parameters.mw_direction = mw_direction

    def set_e_field_v_per_m(self, e_field_v_per_m: NDArray) -> None:
        self._experiment_parameters.e_field_vector_v_per_m = e_field_v_per_m

    def set_zero_field_splitting_hz(self, zero_field_splitting_hz: float) -> None:
        self._experiment_parameters.zero_field_splitting_hz = zero_field_splitting_hz

    def get_experiment_parameters(self) -> OffAxisFieldExperimentParameters:
        return deepcopy(self._experiment_parameters)
