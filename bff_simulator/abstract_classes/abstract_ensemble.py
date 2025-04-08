from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence

from numpy.typing import ArrayLike

import bff_simulator.constants as constants

HYPERFINE_N15_FREQUENCY_HZ = 3e6


class NVOrientation(int, Enum):
    A = 0
    B = 1
    C = 2
    D = 3


class NV14HyperfineField(float, Enum):
    N14_plus = (
        constants.f_h / constants.gammab
    )  # NOTE: Technically, if N14_plus is referring to mI = +1, this is incorrect -- the hyperfine constant should be negative, but is defined as positive in constants.py -- but it won't really change anything.
    N14_0 = 0.0
    N14_minus = -constants.f_h / constants.gammab


class N15HyperfineField(float, Enum):
    N15_up = HYPERFINE_N15_FREQUENCY_HZ / constants.gammab
    N15_down = -HYPERFINE_N15_FREQUENCY_HZ / constants.gammab


@dataclass
class NVSpecies:
    """
    Attributes that define an individual species, potentially within an ensemble.
    """

    rabi_projection: float
    axis_vector: ArrayLike
    residual_bz_field_t: float
    efield_splitting_hz: float
    t2_star_s: float
    weight: float = 1.0


class NVEnsemble(ABC):
    """
    A data class for containing a list of NV species.
    """

    def __init__(self) -> None:
        self._populations: List[NVSpecies] = []

    def _add_nv_species(
        self,
        rabi_projection: float,
        axis_vector: ArrayLike,
        residual_bz_field_t: float,
        efield_splitting_hz: float,
        t2_star_s: float,
        weight: float,
    ):
        self._populations.append(
            NVSpecies(
                rabi_projection,
                axis_vector,
                residual_bz_field_t,
                efield_splitting_hz,
                t2_star_s,
                weight,
            )
        )

    @property
    def populations(self) -> Sequence[NVSpecies]:
        return deepcopy(self._populations)

    def get_metadata_dict(self) -> Dict[str, float]:
        metadata = self._population_metadata_dict()
        metadata.update(self._get_metadata_dict())
        return metadata

    @abstractmethod
    def _get_metadata_dict(self) -> Dict[str, float]:
        pass

    def _population_metadata_dict(self) -> Dict[str, float]:
        metadata = dict()
        for i, nv in enumerate(self._populations):
            metadata.update({f"NV Population {i} {key}": value for key, value in NVEnsemble._nv_metadata(nv).items()})
        return metadata

    @classmethod
    def _nv_metadata(cls, nv: NVSpecies) -> Dict[str, float]:
        metadata = {
            "Rabi Projection": nv.rabi_projection,
            "Residual Bz Field (T)": nv.residual_bz_field_t,
            "E Field Splitting (Hz)": nv.efield_splitting_hz,
            "T2* (s)": nv.t2_star_s,
            "Weight": nv.weight,
        }
        for i, axis in enumerate(["X", "Y", "Z"]):
            # Mypy does not play well with "ArrayLike"
            metadata[f"{axis} Axis"] = nv.axis_vector[i]  # type: ignore
        return metadata
