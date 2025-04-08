from copy import deepcopy
from typing import Dict, List

import numpy as np
from numpy.typing import ArrayLike

from bff_simulator.abstract_classes.abstract_ensemble import (
    NV14HyperfineField,
    NVEnsemble,
    NVOrientation,
    NVSpecies,
)
from bff_simulator.constants import NVaxes_100
from bff_simulator.vector_manipulation import perpendicular_projection

DEFAULT_MW_DIRECTION = np.array([0.362e-3, 0.097e-3, 0.927e-3])
DEFAULT_EFIELD_SPLITTING_HZ = 1e4
DEFAULT_T2_STAR_S = 1e-6
DEFAULT_WEIGHT = 1.0


class HomogeneousEnsemble(NVEnsemble):
    """
    A data class for containing a list of NV species.

    Rather than specifying the rabi_projection for each sub axis, they are
    always calculated from the perpendicular projection of the MW field
    with the NV orientation.

    All other NV parameters (efield, t2_star, weight, mw_field_direction)
    are set by HomogeneousEnsemble's properties.

    Args:
        mw_direction (ArrayLike): MW field vector (len = 3)
        efield_splitting_hz (float): Electric field (strain splitting) frequency
        t2_star_s (float): T2 star dephasing time
        weight: Signal weight for this NV
    """

    def __init__(
        self,
        mw_direction: ArrayLike = DEFAULT_MW_DIRECTION,
        efield_splitting_hz: float = DEFAULT_EFIELD_SPLITTING_HZ,
        t2_star_s: float = DEFAULT_T2_STAR_S,
        weight: float = DEFAULT_WEIGHT,
    ) -> None:
        self._mw_direction = mw_direction
        self._efield_splitting_hz = efield_splitting_hz
        self._t2_star_s = t2_star_s
        self._weight = weight
        self._populations: List[NVSpecies] = []

    @property
    def mw_direction(self):
        return deepcopy(self._mw_direction)

    @mw_direction.setter
    def mw_direction(self, new_mw_direction: ArrayLike):
        assert np.array(new_mw_direction).size == 3, "mw_direction must be a 3-component vector"
        self._mw_direction = new_mw_direction
        for nv_species in self._populations:
            nv_species.rabi_projection = perpendicular_projection(new_mw_direction, nv_species.axis_vector)

    @property
    def efield_splitting_hz(self):
        return deepcopy(self._efield_splitting_hz)

    @efield_splitting_hz.setter
    def efield_splitting_hz(self, new_splitting: float):
        self._efield_splitting_hz = new_splitting
        for nv_species in self._populations:
            nv_species.efield_splitting_hz = new_splitting

    @property
    def t2_star_s(self):
        return deepcopy(self._t2_star_s)

    @t2_star_s.setter
    def t2_star_s(self, new_t2_star_s: float):
        self._t2_star_s = new_t2_star_s
        for nv_species in self._populations:
            nv_species.t2_star_s = new_t2_star_s

    @property
    def weight(self):
        return deepcopy(self._weight)

    @weight.setter
    def weight(self, weight: float):
        for nv_species in self._populations:
            nv_species.weight = weight

    def set_individual_weights(self, weights: List[float]):
        assert len(weights) == len(
            self.populations
        ), f"len(weights) must equal len(populations). Got {len(weights)} and {len(self.populations)}"
        for weight, nv_species in zip(weights, self._populations):
            nv_species.weight = weight

    def add_nv_single_species(self, orientation: NVOrientation, hyperfine_field_t: NV14HyperfineField):
        axis_vector = NVaxes_100[orientation]
        self._add_nv_species(
            perpendicular_projection(self.mw_direction, axis_vector),
            axis_vector,
            hyperfine_field_t,
            self.efield_splitting_hz,
            self.t2_star_s,
            self.weight,
        )

    def add_n14_triplet(self, orientation: NVOrientation):
        """
        Adds an NV triplet with N14 hyperfine states with a given crystallographic
        orientation. All other NV parameters (efield, t2_star, weight, mw_field_direction)
        are set by HomogeneousEnsemble's properties.

        Args:
            orientation (NVOrientation): Diamond vector for the chosen NV triplet
        """
        for hyperfine_field_t in NV14HyperfineField:
            self.add_nv_single_species(orientation, hyperfine_field_t)

    def add_full_diamond_populations(self):
        """
        Adds all 4 NV orientations, each with triplet N14 hyperfine states.
        All other NV parameters (efield, t2_star, weight, mw_field_direction)
        are set by HomogeneousEnsemble's properties.
        """
        for orientation in NVOrientation:
            self.add_n14_triplet(orientation)

    def _get_metadata_dict(self) -> Dict[str, float]:
        metadata = {
            "Ensemble E Field Splitting (Hz)": self._efield_splitting_hz,
            "Ensemble T2* (s)": self._t2_star_s,
            "Ensemble Weight": self._weight,
        }
        for i, axis in enumerate(["X", "Y", "Z"]):
            metadata[f"Ensemble MW Direction {axis}"] = self._mw_direction[i]  # type: ignore

        return metadata
