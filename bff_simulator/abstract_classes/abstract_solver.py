from abc import ABC, abstractmethod
from typing import Dict

from numpy.typing import NDArray

from bff_simulator.abstract_classes.abstract_ensemble import NVSpecies
from bff_simulator.experiment_parameters import ExperimentParameters
from bff_simulator.solver_parameters import SolverParam


class Solver(ABC):
    """
    Base solver for getting the output states of a double quantum pulse sequence.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def solve(self, params: SolverParam) -> NDArray:
        """
        Returns ndarray of populations MxNx3, where first axis is microwave time, second axis is evolution time, and
        the final axis is the three possible spin states ordered as [1, 0, -1]
        """
        pass

    @abstractmethod
    def construct_solver_parameters(
        self, experiment_parameters: ExperimentParameters, nv_species: NVSpecies
    ) -> SolverParam:
        pass

    @abstractmethod
    def get_metadata_dict(self) -> Dict[str, float]:
        pass
