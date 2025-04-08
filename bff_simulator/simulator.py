from copy import deepcopy
from typing import Dict, List, Union

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from bff_simulator.abstract_classes.abstract_ensemble import NVEnsemble
from bff_simulator.abstract_classes.abstract_solver import Solver
from bff_simulator.experiment_parameters import ExperimentParameters
from bff_simulator.solver_parameters import SolverParam


class SimulationHeaders:
    pulse_time = "Pulse Time (s)"
    evolution_time = "Evolution Time (s)"
    total_ms0 = "Total ms=0"

    @staticmethod
    def indexed_nv_fraction_in_zeeman_sublevel(nv_index: int = 0, zeeman_sublevel: int = 0) -> str:
        return f"NV Population {nv_index} State ms={zeeman_sublevel}"


class Simulation:
    def __init__(
        self,
        experiment_params: ExperimentParameters,
        nv_ensemble: NVEnsemble,
        solver: Solver,
    ):
        self._experiment_params = deepcopy(experiment_params)
        self._nv_ensemble = deepcopy(nv_ensemble)
        self._solver = deepcopy(solver)
        self._solver_params_by_species: Union[List[SolverParam], None] = None
        self._normalized_ms0_population: Union[NDArray, None] = None
        self._output_spin_state_by_species: Union[List[NDArray], None] = None

        self._simulate_on_construction()

    @property
    def experiment_params(self) -> ExperimentParameters:
        return deepcopy(self._experiment_params)

    @property
    def nv_ensemble(self) -> NVEnsemble:
        return deepcopy(self._nv_ensemble)

    @property
    def solver_params_by_species(self) -> Union[List[SolverParam], None]:
        return deepcopy(self._solver_params_by_species)

    @property
    def solver(self) -> Solver:
        return deepcopy(self._solver)

    @property
    def ms0_results_by_species(self) -> List[NDArray]:
        assert isinstance(self._output_spin_state_by_species, List)
        return [abs(final_states[..., 1]) ** 2 for final_states in self._output_spin_state_by_species]

    @property
    def ms0_results(self) -> NDArray:
        assert self._normalized_ms0_population is not None
        return deepcopy(self._normalized_ms0_population)

    @property
    def results_by_pop(self) -> Union[List[NDArray], None]:
        return deepcopy(self._output_spin_state_by_species)

    @property
    def dataframe(self) -> DataFrame:
        grid_pulse_times, grid_evo_times = np.meshgrid(
            self._experiment_params.mw_pulse_length_s, self._experiment_params.evolution_time_s, indexing="ij"
        )
        df_dict: Dict[str, NDArray] = {
            SimulationHeaders.pulse_time: grid_pulse_times.flatten(),
            SimulationHeaders.evolution_time: grid_evo_times.flatten(),
        }
        df_dict.update(self._generate_result_dictionary())
        df_dict[SimulationHeaders.total_ms0] = self.ms0_results.flatten()
        df = DataFrame(df_dict)

        df = self._append_metadata_columns(df)
        return df

    def compute_fluorescence(self, baseline_fluorescence: float, contrast: float = 0.3) -> NDArray:
        assert self._normalized_ms0_population is not None
        return baseline_fluorescence * (1 - contrast * (1 - self._normalized_ms0_population))

    def _get_total_ms0_population(self) -> NDArray:
        sum_of_weights = sum(nv_species.weight for nv_species in self.nv_ensemble.populations)
        sum_of_ms0_populations = np.asarray(
            sum(
                (
                    nv_species.weight * ms0_pop
                    for (nv_species, ms0_pop) in zip(self.nv_ensemble.populations, self.ms0_results_by_species)
                )
            )
        )
        return sum_of_ms0_populations / sum_of_weights

    def _simulate_on_construction(self):
        self._output_spin_state_by_species = []
        self._generate_solver_populations()
        for solver_param in self._solver_params_by_species:
            self._output_spin_state_by_species.append(self.solver.solve(solver_param))

        self._normalized_ms0_population = self._get_total_ms0_population()

    def _generate_solver_populations(self):
        self._solver_params_by_species = []
        for nv_species in self.nv_ensemble.populations:
            self._solver_params_by_species.append(
                self._solver.construct_solver_parameters(self.experiment_params, nv_species)
            )

    def _generate_result_dictionary(self) -> Dict[str, NDArray]:
        output_pops: Dict[str, NDArray] = dict()
        assert self.results_by_pop is not None
        for i, result_pops in enumerate(self.results_by_pop):
            for j, label in enumerate([1, 0, -1]):
                output_pops[SimulationHeaders.indexed_nv_fraction_in_zeeman_sublevel(i, label)] = result_pops[
                    :, :, j
                ].flatten()
        return output_pops

    def _append_metadata_columns(self, df: DataFrame) -> DataFrame:
        metadata = self._nv_ensemble.get_metadata_dict()
        metadata.update(self._experiment_params.get_metadata_dict())
        metadata.update(self._solver.get_metadata_dict())

        for key, datum in metadata.items():
            df[key] = datum
        return df
