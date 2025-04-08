from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass
class ExperimentParameters:
    mw_pulse_length_s: Sequence[float]
    evolution_time_s: Sequence[float]
    second_pulse_phase_rad: float
    b_field_vector_t: Sequence[float]
    detuning_hz: float
    rabi_frequency_base_hz: float

    def get_metadata_dict(self) -> Dict[str, float]:
        metadata = {
            "Detuning (Hz)": self.detuning_hz,
            "Base Rabi Frequency (Hz)": self.rabi_frequency_base_hz,
            "Second Pulse Phase (rad)": self.second_pulse_phase_rad,
        }
        for i, axis in enumerate(["X", "Y", "Z"]):
            metadata[f"B {axis} Field (T)"] = self.b_field_vector_t[i]
        return metadata


class ExperimentParametersFactory:
    def __init__(self):
        self._experiment_parameters = ExperimentParameters(
            np.linspace(0, 1e-7, 101),
            np.linspace(0, 1e-6, 101),
            0.0,
            np.array([0, 0, 0]),
            0.0,
            1e8,
        )

    @staticmethod
    def _has_at_least_two_elements(sequence: Sequence[float]) -> None:
        assert len(sequence) >= 2, "Time sequence must have at least two elements"

    def set_mw_pulse_lengths(self, mw_pulse_length_s: Sequence[float]) -> None:
        self._has_at_least_two_elements(mw_pulse_length_s)
        self._experiment_parameters.mw_pulse_length_s = mw_pulse_length_s

    def set_evolution_times(self, evolution_time_s: Sequence[float]) -> None:
        self._has_at_least_two_elements(evolution_time_s)
        self._experiment_parameters.evolution_time_s = evolution_time_s

    def set_second_pulse_phase(self, second_pulse_phase_rad: float) -> None:
        self._experiment_parameters.second_pulse_phase_rad = second_pulse_phase_rad

    def set_b_field_vector(self, b_field_vector_t: Sequence[float]) -> None:
        self._experiment_parameters.b_field_vector_t = b_field_vector_t

    def set_detuning(self, detuning_hz: float) -> None:
        self._experiment_parameters.detuning_hz = detuning_hz

    def set_base_rabi_frequency(self, rabi_frequency_base_hz: float) -> None:
        self._experiment_parameters.rabi_frequency_base_hz = rabi_frequency_base_hz

    def get_experiment_parameters(self) -> ExperimentParameters:
        return deepcopy(self._experiment_parameters)
