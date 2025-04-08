from dataclasses import dataclass
from typing import Sequence


@dataclass
class SolverParam:
    mw_pulse_length_s: Sequence[float]
    evolution_time_s: Sequence[float]

    second_pulse_phase_rad: float
    detuning_hz: float
    rabi_frequency_hz: float

    bz_field_t: float
    bperp_field_t: float

    efield_splitting_hz: float
    t2star_s: float
