from pathlib import Path
from pickle import dump, load

from bff_simulator.simulator import Simulation


def save_simulation(path: Path, simulation: Simulation, overwrite: bool = False):
    if Path.exists(path) and not overwrite:
        raise ValueError("Output path already exists")
    with open(path, mode="wb") as out:
        dump(simulation, out)


def load_simulation(path: Path) -> Simulation:
    with open(path, mode="rb") as src:
        simulation = load(src)
    return simulation
