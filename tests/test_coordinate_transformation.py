import unittest

import numpy as np
from numpy.linalg import norm

from bff_simulator.constants import NVaxes_100
from bff_simulator.vector_manipulation import transform_from_crystal_to_nv_coords


class TestCrystalToNVTransformation(unittest.TestCase):
    def test_parallel_component(self) -> None:
        vector = np.array([1, 5, 2])
        nv_axis = NVaxes_100[3]
        _, _, vz = transform_from_crystal_to_nv_coords(vector, nv_axis)
        self.assertAlmostEqual(vector @ nv_axis, vz)

    def test_perpendicular_component(self) -> None:
        vector = np.array([1, 5, 2])
        nv_axis = NVaxes_100[2]
        vx, vy, _ = transform_from_crystal_to_nv_coords(vector, nv_axis)
        self.assertAlmostEqual(norm(np.cross(vector, nv_axis)), np.sqrt(vx**2 + vy**2))


if __name__ == "__main__":
    unittest.main()
