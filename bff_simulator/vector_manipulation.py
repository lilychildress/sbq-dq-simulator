import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg


def perpendicular_projection(vector_1: ArrayLike, vector_2: ArrayLike):
    unit_vector_1 = np.array(vector_1) / linalg.norm(vector_1)
    unit_vector_2 = np.array(vector_2) / linalg.norm(vector_2)
    return np.sqrt(max(1 - np.dot(unit_vector_1, unit_vector_2) ** 2, 0))


def transform_from_crystal_to_nv_coords(vector_xtal_coords: NDArray, nv_axis_unit_vector: NDArray) -> NDArray:
    a, b, c = nv_axis_unit_vector
    transformation_matrix = np.array(
        (
            (
                (a**2 * c / np.sqrt(a**2 + b**2 + c**2) + b**2) / (a**2 + b**2),
                a * b * (c / np.sqrt(a**2 + b**2 + c**2) - 1) / (a**2 + b**2),
                -a / np.sqrt(a**2 + b**2 + c**2),
            ),
            (
                a * b * (c / np.sqrt(a**2 + b**2 + c**2) - 1) / (a**2 + b**2),
                (a**2 + b**2 * c / np.sqrt(a**2 + b**2 + c**2)) / (a**2 + b**2),
                -b / np.sqrt(a**2 + b**2 + c**2),
            ),
            (a / np.sqrt(a**2 + b**2 + c**2), b / np.sqrt(a**2 + b**2 + c**2), c / np.sqrt(a**2 + b**2 + c**2)),
        )
    )

    return transformation_matrix @ vector_xtal_coords


def get_vector_from_vpar_vperp_and_angle(vpar: float, vperp: float, xy_plane_angle_from_x_rad: float) -> NDArray:
    return np.array([vperp * np.cos(xy_plane_angle_from_x_rad), vperp * np.sin(xy_plane_angle_from_x_rad), vpar])


def convert_spherical_coordinates_to_cartesian(spherical_coordinates: NDArray = np.array([1, 0, 0])) -> NDArray:
    """
    Converts a vector from the form (magnitude, theta, phi) to the cartesian coordinates (x, y, z). Uses the
    physics convention, with theta being the polar angle in radians.
    """
    vec = spherical_coordinates[np.newaxis, :] if spherical_coordinates.ndim == 1 else spherical_coordinates
    return np.vstack(
        [
            vec[:, 0] * np.sin(vec[:, 1]) * np.cos(vec[:, 2]),
            vec[:, 0] * np.sin(vec[:, 1]) * np.sin(vec[:, 2]),
            vec[:, 0] * np.cos(vec[:, 1]),
        ]
    ).T.squeeze()
