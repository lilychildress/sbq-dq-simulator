import numpy as np

from bff_simulator.vector_manipulation import convert_spherical_coordinates_to_cartesian

############################
# NV Hamiltonian parameters
############################
D = 2.870685e9  # Zero-field splitting in Hz
f_h = 2.16e6  # Hyperfine splitting in Hz
gammab = 28.024e9  # Gyromagnetic ratio gamma_e / 2*np.pi in Hz/T
exy = 0.17  # Electric field transverse coupling in Hz/(V/m) from Dolde 2011 Nat Phys 7, 459 https://www.nature.com/articles/nphys1969
ez = 3.5e-3  # Electric field longitudinal coupling in Hz/(V/m) from Dolde 2011 Nat Phys 7, 459 https://www.nature.com/articles/nphys1969

# NV orientations for [100] samples (thickness along z)
# Spherical and cartesian, going from 1st to 4th quadrant, 1st and 3rd pointing towards +z
NVaxes_100_sph = [
    (1, np.arccos(-1 / 3) / 2, 1 * 45.0 * np.pi / 180),
    (1, np.pi - np.arccos(-1 / 3) / 2, 3 * 45.0 * np.pi / 180),
    (1, np.arccos(-1 / 3) / 2, 5 * 45.0 * np.pi / 180),
    (1, np.pi - np.arccos(-1 / 3) / 2, 7 * 45.0 * np.pi / 180),
]
NVaxes_100 = [convert_spherical_coordinates_to_cartesian(np.array(vv)) for vv in NVaxes_100_sph]
