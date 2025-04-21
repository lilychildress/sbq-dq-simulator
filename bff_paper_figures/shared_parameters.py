import numpy as np

# Conversion constants
T_TO_UT = 1e6
T_TO_NT = 1e9
HZ_TO_MHZ = 1e-6
S_TO_US = 1e6
RAD_TO_DEGREE = 360 / (2 * np.pi)

N_HYPERFINE = 3  # Number of hyperfine peaks

# Arbitrary magnetic field angle used in Figure 4
B_THETA_FIG4 = 3 * np.pi / 8
B_PHI_FIG4 = 13 * np.pi / 16

# Nominal MW parameters
RABI_FREQ_BASE_HZ = 100e6
MW_THETA = 0.23957582149174544
MW_PHI = 0.5244528089360776
MW_DIRECTION = np.array([np.sin(MW_THETA) * np.cos(MW_PHI), np.sin(MW_THETA) * np.sin(MW_PHI), np.cos(MW_THETA)])

# Shared experiment settings
E_FIELD_VECTOR_V_PER_CM = np.array([0, 0, 0])
DETUNING_HZ = 0e6
T2STAR_S = 2e-6

# Folder for loading data (machine-specific)
BASE_PATH = "/Users/lilianchildress/Documents/GitHub/sbq-dq-simulator/bff_paper_figures/data/"

# Determines which of the three hyperfine transitions we will be comparing to expected value to extract
# inversion errors for each orientation; 0 is highest-frequency peak (i.e. hyperfine and external field aligned)
PEAK_INDEX = 0

# Determines the Ramsey frequency range over which we will evaluate the double inner product in order to
# geet initial guesses for subsequent fits.
RAMSEY_FREQ_RANGE_INITIAL_GUESS_HZ = np.linspace(0, 10e6, 251)
