import numpy as np
import random

# Set random seed for consistency
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Define constants as per the simulation parameters in the paper
SIMULATION_AREA = (300, 300)  # 300x300 m^2
NUM_SUS = 64  # Number of SUs (can be varied from 30-100)
NUM_CHANNELS = 10  # Number of PU channels
NUM_PUS = 10  # Number of PUs
PU_TRANSMISSION_RANGE = 60  # Transmission range of PUs in meters
SU_TRANSMISSION_RANGE = 40  # Transmission range of SUs in meters
INITIAL_ENERGY = 0.2  # Initial energy of SUs in Joules
SENSING_ENERGY = 1.31e-4  # Energy consumption for channel sensing (J)
SWITCHING_ENERGY = 1e-5  # Energy consumption for channel switching (J)
E_FS = 10e-12  # Energy parameter (J/bit/m^2)
E_ELEC = 50e-9  # Energy parameter (J/bit)
PU_ON_PERIOD_MEAN = [0.5, 2.0]  # Mean value for PU ON period (alpha)
PU_OFF_PERIOD_MEAN = [0.5, 2.0]  # Mean value for PU OFF period (beta)

# Randomly generate SU positions and channels
su_positions = {
    i: (np.random.uniform(0, SIMULATION_AREA[0]), np.random.uniform(
        0, SIMULATION_AREA[1]))
    for i in range(NUM_SUS)
}
su_channels = {
    i: set(random.sample(range(NUM_CHANNELS), random.randint(1, NUM_CHANNELS)))
    for i in range(NUM_SUS)
}
