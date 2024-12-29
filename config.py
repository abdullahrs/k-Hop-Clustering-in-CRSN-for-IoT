# config.py

# General Network Settings
NUM_SUS = 100  # Number of secondary users (adjustable as needed)
AREA_SIZE = (300, 300)  # Dimensions of the simulation area in meters
INITIAL_ENERGY = 0.2  # Initial energy for each SU in joules
NUM_CHANNELS = 10  # Total number of available channels
PU_ALPHA = 2  # PU ON rate (short OFF followed by short ON)
PU_BETA = 2  # PU OFF rate
SENSING_COST = 1.31e-4
# Channel Quality Parameters
EPSILON = 1.5  # Weighting parameter for Pi and Di preferences

# Clustering Parameters
MIN_COMMON_CHANNELS = 2  # Minimum channels required for bichannel connectivity
HOP_LIMIT = 3  # Maximum k-hop limit for clustering

# Simulation Parameters
NUM_ROUNDS = 500  # Number of rounds to simulate

# Other Assumptions
TRANSMISSION_RANGE = 50  # Range for node communication in meters
RECLUSTER_ENERGY = 1.31e-4  # Energy consumption for reclustering
SENSING_COST = 1.31e-4
SWITCHING_COST = 1e-5 # 10^-5 joules
