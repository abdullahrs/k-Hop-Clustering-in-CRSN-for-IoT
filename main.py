import numpy as np
import matplotlib.pyplot as plt
import random

from comparison_algorithms import ComparisonAlgorithm
from constants import *
from k_SACB_EC import k_SACB_EC
from k_SACB_WEC import k_SACB_WEC
from utility_functions import calculate_channel_qualities


channel_qualities = calculate_channel_qualities(NUM_CHANNELS)


result = k_SACB_EC(su_positions, su_channels, channel_qualities, PU_TRANSMISSION_RANGE)

print(result)
print()
print()
print()
print(su_positions)
print()
print(su_channels)
