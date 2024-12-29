# pu_utils.py
from random import random
from config import ALPHA, BETA


def channel_on_probability(alpha=ALPHA, beta=BETA):
    """
    Returns the probability that a channel is ON (available) 
    based on alpha/beta parameters.
    """
    return beta / (alpha + beta)

def simulate_channel_availability(num_channels=5, alpha=ALPHA, beta=BETA):
    """
    For each of the 'num_channels', we decide if it's ON or OFF 
    (available or unavailable).
    """
    on_prob = channel_on_probability(alpha, beta)
    available = set()
    for c in range(num_channels):
        if random() < on_prob:
            available.add(c)
    return available
