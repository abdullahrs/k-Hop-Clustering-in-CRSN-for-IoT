import numpy as np
from itertools import combinations
import random

# Function to calculate channel quality as per Section 3 in the paper.
# This calculation is based on the description provided in the paper,
# which considers idle probability and duration.
def calculate_channel_qualities(num_channels, epsilon=2.0):
    """
    Calculate channel qualities for each channel.

    Parameters:
    - num_channels: Total number of channels.
    - epsilon: Preference parameter for idle probability.

    Returns:
    - channel_qualities: Dictionary of channel qualities (key: channel ID, value: quality).

    Note:
    The channel qualities are treated as constants in the algorithm, but we pass them as parameters
    for generalization in simulations.
    """
    channel_qualities = {}
    for channel in range(num_channels):
        # Random idle probability, adjustable per paper requirements.
        P_i = random.uniform(0.1, 0.9)
        D_i = random.uniform(1, 5)  # Random idle duration, also adjustable.
        # Quality formula, can be adjusted based on model requirements.
        Q_i = (1 + np.log(epsilon * P_i)) * D_i
        channel_qualities[channel] = Q_i
    return channel_qualities


def MWCBG(node, channels, neighbors, su_positions, channel_qualities, neighbor_channels, preference_factor=0.5):
    """
    Maximum Weight Complete Bipartite Graph (MWCBG) for a node with its neighbors and channels.

    Parameters:
    - node: The ID of the node for which the bipartite graph is constructed.
    - channels: Set of available channels for the node.
    - neighbors: List of neighbors for the node.
    - su_positions: Dictionary of SU positions.
    - channel_qualities: Dictionary of channel qualities (key: channel ID, value: quality).
    - neighbor_channels: Dictionary of channels available for each neighbor.
    - preference_factor: Preference between network stability and residual energy.

    Returns:
    - A tuple of (number of common channels, cluster members, max weight).
    """
    max_weight = 0
    best_common_channels = set()
    best_cluster_members = set()
    epsilon = 1e-6  # Small value to prevent division by zero

    for neighbor in neighbors:
        common_channels = channels.intersection(neighbor_channels[neighbor])

        # Skip if fewer than 2 common channels
        if len(common_channels) < 2:
            continue

        best_common_channels.update(common_channels)

        # Initial weight calculation based on channel quality
        weight = preference_factor * sum(channel_qualities[c] for c in common_channels)
        
        # Adjust weight with distances
        for n in neighbors:
            dist = np.linalg.norm(np.array(su_positions[node]) - np.array(su_positions[n]))
            norm_dist = min(dist / (sum(np.linalg.norm(np.array(su_positions[node]) - np.array(su_positions[other])) 
                           for other in neighbors if other != n) + epsilon), 1)
            weight += (1 - norm_dist)

        # Ensure non-negative weights and enforce minimum threshold
        weight = max(weight, 0)
        
        # Update `max_weight` and `best_cluster_members` if this weight is greater
        print("weight :", weight, " ||| max weight :", max_weight)
        if weight > max_weight:
            max_weight = weight
            best_cluster_members = {neighbor}
    print(len(best_common_channels), best_cluster_members, max_weight)
    return len(best_common_channels), best_cluster_members, max_weight


def probabilistic_neighbor_discovery(su_positions, su_channels, transmission_range):
    """
    Simplified probabilistic neighbor discovery for identifying 1-hop neighbors based on:
    - Transmission range
    - Channel overlap

    Parameters:
    - su_positions: Dictionary of SU positions (key: node ID, value: position as (x, y) tuple).
    - su_channels: Dictionary of SU channels (key: node ID, value: set of available channels).
    - transmission_range: Transmission range within which SUs can detect each other.

    Returns:
    - neighbors: Dictionary mapping each SU to a set of neighboring SUs.
    """
    neighbors = {i: set()
                 for i in su_positions}  # Initialize neighbors for each SU

    # Use itertools.combinations to generate unique pairs (su_a, su_b)
    for su_a, su_b in combinations(su_positions.keys(), 2):
        pos_a, pos_b = su_positions[su_a], su_positions[su_b]
        # print("su_a :", su_a, "||| pos_a :", pos_a)
        # print("su_b :", su_b, "||| pos_b :", pos_b)
        # Step 1: Calculate Euclidean distance to check if within transmission range
        distance = np.linalg.norm(np.array(pos_a) - np.array(pos_b))
        if distance <= transmission_range:
            # Step 2: Check for channel overlap between SU_a and SU_b
            if su_channels[su_a].intersection(su_channels[su_b]):
                neighbors[su_a].add(su_b)  # Add SU_b as a neighbor of SU_a
                # Also add SU_a as a neighbor of SU_b (bidirectional link)
                neighbors[su_b].add(su_a)

    return neighbors


# Neighbor Discovery Function for k-hop
def discover_k_hop_neighbors(su_positions, su_channels, transmission_range, num_pus, k=1):
    """
    Discover k-hop neighbors for each SU, using probabilistic neighbor discovery for 1-hop neighbors,
    and then expanding to k-hop.

    Parameters:
    - su_positions: Dictionary of SU positions (key: node ID, value: position as (x, y) tuple).
    - su_channels: Dictionary of SU channels (key: node ID, value: set of available channels).
    - transmission_range: Transmission range for SUs.
    - num_pus: Number of PU channels (M).
    - k: Number of hops to consider.

    Returns:
    - k_hop_neighbors: Dictionary mapping each SU to its discovered k-hop neighbors.
    """
    # Step 1: Initial 1-hop neighbors using probabilistic discovery
    one_hop_neighbors = probabilistic_neighbor_discovery(
        su_positions, su_channels, transmission_range, num_pus
    )

    # Initialize k-hop neighbors starting with 1-hop neighbors
    k_hop_neighbors = {
        node: set(one_hop_neighbors[node]) for node in su_positions}

    # Step 2: Iteratively discover neighbors up to k-hops
    for hop in range(2, k + 1):
        for node in su_positions:
            # Collect neighbors from the previous hop level
            new_neighbors = set()
            for neighbor in k_hop_neighbors[node]:
                new_neighbors.update(one_hop_neighbors.get(neighbor, []))

            # Remove the node itself and previously discovered neighbors
            new_neighbors.discard(node)
            new_neighbors -= k_hop_neighbors[node]

            # Add these newly discovered neighbors to the k-hop set
            k_hop_neighbors[node].update(new_neighbors)

    return k_hop_neighbors
