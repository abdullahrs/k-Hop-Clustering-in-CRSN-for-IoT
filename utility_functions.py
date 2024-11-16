import numpy as np
import itertools
import random


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

    Function to calculate channel quality as per Section 3 in the paper.
    This calculation is based on the description provided in the paper,
    which considers idle probability and duration.
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


def euclidean_distance(node1, node2):
    """Calculate Euclidean distance between two nodes."""
    return np.linalg.norm(np.array(node1) - np.array(node2))


def MWCBG(node_id, node_position, channels_i, neighbors_i, neighbor_data, channel_qualities):
    """
    Perform the MWCBG procedure to find the maximum weight complete bipartite subgraph.

    Parameters:
    - node_id: ID of the current node
    - node_position: Position (x, y) of the current node
    - channels_i: Set of available channels at node i
    - neighbors_i: Set of node i's neighbors
    - neighbor_data: Dictionary with neighbor positions and channels

    Returns:
    - Tuple (selected_channels, selected_neighbors, max_weight)
    """

    # Step 1: Construct the Bipartite Graph
    # G(N_i ∪ C_i, E_i) using available channels C_i and neighboring nodes N_i
    # print("Node id:", node_id)
    edges = []
    for neighbor_id in neighbors_i:
        # print("Looking for channels of neighbor_id:", neighbor_id)
        neighbor_channels = neighbor_data[neighbor_id]["channels"]
        common_channels = channels_i.intersection(neighbor_channels)
        if common_channels:
            for channel in common_channels:
                # Edge (neighbor, channel)
                edges.append((neighbor_id, channel))
    # print(" Step 1: Construct the Bipartite Graph; edges :", edges)
    # Step 2: Identify all possible complete bipartite subgraphs
    # Find all combinations of neighbors and channels to form complete bipartite subgraphs
    max_weight = 0
    PCM_i = set()
    selected_channels = set()

    for neighbor_subset in powerset(neighbors_i):
        # Only proceed if the subset has neighbors
        if not neighbor_subset:
            continue

        # Find common channels for the subset
        common_channels = set.intersection(
            *(neighbor_data[neighbor]["channels"] for neighbor in neighbor_subset)).intersection(channels_i)
        # print(
        #     f"Find common channels for the subset for {neighbor_subset} common channels : {common_channels}")
        if not common_channels:
            continue

        # Step 3: Calculate weight for each complete bipartite subgraph
        # Following the weight formula described in the paper
        subgraph_weight = calculate_weight(
            neighbor_subset, common_channels, node_position, neighbor_data, channel_qualities)
        # print(
        #     f"subgraph_weight : {subgraph_weight} , max_weight : {max_weight}")
        # Step 4: Select the maximum-weight subgraph
        if subgraph_weight > max_weight:
            max_weight = subgraph_weight
            PCM_i = {(neighbor, tuple(common_channels), subgraph_weight)
                     for neighbor in neighbor_subset}
            selected_channels = common_channels

    return selected_channels, PCM_i, max_weight


def calculate_weight(neighbors, channels, node_position, neighbor_data, channel_qualities, preference_factor=0.1):
    """
    Calculate the weight of a complete bipartite subgraph based on the formula in the paper.

    Parameters:
    - neighbors: Set of neighbors in the subgraph
    - channels: Set of channels in the subgraph
    - node_position: Position of the current node
    - neighbor_data: Dictionary with neighbor positions and channels
    - channel_qualities: Dictionary of channel qualities
    - preference_factor: Balance between network stability and energy (nu in the paper)

    Returns:
    - Weight of the subgraph
    """
    # |N_i^q|: number of neighbors in subgraph
    num_neighbors = len(neighbors)

    # |C_i^q|: number of commonly available channels in subgraph
    num_channels = len(channels)

    # Sum of channel quality for each channel in the subgraph
    channel_quality_sum = sum(
        channel_qualities[channel] for channel in channels)

    # Calculate the first component (channel quality weighted by number of neighbors)
    channel_quality_component = preference_factor * \
        (num_neighbors * channel_quality_sum)

    # Calculate the second component (distance term)
    if num_neighbors > 0:
        total_distance = sum(euclidean_distance(
            node_position, neighbor_data[neighbor]["position"]) for neighbor in neighbors)
        distance_component_sum = sum(
            (1 - (euclidean_distance(node_position,
             neighbor_data[neighbor]["position"]) / total_distance))
            for neighbor in neighbors
        )
        distance_component = (1 - preference_factor) * \
            (num_channels * distance_component_sum)
    else:
        distance_component = 0

    # Calculate the total weight as per the formula
    weight = channel_quality_component + distance_component

    return weight


def powerset(iterable):
    """
    Helper function to generate all subsets of an iterable.
    """
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, len(s) + 1))


def calculate_edges(nodes, su_positions, su_channels, su_transmission_range):
    edges = []

    # Iterate over each pair of nodes
    for i, node_i in enumerate(nodes):
        for node_j in nodes[i + 1:]:
            # Calculate the distance between nodes
            distance = np.linalg.norm(
                np.array(su_positions[node_i]) - np.array(su_positions[node_j])
            )

            # Check if the nodes are within transmission range
            if distance <= su_transmission_range:
                # Find the common channels between the nodes
                common_channels = su_channels[node_i].intersection(
                    su_channels[node_j])

                # If there are common channels, add an edge
                if common_channels:
                    edges.append((node_i, node_j, common_channels))

    return edges


def get_k_hop_neighbors(node, edges, k):
    """
    Find all neighbors within k hops from the given node.
    """
    # Initialize with the direct neighbors (1-hop neighbors)
    current_level = {neighbor for n1, neighbor, _ in edges if n1 == node} | {
        n1 for neighbor, n1, _ in edges if neighbor == node}
    all_neighbors = set(current_level)

    # Perform a BFS up to k levels
    for _ in range(k - 1):
        next_level = set()
        for n in current_level:
            next_level |= {neighbor for n1, neighbor, _ in edges if n1 == n} | {
                n1 for neighbor, n1, _ in edges if neighbor == n}
        next_level -= all_neighbors  # Avoid revisiting nodes
        all_neighbors |= next_level
        current_level = next_level
    return all_neighbors


def assign_channels(num_sus, num_channels):
    """
    Assign channels to nodes with a bias towards some overlap.

    Parameters:
    - num_sus: Number of secondary users (nodes)
    - num_channels: Total number of available channels
    - min_channels: Minimum number of channels a node can have
    - max_channels: Maximum number of channels a node can have
    """
    # Step 1: Generate channel weights to bias certain channels
    min_channels = random.randint(0, num_channels)
    max_channels = random.randint(min_channels, num_channels)
    channel_weights = [random.uniform(0.5, 1.5) for _ in range(num_channels)]

    su_channels = {}
    for i in range(num_sus):
        # Step 2: Randomly choose the number of channels for this node
        num_assigned_channels = random.randint(min_channels, max_channels)

        # Step 3: Weighted sampling of channels
        chosen_channels = random.choices(
            range(num_channels),
            weights=channel_weights,
            k=num_assigned_channels
        )
        su_channels[i] = set(chosen_channels)

    return su_channels
