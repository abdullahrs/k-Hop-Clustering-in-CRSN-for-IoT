from constants import NUM_PUS
from utility_functions import MWCBG, discover_k_hop_neighbors, probabilistic_neighbor_discovery


def k_SACB_EC(su_positions, su_channels, channel_qualities, transmission_range):
    clusters = set()  # Final clusters in CRSN
    node_states = {i: "initial" for i in su_positions.keys()}
    available_channels = {i: su_channels[i] for i in su_positions.keys()}

    while any(state not in {"clustered_CM", "clustered_CH"} for state in node_states.values()):
        for node_i in su_positions.keys():
            if node_states[node_i] in {"clustered_CM", "clustered_CH"}:
                continue

            # Step I: Initial clustering
            participant_i = set()
            intermediate_cluster_i = set()
            w_i = 0
            Channel_i = available_channels[node_i]

            # Discover eligible participants
            for neighbor in discover_neighbors(node_i, su_positions, su_channels, transmission_range):
                if node_states[neighbor] not in {"clustered_CM", "clustered_CH"}:
                    if available_channels[node_i].intersection(available_channels[neighbor]):
                        participant_i.add(neighbor)

            # Prepare neighbor channels dictionary for MWCBG
            neighbor_channels = {
                neighbor: available_channels[neighbor] for neighbor in participant_i}

            # Calculate MWCBG and determine potential cluster members
            cmn_i, PCM_i, w_i = MWCBG(
                node_i, Channel_i, participant_i, su_positions, channel_qualities, neighbor_channels)
            # print("cmn_i : ", cmn_i)
            # print("PCM_i : ", PCM_i)
            # print("w_i : ", w_i)
            if (cmn_i >= 2) and (w_i > max((channel_qualities[j] for j in PCM_i if j in channel_qualities), default=0)):
                if node_states[node_i] == "initial":
                    node_states[node_i] = "intermediate_CH"
                    CM_i = PCM_i  # Current intermediate cluster members

                # Notify CMs and update states
                for cm in CM_i:
                    if node_states[cm] == "initial":
                        intermediate_cluster_i.add(cm)
                        node_states[cm] = "clustered_CM"

            # Step II: Edge contraction
            contracted_cluster = intermediate_cluster_i.union({node_i})
            # Update G(V, E) based on edge contraction
            update_graph(contracted_cluster)

            # Step III: Finalize clusters
            if cmn_i < 2:
                clusters.add(frozenset(contracted_cluster))
                node_states[node_i] = "clustered_CH"
            else:
                highest_weight_ch = select_highest_weight_ch(
                    participant_i, {node_i: w_i})
                node_states[node_i] = "clustered_CM" if highest_weight_ch else "clustered_CH"

    return clusters


# Call probabilistic_neighbor_discovery with the full su_positions dictionary
def discover_neighbors(node_id, su_positions, su_channels, transmission_range):
    """
    Discover 1-hop neighbors for a given node using probabilistic neighbor discovery.

    Parameters:
    - node_id: ID of the node for which neighbors are being discovered.
    - su_positions: Dictionary of all SU positions.
    - su_channels: Dictionary of channels available for each SU.
    - transmission_range: The transmission range within which nodes are considered neighbors.

    Returns:
    - Set of neighbors for the specified node.
    """
    # Get all neighbors and then filter for the specific node_id
    all_neighbors = probabilistic_neighbor_discovery(
        su_positions, su_channels, transmission_range)
    result = all_neighbors.get(node_id, set())
    # print("discover_neighbors result :", result)
    return result


def update_graph(contracted_cluster):
    # In practice, this updates the cluster memberships and possibly recalculates available channels
    # Select cluster head as node with smallest ID
    cluster_head = min(contracted_cluster)
    # Update each node’s state and cluster affiliation here, if necessary
    return cluster_head


def select_highest_weight_ch(participant_i, node_weights):
    # Find the CH with the highest weight among neighbors in `participant_i`
    highest_weight_ch = max(
        (node for node in participant_i if node in node_weights),
        key=lambda node: node_weights[node],
        default=None
    )
    return highest_weight_ch
