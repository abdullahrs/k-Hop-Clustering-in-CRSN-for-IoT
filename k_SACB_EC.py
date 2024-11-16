from copy import deepcopy
from math import sqrt

from utility_functions import MWCBG, get_k_hop_neighbors


def k_SACB_EC(nodes, edges, channels, su_positions, channel_qualities, su_transmission_range, k=2):
    """
    k-SACB-EC algorithm for k-hop clustering with edge contraction in a cognitive radio sensor network.

    Parameters:
        nodes: List of nodes in the network.
        edges: Dictionary representing edges between nodes with node IDs as keys and sets of neighboring node IDs as values.
        channels: Dictionary representing available channels for each node.
        k: Number of hops for the clustering algorithm.

    Returns:
        clusters: List of clusters with their members.
    """
    su_channels = deepcopy(channels)
    # Initialize clusters and state for each node
    clusters = []
    # Initialize all nodes as 'initial'
    node_states = {node: 'initial' for node in nodes}

    while any(state == 'initial' for state in node_states.values()):
        for node in nodes:
            if node_states[node] in {'clustered_CM', 'clustered_CH'}:
                continue

            # Step 1: Initialize participants and intermediate cluster for node
            participants_i = set()
            intermediate_cluster_i = set()
            w_i = 0  # Initial weight for node's cluster
            # Get the position of the current node
            node_position = su_positions[node]

            # Step 2: Neighbor Selection - Finding participants within 1-hop range that share at least one channel
            # Map neighbor data with weight and position
            # Filter edges for the specific node
            # Filter out any empty sets if they exist
            # Filter edges for the specific node
            # print(f"Node {node}, channels {channels[node]}")
            k_hop_neighbors = get_k_hop_neighbors(node, edges, k)
            # print(f"Node {node}, {k}-hop-neighbors : {k_hop_neighbors}")
            node_edges = [(neighbor, common_channels) for node_i, neighbor,
                          common_channels in edges if node_i == node and neighbor in k_hop_neighbors]
            # for node_i, neighbor,common_channels in edges:
            #     print(f"node_i: {node_i} node : {node} neighbor : {neighbor}, k_hop_neighbors {k_hop_neighbors}")
            # print(f"Node {node}, node_edges : {node_edges}")

            # Ensure edges are in the correct format and store neighbor data
            neighbor_data = {
                neighbor: {
                    "position": su_positions[neighbor],
                    "channels": neighbor_channels
                }
                for neighbor, neighbor_channels in node_edges
            }
            # print("neighbor_data :", neighbor_data)
            # Neighbor selection step - adding neighbors to participants if they are in initial state and share channels
            for neighbor, neighbor_channels in node_edges:
                # Calculate Euclidean distance
                dx = su_positions[neighbor][0] - node_position[0]
                dy = su_positions[neighbor][1] - node_position[1]
                distance = sqrt(dx**2 + dy**2)
                # print("distance :", distance)
                if (
                    node_states[neighbor] not in {
                        'clustered_CM', 'clustered_CH'}
                    and neighbor_channels.intersection(channels[node])
                    and distance <= su_transmission_range
                ):
                    participants_i.add(neighbor)

            # Step 3: Bipartite Graph Construction and Maximum Weight Calculation using MWCBG
            # print(f"Node {node}, participants_i :", participants_i)
            cmn_i, PCM_i, w_i = MWCBG(
                node, node_position, channels[node], participants_i, neighbor_data, channel_qualities)

            # Step 4: Cluster Head Selection
            # print(f"Step 4: Cluster Head Selection, before the condition ,cmn_i :{cmn_i}, PCM_i : {PCM_i}, w_i :{w_i}")
            if len(cmn_i) >= 2 and PCM_i:
                # Mark node as intermediate cluster head
                # print("Step 4: Cluster Head Selection, in the condition node_states[node]:", node_states[node])
                if node_states[node] == 'initial':
                    node_states[node] = 'intermediate_CH'
                    # Nodes in PCM_i become cluster members
                    CM_i = {p[0] for p in PCM_i}

                    # Send 'join' message to each member in CM_i
                    for member in CM_i:
                        # Adding to intermediate cluster and marking as clustered_CM if it receives join only from this node
                        if node_states[member] == 'initial':
                            node_states[member] = 'clustered_CM'
                            intermediate_cluster_i.add(member)

            # Step 5: Edge Contraction - Updating graph structure
            contracted_node = node  # The node itself will act as the cluster representative
            new_edges = []
            for member in intermediate_cluster_i:
                # Contract edges, excluding edges from `member` back to `node`
                for edge in edges:
                    if edge[0] == member and edge[1] != contracted_node:
                        new_edges.append((contracted_node, edge[1], edge[2]))
                    elif edge[1] == member and edge[0] != contracted_node:
                        new_edges.append((edge[0], contracted_node, edge[2]))
                # Remove old member edges
                edges = [edge for edge in edges if edge[0]
                         != member and edge[1] != member]
            edges += new_edges

            # Set common channels for the contracted node
            channels[contracted_node] = cmn_i.copy()

            # Step 6: Cluster Finalization
            # print("Step 6: Cluster Finalization, before the if condition PCM_i value :", PCM_i)
            if len(cmn_i) < 2 or not any(w_i > weight for _, _, weight in PCM_i):
                # print("Step 6: Cluster Finalization, in the condition value of intermediate_cluster_i: ", intermediate_cluster_i)
                node_states[node] = 'clustered_CH'
                final_cluster = intermediate_cluster_i | {node}
                clusters.append(final_cluster)
    # Validation for clusters are they satisfies the paper requirements
    for cluster in clusters:
        if len(cluster) > 1:
            common_channels = set.intersection(*(su_channels[node] for node in cluster))
            assert len(common_channels) >= 2, "Cluster does not meet the common channels requirement"
    for cluster in clusters:
        for node in cluster:
            k_hop_neighbors = get_k_hop_neighbors(node, edges, k=2)
            assert all(neighbor in cluster for neighbor in k_hop_neighbors if neighbor in clusters), "k-hop constraint not satisfied"
    return clusters
