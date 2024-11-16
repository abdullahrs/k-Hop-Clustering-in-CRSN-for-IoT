from copy import deepcopy

from utility_functions import MWCBG, get_k_hop_neighbors

def k_SACB_WEC(num_sus, su_positions, su_channels, channel_qualities, edges, k_max=5, preference_factor=0.5):
    clusters = []
    visited = set()

    for node in range(num_sus):
        if node in visited:
            continue

        best_cluster = set()
        best_weight = 0
        current_k = 1
        previous_neighbors = set()
        while current_k <= k_max:
            k_hop_neighbors = get_k_hop_neighbors(node, edges, current_k)
            print(f"Node {node}, k={current_k}, neighbors: {k_hop_neighbors}")

            # If the set of neighbors doesn't change, break early
            if k_hop_neighbors == previous_neighbors:
                break
            
            neighbor_data = {
                neighbor: {
                    "position": su_positions[neighbor],
                    "channels": su_channels[neighbor]
                }
                for neighbor in k_hop_neighbors
            }
            
            node_position = su_positions[node]
            channels_i = su_channels[node]
            
            selected_channels, cluster, cluster_weight = MWCBG(
                node, node_position, channels_i, k_hop_neighbors, neighbor_data, channel_qualities)
            
            if not cluster or (cluster_weight <= best_weight and len(cluster) <= len(best_cluster)):
                break

            best_cluster = cluster
            best_weight = cluster_weight
            current_k += 1

        if best_cluster:
            clusters.append({clstr[0] for clstr in best_cluster})
            visited.update({clstr[0] for clstr in best_cluster})

    cluster_heads = []
    for cluster in clusters:
        node_ids = list(cluster)
        if node_ids:
            ch = max(node_ids, key=lambda n: su_positions[n][1])
            cluster_heads.append(ch)
    
    print("Final k value reached:", current_k)
    return clusters, cluster_heads

