from copy import deepcopy
from utility_functions import MWCBG, get_k_hop_neighbors


class kSACBWEC:
    def __init__(self, num_sus, edges, su_positions, su_channels, channel_qualities,  get_available_channels, initial_energy, k_max=5, preference_factor=0.5, sensing_energy=1.31e-4):
        """
        Initialize the k-SACB-WEC algorithm with the given parameters.
        """
        self.num_sus = num_sus
        self.su_positions = su_positions
        self.su_channels = deepcopy(su_channels)
        self.original_edges = deepcopy(edges)
        self.original_channels = deepcopy(su_channels)
        self.su_energies = {i: initial_energy for i in range(self.num_sus)}
        self.channel_qualities = channel_qualities
        self.edges = edges
        self.get_available_channels = get_available_channels
        self.k_max = k_max
        self.preference_factor = preference_factor
        self.sensing_energy = sensing_energy
        self.clusters = []
        self.visited = set()

    def consume_energy(self, node, energy_amount):
        """
        Deduct energy from a given node's energy budget.
        """
        self.su_energies[node] -= energy_amount
        if self.su_energies[node] < 0:
            self.su_energies[node] = 0

    def reset(self):
        self.clusters = []
        self.cluster_heads = []
        self.visited = set()

    def form_clusters(self):
        """
        Main function to execute the k-SACB-WEC clustering algorithm with energy consumption.
        """
        for node in range(self.num_sus):
            if node in self.visited or self.su_energies[node] <= 0:
                continue

            best_cluster = set()
            best_weight = 0
            current_k = 1
            previous_neighbors = set()

            # Expand up to k_max to find the optimal cluster
            while current_k <= self.k_max:
                k_hop_neighbors = get_k_hop_neighbors(
                    node, self.edges, current_k)
                # print("current_k :", current_k, " k_max :", self.k_max)
                # If the set of neighbors doesn't change, break early
                if k_hop_neighbors == previous_neighbors:
                    break
                previous_neighbors = k_hop_neighbors

                # Prepare neighbor data using available channels
                neighbor_data = {
                    neighbor: {
                        "position": self.su_positions[neighbor],
                        "channels": self.get_available_channels(neighbor)
                    }
                    for neighbor in k_hop_neighbors
                }

                node_position = self.su_positions[node]
                channels_i = self.get_available_channels(node)

                # Step 2: Calculate MWCBG for the current k-hop neighborhood
                selected_channels, cluster, cluster_weight = MWCBG(
                    node, node_position, channels_i, k_hop_neighbors, neighbor_data, self.channel_qualities
                )

                # Stop expanding if no improvement is found
                if not cluster or (cluster_weight <= best_weight and len(cluster) <= len(best_cluster)):
                    break

                best_cluster = cluster
                best_weight = cluster_weight
                current_k += 1

            # Deduct energy for clustering process
            self.consume_energy(node, self.sensing_energy)

            # Deduct energy for each member in the cluster
            if best_cluster:
                for clstr in best_cluster:
                    self.consume_energy(clstr[0], energy_amount=0.0005)

                self.clusters.append({clstr[0] for clstr in best_cluster})
                self.visited.update({clstr[0] for clstr in best_cluster})

        return self.clusters

    def select_cluster_heads(self):
        """
        Select Cluster Heads (CH) for each cluster.
        """
        cluster_heads = []
        for cluster in self.clusters:
            node_ids = list(cluster)
            if node_ids:
                # Select the cluster head with the highest y-coordinate
                ch = max(node_ids, key=lambda n: self.su_positions[n][1])
                cluster_heads.append(ch)
        return cluster_heads

    def run(self):
        """
        Execute the algorithm and return the clusters and their cluster heads.
        """
        clusters = self.form_clusters()
        cluster_heads = self.select_cluster_heads()
        return clusters, cluster_heads
