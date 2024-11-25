from copy import deepcopy
from math import sqrt
from utility_functions import MWCBG, get_k_hop_neighbors


class kSACBEC:
    def __init__(self, nodes, edges, channels, su_positions, channel_qualities, su_transmission_range, get_available_channels, initial_energy, sensing_energy=1.31e-4, k=2):
        self.nodes = nodes
        self.edges = deepcopy(edges)
        self.channels = deepcopy(channels)
        self.original_edges = deepcopy(edges)
        self.original_channels = deepcopy(channels)
        self.su_positions = su_positions
        self.channel_qualities = channel_qualities
        self.su_energies = {node: initial_energy for node in self.nodes}
        self.su_transmission_range = su_transmission_range
        self.k = k
        self.node_states = {node: 'initial' for node in nodes}
        self.clusters = []
        self.cluster_heads = []
        self.sensing_energy = sensing_energy
        self.get_available_channels = get_available_channels

    def consume_energy(self, node, energy_amount):
        """Reduce the energy of a given node by a specified amount."""
        self.su_energies[node] -= energy_amount
        if self.su_energies[node] < 0:
            self.su_energies[node] = 0

    def reset(self):
        self.node_states = {node: 'initial' for node in self.nodes}
        self.clusters = []
        self.cluster_heads = []
        self.channels = deepcopy(self.original_channels)
        self.edges = deepcopy(self.original_edges)

    def form_clusters(self):
        """
        Execute the k-SACB-EC clustering algorithm with correct energy consumption.
        """
        while any(state == 'initial' for state in self.node_states.values()):
            progress_made = False
            for node in self.nodes:
                # Skip nodes that are already clustered or have no energy
                if self.node_states[node] in {'clustered_CM', 'clustered_CH'} or self.su_energies[node] <= 0:
                    continue

                participants_i = set()
                intermediate_cluster_i = set()
                node_position = self.su_positions[node]

                # Step 1: Get k-hop neighbors and consume energy for sensing
                k_hop_neighbors = get_k_hop_neighbors(node, self.edges, self.k)
                self.consume_energy(node, self.sensing_energy)
                if self.su_energies[node] <= 0:
                    continue

                # Prepare neighbor data
                node_edges = [
                    (neighbor, common_channels)
                    for node_i, neighbor, common_channels in self.edges
                    if node_i == node and neighbor in k_hop_neighbors
                ]
                neighbor_data = {
                    neighbor: {
                        "position": self.su_positions[neighbor],
                        "channels": self.get_available_channels(neighbor)
                    }
                    for neighbor, neighbor_channels in node_edges
                }

                available_channels = self.get_available_channels(node)

                # Neighbor selection step - adding neighbors to participants if they are in initial state and share channels
                for neighbor, neighbor_channels in node_edges:
                    dx = self.su_positions[neighbor][0] - node_position[0]
                    dy = self.su_positions[neighbor][1] - node_position[1]
                    distance = sqrt(dx**2 + dy**2)

                    if (
                        self.node_states[neighbor] not in {'clustered_CM', 'clustered_CH'} and
                        neighbor_channels.intersection(available_channels) and
                        distance <= self.su_transmission_range
                    ):
                        participants_i.add(neighbor)

                # Step 3: Bipartite Graph Construction and Maximum Weight Calculation using MWCBG
                cmn_i, PCM_i, w_i = MWCBG(
                    node, node_position, available_channels, participants_i, neighbor_data, self.channel_qualities
                )

                # Consume energy for performing MWCBG
                self.consume_energy(node, self.sensing_energy)

                # Step 4: Cluster Head Selection
                if len(cmn_i) >= 2 and PCM_i:
                    if self.node_states[node] == 'initial':
                        self.node_states[node] = 'intermediate_CH'
                        CM_i = {p[0] for p in PCM_i}

                        # Send 'join' message to each member in CM_i
                        for member in CM_i:
                            if self.node_states[member] == 'initial':
                                self.consume_energy(
                                    member, self.sensing_energy)
                                self.node_states[member] = 'clustered_CM'
                                progress_made = True
                                intermediate_cluster_i.add(member)

                # Step 5: Edge Contraction
                contracted_node = node
                new_edges = []
                for member in intermediate_cluster_i:
                    for edge in self.edges:
                        if edge[0] == member and edge[1] != contracted_node:
                            new_edges.append(
                                (contracted_node, edge[1], edge[2]))
                        elif edge[1] == member and edge[0] != contracted_node:
                            new_edges.append(
                                (edge[0], contracted_node, edge[2]))

                    # Remove old member edges
                    self.edges = [
                        edge for edge in self.edges if edge[0] != member and edge[1] != member]

                    # Consume energy for contracting edges
                    self.consume_energy(member, self.sensing_energy)

                self.edges += new_edges
                self.channels[contracted_node] = cmn_i.copy()

                # Step 6: Finalize Cluster
                if len(cmn_i) < 2 or not any(w_i > weight for _, _, weight in PCM_i):
                    self.node_states[node] = 'clustered_CH'
                    final_cluster = intermediate_cluster_i | {node}
                    self.clusters.append(final_cluster)
                    self.cluster_heads.append(
                        self.select_cluster_head(final_cluster))
                    self.consume_energy(node, self.sensing_energy)
            if not progress_made:
                print("progress_made :",progress_made)
                break
        return self.clusters, self.cluster_heads

    def select_cluster_head(self, cluster):
        """Select the cluster head based on the highest residual energy."""
        return max(cluster, key=lambda n: self.su_energies[n])

    def validate_clusters(self):
        """Validate that the formed clusters satisfy the requirements."""
        for cluster in self.clusters:
            if len(cluster) > 1:
                common_channels = set.intersection(
                    *(self.channels[node] for node in cluster))
                assert len(
                    common_channels) >= 2, "Cluster does not meet the common channels requirement"

    def run(self):
        clusters, cluster_heads = self.form_clusters()
        self.validate_clusters()
        return clusters, cluster_heads
