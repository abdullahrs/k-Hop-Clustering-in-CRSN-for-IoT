from copy import deepcopy
import random
import numpy as np

from utility_functions import powerset, euclidean_distance


class NSAC:
    def __init__(self, num_sus, su_positions, su_channels, channel_qualities, edges, get_available_channels, initial_energy, preference_factor=0.5, sensing_energy=1.31e-49):
        """
        Initialize the NSAC algorithm with the given parameters.
        """
        self.num_sus = num_sus
        self.su_positions = su_positions
        self.su_channels = deepcopy(su_channels)
        self.su_energies = {i: initial_energy for i in range(self.num_sus)}
        self.channel_qualities = channel_qualities
        self.edges = edges
        self.get_available_channels = get_available_channels
        self.preference_factor = preference_factor
        self.sensing_energy = sensing_energy

        self.clusters = []
        self.cluster_heads = []

    def reset(self):
        self.clusters = []
        self.cluster_heads = []

    def consume_energy(self, node, amount):
        """
        Deduct the specified amount of energy from the node's remaining energy.
        """
        self.su_energies[node] -= amount
        if self.su_energies[node] < 0:
            self.su_energies[node] = 0

    def spectrum_sensing(self, node):
        """
        Perform spectrum sensing to determine the number of idle channels.
        Consume energy during spectrum sensing.
        """
        if self.su_energies[node] <= 0:
            return set()

        channels = self.get_available_channels(node)
        # Consume energy for spectrum sensing
        self.consume_energy(node, self.sensing_energy)
        return channels

    def calculate_weight(self, node_id, neighbors_i, mcbg_edges):
        """
        Calculate the weight for a node based on NSAC formula.
        """
        # Spectrum stability weight
        spectrum_weight = sum(
            self.channel_qualities[edge[1]] for edge in mcbg_edges)

        # Energy efficiency weight using residual energy
        E_i = self.su_energies[node_id]
        E_neighbors = sum(self.su_energies[neighbor]
                          for neighbor in neighbors_i)
        energy_weight = E_i / \
            (E_i + E_neighbors) if (E_i + E_neighbors) > 0 else 0

        # Final weight calculation using preference factor
        weight = (self.preference_factor * spectrum_weight) + \
            ((1 - self.preference_factor) * energy_weight)
        return weight

    def find_cluster_head(self, nodes):
        """
        Select the Cluster Head (CH) with the highest energy among a set of nodes.
        """
        if not nodes:
            return None
        return max(nodes, key=lambda n: self.su_energies[n])

    def consume_energy_for_communication(self, node, distance):
        """
        Consume energy for communication based on distance.
        """
        energy_cost = 10e-12 * (distance ** 2)
        self.consume_energy(node, energy_cost)

    def form_clusters(self):
        clusters = []
        visited = set()

        for node in range(self.num_sus):
            if node in visited or self.su_energies[node] <= 0:
                continue

            node_position = self.su_positions[node]
            channels_i = self.spectrum_sensing(node)
            if not channels_i:
                continue

            # Get 1-hop neighbors
            neighbors_i = self.get_neighbors(node)
            if not neighbors_i:
                continue

            # Prepare neighbor data using available channels
            neighbor_data = {
                neighbor: {
                    "position": self.su_positions[neighbor],
                    "channels": self.spectrum_sensing(neighbor)
                }
                for neighbor in neighbors_i
            }

            # Step 2 and 3: Calculate MWCBG & Calculate weight using the NSAC-specific formula
            selected_channels, mcbg_edges, weight = self.MWCBG(
                node, node_position, channels_i, neighbors_i, neighbor_data
            )

            if not mcbg_edges:
                continue

            # Step 4: Select the cluster head and form the cluster
            cluster_head = self.find_cluster_head(
                [node] + [edge[0] for edge in mcbg_edges])
            cluster = set([node] + [edge[0] for edge in mcbg_edges])

            for member in cluster:
                if member != cluster_head:
                    distance = euclidean_distance(
                        self.su_positions[cluster_head], self.su_positions[member])
                    self.consume_energy_for_communication(member, distance)

            visited.update(cluster)
            clusters.append(cluster)

        self.clusters = clusters
        self.cluster_heads = [self.find_cluster_head(
            cluster) for cluster in clusters]

    def MWCBG(self, node_id, node_position, channels_i, neighbors_i, neighbor_data):
        edges = []
        for neighbor_id in neighbors_i:
            neighbor_channels = neighbor_data[neighbor_id]["channels"]
            common_channels = channels_i.intersection(neighbor_channels)
            if common_channels:
                for channel in common_channels:
                    edges.append((neighbor_id, channel))

        max_weight = 0
        PCM_i = set()
        selected_channels = set()

        for neighbor_subset in powerset(neighbors_i):
            if not neighbor_subset:
                continue

            common_channels = set.intersection(
                *(neighbor_data[neighbor]["channels"] for neighbor in neighbor_subset)).intersection(channels_i)
            if not common_channels:
                continue

            subgraph_weight = self.calculate_weight(
                node_id, neighbor_subset, edges)

            if subgraph_weight > max_weight:
                max_weight = subgraph_weight
                PCM_i = {(neighbor, tuple(common_channels), subgraph_weight)
                         for neighbor in neighbor_subset}
                selected_channels = common_channels

        return selected_channels, PCM_i, max_weight

    def get_neighbors(self, node):
        """
        Get 1-hop neighbors of a node based on the edges list.
        """
        return {neighbor for neighbor, other, *_ in self.edges if other == node}.union(
            {other for neighbor, other, *_ in self.edges if neighbor == node}
        )

    def run(self):
        """
        Execute the NSAC algorithm and return clusters and cluster heads.
        """
        self.form_clusters()
        return self.clusters, self.cluster_heads


class CogLEACH:
    def __init__(self, nodes, num_sus, su_positions, su_channels, channel_qualities, su_transmission_range, get_available_channels, initial_energy, k_opt=10, sensing_energy=1.31e-49):
        """
        Initialize the CogLEACH algorithm with the given parameters.
        """
        self.nodes = nodes
        self.num_sus = num_sus
        self.su_positions = su_positions
        self.su_channels = deepcopy(su_channels)
        self.original_channels = deepcopy(su_channels)
        self.su_energies = {node: initial_energy for node in self.nodes}
        self.channel_qualities = channel_qualities
        self.k_opt = k_opt
        self.su_transmission_range = su_transmission_range
        self.get_available_channels = get_available_channels
        self.sensing_energy = sensing_energy
        self.clusters = []
        self.cluster_heads = []

    def reset(self):
        self.clusters = []
        self.cluster_heads = []

    def consume_energy(self, node, amount):
        """
        Deduct the specified amount of energy from the node's remaining energy.
        """
        self.su_energies[node] -= amount
        if self.su_energies[node] < 0:
            self.su_energies[node] = 0

    def spectrum_sensing(self, node):
        """
        Perform spectrum sensing to determine the number of idle channels.
        Consume energy during spectrum sensing.
        """
        if self.su_energies[node] <= 0:
            return 0  # Return 0 if the node is out of energy

        channels = self.get_available_channels(node)
        idle_channels_count = len(channels)

        # Consume energy for spectrum sensing
        self.consume_energy(node, self.sensing_energy)

        return idle_channels_count  # Return the count of idle channels

    def calculate_CH_probability(self, idle_channels_count, total_channels):
        """
        Calculate the probability of becoming a cluster head based on the number of idle channels.
        """
        # Calculate alpha using the sum of idle channels across all nodes
        total_idle_channels = sum(self.spectrum_sensing(node)
                                  for node in self.nodes if self.su_energies[node] > 0)

        if total_idle_channels == 0:
            return 0  # Prevent division by zero if there are no idle channels

        alpha = total_channels / total_idle_channels
        return min(self.k_opt * idle_channels_count * alpha / total_channels, 1)

    def elect_cluster_heads(self):
        """
        Elect cluster heads based on the probability function.
        """
        for node in self.nodes:
            if self.su_energies[node] <= 0:
                continue

            # Get the count of idle channels directly
            idle_channels_count = self.spectrum_sensing(node)
            total_channels = len(self.su_channels[node])

            # Calculate CH probability
            prob = self.calculate_CH_probability(
                idle_channels_count, total_channels)

            # Randomly elect CH based on the probability
            if random.random() < prob:
                self.cluster_heads.append(node)

    def consume_energy_for_communication(self, node, distance):
        """
        Consume energy for communication based on distance.
        """
        energy_cost = 10e-12 * (distance ** 2)
        self.consume_energy(node, energy_cost)

    def form_clusters(self):
        """
        Form clusters around the elected cluster heads.
        """
        visited = set()
        for ch in self.cluster_heads:
            if self.su_energies[ch] <= 0:
                continue

            cluster = set()
            for node in self.nodes:
                if node != ch and node not in visited and self.su_energies[node] > 0:
                    common_channels = set(self.su_channels[ch]).intersection(
                        self.su_channels[node])
                    distance = euclidean_distance(
                        self.su_positions[ch], self.su_positions[node])

                    if common_channels and distance <= self.su_transmission_range:
                        cluster.add(node)
                        visited.add(node)

                        # Consume energy for communication
                        self.consume_energy_for_communication(node, distance)

            cluster.add(ch)
            self.clusters.append(cluster)

    def run(self):
        """
        Execute the CogLEACH algorithm to form clusters.
        """
        self.elect_cluster_heads()
        self.form_clusters()
        return self.clusters, self.cluster_heads
