from copy import deepcopy
import random
import numpy as np

from utility_functions import powerset, euclidean_distance

class NSAC:
    def __init__(self, num_sus, su_positions, su_channels, su_energies, channel_qualities, edges, preference_factor=0.5):
        """
        Initialize the NSAC algorithm with the given parameters.
        """
        self.num_sus = num_sus
        self.su_positions = su_positions
        self.su_channels = su_channels
        self.su_energies = deepcopy(su_energies)
        self.channel_qualities = channel_qualities
        self.edges = edges
        self.preference_factor = preference_factor

    def calculate_weight(self, node_id, neighbors_i, mcbg_edges):
        """
        Calculate the weight for a node based on NSAC formula.
        This weight focuses on spectrum stability and energy efficiency.
        """
        # Calculate spectrum stability weight using the edges from MWCBG
        spectrum_weight = sum(self.channel_qualities[edge[1]] for edge in mcbg_edges)

        # Calculate energy efficiency part using residual energy
        E_i = self.su_energies[node_id]
        E_neighbors = sum(self.su_energies[neighbor] for neighbor in neighbors_i)
        energy_weight = E_i / (E_i + E_neighbors) if (E_i + E_neighbors) > 0 else 0

        # Final weight calculation using the preference factor
        weight = (self.preference_factor * spectrum_weight) + \
                 ((1 - self.preference_factor) * energy_weight)
        return weight

    def find_cluster_head(self, nodes):
        if not nodes:
            return None
        return max(nodes, key=lambda n: self.su_energies[n])

    def form_clusters(self):
        """
        Main function to form clusters using the NSAC algorithm.
        """
        clusters = []
        visited = set()

        # Iterate over all nodes to form clusters
        for node in range(self.num_sus):
            if node in visited:
                continue

            # Step 1: Get node position, channels, and neighbors
            node_position = self.su_positions[node]
            channels_i = self.su_channels[node]

            # Get 1-hop neighbors
            neighbors_i = self.get_neighbors(node)
            if not neighbors_i:
                continue

            # Prepare neighbor data
            neighbor_data = {
                neighbor: {
                    "position": self.su_positions[neighbor],
                    "channels": self.su_channels[neighbor]
                }
                for neighbor in neighbors_i
            }

            # Step 2 and 3: Calculate MWCBG & Calculate weight using the NSAC-specific formula
            selected_channels, mcbg_edges, weight = self.MWCBG(
                node, node_position, channels_i, neighbors_i, neighbor_data
            )

            # If no valid bipartite graph found, skip to the next node
            if not mcbg_edges:
                continue

            # Step 4: Select the cluster head and form the cluster
            cluster_head = self.find_cluster_head(
                [node] + [edge[0] for edge in mcbg_edges])
            cluster = set([node] + [edge[0] for edge in mcbg_edges])

            # Mark nodes as visited
            visited.update(cluster)
            clusters.append(cluster)

        # Return the formed clusters and their corresponding cluster heads
        cluster_heads = [self.find_cluster_head(cluster) for cluster in clusters]
        return clusters, cluster_heads

    def get_neighbors(self, node):
        return {neighbor for neighbor, other, *_ in self.edges if other == node}.union(
            {other for neighbor, other, *_ in self.edges if neighbor == node}
        )

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

            subgraph_weight = self.calculate_weight(node_id, neighbor_subset, edges)

            if subgraph_weight > max_weight:
                max_weight = subgraph_weight
                PCM_i = {(neighbor, tuple(common_channels), subgraph_weight) for neighbor in neighbor_subset}
                selected_channels = common_channels

        return selected_channels, PCM_i, max_weight



class CogLEACH:
    def __init__(self, nodes,num_sus, su_positions, su_channels, channel_qualities, su_transmission_range, k_opt=10):
        """
        Initialize the CogLEACH algorithm with the given parameters. SU_TRANSMISSION_RANGE,  NODES
        """
        self.num_sus = num_sus
        self.su_positions = su_positions
        self.su_channels = su_channels
        self.channel_qualities = channel_qualities
        self.k_opt = k_opt  # Optimal number of cluster heads
        self.clusters = []
        self.cluster_heads = []
        self.nodes = nodes
        self.su_transmission_range = su_transmission_range
    
    def spectrum_sensing(self, node):
        """
        Perform spectrum sensing to determine the number of idle channels.
        """
        channels = self.su_channels[node]
        idle_channels = [ch for ch in channels if self.channel_qualities[ch] > 0]
        return len(idle_channels)
    
    def calculate_CH_probability(self, idle_channels_count, total_channels):
        """
        Calculate the probability of becoming a cluster head based on the number of idle channels.
        """
        alpha = total_channels / sum(self.spectrum_sensing(node) for node in self.nodes)
        return min(self.k_opt * idle_channels_count * alpha / total_channels, 1)
    
    def elect_cluster_heads(self):
        """
        Elect cluster heads based on the probability function.
        """
        for node in self.nodes:
            idle_channels_count = self.spectrum_sensing(node)
            total_channels = len(self.su_channels[node])
            
            # Calculate CH probability
            prob = self.calculate_CH_probability(idle_channels_count, total_channels)
            
            # Randomly elect CH based on the probability
            if random.random() < prob:
                self.cluster_heads.append(node)
    
    def form_clusters(self):
        """
        Form clusters around the elected cluster heads.
        """
        visited = set()
        for ch in self.cluster_heads:
            cluster = set()
            for node in self.nodes:
                if node != ch and node not in visited:
                    # Check if the node has common channels with the CH
                    common_channels = set(self.su_channels[ch]).intersection(self.su_channels[node])
                    if common_channels and euclidean_distance(self.su_positions[ch], self.su_positions[node]) <= self.su_transmission_range:
                        cluster.add(node)
                        visited.add(node)
            cluster.add(ch)
            self.clusters.append(cluster)
    
    def execute(self):
        """
        Execute the CogLEACH algorithm to form clusters.
        """
        self.elect_cluster_heads()
        self.form_clusters()
        return self.clusters, self.cluster_heads
