from collections import defaultdict
from itertools import combinations
import networkx as nx

from typing import List, Set

import numpy as np

from base_models import Channel, Node




class MWCBG:
    def __init__(self, preference_factor: float = 0.5):
        self.preference_factor = preference_factor

    def find_maximum_subgraph(
        self, node: "Node", available_channels: Set["Channel"], neighbors: List["Node"]
    ):
        if not neighbors or not available_channels or not node:
            return None, None, float("-inf")

        G = nx.Graph()
        G.add_nodes_from(neighbors, bipartite=0)  # Neighbors
        G.add_nodes_from(available_channels, bipartite=1)  # Channels

        if len(neighbors) == 1 and len(available_channels) >= 2:
            return (
                available_channels,
                neighbors,
                self._calculate_weight(node, neighbors, available_channels),
            )
        # print("neighbors :", len(neighbors))
        for neighbor in neighbors:
            # Filter available_channels for this neighbor
            # Intersection ensures only valid channels
            neighbor_channels = neighbor.channels & available_channels
            for channel in neighbor_channels:
                G.add_edge(neighbor, channel)

        # Project onto neighbors (set 0)
        projected_graph = nx.bipartite.projected_graph(G, neighbors)

        subgraphs = nx.enumerate_all_cliques(projected_graph)

        max_weight = float("-inf")
        best_subgraph = None
        best_common_channels = None

        for subgraph in subgraphs:
            if len(subgraph) < 2:
                continue

            common_channels = set.intersection(
                *[set(neighbor.channels) for neighbor in subgraph]
            )
            if len(common_channels) < 2:
                continue

            weight = self._calculate_weight(node, subgraph, common_channels)
            if weight > max_weight:
                max_weight = weight
                best_subgraph = subgraph
                best_common_channels = common_channels

        return best_common_channels, best_subgraph, max_weight

    def _calculate_weight(
        self,
        cluster_head: Node,
        cluster_members: List[Node],
        common_channels: Set[Channel],
    ) -> float:
        """
        Calculate weight based on channel quality and node distances.
        w = v * |N| * ΣQc + (1 - v) * |C| * Σ(1 - dij / Σdik)
        where:
        - v: preference_factor
        - |N|: number of members
        - ΣQc: sum of channel qualities
        - |C|: number of common channels
        - dij: distance between cluster head and member j
        - Σdik: sum of distances from cluster head to all members
        """
        v = self.preference_factor
        N = len(cluster_members)
        C = len(common_channels)

        if N == 0 or C == 0:
            return float("-inf")

        # Sum of channel qualities
        sum_qc = sum(channel.calculate_quality() for channel in common_channels)

        # Distances from cluster head to each member
        distances = [
            cluster_head.calculate_distance(member) for member in cluster_members
        ]
        sum_dik = sum(distances)
        if sum_dik == 0:
            # Avoid division by zero; assume minimal distance
            sum_dik = 1e-6

        # Calculate Σ(1 - dij / Σdik)
        sum_distance_factor = sum(1 - (dij / sum_dik) for dij in distances)

        # Compute weight
        weight = v * N * sum_qc + (1 - v) * C * sum_distance_factor
        return weight


def test_mwcbg():
    channels = {
        1: Channel(1, alpha=1, beta=2),
        2: Channel(2, alpha=2, beta=1),
        3: Channel(3, alpha=1, beta=1),
    }

    nodes = {
        1: Node(1, x=0, y=0, initial_energy=0.2, transmission_range=50),
        2: Node(2, x=1, y=1, initial_energy=0.2, transmission_range=50),
        3: Node(3, x=2, y=2, initial_energy=0.2, transmission_range=50),
        4: Node(4, x=3, y=6, initial_energy=0.2, transmission_range=50),
    }
    nodes[1].available_channels = [channels[1], channels[2], channels[3]]
    nodes[2].available_channels = [channels[1], channels[2], channels[3]]
    nodes[3].available_channels = [channels[2], channels[3]]
    nodes[4].available_channels = [channels[2], channels[3]]

    mwcbg = MWCBG(preference_factor=0.5)

    node = nodes[1]
    neighbors = [nodes[2], nodes[3], nodes[4]]
    available_channels = {channels[1], channels[2], channels[3]}

    common_channels, best_subgraph, weight = mwcbg.find_maximum_subgraph(
        node=node,
        neighbors=neighbors,
        available_channels=available_channels,
    )
    print(common_channels, best_subgraph, weight)


if __name__ == "__main__":
    test_mwcbg()

class MWCBG_Kerbosch:
    def __init__(self, preference_factor: float = 0.5):
        self.preference_factor = preference_factor

    def find_maximum_subgraph(
        self, node: "Node", available_channels: Set["Channel"], neighbors: List["Node"]
    ):
        if not neighbors or not available_channels or not node:
            return None, None, float("-inf")

        if len(neighbors) == 1 and len(available_channels) >= 2:
            return available_channels, neighbors, self._calculate_weight(node, neighbors, available_channels)

        # Build adjacency graph
        G = nx.Graph()
        G.add_nodes_from(neighbors)
        
        # Create node-to-channels mapping
        node_channels = {n: n.channels & available_channels for n in neighbors}
        
        # Add edges only between nodes with sufficient common channels
        for n1, n2 in combinations(neighbors, 2):
            common = node_channels[n1] & node_channels[n2]
            if len(common) >= 2:
                G.add_edge(n1, n2)

        if not G.edges():
            return None, None, float("-inf")

        # Find maximal cliques
        max_weight = float("-inf")
        best_subgraph = None
        best_common_channels = None

        for clique in self._find_cliques(G, node_channels):
            if len(clique) < 2:
                continue

            common_channels = set.intersection(*[node_channels[n] for n in clique])
            if len(common_channels) < 2:
                continue

            weight = self._calculate_weight(node, clique, common_channels)
            if weight > max_weight:
                max_weight = weight
                best_subgraph = clique
                best_common_channels = common_channels

        return best_common_channels, best_subgraph, max_weight

    def _find_cliques(self, G: nx.Graph, node_channels: dict):
        """Modified clique finding algorithm optimized for our use case"""
        def bronk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    common = set.intersection(*[node_channels[n] for n in R])
                    if len(common) >= 2:
                        yield list(R)
                return

            # Choose pivot from P∪X with most neighbors in P
            pivot_neighbors = set()
            if P:
                pivot = max(P, key=lambda v: len(set(G[v]) & P))
                pivot_neighbors = set(G[pivot]) & P

            # Try remaining vertices
            for v in P - pivot_neighbors:
                v_neighbors = set(G[v])
                yield from bronk(
                    R | {v},
                    P & v_neighbors,
                    X & v_neighbors
                )
                P = P - {v}
                X = X | {v}

        return bronk(set(), set(G.nodes()), set())

    def _calculate_weight(
        self,
        cluster_head: "Node",
        cluster_members: List["Node"],
        common_channels: Set["Channel"],
    ) -> float:
        N = len(cluster_members)
        C = len(common_channels)
        
        if N == 0 or C == 0:
            return float("-inf")

        sum_qc = sum(channel.calculate_quality() for channel in common_channels)
        distances = [cluster_head.calculate_distance(member) for member in cluster_members]
        sum_dik = sum(distances)
        
        if sum_dik == 0:
            sum_dik = 1e-6
            
        sum_distance_factor = sum(1 - (dij / sum_dik) for dij in distances)
        return self.preference_factor * N * sum_qc + (1 - self.preference_factor) * C * sum_distance_factor