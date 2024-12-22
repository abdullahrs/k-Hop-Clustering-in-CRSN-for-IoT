# k_sacb_wec.py

from typing import List, Set, Tuple
from dataclasses import dataclass
from simulation_environment import Node, CRSNEnvironment
from mwcbg import find_MWCBG
import networkx as nx


@dataclass
class HopBasedCluster:
    """Represents a cluster with specific hop count in the network"""

    id: int
    ch: Node  # Cluster Head
    members: Set[Node]  # Cluster Members
    common_channels: Set[int]  # Common channels for all nodes in cluster
    hop_count: int  # Number of hops for this cluster


class KSABWEC:
    """Implementation of k-hop Spectrum Aware clustering Without Edge Contraction"""

    def __init__(self, env: CRSNEnvironment):
        """
        Initialize k-SACB-WEC algorithm

        Args:
            env: CRSN simulation environment
        """
        self.env = env
        self.clusters: List[HopBasedCluster] = []
        self.cluster_id_counter = 0
        self.network_graph = nx.Graph()
        self._initialize_network_graph()

    def _initialize_network_graph(self):
        """Initialize network graph from environment nodes"""
        # Add nodes
        for node in self.env.nodes:
            self.network_graph.add_node(node)

        # Add edges where nodes have common channels
        for node1 in self.env.nodes:
            for node2 in self.env.nodes:
                if node1 != node2:
                    common_channels = (
                        node1.available_channels & node2.available_channels
                    )
                    if len(common_channels) >= 2:  # Bi-channel connectivity requirement
                        distance = node1.calculate_distance(node2)
                        if distance <= node1.transmission_range:
                            self.network_graph.add_edge(node1, node2)

    def _get_k_hop_neighbors(self, node: Node, k: int) -> Set[Node]:
        """Get all neighbors within k hops of the node"""
        neighbors = set()
        current_nodes = {node}

        for hop in range(k):
            next_nodes = set()
            for current in current_nodes:
                next_nodes.update(self.network_graph.neighbors(current))
            next_nodes -= neighbors  # Remove already found neighbors
            next_nodes.discard(node)  # Remove source node
            neighbors.update(next_nodes)
            current_nodes = next_nodes

        return neighbors

    def _find_optimal_hop_count(
        self, node: Node, max_hops: int
    ) -> Tuple[int, float, Set[Node], Set[int]]:
        """
        Find optimal hop count for a node that maximizes clustering weight

        Returns:
            Tuple of (optimal_hop_count, max_weight, best_nodes, best_channels)
        """
        best_hop_count = 1
        max_weight = 0.0
        best_nodes = set()
        best_channels = set()

        for k in range(1, max_hops + 1):
            # Get k-hop neighbors
            k_hop_neighbors = self._get_k_hop_neighbors(node, k)
            if not k_hop_neighbors:
                continue

            # Find maximum weight complete bipartite subgraph
            # Verify common channels across all k-hop neighbors before MWCBG
            common_channels = node.available_channels.copy()
            for neighbor in k_hop_neighbors:
                common_channels &= neighbor.available_channels

            if (
                len(common_channels) >= 2
            ):  # Only proceed if we have bi-channel connectivity
                result = find_MWCBG(
                    node=node,
                    channels=common_channels,  # Use only verified common channels
                    neighbors=k_hop_neighbors,
                )
            else:
                break

            if result.weight > max_weight and len(result.channels) >= 2:
                max_weight = result.weight
                best_hop_count = k
                best_nodes = result.nodes
                best_channels = result.channels

        return best_hop_count, max_weight, best_nodes, best_channels

    def _form_cluster(
        self, ch: Node, members: Set[Node], channels: Set[int], hop_count: int
    ) -> HopBasedCluster:
        """Form a cluster with given parameters"""
        cluster = HopBasedCluster(
            id=self.cluster_id_counter,
            ch=ch,
            members=members | {ch},
            common_channels=channels,
            hop_count=hop_count,
        )

        self.cluster_id_counter += 1

        # Update node states
        ch.state = "clustered_CH"
        for member in members:
            member.state = "clustered_CM"

        return cluster

    def form_clusters(self, max_hops: int = 2) -> List[HopBasedCluster]:
        """
        Form k-hop clusters without edge contraction

        Args:
            max_hops: Maximum number of hops allowed in clusters

        Returns:
            List of formed clusters
        """
        unclustered_nodes = set(self.network_graph.nodes())

        while unclustered_nodes:
            # Find node with best clustering potential
            best_node = None
            best_hop_count = 0
            max_weight = 0.0
            best_members = set()
            best_channels = set()

            for node in unclustered_nodes:
                hop_count, weight, members, channels = self._find_optimal_hop_count(
                    node, max_hops
                )

                # Only consider members that are still unclustered
                members &= unclustered_nodes
                if members and weight > max_weight:
                    best_node = node
                    best_hop_count = hop_count
                    max_weight = weight
                    best_members = members
                    best_channels = channels

            if best_node is None:
                # No more clusters can be formed
                break

            # Form cluster with optimal parameters
            cluster = self._form_cluster(
                ch=best_node,
                members=best_members,
                channels=best_channels,
                hop_count=best_hop_count,
            )

            self.clusters.append(cluster)

            # Remove clustered nodes from unclustered set
            unclustered_nodes -= cluster.members

        # Handle remaining unclustered nodes
        for node in unclustered_nodes:
            # Form single-node clusters
            cluster = self._form_cluster(
                ch=node, members=set(), channels=node.available_channels, hop_count=0
            )
            self.clusters.append(cluster)

        return self.clusters
