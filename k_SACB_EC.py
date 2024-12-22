# k_sacb_ec.py

from typing import List, Set
from dataclasses import dataclass
from simulation_environment import Node, CRSNEnvironment
from mwcbg import find_MWCBG
import networkx as nx

@dataclass
class Cluster:
    """Represents a cluster in the network"""
    id: int
    ch: Node                     # Cluster Head
    members: Set[Node]          # Cluster Members
    common_channels: Set[int]    # Common channels for all nodes in cluster

class KSABEC:
    """Implementation of k-hop Spectrum Aware clustering with Edge Contraction"""
    
    def __init__(self, env: CRSNEnvironment):
        """
        Initialize k-SACB-EC algorithm
        
        Args:
            env: CRSN simulation environment
        """
        self.env = env
        self.clusters: List[Cluster] = []
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
                    common_channels = node1.available_channels & node2.available_channels
                    if len(common_channels) >= 2:  # Bi-channel connectivity requirement
                        distance = node1.calculate_distance(node2)
                        if distance <= node1.transmission_range:
                            self.network_graph.add_edge(node1, node2)
    
    def _contract_edge(self, cluster: Cluster):
        """
        Perform edge contraction for a cluster in the network graph
        """
        # Get all neighbors of cluster members before removal
        cluster_neighbors = set()
        for member in cluster.members:
            if member != cluster.ch:
                cluster_neighbors.update(set(self.network_graph.neighbors(member)))
        
        # Remove all cluster members except CH at once
        self.network_graph.remove_nodes_from(
            [n for n in cluster.members if n != cluster.ch]
        )
        
        # Add edges to CH in batch
        new_edges = [
            (cluster.ch, neighbor) for neighbor in cluster_neighbors 
            if (neighbor not in cluster.members and 
                len(cluster.common_channels & neighbor.available_channels) >= 2)
        ]
        self.network_graph.add_edges_from(new_edges)
    
    def _form_initial_clusters(self) -> List[Cluster]:
        """
        Form initial clusters using MWCBG procedure
        
        Returns:
            List of formed clusters
        """
        initial_clusters = []
        unclustered_nodes = set(self.network_graph.nodes())
        
        while unclustered_nodes:
            # Select node with highest weight among unclustered nodes
            current_node = None
            max_weight = -1
            
            for node in unclustered_nodes:
                neighbors = set(self.network_graph.neighbors(node))
                if not neighbors:
                    continue
                    
                result = find_MWCBG(
                    node=node,
                    channels=node.available_channels,
                    neighbors=neighbors & unclustered_nodes
                )
                
                if result.weight > max_weight:
                    max_weight = result.weight
                    current_node = node
            
            if current_node is None:
                break
                
            # Form cluster with selected node
            neighbors = set(self.network_graph.neighbors(current_node))
            result = find_MWCBG(
                node=current_node,
                channels=current_node.available_channels,
                neighbors=neighbors & unclustered_nodes
            )
            
            if result.nodes:  # If valid cluster can be formed
                # Set up cluster head
                current_node.set_as_cluster_head()
                
                # Add members to cluster
                for member in result.nodes:
                    member.join_cluster(current_node)
                
                cluster = Cluster(
                    id=self.cluster_id_counter,
                    ch=current_node,
                    members=result.nodes | {current_node},
                    common_channels=result.channels
                )
                initial_clusters.append(cluster)
                self.cluster_id_counter += 1
                
                # Update node states
                current_node.state = "clustered_CH"
                for member in result.nodes:
                    member.state = "clustered_CM"
                
                # Remove clustered nodes from unclustered set
                unclustered_nodes -= cluster.members
            else:
                # If no valid cluster can be formed, make single node cluster
                cluster = Cluster(
                    id=self.cluster_id_counter,
                    ch=current_node,
                    members={current_node},
                    common_channels=current_node.available_channels
                )
                initial_clusters.append(cluster)
                self.cluster_id_counter += 1
                current_node.state = "clustered_CH"
                unclustered_nodes.remove(current_node)
        
        return initial_clusters
    
    def _merge_clusters(self, clusters: List[Cluster]) -> List[Cluster]:
        """
        Merge clusters where possible while maintaining bi-channel connectivity
        
        Args:
            clusters: List of clusters to merge
            
        Returns:
            List of merged clusters
        """
        merged = True
        while merged:
            merged = False
            for i, cluster1 in enumerate(clusters):
                for j, cluster2 in enumerate(clusters[i+1:], i+1):
                    common_channels = cluster1.common_channels & cluster2.common_channels
                    
                    # Check if clusters can be merged
                    if len(common_channels) >= 2:
                        # Check if cluster heads are neighbors
                        if self.network_graph.has_edge(cluster1.ch, cluster2.ch):
                            # Merge clusters
                            new_cluster = Cluster(
                                id=self.cluster_id_counter,
                                ch=cluster1.ch if cluster1.ch.residual_energy > cluster2.ch.residual_energy 
                                   else cluster2.ch,
                                members=cluster1.members | cluster2.members,
                                common_channels=common_channels
                            )
                            
                            self.cluster_id_counter += 1
                            clusters.remove(cluster1)
                            clusters.remove(cluster2)
                            clusters.append(new_cluster)
                            merged = True
                            break
                if merged:
                    break
                    
        return clusters
    
    def form_clusters(self, max_hops: int = 2) -> List[Cluster]:
        """
        Form k-hop clusters using edge contraction
        
        Args:
            max_hops: Maximum number of hops allowed in clusters
            
        Returns:
            List of formed clusters
        """
        # Step 1: Form initial clusters
        self.clusters = self._form_initial_clusters()
        
        # Step 2: Contract edges for each cluster
        for cluster in self.clusters:
            self._contract_edge(cluster)
            
        # Step 3: Merge clusters while possible
        self.clusters = self._merge_clusters(self.clusters)
        
        return self.clusters