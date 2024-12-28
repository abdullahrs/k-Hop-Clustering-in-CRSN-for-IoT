# mwcbg.py

from typing import List, Set, Dict, Tuple, Optional
import networkx as nx
from dataclasses import dataclass
from simulation_environment import Node, Channel

@dataclass
class BipartiteGraphResult:
    """Result structure for MWCBG procedure"""
    nodes: Set[Node]          # Selected nodes in maximum weight complete bipartite subgraph
    channels: Set[int]        # Common channels for selected nodes
    weight: float            # Weight of the complete bipartite subgraph

class MWCBG:
    """Maximum Weight Complete Bipartite Graph procedure implementation"""
    
    def __init__(self, node: Node, channels: Set[int], neighbors: Set[Node]):
        """
        Initialize MWCBG procedure
        
        Args:
            node: Source node (i)
            channels: Available channels at node i
            neighbors: Set of node i's neighbors
        """
        self.source_node = node
        self.available_channels = channels
        self.neighbors = neighbors
        self.graph = nx.Graph()
        
    def construct_bipartite_graph(self) -> nx.Graph:
        """
        Construct bipartite graph G(Ni ∪ Ci, Ei) as per paper
        
        Returns:
            Constructed bipartite graph
        """
        # Add nodes to first partition (neighbors)
        self.graph.add_nodes_from(self.neighbors, bipartite=0)
        
        # Add nodes to second partition (channels)
        self.graph.add_nodes_from(self.available_channels, bipartite=1)
        
        # Add edges between neighbors and their available channels
        for neighbor in self.neighbors:
            for channel in self.available_channels:
                if channel in neighbor.available_channels:
                    self.graph.add_edge(neighbor, channel)
                    
        return self.graph
        
    def find_maximum_weight_complete_bipartite(self) -> BipartiteGraphResult:
        """
        Find maximum weight complete bipartite subgraph
        
        Returns:
            BipartiteGraphResult containing selected nodes, channels and weight
        """
        if not self.neighbors or not self.available_channels:
            return BipartiteGraphResult(set(), set(), 0.0)
            
        # Get all possible combinations of nodes and channels
        max_weight = 0.0
        best_nodes = set()
        best_channels = set()
        
        # Try all possible subsets of neighbors
        for n in range(1, len(self.neighbors) + 1):
            for node_subset in self._get_subsets(self.neighbors, n):
                # Find common channels for this subset
                common_channels = self._get_common_channels(node_subset)
                
                if len(common_channels) >= 2:  # Bi-channel connectivity requirement
                    weight = self._calculate_subgraph_weight(node_subset, common_channels)
                    
                    if weight > max_weight:
                        max_weight = weight
                        best_nodes = set(node_subset)
                        best_channels = common_channels
                        
        return BipartiteGraphResult(best_nodes, best_channels, max_weight)
    
    def find_maximum_weight_complete_bipartite(self) -> BipartiteGraphResult:
        """
        Find maximum weight complete bipartite subgraph using NetworkX
        """
        if not self.neighbors or not self.available_channels:
            return BipartiteGraphResult(set(), set(), 0.0)

        # Get the bipartite graph
        G = self.construct_bipartite_graph()
        
        max_weight = 0.0
        best_nodes = set()
        best_channels = set()
        
        # Get node sets from bipartite graph
        nodes_set = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
        channels_set = {n for n, d in G.nodes(data=True) if d['bipartite'] == 1}
        
        # Use maximum biclique algorithm if available in NetworkX
        # Otherwise, use our optimized approach
        for node_subset in nx.enumerate_all_cliques(nx.bipartite.projected_graph(G, nodes_set)):
            if len(node_subset) < 1:
                continue
                
            # Find common channels for this subset
            common_channels = set.intersection(*[set(G.neighbors(n)) for n in node_subset])
            
            if len(common_channels) >= 2:  # Bi-channel connectivity requirement
                weight = self._calculate_subgraph_weight(set(node_subset), common_channels)
                
                if weight > max_weight:
                    max_weight = weight
                    best_nodes = set(node_subset)
                    best_channels = common_channels
        
        return BipartiteGraphResult(best_nodes, best_channels, max_weight)
    
    
    def _get_subsets(self, items: Set, n: int) -> List[Set]:
        """Get all possible subsets of size n from items"""
        if n == 0:
            return [set()]
        return [set(subset) | {item} 
                for item in items 
                for subset in self._get_subsets(items - {item}, n-1)]
    
    def _get_common_channels(self, nodes: Set[Node]) -> Set[int]:
        """Find common channels among given nodes"""
        if not nodes:
            return set()
            
        common = set(next(iter(nodes)).available_channels)
        print("MWCBG common :", common)
        for node in nodes:
            common &= node.available_channels
            
        return common
    
    def _calculate_subgraph_weight(self, nodes: Set[Node], channels: Set[int]) -> float:
        """
        Calculate weight for complete bipartite subgraph using equation (2) from paper
        
        wi = ν(|Ni*| · ∑(Qc)) + (1-ν)(|Ci*| · ∑(1- dj/∑dk))
        """
        if not nodes or not channels:
            return 0.0
            
        ν = 0.5  # preference factor between network stability and residual energy
        
        # Calculate channel quality sum
        quality_sum = sum(self.source_node.calculate_channel_quality(c) for c in channels)
        
        # Calculate distance factor
        distances = [self.source_node.calculate_distance(n) for n in nodes]
        total_distance = sum(distances)
        if total_distance == 0:
            distance_factor = 0
        else:
            distance_factor = sum(1 - d/total_distance for d in distances)
        
        # Calculate final weight using equation (2)
        weight = (
            ν * (len(nodes) * quality_sum) +
            (1-ν) * (len(channels) * distance_factor)
        )
        
        return weight

def find_MWCBG(node: Node, channels: Set[int], neighbors: Set[Node]) -> BipartiteGraphResult:
    """
    Wrapper function to execute MWCBG procedure
    
    Args:
        node: Source node
        channels: Available channels at source node
        neighbors: Set of neighbor nodes
        
    Returns:
        BipartiteGraphResult containing selected nodes, channels and weight
    """
    mwcbg = MWCBG(node, channels, neighbors)
    mwcbg.construct_bipartite_graph()
    return mwcbg.find_maximum_weight_complete_bipartite()