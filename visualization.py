# visualization.py

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import networkx as nx

def plot_network_energy(results: Dict[str, List[float]], 
                       alpha: float, 
                       beta: float):
    """
    Plot network remaining energy over rounds (Figures 7-10)
    
    Args:
        results: Dictionary mapping algorithm names to lists of energy values
        alpha: PU departure rate
        beta: PU arrival rate
    """
    plt.figure(figsize=(10, 6))
    
    for algo_name, energy_values in results.items():
        plt.plot(range(len(energy_values)), energy_values, 
                label=algo_name, marker='o', markersize=2)
    
    plt.xlabel('Number of rounds')
    plt.ylabel('Network remaining energy')
    plt.title(f'Network Remaining Energy (α={alpha}, β={beta})')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_first_node_death(results: Dict[str, int],
                         alpha: float,
                         beta: float):
    """
    Plot number of rounds before first node death (Figures 11-12)
    
    Args:
        results: Dictionary mapping algorithm names to round numbers
        alpha: PU departure rate
        beta: PU arrival rate
    """
    plt.figure(figsize=(10, 6))
    
    algorithms = list(results.keys())
    rounds = list(results.values())
    
    plt.bar(algorithms, rounds)
    plt.xlabel('Algorithms')
    plt.ylabel('Number of rounds')
    plt.title(f'Number of Rounds Before First Node Dead (α={alpha}, β={beta})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_clusters(nodes: List[Any], 
                      clusters: List[Any], 
                      title: str):
    """
    Visualize network topology with clusters (Required cluster visualization)
    
    Args:
        nodes: List of nodes
        clusters: List of clusters
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Create graph for visualization
    G = nx.Graph()
    
    # Add nodes
    for node in nodes:
        G.add_node(node.id, pos=(node.x, node.y))
    
    # Add edges within clusters
    colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))
    
    for cluster_idx, cluster in enumerate(clusters):
        cluster_color = colors[cluster_idx]
        
        # Draw cluster members
        member_positions = {
            node.id: (node.x, node.y) for node in cluster.members
        }
        
        # Draw connections to CH
        for member in cluster.members:
            if member != cluster.ch:
                G.add_edge(cluster.ch.id, member.id)
    
        # Draw nodes
        nx.draw_networkx_nodes(
            G, member_positions, 
            nodelist=member_positions.keys(),
            node_color=[cluster_color],
            node_size=500
        )
        
        # Highlight CH
        nx.draw_networkx_nodes(
            G, {cluster.ch.id: (cluster.ch.x, cluster.ch.y)},
            nodelist=[cluster.ch.id],
            node_color=[cluster_color],
            node_size=800,
            node_shape='s'  # square for CH
        )
    
    # Draw edges
    nx.draw_networkx_edges(G, nx.get_node_attributes(G, 'pos'))
    
    # Add labels
    labels = {node.id: f"N{node.id}" for node in nodes}
    nx.draw_networkx_labels(G, nx.get_node_attributes(G, 'pos'), labels)
    
    plt.title(title)
    plt.axis('on')
    plt.grid(True)
    plt.show()