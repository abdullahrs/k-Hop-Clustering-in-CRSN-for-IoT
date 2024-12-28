# energy_management.py

from dataclasses import dataclass
from typing import Set, List
import math
from simulation_environment import Node

@dataclass
class EnergyParameters:
    """Energy consumption parameters from the paper"""
    E_ELEC: float = 50e-9       # Energy for running radio electronics (J/bit)
    E_FS: float = 10e-12        # Energy for free space propagation (J/bit/m^2)
    E_SENSING: float = 1.31e-4   # Energy for channel sensing (J)
    E_SWITCHING: float = 1e-5    # Energy for channel switching (J)
    PACKET_SIZE: int = 4000     # Size of data packets in bits
    CONTROL_PACKET_SIZE: int = 200  # Size of control packets in bits

class EnergyManager:
    """Manages energy consumption and tracking for the network"""
    
    def __init__(self, nodes: List[Node], energy_params: EnergyParameters = None):
        self.nodes = nodes
        self.params = energy_params or EnergyParameters()
        self.dead_nodes: Set[Node] = set()
        self.rounds_completed = 0
        self.first_node_death_round = None
        
    def calculate_transmission_energy(self, sender: Node, receiver: Node, packet_size: int) -> float:
        """Calculate energy for transmitting data between nodes"""
        distance = math.sqrt(
            (sender.x - receiver.x)**2 + 
            (sender.y - receiver.y)**2
        )
        return (
            packet_size * self.params.E_ELEC + 
            packet_size * self.params.E_FS * (distance ** 2)
        )
    
    def calculate_reception_energy(self, packet_size: int) -> float:
        """Calculate energy for receiving data"""
        return packet_size * self.params.E_ELEC
    
    def consume_energy_for_clustering(self, node: Node, is_ch: bool, member_count: int):
        """Consume energy for clustering operations"""
        if node.residual_energy <= 0:
            return
            
        # Channel sensing energy
        node.residual_energy -= self.params.E_SENSING
        
        if is_ch:
            # CH needs to receive from all members and send aggregated data
            # Control messages for cluster formation
            node.residual_energy -= (
                member_count * self.calculate_reception_energy(self.params.CONTROL_PACKET_SIZE)
            )
            # Data aggregation and transmission
            node.residual_energy -= (
                member_count * self.calculate_reception_energy(self.params.PACKET_SIZE)
            )
        else:
            # Regular node sends data to CH
            if node.cluster_head is not None:
                node.residual_energy -= (
                    self.calculate_transmission_energy(
                        node, 
                        node.cluster_head, 
                        self.params.PACKET_SIZE
                    )
                )
    
    def consume_energy_for_round(self, clusters: List):
        """Consume energy for one round of operation"""
        for cluster in clusters:
            # CH operations
            if cluster.ch.residual_energy > 0:
                self.consume_energy_for_clustering(
                    cluster.ch, 
                    True, 
                    len(cluster.members) - 1  # Subtract 1 to exclude CH
                )
            
            # Member operations
            for member in cluster.members:
                if member != cluster.ch and member.residual_energy > 0:
                    self.consume_energy_for_clustering(
                        member,
                        False,
                        0
                    )
        
        # Update dead nodes
        for node in self.nodes:
            if node.residual_energy <= 0 and node not in self.dead_nodes:
                self.dead_nodes.add(node)
                if self.first_node_death_round is None:
                    self.first_node_death_round = self.rounds_completed
        
        self.rounds_completed += 1
    
    def get_network_energy(self) -> float:
        """Get total remaining energy in the network"""
        return sum(node.residual_energy for node in self.nodes)
    
    def get_alive_nodes_count(self) -> int:
        """Get number of alive nodes"""
        return len(self.nodes) - len(self.dead_nodes)
    
    def reset(self):
        """Reset energy manager state"""
        self.dead_nodes.clear()
        self.rounds_completed = 0
        self.first_node_death_round = None
        for node in self.nodes:
            node.residual_energy = node.initial_energy