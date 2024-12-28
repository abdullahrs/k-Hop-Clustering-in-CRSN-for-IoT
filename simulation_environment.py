# simulation_environment.py

from dataclasses import dataclass
from typing import List, Set, Optional
import random
import math
from pu_activity_model import Channel, PUActivityModel

# Constants from the paper
DEFAULT_SIMULATION_PARAMS = {
    "SIMULATION_TIME": 1000,        # Total simulation time
    "TIME_SLOT": 1,                # Length of each time slot
    "TIME_SCALE": 0.001,           # Time scale for simulation (milliseconds to seconds)
    "AREA_SIZE": (300, 300),           # Simulation area in meters
    "NUM_CHANNELS": 10,                # Number of channels (M in paper)
    "NUM_SUS": 100,                    # Number of secondary users
    "TRANSMISSION_RANGE": 40,          # Transmission range in meters
    "INITIAL_ENERGY": 0.2,             # Initial energy in Joules
    "ENERGY_SENSING": 1.31e-4,         # Energy for channel sensing
    "ENERGY_SWITCHING": 1e-5,          # Energy for channel switching
    "MIN_CHANNELS_PER_NODE": 2,        # Minimum channels per node for bi-channel connectivity
    "EPSILON": 1.5,                    # Parameter for channel quality calculation
    "WEIGHT_PREFERENCE": 0.5,          # Preference factor between network stability and residual energy (v in paper)
    "PU_ALPHA_RANGE": (0.5, 2.0),      # Range for PU departure rate
    "PU_BETA_RANGE": (0.5, 2.0)        # Range for PU arrival rate
}

@dataclass
class NodeState:
    """Enum-like class for node states"""
    INITIAL = "initial"
    INTERMEDIATE_CH = "intermediate_CH"
    CLUSTERED_CM = "clustered_CM"
    CLUSTERED_CH = "clustered_CH"

@dataclass
class Node:
    """Represents a Secondary User (SU) node in the CRSN"""
    id: int
    x: float
    y: float
    available_channels: Set[int]
    initial_energy: float
    transmission_range: float
    
    def __post_init__(self):
        self.residual_energy = self.initial_energy
        self.state = NodeState.INITIAL
        self.cluster_head: Optional['Node'] = None
        self.cluster_members: Set['Node'] = set()
        
    def is_alive(self) -> bool:
        """Check if node has enough energy to operate"""
        return self.residual_energy > 0
        
    def set_as_cluster_head(self):
        """Set node as cluster head"""
        if not self.is_alive():
            return False
        self.state = NodeState.CLUSTERED_CH
        self.cluster_head = self
        self.cluster_members = {self}
        return True
        
    def join_cluster(self, ch: 'Node') -> bool:
        """Join a cluster with given cluster head"""
        if not self.is_alive() or not ch.is_alive():
            return False
        self.state = NodeState.CLUSTERED_CM
        self.cluster_head = ch
        ch.cluster_members.add(self)
        return True

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id
        
    def calculate_distance(self, other_node: 'Node') -> float:
        """Calculate Euclidean distance to another node"""
        return math.sqrt(
            (self.x - other_node.x)**2 + 
            (self.y - other_node.y)**2
        )
        
    def calculate_channel_quality(self, channel: int, epsilon: float = 1.5) -> float:
        """
        Calculate channel quality using equation (1) from paper:
        Qi = (1 + logε Pi)Di
        
        For testing purposes, we'll use a simplified version
        based on channel number and distance from center
        """
        # For testing, return a quality value based on channel number
        # Lower channel numbers are assumed to have better quality
        base_quality = 1.0 / (channel + 1)
        
        # Adjust quality based on distance from center (assumed to be 150,150)
        center_distance = math.sqrt((self.x - 150)**2 + (self.y - 150)**2)
        distance_factor = 1.0 / (1 + center_distance/100)  # Normalize by 100
        
        return base_quality * distance_factor * epsilon

class CRSNEnvironment:
    """Main simulation environment for Cognitive Radio Sensor Network"""
    
    def __init__(self, params: dict = None):
        """
        Initialize CRSN environment with given parameters or defaults
        
        Args:
            params: Dictionary of simulation parameters
        """
        self.params = DEFAULT_SIMULATION_PARAMS.copy()
        if params:
            self.params.update(params)
            
        self.nodes: List[Node] = []
        self.channels: List[Channel] = []
        self.current_time = 0
        self.initialize_network()
        
    def initialize_network(self):
        print(f"Simulation parameters: {self.params}")
        """Initialize the CRSN with randomly positioned nodes and channels"""
        # Initialize channels
        self._initialize_channels()
        # Initialize nodes
        self._initialize_nodes()
        print(f"Initialized CRSN with {len(self.nodes)} nodes and {len(self.channels)} channels")
        
        
    def _initialize_channels(self):
        """Initialize channels with PU activity parameters"""
        for i in range(self.params["NUM_CHANNELS"]):
            alpha = random.uniform(*self.params["PU_ALPHA_RANGE"])
            beta = random.uniform(*self.params["PU_BETA_RANGE"])
            channel = Channel(id=i, alpha=alpha, beta=beta)
            
            # Determine and log the activity pattern
            pattern = PUActivityModel.determine_activity_pattern(alpha, beta)
            print(f"Channel {i} initialized with pattern: {pattern.name} - {pattern.description}")
            
            self.channels.append(channel)
            
    def _initialize_nodes(self):
        """Initialize nodes with random positions and channel assignments"""
        for i in range(self.params["NUM_SUS"]):
            x = random.uniform(0, self.params["AREA_SIZE"][0])
            y = random.uniform(0, self.params["AREA_SIZE"][1])
            
            # Randomly assign available channels
            num_available = random.randint(
                self.params["MIN_CHANNELS_PER_NODE"], 
                self.params["NUM_CHANNELS"]
            )
            available_channels = set(random.sample(
                range(self.params["NUM_CHANNELS"]), 
                num_available
            ))
            
            self.nodes.append(Node(
                id=i,
                x=x,
                y=y,
                available_channels=available_channels,
                initial_energy=self.params["INITIAL_ENERGY"],
                transmission_range=self.params["TRANSMISSION_RANGE"]
            ))
            
    def calculate_channel_quality(self, channel: Channel) -> float:
        """Calculate channel quality using equation from paper"""
        Pon = channel.beta / (channel.alpha + channel.beta)
        D = 1 / channel.alpha  # Average idle duration
        return (1 + math.log(self.params["EPSILON"] * Pon)) * D
        
    def get_neighbors(self, node: Node, hop_count: int = 1) -> Set[Node]:
        """
        Get k-hop neighbors of a node
        
        Args:
            node: Source node
            hop_count: Number of hops to consider
        
        Returns:
            Set of neighbor nodes within hop_count distance
        """
        neighbors = set()
        current_hop_nodes = {node}
        
        for _ in range(hop_count):
            next_hop_nodes = set()
            for current_node in current_hop_nodes:
                for potential_neighbor in self.nodes:
                    if (potential_neighbor not in neighbors and 
                        potential_neighbor != node):
                        # Check if in transmission range
                        distance = self.calculate_distance(current_node, potential_neighbor)
                        # Check common channels
                        common_channels = (
                            current_node.available_channels & 
                            potential_neighbor.available_channels
                        )
                        
                        if (distance <= node.transmission_range and 
                            len(common_channels) >= self.params["MIN_CHANNELS_PER_NODE"]):
                            next_hop_nodes.add(potential_neighbor)
            
            neighbors.update(next_hop_nodes)
            current_hop_nodes = next_hop_nodes
            
        return neighbors

    def calculate_distance(self, node1: Node, node2: Node) -> float:
        """Calculate Euclidean distance between two nodes"""
        return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

    def calculate_weight(self, node: Node, neighbors: Set[Node]) -> float:
        """
        Calculate clustering weight for a node using equation (2) from paper
        
        Args:
            node: Node to calculate weight for
            neighbors: Set of neighbor nodes
        
        Returns:
            Calculated weight value
        """
        if not neighbors:
            return 0
            
        # Get common channels for all neighbors
        common_channels = node.available_channels
        for neighbor in neighbors:
            common_channels &= neighbor.available_channels
        
        if not common_channels:
            return 0
            
        # Calculate channel quality sum
        quality_sum = sum(self.calculate_channel_quality(self.channels[c]) 
                         for c in common_channels)
        
        # Calculate distance factor
        distances = [self.calculate_distance(node, neighbor) 
                    for neighbor in neighbors]
        
        distance_factor = sum(1 - d/sum(distances) for d in distances)
        
        # Calculate final weight using equation (2) from paper
        weight = (
            self.params["WEIGHT_PREFERENCE"] * (len(neighbors) * quality_sum) + 
            (1 - self.params["WEIGHT_PREFERENCE"]) * (len(common_channels) * distance_factor)
        )
        
        return weight
        
    def step(self):
        """
        Advance simulation by one time step and update channel states
        
        Returns:
            Dict mapping channel IDs to their current states
        """
        self.current_time += self.params["TIME_SLOT"] * self.params["TIME_SCALE"]
        channel_states = {}
        
        # Update each channel's state
        for channel in self.channels:
            new_state = channel.update(self.current_time)
            channel_states[channel.id] = new_state
            
        return channel_states
    
    def get_available_channels(self) -> List[Channel]:
        """Get list of currently available channels"""
        return [channel for channel in self.channels if channel.is_available]
    
    def update_node_available_channels(self):
        """Update available channels for each node based on PU activity"""
        available_channels = set(c.id for c in self.get_available_channels())
        for node in self.nodes:
            # Intersect node's assigned channels with currently available channels
            node.available_channels &= available_channels
            
    def run_simulation(self, num_steps: Optional[int] = None):
        """
        Run simulation for specified number of steps or until simulation time is reached
        
        Args:
            num_steps: Number of steps to simulate. If None, uses SIMULATION_TIME parameter
        """
        max_steps = num_steps if num_steps is not None else int(
            self.params["SIMULATION_TIME"] / 
            (self.params["TIME_SLOT"] * self.params["TIME_SCALE"])
        )
        
        channel_state_history = []
        for _ in range(max_steps):
            # Update channel states
            current_states = self.step()
            channel_state_history.append(current_states)
            
            # Update node available channels
            self.update_node_available_channels()
            
        return channel_state_history