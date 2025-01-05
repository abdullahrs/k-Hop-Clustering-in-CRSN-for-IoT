import random

from base_models import Channel, Node

random.seed(23)


class CRSNetwork:
    def __init__(
        self,
        area_size: tuple[float, float] = (300, 300),
        num_nodes: int = 65,
        num_channels: int = 10,
        transmission_range: float = 50.0,
        initial_energy: float = 0.2,
        alpha: float = 2,
        beta: float = 2,
        min_channel_per_node: int = 2,
    ):
        """Initialize CR sensor network

        Args:
            area_size: Size of square deployment area
            num_nodes: Number of nodes to deploy
            num_channels: Number of available channels
            transmission_range: Node transmission range
            initial_energy: Initial node energy
        """
        self.area_size = area_size
        self.num_nodes = num_nodes
        self.num_channels = num_channels
        self.transmission_range = transmission_range
        self.initial_node_energy = initial_energy

        self.alpha = alpha
        self.beta = beta

        self.channels = [
            Channel(channel_id, self.alpha, self.beta)
            for channel_id in range(self.num_channels)
        ]
        self.nodes = []

        # print("CHANNELS:", [channel.id for channel in self.channels])

        # Create nodes with random positions
        for node_id in range(num_nodes):
            x = random.uniform(0, area_size[0])
            y = random.uniform(0, area_size[1])
            num_channels = random.randint(min_channel_per_node, len(self.channels))
            node = Node(node_id, x, y, self.initial_node_energy, transmission_range)
            node.channels = set(random.sample(self.channels, num_channels))
            self.nodes.append(node)

    def update_channel_states(self):
        """Update states of all channels based on PU activity"""
        for channel in self.channels:
            channel.update_state()

    def get_remaining_energy(self):
        """Calculate total remaining network energy"""
        return sum(node.energy for node in self.nodes)

    def get_alive_nodes(self):
        """Get number of nodes that still have energy"""
        return sum(1 for node in self.nodes if node.energy > 0)

    def reset_enviroment(self, **params):
        """Reset enviroment with new parameters or default values"""
        self.__init__(**params)
