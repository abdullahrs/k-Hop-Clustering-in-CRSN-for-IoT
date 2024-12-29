from enum import Enum
import math
import random
import networkx as nx

from config import (
    AREA_SIZE,
    INITIAL_ENERGY,
    NUM_CHANNELS,
    SENSING_COST,
    TRANSMISSION_RANGE,
)

random.seed(23)


class NodeState(Enum):
    INITIAL = "initial"
    INTERMEDIATE_CH = "intermediate_CH"
    CLUSTERED_CH = "clustered_CH"
    CLUSTERED_CM = "clustered_CM"


class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.energy = INITIAL_ENERGY
        self.position = (
            random.uniform(0, AREA_SIZE[0]),
            random.uniform(0, AREA_SIZE[1]),
        )
        self.channels = set(
            random.sample(range(1, NUM_CHANNELS + 1), random.randint(2, NUM_CHANNELS))
        )
        assert len(self.channels) >= 2, "Node initialized with fewer than 2 channels!"

        self.state = NodeState.INITIAL  # Initialize with the 'INITIAL' state
        self.neighbors = set()

    @property
    def is_alive(self):
        """
        Returns True if the node has energy > 0, otherwise False.
        """
        return self.energy > 0

    def consume_energy(self, amount):
        """
        Subtract energy for specific actions (e.g., joining a cluster).
        Args:
            amount (float): Energy to consume.
        """
        self.energy = max(0, self.energy - amount)  # Ensure energy doesn't go below 0


class CRSNetwork:
    def __init__(self, num_sus, alpha=2, beta=2):
        self.nodes = [Node(i) for i in range(num_sus)]
        self.graph = nx.Graph()
        self.alpha = alpha
        self.beta = beta

    def initialize_graph(self):
        """
        Initialize the network graph by adding all nodes and connecting edges only between nodes
        that are within transmission range and have at least one common channel.
        """
        # Add all nodes to the graph (even if they have no edges)
        for node in self.nodes:
            self.graph.add_node(node.node_id)

        # Add edges between nodes that satisfy the transmission range and channel conditions
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                if i != j:
                    # Calculate distance between nodes
                    dist = math.dist(node1.position, node2.position)

                    # Check for transmission range and common channels
                    if dist <= TRANSMISSION_RANGE and node1.channels & node2.channels:
                        # Add edge if both conditions are met
                        self.graph.add_edge(node1.node_id, node2.node_id)
                        node1.neighbors.add(node2)
                        node2.neighbors.add(node1)

    def discover_neighbors(self, node, sensing_cost=SENSING_COST) -> set[Node]:
        """
        Discover neighbors for a given node based on transmission range and common channels,
        only if the node has sufficient energy for sensing.

        Parameters:
            node (Node): The node for which neighbors are to be discovered.
            sensing_cost (float): The energy cost per neighbor discovery.

        Returns:
            Set[Node]: A set of neighbors for the given node, or an empty set if energy is insufficient.
        """
        # Check if the node has enough energy for sensing
        if node.energy <= 0:
            return set()  # No neighbors if energy is insufficient

        # Find neighbors of the node using the network graph
        neighbor_ids = set(self.graph.neighbors(node.node_id))
        neighbors = {self.nodes[neighbor_id] for neighbor_id in neighbor_ids}

        # Deduct sensing energy cost based on the number of neighbors
        total_cost = sensing_cost * len(neighbors)
        if node.energy >= total_cost:
            node.energy -= total_cost
        else:
            neighbors = set()  # No neighbors returned if energy is insufficient
        # print(f"Node {node.node_id}: Discovered Neighbors -> {[n.node_id for n in neighbors]}")
        return neighbors

    def calculate_channel_quality(self, channel, round, epsilon=2):
        """
        Calculate the quality of a channel based on the formula provided in the paper.
        Adjusts for negative log values to ensure positive quality.

        Parameters:
            channel (int): The channel ID.
            epsilon (float): Preference parameter for idle probability and duration.

        Returns:
            float: The quality of the given channel.
        """
        idle_probability = self.get_idle_probability()  # Pi
        idle_duration = round % self.get_idle_duration() * channel  # Di

        # Adjust logarithmic term with a positive bias
        log_term = math.log(idle_probability, epsilon)
        channel_quality = (1 + log_term) * idle_duration

        print(
            f"Channel {channel}: Quality {channel_quality}, Idle Probability {idle_probability}, Idle Duration {idle_duration}"
        )
        return channel_quality

    def construct_mwcbg(self, node, neighbors, round,nu=0.3):
        """
        Construct the Maximum Weight Complete Bipartite Graph (MWCBG) for a given node
        and its neighbors. Calculate weight using Equation (2) from the paper.

        Parameters:
            node (Node): The center node for MWCBG construction.
            neighbors (Set[Node]): The neighbors of the given node.
            nu (float): Preference factor between network stability and residual energy.

        Returns:
            Tuple[Set[int], Set[int], float]: Channels in MWCBG, nodes in MWCBG, and maximum weight (w*).
        """
        # Create a bipartite graph
        bipartite_graph = nx.Graph()

        # Bipartite sets: neighbors (nodes) and channels
        neighbor_ids = {neighbor.node_id for neighbor in neighbors}
        channels = node.channels

        # Add nodes for both sets
        bipartite_graph.add_nodes_from(neighbor_ids, bipartite=0)  # Neighbors
        bipartite_graph.add_nodes_from(channels, bipartite=1)  # Channels

        # Add edges with weights based on the formula
        for neighbor in neighbors:
            for channel in channels:
                if channel in neighbor.channels:
                    # Calculate channel quality
                    channel_quality = self.calculate_channel_quality(channel, round)
                    distance = math.dist(node.position, neighbor.position)
                    normalized_distance = distance / TRANSMISSION_RANGE
                    if normalized_distance > 1:
                        normalized_distance = 1  # Cap to avoid negatives

                    weight = max(0, channel_quality * (1 - normalized_distance))
                    if weight > 0:  # Only add edges with positive weight
                        bipartite_graph.add_edge(
                            neighbor.node_id, channel, weight=weight
                        )

        # Find the MWCBG using maximum weight matching
        mwcbg_edges = nx.algorithms.matching.max_weight_matching(
            bipartite_graph, maxcardinality=True
        )

        # Extract channels and nodes in the MWCBG
        mwcbg_channels = {edge[1] for edge in mwcbg_edges if isinstance(edge[1], int)}
        mwcbg_nodes = {edge[0] for edge in mwcbg_edges if isinstance(edge[0], int)}
        valid_node_ids = {
            neighbor.node_id for neighbor in neighbors
        }  # Only valid neighbor IDs
        mwcbg_nodes = mwcbg_nodes.intersection(
            valid_node_ids
        )  # Retain only valid nodes

        # Validate MWCBG channels
        mwcbg_channels = {
            c
            for c in mwcbg_channels
            if any(c in neighbor.channels for neighbor in neighbors)
        }

        # Validate MWCBG nodes
        valid_node_ids = {n.node_id for n in self.nodes}
        filtered_mwcbg_nodes = {n for n in mwcbg_nodes if n in valid_node_ids}

        # Calculate the weight of the complete bipartite subgraph using Equation (2)
        channel_quality_sum = sum(
            self.calculate_channel_quality(c) for c in mwcbg_channels
        )

        if len(filtered_mwcbg_nodes) == 0:
            node_weight_sum = 0
        else:
            total_distance_sum = sum(
                math.dist(node.position, self.nodes[k].position)
                for k in filtered_mwcbg_nodes
                if k < len(self.nodes)
            )
            if total_distance_sum == 0:
                node_weight_sum = 0  # Avoid division by zero
            else:
                node_weight_sum = sum(
                    1
                    - (
                        math.dist(node.position, self.nodes[n].position)
                        / total_distance_sum
                    )
                    for n in filtered_mwcbg_nodes
                    if n < len(self.nodes)
                )

        mwcbg_weight = nu * (len(filtered_mwcbg_nodes) * channel_quality_sum) + (
            1 - nu
        ) * (len(mwcbg_channels) * node_weight_sum)

        # print(f"Node {node.node_id}: MWCBG -> Channels: {mwcbg_channels}, Nodes: {mwcbg_nodes}, Weight: {mwcbg_weight}")
        return mwcbg_channels, filtered_mwcbg_nodes, mwcbg_weight

    def get_idle_duration(self):
        """
        Calculate the average idle duration (D_i) for a given channel based on PU activity rates.

        Parameters:
            channel (int): The channel ID.

        Returns:
            float: The average idle duration for the channel.
        """
        return 1 / self.beta  # Beta is the OFF rate for the channel

    def get_idle_probability(self):
        """
        Calculate the idle probability for a given channel based on PU activity rates.

        Parameters:
            channel (int): The channel ID (optional if channel-specific alpha, beta exist).
            alpha (float): PU ON rate for the channel.
            beta (float): PU OFF rate for the channel.

        Returns:
            float: Idle probability of the channel.
        """
        # Calculate P_off and P_on based on alpha and beta
        P_on = self.beta / (self.alpha + self.beta)
        return P_on  # Idle probability is P_on as defined
