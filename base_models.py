from enum import Enum, auto
import math
from typing import Set

import numpy as np

np.random.seed(23)


class NodeState(Enum):
    INITIAL = auto()
    INTERMEDIATE_CH = auto()
    CLUSTERED_CH = auto()
    CLUSTERED_CM = auto()


class EnergyConsumption(Enum):
    SENSING = 1.31e-4
    SWITCHING = 1e-5
    IDLE = 1e-4


class Channel:
    """Represents an ON/OFF channel, with rates for departure/arrival."""

    def __init__(self, channel_id: int, alpha: float, beta: float):
        self.id = channel_id
        self.alpha = alpha  # Departure rate (probability of going from ON to OFF)
        self.beta = beta  # Arrival rate (probability of going from OFF to ON)
        self.is_busy = False  # False => ON (idle), True => OFF (busy)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Channel):
            return False
        return self.id == other.id

    def __repr__(self):
        state = "BUSY" if self.is_busy else "IDLE"
        return f"Channel(id={self.id}, state={state})"

    def calculate_idle_probability(self) -> float:
        """P_idle = alpha / (alpha + beta)."""
        return (
            self.alpha / (self.alpha + self.beta)
            if (self.alpha + self.beta) > 0
            else 0.0
        )

    def calculate_quality(self, epsilon: float = 1.2) -> float:
        """
        Computes the channel quality metric.
        (1 + log(P_idle, epsilon)) * D_idle
        D_idle = 1 / alpha
        Clamps the quality to a minimum of 0 to avoid negative values.
        """
        p_idle = self.calculate_idle_probability()
        d_idle = 1 / self.alpha if self.alpha > 0 else 0.0
        if p_idle > 0 and epsilon > 1:
            log_term = math.log(p_idle, epsilon)
        else:
            log_term = 0.0
        raw_quality = (1 + log_term) * d_idle
        return max(raw_quality, 0.0)

    def update_state(self) -> None:
        """
        Updates the channel's state based on alpha and beta.
        - If busy (OFF), it can become idle (ON) with probability alpha.
        - If idle (ON), it can become busy (OFF) with probability beta.

        ~67% chance of switching to the other state
        """
        if self.is_busy:
            # Probability to switch from OFF to ON
            if np.random.rand() * self.alpha * 1.5 < self.alpha:
                self.is_busy = False
        else:
            # Probability to switch from ON to OFF
            if np.random.rand() * self.beta * 1.5 < self.beta:
                self.is_busy = True


class Node:
    """Represents a CRSN node with position, energy, state, and available channels."""

    def __init__(
        self,
        node_id: int,
        x: float,
        y: float,
        initial_energy: float = 0.2,
        transmission_range: float = 50.0,
        available_channels: Set["Channel"] = set(),
    ):
        self.id = node_id
        self.position = (x, y)
        self.energy = initial_energy
        self.state = NodeState.INITIAL
        self.channels = available_channels
        self.transmission_range = transmission_range

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def __repr__(self):
        return f"Node(id={self.id}, energy={self.energy}, state={self.state.name})"

    def calculate_distance(self, other_node: "Node") -> float:
        """Euclidean distance to another node."""
        return math.sqrt(
            (self.position[0] - other_node.position[0]) ** 2
            + (self.position[1] - other_node.position[1]) ** 2
        )

    def consume_energy(self, operation: EnergyConsumption) -> bool:
        """
        Consume energy based on the operation:
          - sensing: 1.31e-4 J
          - switching: 1e-5 J
          - idle: 1e-6 J


        Clamps energy at 0 if negative.
        """

        cost = operation.value
        remaining_energy = self.energy - cost
        self.energy = max(0 ,remaining_energy)
        if remaining_energy > 0:
            return True
        return False

    def is_alive(self) -> bool:
        """Check if node has energy > 0."""
        return self.energy > 0

    def reset_state(self) -> None:
        """Reset node to initial state for reclustering."""
        self.state = NodeState.INITIAL


class Cluster:
    """Represents a cluster with a cluster head and member nodes."""

    def __init__(
        self,
        cluster_head: Node,
        common_channels: Set[Channel],
        members: Set[Node],
        cluster_weight: float,
    ):
        self.cluster_head = cluster_head
        self.common_channels = common_channels
        self.members = set(members)
        self.cluster_weight = cluster_weight

    def __repr__(self):
        return f"Cluster(cluster head id={self.cluster_head.id}, member ids={[member.id for member in self.members]}, common channel ids={[channel.id for channel in self.common_channels]}, cluster weight={self.cluster_weight})"

    def get_total_energy(self) -> float:
        """Calculate total remaining energy in cluster."""
        return sum(member.energy for member in self.members)

    def check_alive(self) -> bool:
        """
        Check if cluster is still viable:
        - Cluster head is alive.
        - Any member is alive.
        """
        if not self.cluster_head.is_alive():
            # Reset all members to INITIAL
            for member in self.members:
                member.reset_state()
            return False

        dead_members = [member for member in self.members if not member.is_alive()]
        if dead_members:
            # Reset all members to INITIAL
            for member in self.members:
                member.reset_state()
            return False

        return True

    def consume_cluster_energy(self, energy_consumption: EnergyConsumption) -> None:
        """
        Simulate energy consumption for one round:
        - Cluster head consumes energy for sensing.
        - Members consume energy for sensing.
        """
        if not self.is_active:
            return

        self.cluster_head.consume_energy(energy_consumption)
        for member in self.members:
            # double check that the member is not the cluster head
            if member.id == self.cluster_head.id:
                continue
            member.consume_energy(energy_consumption)
