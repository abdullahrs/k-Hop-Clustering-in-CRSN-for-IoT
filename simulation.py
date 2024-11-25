from collections import defaultdict
from comparison_algorithms import NSAC, CogLEACH
from constants import RANDOM_SEED
from k_SACB_EC import kSACBEC
from k_SACB_WEC import kSACBWEC
from utility_functions import assign_channels, calculate_channel_qualities, calculate_edges
from copy import deepcopy
import numpy as np
import random


class Replication:
    def __init__(self, simulation_area=(300, 300), num_sus=80, num_channels=10, alpha=2, beta=2, rounds=900, su_transmission_range=40, initial_energy=0.2, sensing_energy=1.31e-4, communication_energy=50e-9):
        self.simulation_area = simulation_area
        self.num_sus = num_sus
        self.nodes = [i for i in range(self.num_sus)]
        self.num_channels = num_channels
        self.alpha = alpha
        self.beta = beta
        self.rounds = rounds
        self.su_transmission_range = su_transmission_range
        self.initial_energy = initial_energy
        self.sensing_energy = sensing_energy
        self.communication_energy = communication_energy
        self.metrics = defaultdict(list)
        self.initialize_environment()

    def initialize_environment(self):
        """
        Initialize SU positions, channels, and energies.
        """
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        self.su_positions = {
            i: (np.random.uniform(0, self.simulation_area[0]), np.random.uniform(
                0, self.simulation_area[1]))
            for i in range(self.num_sus)
        }
        self.su_channels = assign_channels(self.num_sus, self.num_channels)
        self.su_energies = {
            i: self.initial_energy for i in range(self.num_sus)}
        self.channel_qualities = calculate_channel_qualities(self.num_channels)
        self.edges = calculate_edges(
            self.nodes, self.su_positions, self.su_channels, self.su_transmission_range)
        # Initialize channel states and timers with random values
        self.channel_states = {
            ch: 'OFF' for ch in range(self.num_channels)
        }
        self.channel_timers = {
            ch: np.random.exponential(
                self.alpha if self.channel_states[ch] == 'OFF' else self.beta)
            for ch in range(self.num_channels)
        }

    def update_channel_states(self):
        """
        Update the state of each channel (ON/OFF) based on alpha and beta values.
        """
        print("BEFORE UPDATE : self.channel_states :", self.channel_states)
        print("BEFORE UPDATE : self.channel_timers :", self.channel_timers)
        for ch in range(self.num_channels):
            # If the timer is zero or less, switch the channel state
            if self.channel_timers[ch] <= 0:
                if self.channel_states[ch] == 'ON':
                    # Switch to 'OFF' and set a new timer using beta
                    self.channel_states[ch] = 'OFF'
                    self.channel_timers[ch] = np.random.exponential(self.beta)
                else:
                    # Switch to 'ON' and set a new timer using alpha
                    self.channel_states[ch] = 'ON'
                    self.channel_timers[ch] = np.random.exponential(self.alpha)
            else:
                # Decrease the timer for channels that are still active
                self.channel_timers[ch] -= 1
        print("AFTER UPDATE : self.channel_states :", self.channel_states)
        print("AFTER UPDATE : self.channel_timers :", self.channel_timers)

    def get_available_channels(self, node):
        """
        Return channels that are currently OFF (available) for a given node.
        """
        channels = self.su_channels[node]
        available_channels = {
            ch for ch in channels if self.channel_states[ch] == 'OFF'}
        return available_channels

    def simulate_rounds(self):
        """
        Run all algorithms for multiple rounds and collect metrics.
        """
        algorithms = {
            'k-SACB-EC': kSACBEC,
            'k-SACB-WEC': kSACBWEC,
            'NSAC': NSAC,
            'CogLEACH': CogLEACH
        }

        for algo_name, algo_class in algorithms.items():
            print(f"Running {algo_name}...")
            algo_instance = self.initialize_algorithm(algo_class)

            for round_num in range(self.rounds):
                print(f"Running {algo_name} round {round_num}...")
                self.update_channel_states()
                clusters, cluster_heads = algo_instance.run()
                print(f"<{round_num}> clusters : {clusters}")
                # return

                remaining_energy = sum(algo_instance.su_energies.values())
                print(f"<{round_num}> remaining_energy :",remaining_energy)
                alive_nodes = sum(
                    1 for e in algo_instance.su_energies.values() if e > 0)
                print(f"<{round_num}> alive_nodes :",alive_nodes)
                # Store metrics
                self.metrics[f"{algo_name}_energy"].append(remaining_energy)
                self.metrics[f"{algo_name}_alive"].append(alive_nodes)
                algo_instance.reset()

    def initialize_algorithm(self, algo_class):
        """
        Initialize the specified algorithm class with the current simulation state.
        """
        print("Initial SU Channels :", self.su_channels)
        if algo_class == kSACBEC:
            return kSACBEC(
                nodes=self.nodes,
                edges=self.edges,
                channels=deepcopy(self.su_channels),
                su_positions=self.su_positions,
                channel_qualities=self.channel_qualities,
                su_transmission_range=self.su_transmission_range,
                get_available_channels=self.get_available_channels,
                initial_energy=self.initial_energy,
                sensing_energy=self.sensing_energy,
            )
        elif algo_class == kSACBWEC:
            return kSACBWEC(
                num_sus=self.num_sus,
                su_positions=self.su_positions,
                initial_energy=self.initial_energy,
                su_channels=deepcopy(self.su_channels),
                channel_qualities=self.channel_qualities,
                edges=self.edges,
                get_available_channels=self.get_available_channels,
                preference_factor=0.5
            )
        elif algo_class == NSAC:
            return NSAC(
                num_sus=self.num_sus,
                su_positions=self.su_positions,
                su_channels=deepcopy(self.su_channels),
                channel_qualities=self.channel_qualities,
                get_available_channels=self.get_available_channels,
                edges=self.edges,
                preference_factor=0.5,
                initial_energy=self.initial_energy,
                sensing_energy=self.sensing_energy,
            )
        elif algo_class == CogLEACH:
            return CogLEACH(
                nodes=self.nodes,
                num_sus=self.num_sus,
                su_positions=self.su_positions,
                su_channels=deepcopy(self.su_channels),
                channel_qualities=self.channel_qualities,
                su_transmission_range=self.su_transmission_range,
                get_available_channels=self.get_available_channels,
                initial_energy=self.initial_energy,
                sensing_energy=self.sensing_energy
            )

    def run(self):
        """
        Execute the simulation for all algorithms.
        """
        self.simulate_rounds()
        print("Simulation complete.")
        return self.metrics
