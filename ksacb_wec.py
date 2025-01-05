from collections import defaultdict
from base_models import Channel, Cluster, EnergyConsumption, Node, NodeState
from crsn_network import CRSNetwork
from mwcbg import MWCBG
from utilities import get_k_hop_neighbors


class KSCABWEC:
    def __init__(
        self,
        network: CRSNetwork,
        mwcbg: MWCBG,
    ) -> None:
        self.network = network
        self.mwcbg = mwcbg
        self.clusters = {}
        self.neighbor_relation = {}
        self.previous_clusters = {}

    def run(self):
        self.form_clusters()
        self.finalize_clusters()
        self.deduct_idle_energy()
        self.prune_dead_clusters()
        self.revalidate_clusters()
        self.network.update_channel_states()
        disappeared_clusters = 0
        new_clusters = 0
        if len(self.previous_clusters.keys()) == 0:
            self.previous_clusters = dict(self.clusters)
        else:
            previous_cluster_ids = set(self.previous_clusters.keys())
            current_cluster_ids = set(self.clusters.keys())

            # Clusters that disappeared
            disappeared_clusters = len(previous_cluster_ids - current_cluster_ids)
            # New clusters that formed
            new_clusters = len(current_cluster_ids - previous_cluster_ids)

            # Update previous clusters for next iteration
            self.previous_clusters = dict(self.clusters)

        return disappeared_clusters + new_clusters

    def form_clusters(self):
        for node in self.network.nodes:
            if node.state != NodeState.INITIAL:
                continue

            max_weight = float("-inf")
            max_cmni = None
            max_pcmi = None
            relations = defaultdict(set)
            k = 1

            while True:
                neighbors, relations = get_k_hop_neighbors(
                    node,
                    self.network.nodes,
                    k,
                    transmission_range=self.network.transmission_range,
                )

                # print(
                #     f"k : {k} node : {node.id} neighbors : {[neighbor.id for neighbor in neighbors]}, max_pcmi : {[member.id for member in max_pcmi] if max_pcmi else []}"
                # )
                # if no neighbors found break or,
                # length of the current PCMi is equal to the number of neighbors then break
                # because max_pcmi is sub set of the neighbors if no new exploration exists in
                # this iteration it mean next iteration will not have any new exploration
                # because get k hop neighbors basically searchin in BFS manner
                if not neighbors or (
                    max_pcmi is not None and set(neighbors) == set(max_pcmi)
                ):
                    break

                available_channels = node.channels.union(
                    *[neighbor.channels for neighbor in neighbors]
                )

                # print(
                #     f"available_channels : {[channel.id for channel in available_channels]}"
                # )
                # print(f"neighbors : {[neighbor.id for neighbor in neighbors]}")
                # common channels, cluster members, maximum weight of the subgraph
                cmni, PCMi, wi = self.mwcbg.find_maximum_subgraph(
                    node, available_channels, neighbors
                )

                # print(
                #     f"cmni : {[member.id for member in cmni] if cmni else []}, PCMi : {[member.id for member in PCMi] if PCMi else []}, wi : {wi}"
                # )

                if not cmni or not PCMi:
                    break

                if wi <= max_weight:
                    break
                if len(cmni) >= 2 and wi > max_weight:
                    max_weight = wi
                    max_cmni = cmni
                    max_pcmi = PCMi

                if k == 5:
                    break
                
                k += 1
                # print("k :", k, wi, max_weight)
                # print("--------------------------------")

            if not max_pcmi or not max_cmni:
                continue
            

            node.state = NodeState.CLUSTERED_CH
            node.consume_energy(EnergyConsumption.SWITCHING)

            for member in max_pcmi:
                member.state = NodeState.CLUSTERED_CM
                member.consume_energy(EnergyConsumption.SWITCHING)

            pcmi_set = set(PCMi) if PCMi else set()
            cluster = Cluster(
                cluster_head=node,
                common_channels=max_cmni,
                members=pcmi_set | {node},
                cluster_weight=max_weight,
            )
            self.clusters[node.id] = cluster
            self.neighbor_relation.update(relations)

    def finalize_clusters(self):
        # If node is in initial state, form a standalone cluster
        for node in self.network.nodes:
            if node.state == NodeState.INITIAL:
                node.state = NodeState.CLUSTERED_CH
                node.consume_energy(EnergyConsumption.SWITCHING)
                self.clusters[node.id] = Cluster(node, node.channels, set([node]), 0)

    def deduct_idle_energy(self):
        for cluster in self.clusters.values():
            for node in cluster.members:
                node.consume_energy(EnergyConsumption.IDLE)
                cluster.cluster_head.consume_energy(EnergyConsumption.IDLE)

    def prune_dead_clusters(self) -> int:
        dead_clusters = set()

        for cluster_id, cluster in self.clusters.items():
            if not cluster.cluster_head.is_alive():
                dead_clusters.add(cluster_id)

                for member in cluster.members:
                    if member.id != cluster.cluster_head.id:
                        member.state = NodeState.INITIAL

        for cluster_id in dead_clusters:
            del self.clusters[cluster_id]

        return len(dead_clusters)

    def revalidate_clusters(self) -> int:
        clusters = list(self.clusters.values())

        for cluster in clusters:  # Use list to safely modify clusters
            invalid_members = set()

            # Handle dead members
            for member in set(cluster.members):
                if member.is_alive():
                    continue

                # Check if member still has a valid connection to the cluster
                is_leaf_node = self._is_leaf_node(member)

                if is_leaf_node:
                    member.state = NodeState.INITIAL
                    invalid_members.add(member)
                else:
                    self._prune_subtree(member, cluster, invalid_members)

            # Remove invalid members from the cluster
            cluster.members -= invalid_members

    def _is_leaf_node(self, node: Node) -> bool:
        """Check if a node is an internal node."""
        neighbors = self.neighbor_relation.get(node.id, [])
        return len(neighbors) == 1

    def _prune_subtree(self, node: Node, cluster: Cluster, invalid_members: set):
        """Recursively prune all members dependent on the given node."""
        queue = [node]

        while queue:
            current = queue.pop(0)
            neighbors = self.neighbor_relation.get(current.id, [])

            invalid_members.add(current)
            current.state = NodeState.INITIAL

            for neighbor in neighbors:
                if neighbor in cluster.members and neighbor not in invalid_members:
                    neighbor_neighbors = self.neighbor_relation.get(neighbor.id, [])
                    if cluster.cluster_head in neighbor_neighbors:
                        continue
                    queue.append(neighbor)


def test_kscabwec():
    # Define channels
    channels = [Channel(channel_id=i, alpha=2.0, beta=2.0) for i in range(5)]

    # Define nodes
    nodes = [
        Node(
            node_id=1,
            x=0,
            y=0,
            available_channels={channels[0], channels[1]},
            initial_energy=0.2,
        ),
        Node(
            node_id=2,
            x=5,
            y=0,
            available_channels={channels[0], channels[1], channels[2]},
            initial_energy=0.2,
        ),
        Node(
            node_id=3,
            x=10,
            y=0,
            available_channels={channels[1], channels[2]},
            initial_energy=0.2,
        ),
        Node(
            node_id=4,
            x=15,
            y=0,
            available_channels={channels[3], channels[4]},
            initial_energy=0.2,
        ),
        Node(
            node_id=5,
            x=20,
            y=0,
            available_channels={channels[3], channels[4]},
            initial_energy=0.2,
        ),
        Node(
            node_id=6,
            x=0,
            y=10,
            available_channels={channels[1], channels[2]},
            initial_energy=0.2,
        ),
        Node(
            node_id=7,
            x=25,
            y=0,
            available_channels={channels[3], channels[4]},
            initial_energy=0.2,
        ),
        Node(
            node_id=8,
            x=15,
            y=5,
            available_channels={channels[3], channels[4]},
            initial_energy=0.2,
        ),
        Node(
            node_id=9,
            x=15,
            y=10,
            available_channels={channels[3], channels[4]},
            initial_energy=0.2,
        ),
    ]

    # Initialize network
    network = CRSNetwork(
        area_size=(30, 10),
        num_nodes=len(nodes),
        num_channels=len(channels),
        transmission_range=6.0,  # Ensure neighbors form distinct clusters
        initial_energy=0.2,
    )
    network.nodes = nodes
    network.channels = channels

    # Initialize MWCBG and K-SACB-WEC algorithm
    mwcbg = MWCBG(preference_factor=0.5)
    algorithm = KSCABWEC(network, mwcbg)

    # Run the algorithm
    algorithm.form_clusters()
    algorithm.finalize_clusters()

    # Print results
    print("Clusters formed:")
    for cluster_id, cluster in algorithm.clusters.items():
        print(f"Cluster ID: {cluster_id}")
        print(f"  CH: Node {cluster.cluster_head.id}")
        print(f"  Members: {[node.id for node in cluster.members]}")
        print(f"  Common Channels: {[ch.id for ch in cluster.common_channels]}")


def test_kscabwec_2():
    # Define channels
    channels = [Channel(channel_id=i, alpha=2.0, beta=2.0) for i in range(4)]

    # Define nodes
    nodes = [
        Node(
            node_id=1,
            x=0,
            y=0,
            available_channels={channels[0], channels[1]},
            initial_energy=0.2,
        ),
        Node(
            node_id=2,
            x=5,
            y=0,
            available_channels={channels[0], channels[1], channels[2], channels[3]},
            initial_energy=0.2,
        ),
        Node(
            node_id=3,
            x=10,
            y=0,
            available_channels={channels[2], channels[3]},
            initial_energy=0.2,
        ),
    ]

    # Initialize network
    network = CRSNetwork(
        area_size=(50, 50),
        num_nodes=len(nodes),
        num_channels=len(channels),
        transmission_range=6.0,  # Ensure neighbors form distinct clusters
        initial_energy=0.2,
    )
    network.nodes = nodes
    network.channels = channels

    # Initialize MWCBG and K-SACB-WEC algorithm
    mwcbg = MWCBG(preference_factor=0.5)
    algorithm = KSCABWEC(network, mwcbg)

    # Run the algorithm
    algorithm.form_clusters()
    algorithm.finalize_clusters()

    # Print results
    print("Clusters formed:")
    for cluster_id, cluster in algorithm.clusters.items():
        print(f"Cluster ID: {cluster_id}")
        print(f"  CH: Node {cluster.cluster_head.id}")
        print(f"  Members: {[node.id for node in cluster.members]}")
        print(f"  Common Channels: {[ch.id for ch in cluster.common_channels]}")
    for node_id, neighbors in algorithm.neighbor_relation.items():
        print(f"Node {node_id.id} neighbors: {[neighbor.id for neighbor in neighbors]}")


def test_wec_multi_iteration():
    # Define channels
    channels = [Channel(channel_id=i, alpha=2.0, beta=2.0) for i in range(5)]

    # Define nodes
    nodes = [
        Node(
            node_id=1,
            x=0,
            y=0,
            available_channels={channels[0], channels[1]},
            initial_energy=0.2,
        ),
        Node(
            node_id=2,
            x=10,
            y=0,
            available_channels={channels[0], channels[1]},
            initial_energy=0.2,
        ),
        Node(
            node_id=3,
            x=20,
            y=0,
            available_channels={channels[0], channels[1]},
            initial_energy=0.2,
        ),
        Node(
            node_id=4,
            x=30,
            y=0,
            available_channels={channels[0], channels[1]},
            initial_energy=0.2,
        ),
        Node(
            node_id=5,
            x=24,
            y=0,
            available_channels={channels[1], channels[2]},
            initial_energy=0.2,
        ),
        Node(
            node_id=6,
            x=36,
            y=0,
            available_channels={channels[1], channels[2]},
            initial_energy=0.2,
        ),
    ]

    # Initialize network
    network = CRSNetwork(
        area_size=(60, 60),
        num_nodes=len(nodes),
        num_channels=len(channels),
        transmission_range=15.0,  # Ensure merging is possible
        initial_energy=0.2,
    )

    network.nodes = nodes
    network.channels = channels

    # Initialize MWCBG and WEC algorithm
    mwcbg = MWCBG(preference_factor=0.5)
    algorithm = KSCABWEC(network, mwcbg)

    # Simulation parameters
    num_iterations = 900
    energy_log = []
    reclustering_count = 0

    # Run the simulation
    for iteration in range(1, num_iterations + 1):
        print(f"--- Iteration {iteration} ---")

        algorithm.run()

        # Track total energy and reclustering count
        total_energy = sum(node.energy for node in nodes if node.is_alive())
        energy_log.append(total_energy)

        # Count active clusters
        active_clusters = len(algorithm.clusters)

        # Print current state
        # print(f"Total Energy Remaining: {total_energy:.4f}")
        # print(f"Active Clusters: {active_clusters}")
        # print("Cluster Details:")
        # for cluster_id, cluster in algorithm.clusters.items():
        #     print(f"  Cluster ID: {cluster_id}")
        #     print(f"    CH: Node {cluster.cluster_head.id}")
        #     print(f"    Members: {[node.id for node in cluster.members]}")
        #     print(f"    Energies: {[node.energy for node in cluster.members]}")

        # Check if no clusters are left
        if active_clusters == 0:
            print("All clusters have disbanded. Simulation ends.")
            break
    print(f"Total Energy Remaining: {total_energy:.4f}")
    print(f"Active Clusters: {active_clusters}")
    print("Cluster Details:")
    for cluster_id, cluster in algorithm.clusters.items():
        print(f"  Cluster ID: {cluster_id}")
        print(f"    CH: Node {cluster.cluster_head.id}")
        print(f"    Members: {[node.id for node in cluster.members]}")
        print(f"    Total energy: {sum([node.energy for node in cluster.members])}")
        # print(f"    Energies: {[node.energy for node in cluster.members]}")
    # Analyze results
    print("\n--- Simulation Summary ---")
    print(f"Total Iterations: {iteration}")
    print(f"Reclustering Events: {reclustering_count}")
    print(f"Final Total Energy: {energy_log[-1]:.4f}")
    # print(f"Energy Log: {energy_log}")


if __name__ == "__main__":
    # test_kscabwec()
    # test_kscabwec_2()
    test_wec_multi_iteration()
