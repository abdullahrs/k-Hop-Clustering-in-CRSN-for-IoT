from typing import Set
from base_models import Channel, Cluster, EnergyConsumption, Node, NodeState
from crsn_network import CRSNetwork
from mwcbg import MWCBG
from utilities import get_k_hop_neighbors


class KSCABEC:
    def __init__(
        self,
        network: CRSNetwork,
        mwcbg: MWCBG,
    ) -> None:
        self.network = network
        self.mwcbg = mwcbg
        self.clusters = {}
        self.previous_clusters = {}

    def run(self):
        self.initial_form_clustering()
        intermediate_nodes, channel_node_relations = self.form_edge_contraction()

        merged_clusters = self.merge_clusters(
            intermediate_nodes, channel_node_relations
        )

        self.finalize_clusters(merged_clusters)
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

    def initial_form_clustering(self):
        unclustered_nodes = set(self.network.nodes)

        while unclustered_nodes:
            current_node = None
            common_channels = []
            members = set()
            max_weight = -1
            for node in unclustered_nodes:
                if node.state != NodeState.INITIAL:
                    continue

                neighbors, _ = get_k_hop_neighbors(
                    node,
                    self.network.nodes,
                    1,
                    transmission_range=self.network.transmission_range,
                )

                # print(
                #     f"node : {node.id} NEIGHBORS:",
                #     [neighbor.id for neighbor in neighbors],
                # )

                node.consume_energy(EnergyConsumption.SENSING)
                for neighbor in neighbors:
                    neighbor.consume_energy(EnergyConsumption.SENSING)

                available_neighbors = [
                    neighbor
                    for neighbor in neighbors
                    if neighbor.state == NodeState.INITIAL
                ]

                available_channels = node.channels.union(
                    *[neighbor.channels for neighbor in available_neighbors]
                )

                # common channels, cluster members, maximum weight of the subgraph
                cmni, PCMi, wi = self.mwcbg.find_maximum_subgraph(
                    node, available_channels, available_neighbors
                )

                if not cmni or not PCMi:
                    continue


                if len(cmni) >= 2 and wi > max_weight:
                    common_channels = cmni
                    current_node = node
                    max_weight = wi
                    members = PCMi

            # print("CLUSTER FOUND :", [node.id for node in members])
            # No cluster found
            if current_node is None:
                break
            if members:
                current_node.consume_energy(EnergyConsumption.SENSING)
                if current_node.state == NodeState.INITIAL:
                    current_node.consume_energy(EnergyConsumption.SWITCHING)
                    current_node.state = NodeState.INTERMEDIATE_CH

                for member in members:
                    if member.state == NodeState.INITIAL:
                        member.consume_energy(EnergyConsumption.SWITCHING)
                        member.state = NodeState.CLUSTERED_CM

                cluster = Cluster(
                    cluster_head=current_node,
                    common_channels=common_channels,
                    members=set(members) | {current_node},
                    cluster_weight=max_weight,
                )
                self.clusters[current_node.id] = cluster
                unclustered_nodes -= set(members) | {current_node}
            else:
                # single node cluster
                # current_node.state = NodeState.CLUSTERED_CH
                # current_node.consume_energy(EnergyConsumption.SWITCHING)
                # self.clusters[current_node.id] = Cluster(
                #     cluster_head=current_node,
                #     common_channels=current_node.channels,
                #     members={current_node},
                #     cluster_weight=0,
                # )
                # unclustered_nodes.remove(current_node)
                pass

    def form_edge_contraction(self):
        intermediate_nodes = set()
        intermediate_channel_node_relation = {}
        for cluster in self.clusters.values():
            if cluster.cluster_head.state == NodeState.INTERMEDIATE_CH:
                cluster.cluster_head.consume_energy(EnergyConsumption.SENSING)
                cluster_common_channels = cluster.common_channels
                if not cluster_common_channels:
                    continue

                cluster_position = cluster.cluster_head.position

                intermediate_node = Node(
                    node_id=cluster.cluster_head.id,
                    x=cluster_position[0],
                    y=cluster_position[1],
                    initial_energy=cluster.cluster_head.energy,
                    available_channels=cluster_common_channels,
                )
                intermediate_node.state = NodeState.INTERMEDIATE_CH
                intermediate_node.consume_energy(EnergyConsumption.SWITCHING)
                intermediate_node.consume_energy(EnergyConsumption.SENSING)
                intermediate_nodes.add(intermediate_node)
                for channel in cluster_common_channels:
                    if channel in intermediate_channel_node_relation:
                        intermediate_channel_node_relation[channel].append(
                            intermediate_node
                        )
                    else:
                        intermediate_channel_node_relation[channel] = [
                            intermediate_node
                        ]
        return intermediate_nodes, intermediate_channel_node_relation

    def merge_clusters(
        self, intermediate_nodes, channel_node_relations
    ) -> Set[Cluster]:
        unclustered_nodes = set(intermediate_nodes)
        merged_clusters = set()
        clustered_nodes = set()
        while unclustered_nodes:
            current_node = unclustered_nodes.pop()
            if current_node in clustered_nodes:
                continue

            neighbors = set()
            available_channels = set()
            for channel in current_node.channels:
                if channel in channel_node_relations:
                    available_channels.add(channel)
                    neighbors.update(channel_node_relations[channel])

            for neighbor in neighbors:
                neighbor.consume_energy(EnergyConsumption.SENSING)

            # control if the neighbors are already clustered
            available_neighbors = {
                neighbor
                for neighbor in neighbors
                if neighbor not in clustered_nodes and neighbor.id != current_node.id
            }

            if not available_neighbors:
                clustered_nodes.add(current_node)
                cls = Cluster(
                    current_node,
                    current_node.channels,
                    {current_node},
                    0,
                )
                # Make the current node a CH
                current_node.consume_energy(EnergyConsumption.SWITCHING)
                merged_clusters.add(cls)
                continue

            # common channels, cluster members, maximum weight of the subgraph
            cmni, PCMi, wi = self.mwcbg.find_maximum_subgraph(
                current_node, available_channels, available_neighbors
            )

            if not cmni or not PCMi:
                continue

            for member in PCMi:
                member.consume_energy(EnergyConsumption.SENSING)

            if len(cmni) >= 1 and PCMi:
                new_ch = max(
                    set(PCMi) | {current_node},
                    # Node with highest energy becomes CH,
                    # if energies is the same, prioritize nodes with more channels
                    # else if number of available channels are same, the node with the smallest id becomes CH
                    key=lambda n: (
                        n.energy,
                        len(n.channels),
                        -n.id,
                    ),
                )
                new_ch.state = NodeState.CLUSTERED_CH
                new_ch.consume_energy(EnergyConsumption.SWITCHING)

                # Update states of other nodes in the subgraph
                new_cluster_members = set()
                for member in PCMi:
                    new_ch.consume_energy(EnergyConsumption.SENSING)
                    if member != new_ch:
                        if member.state == NodeState.INTERMEDIATE_CH:
                            member.consume_energy(EnergyConsumption.SENSING)
                        member.state = NodeState.CLUSTERED_CM
                        member.consume_energy(EnergyConsumption.SWITCHING)
                        clustered_nodes.add(member)
                        new_cluster_members.add(member)

                # Add the merged cluster as a single node in the next iteration
                cls = Cluster(new_ch, cmni, new_cluster_members | {new_ch}, wi)
                merged_clusters.add(cls)
                unclustered_nodes -= set(PCMi) | {new_ch}
                if current_node not in clustered_nodes:
                    clustered_nodes.add(current_node)

        return merged_clusters

    def finalize_clusters(self, merged_clusters: Set[Cluster]):
        for cluster in merged_clusters:
            cluster.cluster_head.consume_energy(EnergyConsumption.SENSING)
            # Ensure the intermediate cluster head becomes a final CH
            if cluster.cluster_head.state == NodeState.INTERMEDIATE_CH:
                cluster.cluster_head.state = NodeState.CLUSTERED_CH
                cluster.cluster_head.consume_energy(EnergyConsumption.SWITCHING)

            # Expand the cluster by merging nodes from old clusters
            nodes_to_merge = set()
            for member in cluster.members:
                if member.id == cluster.cluster_head.id:
                    continue

                # Fetch the old cluster of the member
                old_cluster = self.clusters.get(member.id)
                if old_cluster is None:
                    continue  # Skip if the old cluster is already processed

                # Update the old cluster head's state
                old_cluster.cluster_head.state = NodeState.CLUSTERED_CM
                old_cluster.cluster_head.consume_energy(EnergyConsumption.SWITCHING)

                # Collect all nodes from the old cluster to merge
                nodes_to_merge.add(old_cluster.cluster_head)
                nodes_to_merge.update(old_cluster.members)

                # Remove the old cluster from the cluster dictionary
                del self.clusters[old_cluster.cluster_head.id]

            # Merge the nodes into the current cluster
            expanded_cluster = self.clusters.get(cluster.cluster_head.id)
            if expanded_cluster:
                expanded_cluster.members.update(nodes_to_merge)
            else:
                cluster.cluster_head.consume_energy(EnergyConsumption.SENSING)
                expanded_cluster = Cluster(
                    cluster_head=cluster.cluster_head,
                    common_channels=cluster.common_channels,
                    members=nodes_to_merge | {cluster.cluster_head},
                    cluster_weight=cluster.cluster_weight,
                )
            # Update the cluster dictionary
            self.clusters[cluster.cluster_head.id] = expanded_cluster
        for cluster in self.clusters.values():
            cluster.cluster_head.state = NodeState.CLUSTERED_CH

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
            if not cluster.cluster_head.is_alive():
                # If the CH is dead, disband the entire cluster
                for member in cluster.members:
                    member.state = NodeState.INITIAL
                del self.clusters[cluster.cluster_head.node_id]
                continue

            invalid_members = set()
            intermediate_ch_related_members = set()

            for member in set(cluster.members):
                if not member.is_alive():
                    # Mark dead members as INITIAL and add to invalid_members
                    member.state = NodeState.INITIAL
                    invalid_members.add(member)
                    continue

                distance_to_ch = member.calculate_distance(cluster.cluster_head)
                if distance_to_ch > self.network.transmission_range:
                    # Check if the member is connected through an intermediate CH
                    for intermediate in invalid_members:
                        if (
                            member.calculate_distance(intermediate)
                            <= self.network.transmission_range
                            and intermediate.calculate_distance(cluster.cluster_head)
                            > self.network.transmission_range
                        ):
                            intermediate_ch_related_members.add(member)
                            member.state = NodeState.INITIAL
                            break

            # Remove invalid members and update the cluster
            cluster.members -= invalid_members
            cluster.members -= intermediate_ch_related_members

            if len(cluster.members) == 1:  # If only the CH remains, disband the cluster
                cluster.cluster_head.state = NodeState.INITIAL
                del self.clusters[cluster.cluster_head.id]

        return len(clusters) + len(self.clusters)


def test_intermediate_ch_merging():
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

    # Initialize MWCBG and K-SACB-EC algorithm
    mwcbg = MWCBG(preference_factor=0.5)
    algorithm = KSCABEC(network, mwcbg)

    # Run the algorithm
    algorithm.run()

    # Print results
    print("Clusters after merging intermediate CHs:")
    for cluster_id, cluster in algorithm.clusters.items():
        print(f"Cluster ID: {cluster_id}")
        print(f"  CH: Node {cluster.cluster_head.id}")
        print(f"  Members: {[node.id for node in cluster.members]}")
        print(f"  Common Channels: {[ch.id for ch in cluster.common_channels]}")


def test_over_iterations():
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

    # Initialize MWCBG and K-SACB-EC algorithm
    mwcbg = MWCBG(preference_factor=0.5)
    algorithm = KSCABEC(network, mwcbg)

    # Run the algorithm over multiple iterations
    max_iterations = 1000
    counter = 0
    for iteration in range(max_iterations):
        print(f"--- Iteration {iteration + 1} ---")
        reclustering_count = algorithm.run()
        if reclustering_count > 0:
            counter += reclustering_count
        if iteration == 881:
            print([node.id for node in algorithm.network.nodes])
            print([node.energy for node in algorithm.network.nodes])
        # Print cluster states
        # print("Clusters:")
        # for cluster_id, cluster in algorithm.clusters.items():
        #     print(f"  Cluster ID: {cluster_id}")
        #     print(f"    CH: Node {cluster.cluster_head.id}")
        #     print(f"    Members: {[node.id for node in cluster.members]}")
        #     print(f"    Energies: {[node.energy for node in cluster.members]}")
        #     print(f"    Common Channels: {[ch.id for ch in cluster.common_channels]}")

    # Print final cluster results
    print("\nFinal Clusters: ", counter)
    for cluster_id, cluster in algorithm.clusters.items():
        print(f"Cluster ID: {cluster_id}")
        print(f"  CH: Node {cluster.cluster_head.id}")
        print(f"  Members: {[node.id for node in cluster.members]}")
        print(f"  Energies: {[node.energy for node in cluster.members]}")
        print(f"  Common Channels: {[ch.id for ch in cluster.common_channels]}")


if __name__ == "__main__":
    # test_intermediate_ch_merging()
    test_over_iterations()
