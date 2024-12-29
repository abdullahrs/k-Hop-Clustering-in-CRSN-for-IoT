import math
from config import SENSING_COST, SWITCHING_COST, TRANSMISSION_RANGE
from crsn_enviroment import Node, NodeState


class KSACBEC:
    def __init__(self, network, metrics):
        self.network = network
        self.metrics = metrics
        self.clusters = {}  # Final clusters formed in the network
        self.contracted_edges = {}

    def run(self):
        """
        Main method to form clusters using the KSACBEC algorithm.
        """
        reclustered = False
        for node in self.network.nodes:
            if not node.is_alive:
                continue
            # Step 1: Discover neighbors within transmission range
            neighbors = self.discover_neighbors_within_range(node)
            if not neighbors:
                continue

            # Deduct sensing cost for neighbors
            for neighbor in neighbors:
                neighbor.consume_energy(SENSING_COST)

            if node.state in [NodeState.CLUSTERED_CH, NodeState.CLUSTERED_CM]:
                continue  # Skip already clustered nodes
            # Step 2: MWCBG computation
            common_channels, participant_node_ids, weight = (
                self.network.construct_mwcbg(node, neighbors)
            )

            if len(common_channels) >= 2 and weight > 0:  # Minimum channels constraint
                if node.state == NodeState.INITIAL:
                    node.state = NodeState.INTERMEDIATE_CH
                    node.energy -= SWITCHING_COST

                # Step 3: Form intermediate cluster
                cluster_members = set()
                for participant_id in participant_node_ids:
                    participant = self.network.nodes[participant_id]
                    if participant.state in [
                        NodeState.INITIAL,
                        NodeState.INTERMEDIATE_CH,
                    ]:
                        participant.state = NodeState.CLUSTERED_CM
                        participant.energy -= SWITCHING_COST
                        cluster_members.add(participant)

                # Update node's state
                node.energy -= SWITCHING_COST
                cluster = {
                    "cluster_head": node,
                    "cluster_members": cluster_members,
                    "common_channels": common_channels,
                }
                if (
                    node.node_id not in self.clusters
                    or self.clusters[node.node_id] != cluster
                ):
                    node.state = NodeState.CLUSTERED_CH  # just in
                    self.clusters[node.node_id] = cluster
                    reclustered = True
                # Step 4: Apply edge contraction for the cluster
                self.apply_edge_contraction(node, cluster_members, common_channels)
            else:
                # Fallback: Treat as single cluster head
                node.state = NodeState.CLUSTERED_CH

        # Finalize clusters and update metrics
        self.finalize_clusters()
        if reclustered:
            self.metrics.record_recluster_event()
        self.metrics.update_round(self.network, self.clusters)
        print(
            "cluster length :",
            len(self.clusters),
            "recluster count :",
            self.metrics.recluster_count,
        )
        self.reset_dead_cluster_heads()
        return self.clusters

    def apply_edge_contraction(self, cluster_head, cluster_members, common_channels):
        """
        Simulate edge contraction by updating an overlay structure.
        """
        contracted_cluster = {
            "cluster_head": cluster_head.node_id,
            "members": {member.node_id for member in cluster_members},
            "common_channels": common_channels,
        }
        self.contracted_edges[cluster_head.node_id] = contracted_cluster

    def reset_dead_cluster_heads(self):
        """
        Check for dead cluster heads and reset the state of their cluster members.
        """
        clusters_to_remove = []

        for cluster_id, cluster in self.clusters.items():
            ch = cluster["cluster_head"]
            if ch.energy <= 0:  # CH is dead
                clusters_to_remove.append(cluster_id)

                # Reset the state of cluster members
                for member in cluster["cluster_members"]:
                    member.state = NodeState.INITIAL  # Reset to initial state

        # Remove clusters with dead CHs
        for cluster_id in clusters_to_remove:
            del self.clusters[cluster_id]

    def discover_neighbors_within_range(self, node):
        """
        Identify neighbors within the transmission range, factoring in alpha and beta effects.
        """
        neighbors = []
        for other_node in self.network.nodes:
            if other_node.node_id == node.node_id:
                continue
            distance = math.dist(node.position, other_node.position)
            if distance <= TRANSMISSION_RANGE:
                # Filter neighbors based on dynamic channel availability
                shared_channels = {ch for ch in node.channels & other_node.channels}
                if shared_channels:
                    neighbors.append(other_node)
        return neighbors

    def finalize_clusters(self):
        """
        Finalize cluster states: Ensure all nodes are either Clustered_CH or Clustered_CM.
        """
        for node in self.network.nodes:
            if node.state not in [NodeState.CLUSTERED_CH, NodeState.CLUSTERED_CM]:
                node.state = NodeState.CLUSTERED_CM  # Default to single node cluster
