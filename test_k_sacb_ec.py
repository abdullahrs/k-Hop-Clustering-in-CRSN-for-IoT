from crsn_enviroment import CRSNetwork, Node, NodeState
from k_sacb_ec import KSACBEC
from network_metrics import CRSNMetrics
from random import seed


import matplotlib.pyplot as plt

seed(23)  # For reproducibility


def test_ksacbec():
    # Setup: Create a small CRSNetwork with predefined nodes
    num_sus = 5
    network = CRSNetwork(num_sus, alpha=2, beta=0.5)
    network.initialize_graph()

    # Initialize Metrics
    metrics = CRSNMetrics()

    # Initialize k-SACB-EC
    ksacbec = KSACBEC(network, metrics)

    # Run clustering
    clusters = ksacbec.run()

    # Test Case 1: Verify clusters satisfy minimum common channels
    for cluster in clusters:
        assert len(cluster["common_channels"]) >= 2, (
            f"Cluster with CH {cluster['cluster_head'].node_id} does not satisfy "
            f"minimum common channels: {cluster['common_channels']}"
        )
        print(
            f"Cluster Head: {cluster['cluster_head'].node_id}, Members: {cluster['cluster_members']}, Common Channels: {cluster['common_channels']}"
        )

    # Test Case 2: Verify edge contraction results
    for cluster in clusters:
        if len(cluster["cluster_members"]) > 1:
            supernode_channels = cluster["cluster_head"].channels
            for member_id in cluster["cluster_members"]:
                assert supernode_channels.issubset(
                    network.nodes[member_id].channels
                ), f"Cluster head channels {supernode_channels} are not a subset of member {member_id} channels"

    # Test Case 3: Verify all nodes reach final states
    for node in network.nodes:
        assert node.state in (
            NodeState.CLUSTERED_CM,
            NodeState.CLUSTERED_CH,
        ), f"Node {node.node_id} has invalid final state: {node.state}"
    print("Energy History:", metrics.energy_history)
    print("Cluster History:", metrics.cluster_history)
    print("Alive Nodes:", metrics.alive_history)
    print("Recluster Intervals:", metrics.recluster_intervals)
    print("Clusters:", clusters)

    print("All k-SACB-EC test cases passed!")


def visualize_network(network):
    """
    Visualize the network before clustering. Displays:
    - Node positions.
    - Node IDs.
    - Channels available to each node.
    Args:
        network (CRSNetwork): The network to visualize.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    # positions = {node.node_id: node.position for node in network.nodes}
    # print(positions)
    # Plot nodes
    for node in network.nodes:
        x, y = node.position
        ax.scatter(x, y, s=100, edgecolors="black")  # label=f"Node {node.node_id}",
        ax.text(
            x + 0.5,
            y + 0.5,
            f"ID: {node.node_id}, {node.position}\nCh: {list(node.channels)}",
            fontsize=8,
            color="blue",
        )

    # Set graph properties
    ax.set_title("Network Visualization Before Clustering")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True)
    # plt.legend()
    plt.show()


def visualize_clusters(network, clusters):

    pos = {node.node_id: node.position for node in network.nodes}
    plt.figure(figsize=(10, 10))

    # Draw nodes
    for node in network.nodes:
        if node.state == NodeState.CLUSTERED_CH:
            color = "red"
        elif node.state == NodeState.CLUSTERED_CM:
            color = "blue"
        else:
            color = "purple"
        # color = "red" if node.state == NodeState.CLUSTERED_CH else "blue"
        plt.scatter(*node.position, color=color, label=f"Node {node.node_id}")

    # Draw edges
    for cluster in clusters.values():
        cluster_head = cluster["cluster_head"]
        cluster_head_pos = pos[cluster_head.node_id]
        for member in cluster["cluster_members"]:
            if member.node_id != cluster_head.node_id:  # Avoid self-loops
                member_pos = pos[member.node_id]
                plt.plot(
                    [cluster_head_pos[0], member_pos[0]],
                    [cluster_head_pos[1], member_pos[1]],
                    linestyle="--",
                    color="green",
                    alpha=0.7,
                )

    plt.show()


def test_ksacbec_multiple_rounds():
    """
    Test the KSACBEC algorithm over multiple rounds to validate its behavior
    and ensure that it adheres to the paper's requirements.
    """
    # Setup: Create a CRSNetwork with predefined nodes
    num_sus = 100  # Larger network for multiple rounds
    network = CRSNetwork(num_sus, alpha=2, beta=0.5)
    network.initialize_graph()

    # Initialize Metrics
    metrics = CRSNMetrics()

    # Initialize k-SACB-EC
    ksacbec = KSACBEC(network, metrics)

    # Number of rounds to simulate
    num_rounds = 900

    for round_number in range(num_rounds):
        print(f"--- Round {round_number + 1} ---")
        
        # Run the clustering algorithm for the current round
        clusters = ksacbec.run()
        
        # Assertions per round
        assert metrics.round_counter == round_number + 1, "Round counter mismatch."
        
        # Ensure no invalid cluster states
        for cluster in clusters.values():
            ch = cluster["cluster_head"]
            assert ch.state == NodeState.CLUSTERED_CH, f"Cluster head {ch.node_id} is not in CLUSTERED_CH state."
            for member in cluster["cluster_members"]:
                assert member.state == NodeState.CLUSTERED_CM, f"Cluster member {member.node_id} is not in CLUSTERED_CM state."

        # Check that nodes' energy is decreasing
        total_energy = sum(node.energy for node in network.nodes)
        assert total_energy <= metrics.energy_history[-1], "Energy did not decrease as expected."

        # Verify no duplicate cluster members
        all_members = set()
        for cluster in clusters.values():
            for member in cluster["cluster_members"]:
                assert member.node_id not in all_members, f"Duplicate member {member.node_id} in clusters."
                all_members.add(member.node_id)

        # Validate reclustering frequency (basic check)
        if metrics.recluster_intervals:
            assert metrics.recluster_intervals[-1] >= 1, f"Invalid recluster interval. {metrics.recluster_intervals[-1]}"

        # Stop simulation if all nodes are dead
        if sum(node.is_alive for node in network.nodes) == 0:
            print("All nodes are dead. Ending simulation.")
            break

    # Final Assertions after all rounds
    assert metrics.round_counter <= num_rounds, "Simulation exceeded the maximum number of rounds."
    print(f"Final Reclustering Count: {metrics.recluster_count}")
    print(f"Final Round Count: {metrics.round_counter}")
    print(f"Remaining Energy: {metrics.energy_history[-1]}")
    print(f"Number of Alive Nodes: {metrics.alive_history[-1]}")
    print(f"Number of Final Clusters: {len(clusters)}")
    print(f"Reclustering frequancy: {metrics.recluster_count/metrics.round_counter}")

# Run the test
test_ksacbec_multiple_rounds()
