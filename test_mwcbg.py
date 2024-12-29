import math
from config import SENSING_COST
from crsn_enviroment import CRSNetwork


def test_discover_neighbors():
    # Setup: Create a CRSNetwork and initialize the graph
    num_sus = 10
    crs_network = CRSNetwork(num_sus, alpha=2, beta=2)
    crs_network.initialize_graph()

    # Test Case 1: Zero energy
    node_zero_energy = crs_network.nodes[0]
    node_zero_energy.energy = 0
    assert crs_network.discover_neighbors(node_zero_energy) == set(), "Failed: Zero energy node should return no neighbors"

    # Test Case 2: Sufficient energy (with neighbors)
    node_sufficient_energy = crs_network.nodes[1]
    node_sufficient_energy.energy = 1.0  # High enough energy
    neighbors = crs_network.discover_neighbors(node_sufficient_energy)
    # Check if neighbors exist in the graph
    if len(neighbors) > 0:
        expected_energy = 1.0 - SENSING_COST * len(neighbors)
        assert node_sufficient_energy.energy == expected_energy, "Failed: Energy deduction incorrect for sufficient energy node"
    else:
        assert True, "Node with sufficient energy has no neighbors (valid case if no common channels or out of range)"

    # Test Case 3: Insufficient energy
    node_insufficient_energy = crs_network.nodes[2]
    node_insufficient_energy.energy = SENSING_COST / 2  # Less than needed for all neighbors
    neighbors = crs_network.discover_neighbors(node_insufficient_energy)
    assert neighbors == set(), "Failed: Node with insufficient energy should return no neighbors"
    assert node_insufficient_energy.energy == SENSING_COST / 2, "Failed: Energy should not change for insufficient energy node"

    # Test Case 4: No neighbors
    node_no_neighbors = crs_network.nodes[3]
    crs_network.graph.remove_edges_from(list(crs_network.graph.edges(node_no_neighbors.node_id)))
    assert crs_network.discover_neighbors(node_no_neighbors) == set(), "Failed: Node with no neighbors should return no neighbors"

    print("All discover_neighbors test cases passed!")


def test_construct_mwcbg():
    # Setup: Create a CRSNetwork
    num_sus = 10
    crs_network = CRSNetwork(num_sus, alpha=2, beta=2)
    crs_network.initialize_graph()

    # Test Case 1: Node with no neighbors
    node_no_neighbors = crs_network.nodes[0]
    crs_network.graph.remove_edges_from(
        list(crs_network.graph.edges(node_no_neighbors.node_id))
    )
    neighbors = crs_network.discover_neighbors(node_no_neighbors)
    channels, nodes, weight = crs_network.construct_mwcbg(node_no_neighbors, neighbors)
    assert (
        channels == set()
    ), "Failed: Channels should be empty for node with no neighbors"
    assert nodes == set(), "Failed: Nodes should be empty for node with no neighbors"
    assert weight == 0, "Failed: Weight should be zero for node with no neighbors"

    # Test Case 2: Node with shared channels
    node_shared_channels = crs_network.nodes[1]
    neighbors = crs_network.discover_neighbors(node_shared_channels)
    channels, nodes, weight = crs_network.construct_mwcbg(
        node_shared_channels, neighbors
    )
    assert (
        len(channels) > 0
    ), "Failed: Channels should not be empty for node with shared channels"
    assert (
        len(nodes) > 0
    ), "Failed: Nodes should not be empty for node with shared channels"
    assert weight > 0, "Failed: Weight should be positive for valid MWCBG"

    # Test Case 3: Node with insufficient shared channels
    node_insufficient_channels = crs_network.nodes[2]
    for neighbor in crs_network.discover_neighbors(node_insufficient_channels):
        neighbor.channels.clear()  # Remove all channels from neighbors
    neighbors = crs_network.discover_neighbors(node_insufficient_channels)
    channels, nodes, weight = crs_network.construct_mwcbg(
        node_insufficient_channels, neighbors
    )
    assert (
        channels == set()
    ), "Failed: Channels should be empty when no shared channels exist"
    assert nodes == set(), "Failed: Nodes should be empty when no shared channels exist"
    assert weight == 0, "Failed: Weight should be zero when no shared channels exist"

    # Test Case 4: Weight calculation
    node_weight_check = crs_network.nodes[3]
    neighbors = crs_network.discover_neighbors(node_weight_check)
    channels, nodes, weight = crs_network.construct_mwcbg(node_weight_check, neighbors)
    calculated_weight = sum(
        crs_network.calculate_channel_quality(channel) for channel in channels
    )  # Replace with exact formula if needed
    assert math.isclose(
        weight, calculated_weight, rel_tol=1e-5
    ), "Failed: Weight calculation does not match expected value"

    print("All MWCBG test cases passed!")

def test_mwcbg_large():
    """
    Test MWCBG construction on a large network.
    Verify the properties of the constructed MWCBG.
    """
    # Setup: Large network
    num_sus = 100  # Large number of nodes
    crs_network = CRSNetwork(num_sus, alpha=2, beta=0.5)
    crs_network.initialize_graph()

    # Select a test node and ensure valid shared channels
    node = crs_network.nodes[0]
    for neighbor in crs_network.nodes:
        neighbor.channels = neighbor.channels.union(node.channels)  # Ensure shared channels
    neighbors = crs_network.discover_neighbors(node)

    # Debugging: Validate neighbors and channels
    print(f"Test Node: {node.node_id}")
    print(f"Neighbors: {[n.node_id for n in neighbors]}")
    print(f"Node Channels: {node.channels}")
    for neighbor in neighbors:
        print(f"Neighbor {neighbor.node_id} Channels: {neighbor.channels}")

    # Construct MWCBG
    channels, nodes, weight = crs_network.construct_mwcbg(node, neighbors, nu=0.5)

    # Assertions
    # 1. Channels and nodes should not be empty if neighbors exist
    if len(neighbors) > 0:
        assert len(channels) > 0, "Failed: MWCBG channels should not be empty for a node with neighbors"
        assert len(nodes) > 0, "Failed: MWCBG nodes should not be empty for a node with neighbors"

    # 2. Validate that all channels and nodes are valid
    for channel in channels:
        assert any(channel in neighbor.channels for neighbor in neighbors), \
            f"Failed: Channel {channel} is not shared with any neighbor"
    for node_id in nodes:
        assert node_id in {n.node_id for n in neighbors}, \
            f"Failed: Node {node_id} is not a valid neighbor"

    # 3. Validate that the weight is consistent with Equation (2)
    channel_quality_sum = sum(crs_network.calculate_channel_quality(c) for c in channels)
    total_distance_sum = sum(
        math.dist(node.position, crs_network.nodes[n].position) for n in nodes
    ) if len(nodes) > 0 else 0

    node_weight_sum = 0
    if total_distance_sum > 0:
        node_weight_sum = sum(
            1 - (math.dist(node.position, crs_network.nodes[n].position) / total_distance_sum)
            for n in nodes
        )
    expected_weight = 0.5 * (len(nodes) * channel_quality_sum) + 0.5 * (len(channels) * node_weight_sum)
    assert math.isclose(weight, expected_weight, rel_tol=1e-5), \
        f"Failed: MWCBG weight {weight} does not match expected {expected_weight}"

    print("test_mwcbg_large passed!")

test_discover_neighbors()
test_construct_mwcbg()
test_mwcbg_large()