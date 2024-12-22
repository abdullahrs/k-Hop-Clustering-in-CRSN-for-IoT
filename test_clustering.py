# test_clustering.py


from k_sacb_ec import KSABEC
from k_sacb_wec import KSABWEC
from simulation_environment import CRSNEnvironment, Node


def create_test_network():
    """
    Create a test network with specific topology to test both algorithms

    Network structure:
    - Central cluster: nodes close together with many common channels
    - Bridge nodes: connecting different parts with limited channels
    - Peripheral nodes: farther away with varying channel availability

    This setup allows testing:
    - EC: Edge contraction with varying node densities
    - WEC: Different hop counts and optimal path finding
    """

    params = {
        "AREA_SIZE": (300, 300),
        "NUM_CHANNELS": 8,
        "TRANSMISSION_RANGE": 60,  # Increased for better connectivity
        "INITIAL_ENERGY": 0.2,
    }

    env = CRSNEnvironment(params)

    # Clear existing nodes
    env.nodes = []

    # Create nodes with specific positions and channels
    nodes = [
        # Central cluster (dense group with good channel availability)
        Node(
            id=1,
            x=150,
            y=150,
            available_channels={1, 2, 3, 4},
            initial_energy=0.2,
            transmission_range=60,
        ),
        Node(
            id=2,
            x=160,
            y=160,
            available_channels={1, 2, 3, 5},
            initial_energy=0.2,
            transmission_range=60,
        ),
        Node(
            id=3,
            x=140,
            y=160,
            available_channels={1, 2, 3, 4},
            initial_energy=0.2,
            transmission_range=60,
        ),
        # Bridge nodes (connecting different parts)
        Node(
            id=4,
            x=200,
            y=150,
            available_channels={2, 3, 4, 6},
            initial_energy=0.2,
            transmission_range=60,
        ),
        Node(
            id=5,
            x=150,
            y=200,
            available_channels={1, 3, 5, 6},
            initial_energy=0.2,
            transmission_range=60,
        ),
        # Peripheral nodes (testing hop count scenarios)
        Node(
            id=6,
            x=240,
            y=150,
            available_channels={3, 4, 6, 7},
            initial_energy=0.2,
            transmission_range=60,
        ),
        Node(
            id=7,
            x=150,
            y=240,
            available_channels={3, 5, 6, 8},
            initial_energy=0.2,
            transmission_range=60,
        ),
        Node(
            id=8,
            x=250,
            y=250,
            available_channels={4, 6, 7, 8},
            initial_energy=0.2,
            transmission_range=60,
        ),
        # Isolated node with common channels (testing reachability)
        Node(
            id=9,
            x=280,
            y=280,
            available_channels={3, 4, 6, 7},
            initial_energy=0.2,
            transmission_range=60,
        ),
    ]

    env.nodes = nodes
    return env


def verify_cluster_properties(cluster, algorithm_type):
    """Verify that cluster meets all required properties"""
    verifications = []

    # Check if cluster has a valid CH
    verifications.append(("Has valid CH", cluster.ch is not None))

    # Check if cluster has members
    verifications.append(("Has members", len(cluster.members) > 0))

    # Check if CH is part of members
    verifications.append(("CH is in members", cluster.ch in cluster.members))

    if algorithm_type == "EC":
        # For EC, all members must share common channels
        verifications.append(
            ("Has bi-channel connectivity", len(cluster.common_channels) >= 2)
        )
        all_have_common = all(
            cluster.common_channels.issubset(member.available_channels)
            for member in cluster.members
        )
        verifications.append(("All members have common channels", all_have_common))
    else:  # WEC
        # For WEC, verify hop-by-hop connectivity
        has_connectivity = True
        visited = set([cluster.ch])
        to_visit = set([cluster.ch])

        while to_visit:
            current = to_visit.pop()
            for member in cluster.members:
                if member not in visited:
                    # Check if there's bi-channel connectivity between these nodes
                    common = current.available_channels & member.available_channels
                    if len(common) >= 2:
                        visited.add(member)
                        to_visit.add(member)

        has_connectivity = len(visited) == len(cluster.members)
        verifications.append(
            ("Has hop-by-hop bi-channel connectivity", has_connectivity)
        )
        verifications.append(
            (
                "Has valid hop count",
                hasattr(cluster, "hop_count") and cluster.hop_count >= 0,
            )
        )

    return verifications


def test_clustering_algorithms():
    """Test both clustering algorithms with the test network"""

    # Create test environment
    env = create_test_network()

    print("\nTesting k-SACB-EC Algorithm:")
    print("=" * 30)

    # Test EC algorithm
    ec_algorithm = KSABEC(env)
    ec_clusters = ec_algorithm.form_clusters()

    print(f"\nNumber of EC clusters formed: {len(ec_clusters)}")
    for i, cluster in enumerate(ec_clusters):
        print(f"\nCluster {i+1}:")
        print(f"CH: Node {cluster.ch.id}")
        print(f"Members: {[n.id for n in cluster.members]}")
        print(f"Common Channels: {cluster.common_channels}")

        print("\nVerifications:")
        verifications = verify_cluster_properties(cluster, "EC")
        for desc, passed in verifications:
            print(f"{desc}: {'✓' if passed else '✗'}")

    print("\nTesting k-SACB-WEC Algorithm:")
    print("=" * 30)

    # Test WEC algorithm
    wec_algorithm = KSABWEC(env)
    wec_clusters = wec_algorithm.form_clusters()

    print(f"\nNumber of WEC clusters formed: {len(wec_clusters)}")
    for i, cluster in enumerate(wec_clusters):
        print(f"\nCluster {i+1}:")
        print(f"CH: Node {cluster.ch.id}")
        print(f"Members: {[n.id for n in cluster.members]}")
        print(f"Common Channels: {cluster.common_channels}")
        print(f"Hop Count: {cluster.hop_count}")

        print("\nVerifications:")
        verifications = verify_cluster_properties(cluster, "WEC")
        for desc, passed in verifications:
            print(f"{desc}: {'✓' if passed else '✗'}")


if __name__ == "__main__":
    test_clustering_algorithms()
