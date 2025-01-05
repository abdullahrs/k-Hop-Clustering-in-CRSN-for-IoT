from collections import defaultdict, deque
from typing import List, Set
from base_models import Node, NodeState


def get_k_hop_neighbors(
    node: Node,
    nodes: List[Node],
    k: int,
    min_common_channels: int = 2,
    transmission_range: float = 50.0,
) -> Set[Node]:
    if k < 1:
        return set()

    # Initialize queue for BFS and visited set
    queue = deque([(node, 0)])  # (current_node, current_hop)
    visited = set([node])  # Start with the current node

    k_hop_neighbors = set() # Collect neighbors up to k hops
    relations = defaultdict(set)  # Store node relations

    while queue:
        current_node, current_hop = queue.popleft()

        # If beyond k hops, stop processing
        if current_hop >= k:
            break

        # Find immediate neighbors based on channel intersection
        for candidate in nodes:
            if candidate in visited:
                continue
            distance = current_node.calculate_distance(candidate)
            if distance > transmission_range:
                continue
            intersection = candidate.channels.intersection(current_node.channels)
            # print(f"Node {current_node.id} to Node {candidate.id} distance: {distance}, candidate.channels {[c.id for c in candidate.channels]} current_node.channels {[c.id for c in current_node.channels]} intersection {[c.id for c in intersection]}")
            if len(intersection) >= min_common_channels:
                visited.add(candidate)
                queue.append((candidate, current_hop + 1))
                k_hop_neighbors.add(candidate)
                # Add bidirectional relations
                relations[current_node].add(candidate)
                relations[candidate].add(current_node)

    return k_hop_neighbors, relations


def test_get_k_hop_neighbors():
    node1 = Node(1, x=0, y=0, available_channels={1, 2, 3})
    node2 = Node(2, x=10, y=0, available_channels={2, 3})
    node3 = Node(3, x=5, y=5, available_channels={3, 4})
    node4 = Node(4, x=20, y=20, available_channels={4, 5})
    node5 = Node(5, x=20, y=0, available_channels={2, 3})

    nodes = [node1, node2, node3, node4, node5]

    result = get_k_hop_neighbors(
        node1, nodes, k=2, min_common_channels=2, transmission_range=12
    )
    print(result)


if __name__ == "__main__":
    test_get_k_hop_neighbors()
