from collections import defaultdict, deque

def build_adjacency_map(start_node, graph):
    adj_map = defaultdict(set)
    
    def get_neighbors(node, k):
        visited = {start_node}
        queue = deque([(node, 0)])
        while queue:
            curr, depth = queue.popleft()
            if depth == k:
                continue
                
            for neighbor in graph[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    adj_map[curr].add(neighbor)
                    adj_map[neighbor].add(curr)

    get_neighbors(start_node, 1)
    return dict(adj_map)

# Example usage:
graph = {
    'node_1': ['node_2', 'node_3'],
    'node_2': ['node_1', 'node_4', 'node_5'],
    'node_3': ['node_1', 'node_6'],
    'node_4': ['node_2'],
    'node_5': ['node_2'],
    'node_6': ['node_3', 'node_7'],
    'node_7': ['node_6']
}

result = build_adjacency_map('node_1', graph)

print(result)