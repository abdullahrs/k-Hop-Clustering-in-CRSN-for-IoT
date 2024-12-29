# mwcbg.py
from collections import deque
from config import NU


class MWCBG:
    """
    A helper to handle the BFS-based MWCBG selection.
    We remove 'mode' and always do BFS up to k hops.
    """

    def __init__(self, network, nu=NU):
        """
        :param network: CRSNNetwork object
        :param nu: the preference factor (ν)
        """
        self.network = network
        self.nu = nu

    def _bfs_k_hop_neighbors(self, center_node_id: int, k: int = 1):
        """
        Standard BFS to find all neighbors up to 'k' hops
        from the 'center_node_id'.
        """
        G = self.network.graph
        visited = set([center_node_id])
        queue = deque([(center_node_id, 0)])
        k_hop_neighbors = set()

        while queue:
            current, depth = queue.popleft()
            if depth >= k:
                continue

            for nbr in G.neighbors(current):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, depth + 1))
                    # neighbor is within k hops
                    k_hop_neighbors.add(nbr)

        return k_hop_neighbors

    def compute_weight(self, center_node_id: int, neighbors: set):
        """
        Computes the MWCBG weight w_i^o for the center node and a neighbor set.

        Returns: (weight_value, common_channels)
        """
        if not neighbors:
            return 0.0, set()

        center_node = self.network.nodes[center_node_id]

        # 1) Common channels among center_node + all neighbors
        common_channels = set(center_node.channels)
        for nbr_id in neighbors:
            nbr_node = self.network.nodes[nbr_id]
            common_channels = common_channels.intersection(nbr_node.channels)

        if not common_channels:
            # If no common channels, the subgraph isn't valid => weight=0
            return 0.0, set()

        # 2) Sum channel qualities for the center node
        #    (assuming we only use center node's perspective)
        channel_quality_sum = 0.0
        for ch in common_channels:
            channel_quality_sum += center_node.calculate_channel_quality(ch)

        # 3) Distance factor sums
        dist_sum = 0.0
        for nbr_id in neighbors:
            dist_sum += center_node.distance_to(self.network.nodes[nbr_id])

        distance_factor_sum = 0.0
        for nbr_id in neighbors:
            dij = center_node.distance_to(self.network.nodes[nbr_id])
            if dist_sum > 0:
                distance_factor_sum += 1.0 - dij / dist_sum
            else:
                distance_factor_sum += 1.0  # edge case if dist_sum=0

        # 4) MWCBG formula:
        #    w_i^o = ν( |N_i^o| * ∑ Q_c )
        #           + (1 - ν)( |C_i^o| * ∑(1 - d_ij / ∑ d_ik) )
        Ni_o = len(neighbors)
        Ci_o = len(common_channels)

        w_io = (self.nu * (Ni_o * channel_quality_sum)) + (
            (1 - self.nu) * (Ci_o * distance_factor_sum)
        )

        return w_io, common_channels

    def select_best_subgraph(self, center_node_id: int, k: int = 1):
        """
        1) Gather k-hop neighbors of 'center_node_id' via BFS
        2) Compute MWCBG weight using that entire neighbor set
        3) Return: (best_weight, best_common_channels, neighbor_set)
        """
        if center_node_id not in self.network.graph.nodes:
            return 0.0, set(), set()

        neighbors_khop = self._bfs_k_hop_neighbors(center_node_id, k=k)
        if not neighbors_khop:
            return 0.0, set(), set()

        w_io, common_channels = self.compute_weight(center_node_id, neighbors_khop)
        return w_io, common_channels, neighbors_khop
