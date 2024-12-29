# network_metrics.py
class CRSNMetrics:
    """
    Tracks CRSN metrics across multiple rounds of simulation:
      1. Network remaining energy (energy_history).
      2. Number of alive nodes (alive_history).
      3. Number of clusters (cluster_history).
      4. Rounds before first node dies (first_node_dead_round).
      5. Reclustering frequency (recluster_count).
      6. Reclustering intervals (recluster_intervals).
    """

    def __init__(self):
        # Round index
        self.round_counter = 0

        # Energy / Node metrics
        self.energy_history = []  # total energy over rounds
        self.alive_history = []  # number of alive nodes over rounds
        self.cluster_history = []  # number of clusters over rounds
        self.first_node_dead_round = None

        # Reclustering-related
        self.recluster_count = 0
        self.recluster_rounds = []  # record which rounds we performed reclustering
        self.recluster_intervals = (
            []
        )  # differences between consecutive recluster rounds

    def update_round(self, network, clusters):
        """
        Call once per simulation round (or after each final cluster formation):
          - increments self.round_counter
          - records total network energy
          - records number of alive nodes
          - records number of clusters
          - checks if first node died this round
        """
        self.round_counter += 1

        # 1) Total energy
        total_energy = sum(n.energy for n in network.nodes)
        # 2) Alive nodes
        alive_nodes = sum(n.is_alive for n in network.nodes)
        # 3) Cluster count
        num_clusters = len(clusters)

        self.energy_history.append(total_energy)
        self.alive_history.append(alive_nodes)
        self.cluster_history.append(num_clusters)

        # 4) If this is the first time a node died, record the round
        if self.first_node_dead_round is None and alive_nodes < len(network.nodes):
            self.first_node_dead_round = self.round_counter

    def record_recluster_event(self):
        """
        Whenever we do a reclustering call, we do:
          - increment total count
          - compute interval vs. the last recluster round
        """
        self.recluster_count += 1

        # If we have done a previous recluster, measure interval
        if self.recluster_rounds:
            interval = self.round_counter - self.recluster_rounds[-1]
            self.recluster_intervals.append(interval)

        # Record that we reclustered on the current round count
        self.recluster_rounds.append(self.round_counter)

    @property
    def rounds_before_first_node_dead(self):
        """
        For convenience, either return the exact round
        or 0 if no node died by the end of the simulation.
        """
        return self.first_node_dead_round if self.first_node_dead_round else 0
