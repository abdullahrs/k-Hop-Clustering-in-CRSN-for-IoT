class MetricsTracker:
    def __init__(self):
        self.total_energy = []
        self.alive_nodes = []
        self.cluster_count = []
        self.cluster_size_distribution = []
        self.reclustering_events = 0
        self.reclustering_rounds = []
        self.first_node_death_round = None
        self.cluster_energy_distribution = []
        self.average_cluster_weight = []
        self.reclustering_intervals = []
        self.reclustering_frequency = 0

    def update_metrics(self, iteration: int, algorithm):
        # Update total energy
        total_energy = sum(
            node.energy for node in algorithm.network.nodes if node.is_alive()
        )
        self.total_energy.append(total_energy)

        # Update alive nodes
        alive_nodes = sum(1 for node in algorithm.network.nodes if node.is_alive())
        self.alive_nodes.append(alive_nodes)

        # Record first node death round
        if self.first_node_death_round is None and alive_nodes < len(
            algorithm.network.nodes
        ):
            self.first_node_death_round = iteration

        # Update cluster count and size distribution
        self.cluster_count.append(len(algorithm.clusters))
        self.cluster_size_distribution.append(
            [len(cluster.members) for cluster in algorithm.clusters.values()]
        )

        # Update cluster energy distribution
        cluster_energy = [
            sum(node.energy for node in cluster.members)
            for cluster in algorithm.clusters.values()
        ]
        self.cluster_energy_distribution.append(cluster_energy)

        # Update average cluster weight
        if algorithm.clusters:
            avg_weight = sum(
                cluster.cluster_weight for cluster in algorithm.clusters.values()
            ) / len(algorithm.clusters)
        else:
            avg_weight = 0
        self.average_cluster_weight.append(avg_weight)

    def record_reclustering_event(self, iteration: int, number_of_dead_clusters: int):
        if number_of_dead_clusters > 0:
            # Update the reclustering event count
            self.reclustering_events += number_of_dead_clusters
            self.reclustering_rounds.append(iteration)

            # Calculate reclustering intervals
            if len(self.reclustering_rounds) > 1:
                interval = self.reclustering_rounds[-1] - self.reclustering_rounds[-2]
                self.reclustering_intervals.append(interval)

            # Update reclustering frequency
            total_iterations = len(self.total_energy)
            self.reclustering_frequency = (
                self.reclustering_events / total_iterations if total_iterations > 0 else 0
            )

    def summarize_metrics(self):
        summary = {
            "Total Iterations": len(self.total_energy),
            "Reclustering Events": self.reclustering_events,
            "Reclustering Frequency": self.reclustering_frequency,
            "First Node Death Round": self.first_node_death_round,
            "Final Total Energy": self.total_energy[-1] if self.total_energy else 0,
            "Final Cluster Count": self.cluster_count[-1] if self.cluster_count else 0,
            "Average Reclustering Interval": (
                sum(self.reclustering_intervals) / len(self.reclustering_intervals)
                if self.reclustering_intervals
                else None
            ),
        }
        return summary
