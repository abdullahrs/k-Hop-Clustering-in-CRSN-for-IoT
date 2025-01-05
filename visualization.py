import matplotlib.pyplot as plt
import numpy as np


def visualize_nodes(nodes, iteration=None):
    """
    Visualize nodes as dots in their spatial positions.

    Args:
        nodes: List of nodes with positions (x, y) and states.
        iteration: Current iteration number (optional).
    """
    plt.figure(figsize=(10, 8))

    # Define colors based on states
    for node in nodes:
        color = (
            "gray"
            if node.state == "INITIAL"
            else "red" if node.state == "CLUSTERED_CH" else "blue"
        )
        plt.scatter(
            node.position[0], node.position[1], c=color, s=100, label=f"Node {node.id}"
        )
        plt.text(
            node.position[0] + 0.2,
            node.position[1] + 0.2,
            f"{node.id} {[c.id for c in node.channels]}",
            fontsize=9,
        )

    # Title and labels
    plt.title(
        f"Node Positions at Iteration {iteration}" if iteration else "Node Positions"
    )
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()


class NetworkVisualization:
    def __init__(self, metrics_ec, metrics_wec):
        """
        Initialize with metrics data for EC and WEC algorithms.

        Args:
            metrics_ec (MetricsTracker): Metrics for the EC algorithm.
            metrics_wec (MetricsTracker): Metrics for the WEC algorithm.
        """
        self.metrics_ec = metrics_ec
        self.metrics_wec = metrics_wec

    def plot_energy_over_time(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_ec.total_energy, label="EC", linestyle="--", color="red")
        plt.plot(
            self.metrics_wec.total_energy, label="WEC", linestyle="-", color="blue"
        )
        plt.title("Network Remaining Energy Over Time")
        plt.xlabel("Iterations")
        plt.ylabel("Total Energy")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_alive_nodes_over_time(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_ec.alive_nodes, label="EC", linestyle="--", color="red")
        plt.plot(self.metrics_wec.alive_nodes, label="WEC", linestyle="-", color="blue")
        plt.title("Number of Alive Nodes Over Time")
        plt.xlabel("Iterations")
        plt.ylabel("Alive Nodes")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_first_node_death(self):
        values = [
            self.metrics_ec.first_node_death_round,
            self.metrics_wec.first_node_death_round,
        ]
        algorithms = ["EC", "WEC"]
        plt.figure(figsize=(8, 6))
        plt.bar(algorithms, values, color=["red", "blue"])
        plt.title("Number of Rounds Before First Node Dies")
        plt.ylabel("Rounds")
        plt.show()

    def plot_reclustering_frequency(self):
        reclustering_freq_ec = self.metrics_ec.reclustering_events / len(
            self.metrics_ec.total_energy
        )
        reclustering_freq_wec = self.metrics_wec.reclustering_events / len(
            self.metrics_wec.total_energy
        )
        values = [reclustering_freq_ec, reclustering_freq_wec]
        algorithms = ["EC", "WEC"]
        plt.figure(figsize=(8, 6))
        plt.bar(algorithms, values, color=["red", "blue"])
        plt.title("Reclustering Frequency")
        plt.ylabel("Frequency")
        plt.show()
        
    def num_of_clusters(self):
        
        # Assuming `metrics_tracker` is your instance
        ec_cluster_count = self.metrics_ec.cluster_count

        # Calculate min, avg, and max
        ec_min_clusters = min(ec_cluster_count)
        ec_avg_clusters = sum(ec_cluster_count) / len(ec_cluster_count)
        ec_max_clusters = max(ec_cluster_count)
        
        wec_cluster_count = self.metrics_wec.cluster_count
        wec_min_clusters = min(wec_cluster_count)
        wec_avg_clusters = sum(wec_cluster_count) / len(wec_cluster_count)
        wec_max_clusters = max(wec_cluster_count)
        
        labels = ['Minimum', 'Average', 'Maximum']
        ec_values = [ec_min_clusters, ec_avg_clusters, ec_max_clusters]
        wec_values = [wec_min_clusters, wec_avg_clusters, wec_max_clusters]

        x = np.arange(len(labels))  # Label locations
        width = 0.35  # Bar width

        # Plot
        _, ax = plt.subplots()
        ax.bar(x - width/2, ec_values, width, label='EC', color='red')
        ax.bar(x + width/2, wec_values, width, label='WEC', color='blue')

        # Add text for labels, title, and axes
        ax.set_ylabel('Number of Clusters')
        ax.set_title('Cluster Count Metrics: EC vs WEC')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.show()
