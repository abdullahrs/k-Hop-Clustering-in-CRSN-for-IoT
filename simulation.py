from crsn_network import CRSNetwork
from ksacb_wec import KSCABWEC
from kscab_ec import KSCABEC
from metrics_tracker import MetricsTracker
from mwcbg import MWCBG_Kerbosch, MWCBG
from visualization import NetworkVisualization, visualize_nodes
from copy import deepcopy


def run_simulation(
    algorithm,  # KSCABEC or KSCABWEC
    num_iterations,
    metrics_tracker: MetricsTracker,
):
    for iteration in range(num_iterations):
        if iteration % 100 == 0:
            print(f"{algorithm.__class__.__name__} Iteration {iteration}")
        recluster_count = algorithm.run()
        metrics_tracker.update_metrics(iteration, algorithm)
        metrics_tracker.record_reclustering_event(iteration, recluster_count)


def main():
    network_ec = CRSNetwork(
        alpha=2,
        beta=2,
    )
    network_wec = CRSNetwork(
        alpha=2,
        beta=2,
    )
    # Make the nodes and channels of the WEC network the same as the EC network
    # for the sake of comparison
    network_wec.nodes = deepcopy(network_ec.nodes)
    network_wec.channels = deepcopy(network_ec.channels)

    mwcbg_ec = MWCBG_Kerbosch()
    mwcbg_wec = MWCBG_Kerbosch()

    metrics_tracker_ec = MetricsTracker()
    metrics_tracker_wec = MetricsTracker()

    run_simulation(KSCABEC(network_ec, mwcbg_ec), 2000, metrics_tracker_ec)
    run_simulation(KSCABWEC(network_wec, mwcbg_wec), 2000, metrics_tracker_wec)

    vis = NetworkVisualization(metrics_tracker_ec, metrics_tracker_wec)
    vis.plot_energy_over_time()
    vis.plot_alive_nodes_over_time()
    vis.plot_first_node_death()
    vis.plot_reclustering_frequency()
    vis.num_of_clusters()
    print("--------------------------------")
    print(metrics_tracker_ec.summarize_metrics())
    print("--------------------------------")
    print(metrics_tracker_wec.summarize_metrics())
    print("--------------------------------")


if __name__ == "__main__":
    main()
