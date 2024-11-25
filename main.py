from comparison_algorithms import NSAC, CogLEACH
from constants import *
from k_SACB_EC import kSACBEC
from k_SACB_WEC import kSACBWEC
from simulation import Replication
from utility_functions import calculate_channel_qualities, calculate_edges
from visualization import plot_metrics, visualize_data


# print("channel_usage :", channel_usage)
# channel_qualities = calculate_channel_qualities(NUM_CHANNELS)
# # Calculate edges based on positions, channels, and range
# edges = calculate_edges(NODES, su_positions,
#                         su_channels, SU_TRANSMISSION_RANGE)

# k_SACB_EC PART

# kSACBEC_algo = kSACBEC(NODES, edges, su_channels.copy(),
#                      su_positions, channel_qualities,
#                      SU_TRANSMISSION_RANGE, INITIAL_ENERGY, SENSING_ENERGY)
# clusters = kSACBEC_algo.run()

# print("su_positions :", su_positions)
# print("su_channels :", su_channels)
# print("clusters :", clusters)

# visualize_data(su_channels=su_channels, data=su_positions, clusters=clusters)

# k_SACB_WEC PART

# kSACBWEC_algo = kSACBWEC(
#     num_sus=len(NODES),
#     su_positions=su_positions,
#     su_channels=su_channels,
#     channel_qualities=channel_usage,
#     su_energies=su_energies,
#     edges=edges,
#     k_max=5,
#     preference_factor=0.5
# )

# wec_clusters, wec_cluster_heads = kSACBWEC_algo.run()

# print("wec_clusters :", wec_clusters)
# print("wec_cluster_heads :", wec_cluster_heads)

# visualize_data(su_channels=su_channels, data=su_positions, clusters=wec_clusters)

# NSAC PART

# Initialize NSAC with your existing data
# nsac_algo = NSAC(
#     num_sus=NUM_SUS,
#     su_positions=su_positions,
#     su_channels=su_channels,
#     su_energies=su_energies,
#     channel_qualities=channel_qualities,
#     edges=edges
# )

# # # Form clusters using NSAC
# nsac_clusters, nsac_cluster_heads = nsac_algo.form_clusters()

# # # Output the results
# print("NSAC Clusters:", nsac_clusters)
# print("NSAC Cluster Heads:", nsac_cluster_heads)


# CogLEACH

# cog_leach_algo = CogLEACH(
#     NODES, NUM_SUS, su_positions, su_channels, channel_qualities, SU_TRANSMISSION_RANGE,
# )

# cog_leach_clusters, cog_leach_cluster_heads = cog_leach_algo.execute()

# # # Output the results
# print("CogLEACH Clusters:", cog_leach_clusters)
# print("CogLEACH Cluster Heads:", cog_leach_cluster_heads)

alpha = 2
beta = 2

simulation = Replication(
    simulation_area=SIMULATION_AREA,
    num_sus=NUM_SUS,
    num_channels=NUM_CHANNELS,
    su_transmission_range=SU_TRANSMISSION_RANGE,
    initial_energy=INITIAL_ENERGY,
    sensing_energy=SENSING_ENERGY,
)

metrics = simulation.run()
print(metrics)

plot_metrics(metrics, alpha, beta)