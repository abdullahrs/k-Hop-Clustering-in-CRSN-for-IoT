from comparison_algorithms import NSAC, CogLEACH
from constants import *
from k_SACB_EC import k_SACB_EC
from k_SACB_WEC import k_SACB_WEC
from utility_functions import calculate_channel_qualities, calculate_edges
from visualization import visualize_data


print("channel_usage :", channel_usage)
channel_qualities = calculate_channel_qualities(NUM_CHANNELS)
# Calculate edges based on positions, channels, and range
edges = calculate_edges(NODES, su_positions,
                        su_channels, SU_TRANSMISSION_RANGE)

# k_SACB_EC PART

# # Pass edges to k_SACB_EC
# clusters = k_SACB_EC(NODES, edges, su_channels.copy(),
#                      su_positions, channel_qualities,
#                      SU_TRANSMISSION_RANGE, k=2)


# print("su_positions :", su_positions)
# print("su_channels :", su_channels)
# print("clusters :", clusters)

# visualize_data(su_channels=su_channels, data=su_positions, clusters=clusters)

# k_SACB_WEC PART

# wec_clusters, wec_cluster_heads = k_SACB_WEC(NUM_SUS, su_positions, su_channels, channel_qualities, edges)

# print("wec_clusters :",wec_clusters)
# print("wec_cluster_heads :",wec_cluster_heads)

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

# # Form clusters using NSAC
# nsac_clusters, nsac_cluster_heads = nsac_algo.form_clusters()

# # Output the results
# print("NSAC Clusters:", clusters)
# print("NSAC Cluster Heads:", cluster_heads)


# CogLEACH

cog_leach_algo = CogLEACH(
    NODES, NUM_SUS, su_positions, su_channels, channel_qualities, SU_TRANSMISSION_RANGE,
)

cog_leach_clusters, cog_leach_cluster_heads = cog_leach_algo.execute()

# # Output the results
print("CogLEACH Clusters:", cog_leach_clusters)
print("CogLEACH Cluster Heads:", cog_leach_cluster_heads)