import matplotlib.pyplot as plt

def visualize_data(data, su_channels, clusters = []):
    """
    Visualizes data from a dictionary of the form {int: (float, float)}
    and labels the points that have an associated value in the su_channels dictionary.
    Additionally, it highlights the points that belong to the same cluster.

    Parameters:
    data (dict): A dictionary where the keys are integers and the values are tuples of two floats.
    su_channels (dict): A dictionary where the keys are integers and the values are sets of integers.
    clusters (list): A list of sets, where each set represents a cluster of points.
    """
    # Extract the x and y values from the dictionary
    x_values = [x for x, _ in data.values()]
    y_values = [y for _, y in data.values()]
    
    # Create the figure and axis objects
    _, ax = plt.subplots(figsize=(4, 4))
    
    # Plot the scatter points
    ax.scatter(x_values, y_values, s=50, c='#8884d8')

    # Add labels for the points that have an associated value in su_channels
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        if i in su_channels:
            ax.text(x, y, "ACHs:"+str(su_channels[i]), fontsize=8, ha='center', va='bottom')
    
    if len(clusters) > 0:
        # Highlight the points that belong to the same cluster
        cluster_colors = ['#FF6B6B', '#FFA500', '#9B59B6', 
                          '#1ABC9C', '#3498DB', '#E74C3C', 
                          '#FFCC00FF', '#2980B9', '#7796AAFF', 
                          '#1F0603FF', '#3D361CFF', '#9FC3DBFF',
                          '#1C031FFF', '#D48750FF', '#7B7BFF9A',
                          '#7E687FFF', '#002842FF', '#00D0FFFF', ]
        for i, cluster in enumerate(clusters):
            cluster_x = [x_values[j] for j in cluster]
            cluster_y = [y_values[j] for j in cluster]
            ax.scatter(cluster_x, cluster_y, s=100, c=cluster_colors[i % len(cluster_colors)], alpha=0.7)
            for point_idx in cluster:
                ax.text(x_values[point_idx], y_values[point_idx], f"Cluster {i}, Node {point_idx}", fontsize=8, ha='center', va='top')

    # Set the axis limits and labels
    ax.set_xlim([min(x_values) - 1, max(x_values) + 1])
    ax.set_ylim([min(y_values) - 1, max(y_values) + 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Data Visualization')
    
    # Show the plot
    plt.show()