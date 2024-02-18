import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull, distance_matrix


# Function to calculate nearest neighbor statistics
def calculate_nearest_neighbor_stats(df):
    # Calculate the distance matrix
    dist_matrix = distance_matrix(df[['X', 'Y']], df[['X', 'Y']])

    # Set the diagonal to np.inf to ignore zero distance to self
    np.fill_diagonal(dist_matrix, np.inf)

    # Find the minimum distance for each point
    nearest_neighbor_distances = np.min(dist_matrix, axis=1)

    # Calculate the mean and standard deviation
    mean_distance = np.mean(nearest_neighbor_distances)
    std_distance = np.std(nearest_neighbor_distances)

    return mean_distance, std_distance


# Function to calculate convex hull area
def convex_hull_area(df):
    # Calculate the convex hull
    points = df[['X', 'Y']].to_numpy()
    if len(points) >= 3:  # Convex Hull requires at least 3 points
        hull = ConvexHull(points)
        return hull.area
    else:
        return None

# Load the data and convert coordinates from cm to m
file_paths = ['./datasets/env/line1.csv', './datasets/env/line2.csv', './datasets/env/line3.csv']
# Calculate nearest neighbor statistics and convex hull area for each dataset
results_nn_stats = []
results_hull_area = []

for file_path in file_paths:
    # Load the data and convert coordinates from cm to m
    df = pd.read_csv(file_path)
    df[['X', 'Y']] = df[['X', 'Y']] / 100  # Convert cm to m

    # Calculate nearest neighbor statistics
    mean_dist, std_dist = calculate_nearest_neighbor_stats(df)
    results_nn_stats.append((file_path, mean_dist, std_dist))

    # Calculate convex hull area
    hull_area = convex_hull_area(df)
    results_hull_area.append((file_path, hull_area))

# Compile results
print(results_nn_stats, results_hull_area)
