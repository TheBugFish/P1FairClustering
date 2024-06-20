import numpy as np

def squared_euclidean_distance(a,b):
    return sum([(a1 - a2) ** 2 for a1, a2 in zip(a,b)])

def solve_k_centers(data, n_clusters=5):

    cluster_centers = []

    max_distance = 0
    max_distance_index = 0

    is_part_of_cluster = [None] * data.shape[0]
    distance_to_point = [None] * data.shape[0]

    for i in range(n_clusters):
    
        cluster_centers.append(data.iloc[max_distance_index].tolist())

        max_distance = 0
        max_distance_index = 0

        idx = 0
        for row in data.iterrows():
            _, attributes = row
            
            point = attributes.tolist()

            point_distance = squared_euclidean_distance(cluster_centers[i], point)

            if i == 0 or point_distance < distance_to_point[idx]:
                is_part_of_cluster[idx] = i
                distance_to_point[idx] = point_distance

            if distance_to_point[i] > max_distance:
                max_distance = distance_to_point[idx]
                max_distance_index = idx

            idx+=1

    return np.sqrt(max_distance), cluster_centers, is_part_of_cluster