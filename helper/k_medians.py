import numpy as np
import random
from helper.math_helper import math_helper
from helper.data_loader import data_loader
from scipy.spatial.distance import pdist, squareform

class k_medians:

    def __init__(self, dataset_name, data_loader: data_loader, runs = 5):
        self.dataset_name = dataset_name
        self.runs = runs
        self.dataLoader = data_loader
        self.mathHelper = math_helper(data_loader)

    def fit(self, data, n_clusters):

        all_pair_distance = squareform(pdist(data[self.dataLoader.get_config()[self.dataset_name]['distance_columns']].values, 'euclidean'))

        num_points = len(data.index)

        self.best_cluster_centers  = [None] * n_clusters
        self.cluster_assignment = [None] * num_points
        self.costs = None

        for run in range(0, self.runs):

            cluster_centers = []

            accumulative_prob = np.cumsum([1/num_points] * num_points)
            weights = [None] * num_points

            for cluster in range(0, n_clusters):
                new_cluster = None

                while new_cluster is None or new_cluster in cluster_centers:
                    rand = random.uniform(0, 1) - 1e-9
                    new_cluster = np.searchsorted(accumulative_prob, rand)
                cluster_centers.append(new_cluster)

                running_sum = 0
                accumulative_prob = []
                for point in range(0, num_points):
                    if cluster == 0 or all_pair_distance[point][cluster_centers[cluster]] < weights[point]:
                        weights[point] = all_pair_distance[point][cluster_centers[cluster]]

                    running_sum = running_sum + weights[point]
                    accumulative_prob.append(running_sum)
                accumulative_prob = np.divide(accumulative_prob, running_sum)
            
        assignment = [None] * num_points
        assignment_tmp = [None] * num_points
        cost = 0

        for iter in range(2,5):

            updated_sln = True
            while updated_sln is True:
                updated_sln = False

                cost = 0
                for point in range(0, num_points):
                    assignment[point] = 0
                    assignment_tmp[point] = None
                    connection_cost = all_pair_distance[point][cluster_centers[0]]

                    for c in range(1, n_clusters):
                        if all_pair_distance[point][cluster_centers[c]] < connection_cost:
                            # The previously closest center is now the second-closest center
                            assignment_tmp[point] = assignment[point]
                            # Update the closest center
                            assignment[point] = c
                            connection_cost = all_pair_distance[point][cluster_centers[c]]
                        if assignment_tmp[point] is None:
                            assignment_tmp[point] = c
                    cost = cost + connection_cost

                for new_c in range(0,num_points):
                    # Running cost of swapping new_c with each of the current centers
                    swap_cost = np.array([0] * n_clusters)

                    # For all points, compute the connection cost of swapping new_c with each one of the current centers
                    for p in range(0,num_points):
                        connection_cost = np.array([all_pair_distance[p][new_c]]* n_clusters)
                        # If p does not go to this new_c, it has to go to pred[p]
                        c = cluster_centers[assignment[p]]
                        if all_pair_distance[p][new_c] > all_pair_distance[p][c]:
                            connection_cost = np.array([all_pair_distance[p][c]]* n_clusters)
                            sub_c = cluster_centers[assignment_tmp[p]]
                            # But if pred[p] is thrown away, p has to choose between sub_c and new_c
                            connection_cost[assignment[p]] = min(all_pair_distance[p][new_c], all_pair_distance[p][sub_c])
                        swap_cost = np.add(swap_cost, connection_cost)

                    # Find the center for which the swapping cost of new_c is minimum
                    new_cost, c = min((new_cost, c) for (c, new_cost) in enumerate(swap_cost))
                    # Check if this new_c is good for substitution
                    if new_cost < (1-1/(2**iter))* cost:
                        cluster_centers[c] = new_c
                        updated_sln = True
                        # Break the loop to allow for iter to be incremented and allow for smaller improvements
                        break

        if self.costs is None or cost < self.costs:
            self.costs = cost
            self.cluster_assignment = assignment[:]
            self.best_cluster_centers = cluster_centers[:] 
    
    def get_results(self):
        #TODO
        self.cluster_assignment = [int(x) for x in self.cluster_assignment]
        cluster_mapping = {}
        for cluster in set(self.cluster_assignment):
            cluster_mapping[cluster] = [i for i, x in enumerate(self.cluster_assignment) if x == cluster]
        return self.costs, cluster_mapping
