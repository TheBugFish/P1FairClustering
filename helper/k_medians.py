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

        medians = data.sample(n=n_clusters)
        for i in range(self.maxRuns):

            medians_old = medians.copy()

            #assign points to clusters
            for j in range(len(data)):
                distances = []
                for _, median in medians.iterrows():
                    distances.append(self.mathHelper.compute_distance(data.iloc[j], median, self.dataset_name))
                idx = np.argmin(distances)
                self.cluster_assignment[j] = idx

            #update cluster centers
            for j in range(n_clusters):
                cluster_element_locs = [idx for idx, n in enumerate(self.cluster_assignment) if n == j]
                new_median = data.iloc[cluster_element_locs]

                if len(new_median > 0):
                    medians.iloc[j] = new_median.median()
                cost = sum(self.mathHelper.compute_distance(elem, medians.iloc[j], self.dataset_name) for _, elem in new_median.iterrows())
                self.costs[j] = cost

                #stop condition
                median_difference = sum(medians.sum(axis=1).values - medians_old.sum(axis=1).values)
                if(median_difference ==  0 or i >= self.maxRuns):
                    break
    
    def get_results(self):
        self.cluster_assignment = [int(x) for x in self.cluster_assignment]
        cluster_mapping = {}
        for cluster in set(self.cluster_assignment):
            cluster_mapping[cluster] = [i for i, x in enumerate(self.cluster_assignment) if x == cluster]
        return self.costs, cluster_mapping
