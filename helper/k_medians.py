import numpy as np
from helper.math_helper import math_helper
from helper.data_loader import data_loader

class k_medians:

    def __init__(self, dataset_name, data_loader: data_loader, maxRuns = 100):
        self.dataset_name = dataset_name
        self.maxRuns = maxRuns
        self.mathHelper = math_helper(data_loader)

    def fit(self, data, n_clusters):
        self.cluster_assignment = np.zeros(len(data))
        self.costs = [None] * n_clusters

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
