import numpy as np
from itertools import groupby
from helper.math_helper import math_helper
from helper.data_loader import data_loader

#2-approximation according to Gonzalez
#same method used in the papers
class k_center:

    def __init__(self, dataset_name, data_loader: data_loader):
        self.dataset_name = dataset_name
        self.mathHelper = math_helper(data_loader)


    def fit(self, data, n_clusters=5):

        self.cluster_assignment = [None] * len(data)
        distance_to_assigned_cluster = [None] * len(data)
        self.centers = []
        max_distance_index = 0
        max_distance = 0

        for i in range(n_clusters):
            self.centers.append(data.iloc[max_distance_index])

            max_distance = 0
            max_distance_index = 0
            for row in data.iterrows():
                index, point = row

                new_point_distance = self.mathHelper.squared_euclidean_distance(point, self.centers[i], self.dataset_name)

                if i == 0 or new_point_distance < distance_to_assigned_cluster[index]:
                    self.cluster_assignment[index] = i
                    distance_to_assigned_cluster[index] = new_point_distance

                if distance_to_assigned_cluster[index] > max_distance:
                    max_distance = distance_to_assigned_cluster[index]
                    max_distance_index = index

        self.costs = np.sqrt(max_distance)

    def get_results(self):
        cluster_mapping = {}
        for cluster in range(len(self.centers)):
            indices = [i for i in range(len(self.cluster_assignment)) if self.cluster_assignment[i] == cluster]
            cluster_mapping[self.centers[cluster]] = indices

        return self.costs, cluster_mapping