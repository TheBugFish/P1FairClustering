import numpy as np
from itertools import groupby
from helper.math_helper import math_helper
from helper.data_loader import data_loader

class k_center:

    def __init__(self, dataset_name, data_loader: data_loader):
        self.dataset_name = dataset_name
        self.mathHelper = math_helper(data_loader)

    def fit(self, data, n_clusters=5):
        self.centers = [int(np.random.randint(0, len(data), 1))]
        self.costs = []
        while True:
            remaining_points = list(set(range(0, len(data))) - set(self.centers))
            point_center = [(i, min([self.mathHelper.compute_distance(data.iloc[i], data.iloc[j], self.dataset_name) for j in self.centers])) for i in remaining_points]
            point_center = sorted(point_center, key=lambda x: x[1], reverse=True)
            self.costs.append(point_center[0][1])
            if(len(self.centers) < n_clusters):
                self.centers.append(point_center[0][0])
            else:
                break

        self.point_mapping = [(i, sorted([(j, self.mathHelper.compute_distance(data.iloc[i], data.iloc[j], self.dataset_name)) for j in self.centers], key=lambda x: x[1], 
						   reverse=False)[0][0]) for i in range(len(data))]

    def get_results(self):
        cluster_mapping = {key: [v[0] for v in val] for key, val in groupby(
        sorted(self.point_mapping, key=lambda ele: ele[1]), key=lambda ele: ele[1])}

        return self.costs, cluster_mapping