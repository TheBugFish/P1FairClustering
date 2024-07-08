from helper.data_loader import data_loader
from helper.math_helper import math_helper

class Fairlet_decomposition:

    def __init__(self, dataset, blues, reds, dataset_name, data_loader: data_loader):
        self.dataset = dataset
        self.blues = blues
        self.reds = reds
        self.dataset_name = dataset_name
        self.config = data_loader.get_config()
        self.mathHelper = math_helper(data_loader)

        self.blue_count = len(self.blues)
        self.red_count = len(self.reds)

    def getDistances(self, distanceMetric = "euclidean"):
        return self.mathHelper.get_distances(self.dataset, self.blues, self.reds, self.dataset_name, distanceMetric)
    
    def getDistanceMetric(self, clustering_method):
        if clustering_method == "k-centers":
            return "euclidean"
        elif clustering_method == "k-medians":
            return "sqeuclidean"
