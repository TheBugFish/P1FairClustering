from helper.data_loader import data_loader
import numpy as np

class math_helper:

    def __init__(self, data_loader: data_loader):
        self.config = data_loader.get_config()

    #compute euclidean distance
    def compute_distance(self, a, b, dataset_name):
        res = 0
        for distance_column in self.config[dataset_name]['distance_columns']:
            res += (a[distance_column] - b[distance_column]) ** 2

        return np.sqrt(res)
    
    #returns a 2-dimensional array of distances between all red and blue points
    def get_distances(self, dataset, a, b, dataset_name):

        distances = [[0]* len(b)] * len(a)

        for idx_blue, i in enumerate(a):
            for idx_red, j in enumerate(b):
                distances[idx_blue][idx_red] = self.compute_distance(dataset.loc[i], dataset.loc[j], dataset_name=dataset_name)

        return distances
    
    #manhattan distance
    def manhattan_distance(self, X, points):
        p_size = points.shape[0]
        n = X.shape[0]
        distance = np.zeros((n,p_size),dtype='int32')

        for p in range(p_size):
            for i in range(n):
                distance[i,p] = abs(X[i,0]-points[p,0])+abs(X[i,1]-points[p,1])
        return distance

    def get_balance(self, data, sensitive_column, dataset_name):
        if(data[self.config[dataset_name]['sensitive_column']].std()==0):
            return 0
        return min(data.value_counts(sensitive_column)[0]/data.value_counts(sensitive_column)[1],data.value_counts(sensitive_column)[1]/data.value_counts(sensitive_column)[0])