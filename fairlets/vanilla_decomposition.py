from fairlets.Fairlet_decomposition import Fairlet_decomposition
from helper.k_center import k_center
from helper.k_medians import k_medians
from math import gcd
import numpy as np
import random
import time

from helper.data_loader import data_loader

class Vanilla_decomposition(Fairlet_decomposition):

    def __init__(self, dataset, blues, reds, dataset_name, data_loader: data_loader):
        super().__init__(dataset, blues, reds, dataset_name, data_loader)
        self.swapped_colors = False

    def testValidity(self, p, q):
        if(gcd(p, q)) and p <= q and min(float(self.red_count / self.blue_count), float(self.blue_count / self.red_count)) >= float(p / q):
           return True
        else:
            return False

    def makeFairlet(self, points, fairlets, fairlet_centers, costs, distanceMetric):

        cost_list = [
            (i, max([self.mathHelper.compute_distance(self.dataset.loc[i], self.dataset.loc[j], self.dataset_name, distanceMetric) for j in points])) for i in points
        ]
        cost_list = sorted(cost_list, key=lambda x: x[1], reverse=False)
        center, cost = cost_list[0][0], cost_list[0][1]

        # Adding the shortlisted points to the fairlets
        fairlets.append(points)
        fairlet_centers.append(center)
        costs.append(cost)

        return

    def decompose(self, p, q, randomState = 42, clustering_method = "k-centers"):
        distanceMetric = Fairlet_decomposition.getDistanceMetric(self, clustering_method)
        validity = self.testValidity(p, q)

        assert validity, "please make sure that the inputs are valid"

        fairlets = []
        fairlet_centers = []
        fairlet_costs = []

        if self.red_count < self.blue_count:  # We want the reds to be bigger in size as they correspond to 'q' parameter
            temp = self.blues
            self.blues = self.reds
            self.reds = temp
            self.swapped_colors = True

            self.red_count = len(self.reds)
            self.blue_count = len(self.blues)

        random.seed(randomState)
        random.shuffle(self.reds)
        random.shuffle(self.blues)
        
        b = 0
        r = 0

        while (
            ((self.red_count - r) - (self.blue_count - b)) >= (q - p)
            and (self.red_count - r) >= q
            and (self.blue_count - b) >= p
        ):
            self.makeFairlet(
                self.reds[r : (r + q)] + self.blues[b : (b + p)],
                fairlets,
                fairlet_centers,
                fairlet_costs,
                distanceMetric
            )
            r += q
            b += p
        if ((self.red_count - r) + (self.blue_count - b)) >= 1 and ((self.red_count - r) + (self.blue_count - b)) <= (p + q):
            self.makeFairlet(
                self.reds[r:] + self.blues[b:],
                fairlets,
                fairlet_centers,
                fairlet_costs,
                distanceMetric
            )
            r = self.red_count
            b = self.blue_count
        elif ((self.red_count - r) != (self.blue_count - b)) and ((self.blue_count - b) >= p):
            self.makeFairlet(
                self.reds[r : r + (self.red_count - r) - (self.blue_count - b) + p]
                + self.blues[b : (b + p)],
                fairlets,
                fairlet_centers,
                fairlet_costs,
                distanceMetric
            )
            r += (self.red_count - r) - (self.blue_count - b) + p
            b += p
        assert (self.red_count - r) == (self.blue_count - b), "Error in computing fairlet decomposition."
        for i in range(self.red_count - r):
            self.makeFairlet(
                [self.reds[r + i], self.blues[b + i]],
                fairlets,
                fairlet_centers,
                fairlet_costs,
                distanceMetric
            )

        print("%d fairlets have been identified." % (len(fairlet_centers)))
        assert len(fairlets) == len(fairlet_centers)
        assert len(fairlet_centers) == len(fairlet_costs)

        return fairlets, fairlet_centers, fairlet_costs

    def get_cluster_balances(self, dataset, cluster_mapping, fairlet_information=None, fair=False):
        clusters = []
        for center in cluster_mapping.keys():
            clusters.append(dataset.iloc[cluster_mapping[center]])

        if fair==True:
            all_clustered_point_idx = []
            for cluster_point in clusters:
                clustered_points = cluster_point.index
                all_points = []
                for point in clustered_points:
                    for fairlet_points in fairlet_information:
                        if point in fairlet_points:
                            all_points.append(fairlet_points)
                all_clustered_point_idx.append(all_points)

            all_clustered_points_lists = []
            for all_clustered_point in all_clustered_point_idx:
                all_clustered_points_lists.append([x for xs in all_clustered_point for x in xs])
    
            clusters = []
            for all_clustered_points_list in all_clustered_points_lists:
                clusters.append(self.dataset[self.dataset.index.isin(all_clustered_points_list)])

        balances = []
        for cluster in clusters:
            if len(cluster) > 1:
                balances.append(self.mathHelper.get_balance(cluster, self.config[self.dataset_name]['sensitive_column'][0], self.dataset_name))
        return balances

    def get_fairlet_center_dataframe(self, centers, drop=True):
        if drop == True:
            return self.dataset[self.dataset.index.isin(centers)].drop(self.config[self.dataset_name]['sensitive_column'][0], axis=1)
        else:
            return self.dataset[self.dataset.index.isin(centers)]

    def makeFairlets(self, p, q, randomState = 42, clustering_method = "k-centers"):
        fairlet_start_time = time.time()
        self.information, self.fairlet_centers, fairlet_costs = self.decompose(p, q, randomState, clustering_method)
        self.fairlet_center_sampled_dataset = self.get_fairlet_center_dataframe(self.fairlet_centers)
        self.fairlet_centers_dataset = self.get_fairlet_center_dataframe(self.fairlet_centers, False)
        return (time.time() - fairlet_start_time)
    
    def CalculateClusterCostAndBalance(self, k_centers_instance: k_center, k_medians_instance: k_medians, clustering_method = "k-centers", starting_num_clusters=3, max_num_clusters=20, balance_evaluation="min"):
        unfair_costs = []
        unfair_balances = []
        unfair_durations = []

        fair_costs = []
        fair_balances = []
        fair_durations = []

        cluster_counts = []

        for cluster_count in range(starting_num_clusters, max_num_clusters+1):
            cluster_counts.append(cluster_count)

            if(clustering_method == "k-centers"):
                unfair_start_time = time.time()
                k_centers_instance.fit(self.dataset, cluster_count)
                vanilla_costs, vanilla_cluster_mapping = k_centers_instance.get_results()
                unfair_durations.append(time.time() - unfair_start_time)
                unfair_costs.append(vanilla_costs)
                unfair_balance = self.get_cluster_balances(self.dataset,vanilla_cluster_mapping)
                if(balance_evaluation == "min"):
                    unfair_balances.append(min(unfair_balance))
                elif(balance_evaluation == "mean"):
                    unfair_balances.append(np.average(unfair_balance))

                fair_start_time = time.time()
                k_centers_instance.fit(self.fairlet_center_sampled_dataset, cluster_count)
                fairlet_costs, fairlet_cluster_mapping = k_centers_instance.get_results()
                fair_durations.append(time.time() - fair_start_time)
                fair_costs.append(fairlet_costs)
                fair_balance = self.get_cluster_balances(self.fairlet_centers_dataset, fairlet_cluster_mapping, self.information, True)
                if(balance_evaluation == "min"):
                    fair_balances.append(min(fair_balance))
                elif(balance_evaluation == "mean"):
                    fair_balances.append(np.average(fair_balance))

            elif(clustering_method == "k-medians"):
                unfair_start_time = time.time()
                k_medians_instance.fit(self.dataset, cluster_count)
                vanilla_costs, vanilla_cluster_mapping = k_medians_instance.get_results()
                unfair_durations.append(time.time() - unfair_start_time)
                unfair_costs.append(vanilla_costs)
                unfair_balance = self.get_cluster_balances(self.dataset,vanilla_cluster_mapping)
                if(balance_evaluation == "min"):
                    unfair_balances.append(min(unfair_balance))
                elif(balance_evaluation == "mean"):
                    unfair_balances.append(np.average(unfair_balance))

                fair_start_time = time.time()
                k_medians_instance.fit(self.fairlet_center_sampled_dataset, cluster_count)
                fairlet_costs, fairlet_cluster_mapping = k_medians_instance.get_results()
                fair_durations.append(time.time() - fair_start_time)
                fair_costs.append(fairlet_costs)
                fair_balance = self.get_cluster_balances(self.fairlet_centers_dataset, fairlet_cluster_mapping, self.information, True)
                if(balance_evaluation == "min"):
                    fair_balances.append(min(fair_balance))
                elif(balance_evaluation == "mean"):
                    fair_balances.append(np.average(fair_balance))

        return cluster_counts, unfair_costs, unfair_balances, unfair_durations, fair_costs, fair_balances, fair_durations