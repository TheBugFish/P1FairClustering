import networkx as nx
import time
import numpy as np
from helper.math_helper import math_helper
from helper.data_loader import data_loader
from helper.k_center import k_center
from helper.k_medians import k_medians

class fairlet_decomposition:

    def __init__(self, dataset, blues, reds, dataset_name, data_loader: data_loader):
        self.dataset = dataset
        self.blues = blues
        self.reds = reds
        self.dataset_name = dataset_name
        self.config = data_loader.get_config()
        self.mathHelper = math_helper(data_loader)

        self.blue_count = len(self.blues)
        self.red_count = len(self.reds)

    def get_distances(self):
        return self.mathHelper.get_distances(self.dataset, self.blues, self.reds, self.dataset_name)

    def create_MCF(self, distances, clustering_method="k-centers", t=2, T=400, maxCost=1000000):
        #supply = negative demand
        #cost = weight

        G = nx.DiGraph()
        #add special nodes beta and rho and an edge between them
        G.add_node('beta', demand=(-1*self.red_count))
        G.add_node('rho', demand=self.blue_count)
        G.add_edge('beta','rho', weight=0, capacity=min(self.red_count,self.blue_count))

        #create a node for each b and r
        for i in range(self.blue_count):
            G.add_node('b%d'%(i+1), demand=-1)
            G.add_edge('beta','b%d'%(i+1), weight=0, capacity=t-1)
        for i in range(self.red_count):
            G.add_node('r%d'%(i+1), demand=1)
            G.add_edge('r%d'%(i+1), 'rho', weight=0, capacity=t-1)


        #create t' copies of the b and r nodes
        for i in range(self.blue_count):
            for extra_node_count in range(t):
                G.add_node('b%d_%d'%(i+1, extra_node_count+1), demand=0)
                G.add_edge('b%d'%(i+1),'b%d_%d'%(i+1,extra_node_count+1), weight=0, capacity=1)
        for i in range(self.red_count):
            for extra_node_count in range(t):
                G.add_node('%d_%d'%(i+1, extra_node_count+1), demand=0)
                G.add_edge('r%d_%d'%(i+1, extra_node_count+1), 'r%d'%(i+1), weight=0, capacity=1)

        #add edges between the t' additional b and r nodes
        for i in range(self.blue_count):
            for k in range(t):
                for j in range(self.red_count):
                    for l in range(t):
                        distance = distances[i][j]
                        if(distance <= T):
                            if(clustering_method == "k-centers"):
                                G.add_edge('b%d_%d'%(i+1, k+1), 'r%d_%d'%(j+1, l+1), weight=1, capacity=1)
                            elif(clustering_method == "k-medians"):
                                G.add_edge('b%d_%d'%(i+1, k+1), 'r%d_%d'%(j+1, l+1), weight=distance, capacity=1)
                        else: 
                            G.add_edge('b%d_%d'%(i+1, k+1), 'r%d_%d'%(j+1, l+1), weight=maxCost, capacity=1)

        self.G = G

    def get_fairlets(self, flowDictionary):
        fairlets = []

        for dictKey in flowDictionary.keys():
            if "b" in dictKey and "_" in dictKey:
                if sum(flowDictionary[dictKey].values()) >= 1:
                    for r_dictKey in flowDictionary[dictKey].keys():
                        if flowDictionary[dictKey][r_dictKey] == 1:
                            if not any(dictKey.split('_')[0] in d['blues'] for d in fairlets)  and not any(r_dictKey.split('_')[0] in d['reds'] for d in fairlets):
                                fairlets.append({'blues': [dictKey.split('_')[0]], 'reds': [r_dictKey.split('_')[0]]})
                            elif any(dictKey.split('_')[0] in d['blues'] for d in fairlets)  and not any(r_dictKey.split('_')[0] in d['reds'] for d in fairlets):
                                for fairlet in fairlets:
                                    if dictKey.split('_')[0] in fairlet['blues']:
                                        fairlet['reds'].append(r_dictKey.split('_')[0])
                            elif not any(dictKey.split('_')[0] in d['blues'] for d in fairlets)  and any(r_dictKey.split('_')[0] in d['reds'] for d in fairlets):
                                for fairlet in fairlets:
                                    if r_dictKey.split('_')[0] in fairlet['reds']:
                                        fairlet['blues'].append(dictKey.split('_')[0])

        return fairlets
    
    def get_fairlet_information(self, flowDictionary):
        fairlets = self.get_fairlets(flowDictionary)

        fairlet_information = []
        fairlet_centers = []
        fairlet_costs = []

        for fairlet in fairlets:
            fairlet_distances = {}
            distances = []
            for blue in fairlet['blues']:
                for blue2 in fairlet['blues']:
                    if blue != blue2:
                        distances.append(self.mathHelper.compute_distance(self.dataset.loc[self.blues[int(blue[1:])-1]], self.dataset.loc[self.blues[int(blue2[1:])-1]], dataset_name=self.dataset_name))
                for red in fairlet['reds']:
                    distances.append(self.mathHelper.compute_distance(self.dataset.loc[self.blues[int(blue[1:])-1]], self.dataset.loc[self.reds[int(red[1:])-1]], dataset_name=self.dataset_name))
                fairlet_distances[blue] = max(distances)
                distances = []

            for red in fairlet['reds']:
                for blue in fairlet['blues']:
                    distances.append(self.mathHelper.compute_distance(self.dataset.loc[self.reds[int(red[1:])-1]], self.dataset.loc[self.blues[int(blue[1:])-1]], dataset_name=self.dataset_name))
                for red2 in fairlet['reds']:
                    if red != red2:
                        distances.append(self.mathHelper.compute_distance(self.dataset.loc[self.reds[int(red[1:])-1]], self.dataset.loc[self.reds[int(red2[1:])-1]], dataset_name=self.dataset_name))
                fairlet_distances[red] = max(distances)

            center = min(fairlet_distances, key=fairlet_distances.get)
            fairlet_centers.append(center)
            fairlet_costs.append(fairlet_distances[center])
            fairlet_information.append(fairlet_distances)

        return fairlet_information, fairlet_centers, fairlet_costs
    
    def get_fairlet_center_dataframe(self, center_codes_list, drop=True):
        indices = []
        for fairlet_center_code in center_codes_list:
            if(fairlet_center_code[:1] == 'r'):
                indices.append(self.reds[int(fairlet_center_code[1:])-1])
            else:
                indices.append(self.blues[int(fairlet_center_code[1:])-1])

        if drop == True:
            return self.dataset[self.dataset.index.isin(indices)].drop(self.config[self.dataset_name]['sensitive_column'], axis=1)
        else:
            return self.dataset[self.dataset.index.isin(indices)]
    
    def get_cluster_balances(self, dataset, cluster_mapping, fairlet_information=None, fair=False):
        clusters = []
        for center in cluster_mapping.keys():
            clusters.append(dataset.iloc[cluster_mapping[center]])

        if fair==True:
            coded_points_fairlets = []
            for fairlet in fairlet_information:
                coded_points = fairlet.keys()
                fairlet_indices = []
                for coded_point in coded_points:
                    if(coded_point[:1] == 'r'):
                        fairlet_indices.append(self.reds[int(coded_point[1:])-1])
                    else:
                        fairlet_indices.append(self.blues[int(coded_point[1:])-1])
                coded_points_fairlets.append(fairlet_indices)

            all_clustered_point_idx = []
            for cluster_point in clusters:
                clustered_points = cluster_point.index
                all_points = []
                for point in clustered_points:
                    for coded_points_fairlet in coded_points_fairlets:
                        if point in coded_points_fairlet:
                            all_points.append(coded_points_fairlet)
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
    

    def get_cluster_information(self, clustering_method="k-centers", t=2, T=400):
        distances = self.get_distances()
        self.create_MCF(distances, clustering_method, t, T)
        flowCost, flowDictionary = nx.network_simplex(self.G)
        self.information, centers, costs = self.get_fairlet_information(flowDictionary)
        self.fairlet_center_sampled_dataset = self.get_fairlet_center_dataframe(centers)
        self.fairlet_centers_dataset = self.get_fairlet_center_dataframe(centers, False)

    def CalculateClusterCostAndBalance(self, k_centers_instance: k_center, k_medians_instance: k_medians, clustering_method = "k-centers", starting_num_clusters=3, max_num_clusters=20, balance_evaluation="min"):
        unfair_costs = []
        unfair_balances = []
        unfair_durations = []

        fair_costs = []
        fair_balances = []
        fair_durations = []

        cluster_counts = []

        for cluster_count in range(starting_num_clusters, max_num_clusters+1):
            print("cluster count: " + str(cluster_count))
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
                unfair_costs.append(vanilla_costs[np.argmax(vanilla_costs)])
                unfair_balance = self.get_cluster_balances(self.dataset,vanilla_cluster_mapping)
                if(balance_evaluation == "min"):
                    unfair_balances.append(min(unfair_balance))
                elif(balance_evaluation == "mean"):
                    unfair_balances.append(np.average(unfair_balance))

                fair_start_time = time.time()
                k_medians_instance.fit(self.fairlet_center_sampled_dataset, cluster_count)
                fairlet_costs, fairlet_cluster_mapping = k_medians_instance.get_results()
                fair_durations.append(time.time() - fair_start_time)
                fair_costs.append(vanilla_costs[np.argmax(fairlet_costs)])
                fair_balance = self.get_cluster_balances(self.fairlet_centers_dataset, fairlet_cluster_mapping, self.information, True)
                if(balance_evaluation == "min"):
                    fair_balances.append(min(fair_balance))
                elif(balance_evaluation == "mean"):
                    fair_balances.append(np.average(fair_balance))

        return cluster_counts, unfair_costs, unfair_balances, unfair_durations, fair_costs, fair_balances, fair_durations
