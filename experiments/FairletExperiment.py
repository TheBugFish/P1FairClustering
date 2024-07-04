from fairlets.fairlets import fairlet_decomposition
from helper.data_loader import data_loader
from helper.math_helper import math_helper
from helper.k_center import k_center
from helper.k_medians import k_medians

#get valid randomStates for the Fairlet experiment meaning where balance >= 0.5
def FindValidRandomStates(dataset, dataset_name, loader: data_loader, iterations=5, randomState=42):
    mathHelper = math_helper(loader)
    config = loader.get_config()
    validRandomStates = []

    foundRandomStates = 0
    while(foundRandomStates < iterations):

        sample_balance = 0
        while(sample_balance < 0.5):
            sampled_dataset = loader.sample_data(dataset, dataset_name, randomState)
            sample_balance = mathHelper.get_balance(sampled_dataset, config[dataset_name]['sensitive_column'][0], dataset_name)
            randomState+=1
        validRandomStates.append(randomState-1)
        foundRandomStates+=1

    return validRandomStates

def DoFairletExperiment(dataset, dataset_name, loader: data_loader,  randomStates, fair_algorithm="fairlets", clustering_method="k-centers", starting_num_clusters=3, max_num_clusters=20, balance_evaluation="min"):

    if fair_algorithm == "fairlets":
        cluster_counts = []
        unfair_costs = []
        unfair_balances = []
        unfair_durations = []
        fair_costs = []
        fair_balances = []
        fair_durations = []
        fairlet_durations = []

        for randomState in randomStates:
            sampled_dataset = loader.sample_data(dataset, dataset_name, randomState)
            reds, blues = loader.red_blue_split(sampled_dataset, dataset_name)
            fairletDecomposition = fairlet_decomposition(sampled_dataset, blues, reds, dataset_name, loader)
            fairlet_duration = fairletDecomposition.get_cluster_information(clustering_method=clustering_method, t=2, T=400)
            fairlet_durations.append(fairlet_duration)

            if clustering_method == "k-centers":
                k_centers_instance = k_center(dataset_name, loader)
                cluster_count, unfair_cost, unfair_balance, unfair_duration, fair_cost, fair_balance, fair_duration = fairletDecomposition.CalculateClusterCostAndBalance(k_centers_instance, None, clustering_method, starting_num_clusters, max_num_clusters, balance_evaluation)

                cluster_counts.append(cluster_count)
                unfair_costs.append(unfair_cost)
                unfair_balances.append(unfair_balance)
                unfair_durations.append(unfair_duration)
                fair_costs.append(fair_cost)
                fair_balances.append(fair_balance)
                fair_durations.append(fair_duration)

            elif clustering_method == "k-medians":
                k_medians_instance = k_medians(dataset_name, loader)
                cluster_count, unfair_cost, unfair_balance, unfair_duration, fair_cost, fair_balance, fair_duration = fairletDecomposition.CalculateClusterCostAndBalance(None, k_medians_instance, clustering_method, starting_num_clusters, max_num_clusters, balance_evaluation)

                cluster_counts.append(cluster_count)
                unfair_costs.append(unfair_cost)
                unfair_balances.append(unfair_balance)
                unfair_durations.append(unfair_duration)
                fair_costs.append(fair_cost)
                fair_balances.append(fair_balance)
                fair_durations.append(fair_duration)
            else:
                print("enter valid clustering algorithm")

        return cluster_counts, unfair_costs, unfair_balances, unfair_durations, fair_costs, fair_balances, fair_durations, fairlet_durations
    else:
        print("enter valid fair clustering algorithm")


def PlotAlogirthmResults(fairlet_results: tuple, FAfC_results, bla, balance_evaluation = "min"):
    print("TODO")