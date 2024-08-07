import matplotlib.pyplot as plt
import numpy as np

class plotHelper:
     
    def __init__(self):
        pass
     
    def plot_costs_and_balance(self, cluster_counts, balances, costs, step_size, title):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
        ax[0].plot(costs, marker='.', color='blue')
        ax[0].set_xticks(list(range(0, len(cluster_counts), step_size)))
        ax[0].set_xticklabels(list(range(min(cluster_counts), max(cluster_counts)+1, step_size)), fontsize=12)
        ax[0].set_ylabel("costs")
        ax[1].plot(balances, marker='x', color='saddlebrown')
        ax[1].set_xticks(list(range(0, len(cluster_counts), step_size))) 
        ax[1].set_xticklabels(list(range(min(cluster_counts), max(cluster_counts)+1, step_size)), fontsize=12)
        ax[1].set_ylabel("min balance of all the clusters")
        fig.supxlabel("Number of clusters")
        fig.suptitle(title)
        plt.show()

    def plot_durations(self, cluster_counts, unfair_durations, fair_durations, full_fair_durations):
        x_axis = np.arange(min(cluster_counts), max(cluster_counts)+1)
        plt.bar(x_axis - 0.4, unfair_durations, 0.4, label = "unfair durations")
        plt.bar(x_axis + 0, fair_durations, 0.4, label = "fair durations")
        plt.bar(x_axis + 0.4, full_fair_durations, 0.4, "fair durations incl. fairlet decomp.")
        plt.bar(x_axis)

        plt.xticks(x_axis, cluster_counts)
        plt.xlabel("Number of Clusters") 
        plt.ylabel("duration in seconds") 
        plt.title("Comparison of clustering durations") 
        plt.legend() 
        plt.show() 