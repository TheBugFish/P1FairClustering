from ucimlrepo import fetch_ucirepo
import numpy as np
import json

class data_loader:

    def __init__(self, configPath):
        #get config file
        self.configPath = configPath
        config_file = open(self.configPath)
        self.config = json.load(config_file)
        config_file.close()

    def get_config(self):
        return self.config

    def load_dataset(self, dataset_name):
        # fetch dataset
        dataset = None
        if(dataset_name == "adult"):
            dataset = fetch_ucirepo(id=2) 
        elif(dataset_name == "bank"):
            dataset = fetch_ucirepo(id=222)
        else:
            print("please enter a valid dataset name")

        X = dataset.data.features
        y = dataset.data.targets

        return X, y

    def prepare_dataset(self, dataset, dataset_name):
        dataset = dataset[dataset[self.config[dataset_name]['sensitive_column']].isin(self.config[dataset_name]['sensitive_values'])]
        dataset = dataset[[self.config[dataset_name]['sensitive_column']] + self.config[dataset_name]['distance_columns']].copy()
        dataset[self.config[dataset_name]['sensitive_column']] = np.where(dataset[self.config[dataset_name]['sensitive_column']] == self.config[dataset_name]['sensitive_values'][0], 0, 1)

        return dataset
    
    def sample_data(self, dataset, dataset_name, random_state):
        return dataset.sample(n=self.config[dataset_name]['subset_size'], random_state=random_state)
    
    def red_blue_split(self, dataset, dataset_name):
        reds = list(dataset[dataset[self.config[dataset_name]['sensitive_column']]==0].index)
        blues = list(dataset[dataset[self.config[dataset_name]['sensitive_column']]==1].index)
        return reds, blues