from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import json

class data_loader:

    def __init__(self, configPath):
        #get config file
        self.configPath = configPath
        self.scaler = MinMaxScaler()
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

    def prepare_dataset(self, dataset, dataset_name, complexity = "simple"):
        if(complexity == "simple"):
            if(len(self.config[dataset_name]['sensitive_column']) == 1):
                dataset = dataset[dataset[self.config[dataset_name]['sensitive_column'][0]].isin(self.config[dataset_name]['sensitive_values'][0])]
                dataset = dataset[self.config[dataset_name]['sensitive_column'] + self.config[dataset_name]['distance_columns']].copy()
                dataset[self.config[dataset_name]['sensitive_column'][0]] = np.where(dataset[self.config[dataset_name]['sensitive_column'][0]] == self.config[dataset_name]['sensitive_values'][0][0], 0, 1)
            else:
                dataset = dataset[dataset[self.config[dataset_name]['sensitive_column'][0]].isin(self.config[dataset_name]['sensitive_values'][0])]
                dataset = dataset[[self.config[dataset_name]['sensitive_column'][0]] + self.config[dataset_name]['distance_columns']].copy()
                dataset[self.config[dataset_name]['sensitive_column'][0]] = np.where(dataset[self.config[dataset_name]['sensitive_column'][0]] == self.config[dataset_name]['sensitive_values'][0][0], 0, 1)
        elif(complexity == "extended"):
            dataset = dataset[self.config[dataset_name]['sensitive_column'] + self.config[dataset_name]['distance_columns']]

            for i in range(len(self.config[dataset_name]['sensitive_column'])):
                dataset = dataset[dataset[self.config[dataset_name]['sensitive_column'][i]].isin(self.config[dataset_name]['sensitive_values'][i])]

            for i in range(len(self.config[dataset_name]['sensitive_column'])):
                dataset[self.config[dataset_name]['sensitive_column'][i]] = dataset[self.config[dataset_name]['sensitive_column'][i]].apply(lambda x: self.config[dataset_name]['sensitive_values'][i].index(x))
        return dataset
    
    def sample_data(self, dataset, dataset_name, random_state):
        return dataset.sample(n=self.config[dataset_name]['subset_size'], random_state=random_state)
    
    def normalize_data(self, dataset, dataset_name, complexity="simple"):
        original_indices = dataset.index
        distance_columns_normalized = pd.DataFrame(self.scaler.fit_transform(dataset[self.config[dataset_name]['distance_columns']]), columns=self.config[dataset_name]['distance_columns'], index=original_indices)
        if complexity == 'simple':
            normalized_data = dataset[self.config[dataset_name]['sensitive_column'][0]].to_frame().join(distance_columns_normalized)
            return normalized_data
        elif complexity == 'extended':
            normalized_data = dataset[self.config[dataset_name]['sensitive_column']].join(distance_columns_normalized)
            return normalized_data
        else:
            print("use valid complexity measure")

    def red_blue_split(self, dataset, dataset_name):
        reds = list(dataset[dataset[self.config[dataset_name]['sensitive_column'][0]]==0].index)
        blues = list(dataset[dataset[self.config[dataset_name]['sensitive_column'][0]]==1].index)
        return reds, blues