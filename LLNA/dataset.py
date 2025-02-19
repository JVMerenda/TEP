#%%
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def downsample_binary_series(series, chunk_size):
    """Downsample a binary time series by taking the mean of each chunk with positional encoding"""
    
    n_chunks = series.shape[0] // chunk_size
    downsampled = np.zeros((n_chunks, series.shape[1]))
    
    for i in range(n_chunks):
        chunk = series[i*chunk_size:(i+1)*chunk_size]
        chunk_mean = np.mean(chunk, axis=0)
        
            
        downsampled[i] = chunk_mean
    
    return downsampled

def normalize_time_series(series):
    """
    Normalize each time series independently
    Input shape: (nodes, timesteps) or (timesteps, nodes)
    """
    # Ensure series is (timesteps, nodes)
    if series.shape[0] < series.shape[1]:
        series = series.T
        
    mean = series.mean(axis=0, keepdims=True)
    std = series.std(axis=0, keepdims=True)
    std[std == 0] = 1.0  # Prevent division by zero
    
    normalized = (series - mean) / std
    return normalized.T  # Return to original format

class EdgePredictionDataset(Dataset):
    def __init__(self, 
                 networks_dir, 
                 teps_dir, 
                 network_type='random',
                 num_train=None,
                 num_test=None,
                 is_training=True
        ):  # This will determine the final number of timesteps
        """
        Args:
            networks_dir (str): Directory containing network adjacency lists
            teps_dir (str): Directory containing TEP files
            network_type (str): Type of network to filter
            num_train (int): Number of networks to use for training
            num_test (int): Number of networks to use for testing
            is_training (bool): Whether this is training or testing set
            chunk_size (int): Size of chunks for downsampling. 
                            The number of timesteps will be 1000/chunk_size
        """
        self.networks_dir = networks_dir
        self.teps_dir = teps_dir
        self.network_type = network_type
        self.is_training = is_training
        self.num_train = num_train
        self.num_test = num_test
        
        # Get all TEP files of the specified type
        self.tep_files = [f for f in os.listdir(teps_dir) 
                         if f.startswith(network_type) and f.endswith('_tep.csv')]
        self.tep_files.sort()
        
        # Split into train and test
        total_networks = len(self.tep_files)
        if num_train is None:
            num_train = int(0.8 * total_networks)
        if num_test is None:
            num_test = total_networks - num_train
            
        if is_training:
            self.tep_files = self.tep_files[:num_train]
        else:
            self.tep_files = self.tep_files[-num_test:]
            
        self.networks = []
        self.teps = []
        self.edge_indices = []
        self.labels = []
        
        self._load_data()
        
    def _load_data(self):
        
        for tep_file in self.tep_files:
            network_file = tep_file.replace('_tep.csv', '.txt')
            
            try:
                # Load network and TEP data
                G = nx.read_adjlist(os.path.join(self.networks_dir, network_file))
                G = nx.convert_node_labels_to_integers(G)
                
                tep_data = pd.read_csv(os.path.join(self.teps_dir, tep_file))
                tep_data = tep_data.values  # shape (N, 1000)
                
                
                if len(G.nodes()) != tep_data.shape[1]:
                    print(f"Skipping {tep_file}: Mismatch in number of nodes")
                    continue
                
                n_nodes = len(G.nodes())
                
                # Get all possible node pairs
                for i in range(n_nodes):
                    for j in range(i+1, n_nodes):
                        self.edge_indices.append((len(self.networks), i, j))
                        self.labels.append(1 if G.has_edge(i, j) else 0)
                
                self.networks.append(G)
                self.teps.append(tep_data)
                
            except Exception as e:
                print(f"Error processing {tep_file}: {str(e)}")
                continue
        
        if len(self.networks) == 0:
            raise ValueError("No valid network-TEP pairs found!")
        
    def __len__(self):
        return len(self.edge_indices)
    
    def __getitem__(self, idx):
        network_idx, i, j = self.edge_indices[idx]
        label = self.labels[idx]
        
        # Get time series for both nodes
        tep_data = self.teps[network_idx]
        series_i = tep_data[:,i]  # shape: (1000/chunk_size,)
        series_j = tep_data[:,j]  # shape: (1000/chunk_size,)
        
        # Stack the two time series
        pair_series = np.stack([series_i, series_j])  # shape: (2, 1000/chunk_size)
        
        return torch.FloatTensor(pair_series), torch.FloatTensor([label])


def balance_dataset(dataset, undersample=True):
    """
    Balance the dataset by either undersampling the majority class
    or oversampling the minority class
    
    Args:
        dataset: EdgePredictionDataset instance
        undersample: If True, undersample majority class. If False, oversample minority class
    
    Returns:
        Balanced EdgePredictionDataset
    """
    # Get all indices and their corresponding labels
    all_indices = np.arange(len(dataset))
    all_labels = np.array([dataset.labels[i] for i in range(len(dataset))])
    
    # Separate indices by class
    pos_indices = all_indices[all_labels == 1]
    neg_indices = all_indices[all_labels == 0]
    
    # Get counts
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    print(f"Original dataset distribution:")
    print(f"Positive samples: {n_pos}")
    print(f"Negative samples: {n_neg}")
    print(f"Ratio (pos/neg): {n_pos/n_neg:.3f}")
    
    if undersample:
        # Undersample majority class
        if n_pos > n_neg:
            pos_indices = np.random.choice(pos_indices, size=n_neg, replace=False)
        else:
            neg_indices = np.random.choice(neg_indices, size=n_pos, replace=False)
    else:
        # Oversample minority class
        if n_pos < n_neg:
            pos_indices = np.random.choice(pos_indices, size=n_neg, replace=True)
        else:
            neg_indices = np.random.choice(neg_indices, size=n_pos, replace=True)
    
    # Combine indices and shuffle
    balanced_indices = np.concatenate([pos_indices, neg_indices])
    np.random.shuffle(balanced_indices)
    
    # Create new balanced dataset
    balanced_dataset = EdgePredictionDataset(
        networks_dir=dataset.networks_dir,
        teps_dir=dataset.teps_dir,
        network_type=dataset.network_type,
        num_train=dataset.num_train,
        num_test=dataset.num_test,
        is_training=dataset.is_training
    )
    
    # Copy the loaded data
    balanced_dataset.networks = dataset.networks
    balanced_dataset.teps = dataset.teps
    
    # Update indices and labels
    balanced_dataset.edge_indices = [dataset.edge_indices[i] for i in balanced_indices]
    balanced_dataset.labels = [dataset.labels[i] for i in balanced_indices]
    
    print(f"\nBalanced dataset distribution:")
    n_pos = sum(balanced_dataset.labels)
    n_neg = len(balanced_dataset.labels) - n_pos
    print(f"Positive samples: {n_pos}")
    print(f"Negative samples: {n_neg}")
    print(f"Ratio (pos/neg): {n_pos/n_neg:.3f}\n\n")
    
    return balanced_dataset


