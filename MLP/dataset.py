import torch
from torch.utils.data import Dataset, ConcatDataset
import itertools
import numpy as np

class PairwiseTEPDataset(Dataset):
    """
    Enumerate all pairs of nodes (i,j) from a single TEP and adjacency matrix.
    Each sample is an input of shape (T, 2) and a label y ∈ {0,1}.
    """
    def __init__(self, tep, adjacency_matrix):
        """
        Args:
            tep: np.ndarray of shape (T, N) => (timesteps, number_of_nodes).
            adjacency_matrix: np.ndarray of shape (N, N), adjacency[i,j] = 0 or 1.
        """
        super().__init__()
        self.tep = tep  # shape (T, N)
        self.adj = adjacency_matrix  # shape (N, N)
        
        # We'll gather all (i < j) node pairs
        self.n_nodes = tep.shape[1]
        self.pairs = []
        for i, j in itertools.combinations(range(self.n_nodes), 2):
            self.pairs.append((i, j))
    
    def __len__(self):
        return len(self.pairs)  # typically N*(N-1)/2 for an undirected graph
    
    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        # Build shape (T, 2) => each row is [TEP_t(i), TEP_t(j)]
        # (Time dimension T is along axis=0 in self.tep)
        pair_data = self.tep[:, [i, j]]  # shape (T, 2)
        
        # Convert adjacency to 0 or 1
        label = self.adj[i, j]  # 0 or 1
        
        # We'll return them as torch Tensors
        pair_data_torch = torch.tensor(pair_data, dtype=torch.float)
        label_torch = torch.tensor(label, dtype=torch.float)
        return pair_data_torch, label_torch


def create_full_dataset(file_paths, train=True):
    # file_paths: a list of .npz paths
    # We create a list of PairwiseTEPDataset objects and then Concat them.
    datasets = []
    file_count = 20 if train else 5
    
    for path in file_paths:
        data = np.load(path)
        adjacency_vector = data["graph"]  # shape (4950,)
        # convert to full (N, N)
        adjacency_matrix = vector_to_adjacency(adjacency_vector, n_nodes=100)
        
        # For training TEPs, e.g. arr_0..arr_19
        for k in range(file_count):
            tep = data[f"arr_{k}"]  # shape (T=1000, N=100)
            ds = PairwiseTEPDataset(tep, adjacency_matrix)
            datasets.append(ds)
    
    # Merge them all
    full_dataset = ConcatDataset(datasets)
    return full_dataset

def vector_to_adjacency(vec, n_nodes=100):
    """
    Turns a length-(n_nodes*(n_nodes-1)/2) upper-triangular vector
    into a full adjacency matrix (symmetric, zeros on diag).
    """
    adj = np.zeros((n_nodes, n_nodes), dtype=vec.dtype)
    # Fill the upper triangle
    tri_indices = np.triu_indices(n_nodes, k=1)
    adj[tri_indices] = vec
    # Make symmetric
    adj = adj + adj.T
    return adj