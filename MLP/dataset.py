import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, TensorDataset, DataLoader
import itertools
import numpy as np
from sklearn.preprocessing import StandardScaler

class PairwiseSlidingDataset(Dataset):
    """
    For each node pair (i<j), we split its time series (T,2) into multiple
    windows of length `window_size`, stepping by `stride`.
    Each window is one sample, labeled by adjacency[i,j].
    """
    def __init__(self, tep, adjacency_matrix, window_size=100, stride=50):
        """
        Args:
            tep: np.ndarray of shape (T, N).
            adjacency_matrix: np.ndarray of shape (N, N) => adjacency[i,j] in {0,1}.
            window_size: number of timesteps in each subsequence window.
            stride: number of timesteps to move between consecutive windows.
        """
        super().__init__()
        self.tep = tep               # shape (T, N)
        self.adj = adjacency_matrix  # shape (N, N)
        
        self.T, self.N = self.tep.shape
        self.window_size = window_size
        self.stride = stride
        
        # We'll build a list of samples. Each entry is (i, j, start_t)
        # meaning node pair (i,j) and the window [start_t : start_t+window_size)
        self.samples = []
        
        # Go through all (i<j) node pairs
        for i, j in itertools.combinations(range(self.N), 2):
            label = self.adj[i, j]  # 0 or 1
            # If label is not strictly 0/1, you might binarize or skip as needed
            
            # For each pair, we create subwindows
            start_idx = 0
            while start_idx + self.window_size <= self.T:
                self.samples.append((i, j, start_idx, label))
                start_idx += self.stride
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        i, j, start_t, label = self.samples[idx]
        
        # shape (window_size, 2) => [ [tep[t,i], tep[t,j]] for t in window ]
        pair_window = self.tep[start_t : start_t + self.window_size, [i, j]]  
        
        x_torch = torch.tensor(pair_window, dtype=torch.float)
        y_torch = torch.tensor(label,       dtype=torch.float)
        return x_torch, y_torch


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


def create_full_dataset(file_paths, train=True, window_size=100, stride=50, chunk_size=100):
    datasets = []
    file_count = 20 if train else 5
    
    for path in file_paths:
        data = np.load(path)
        adjacency_vector = data["graph"]
        adjacency_matrix = vector_to_adjacency(adjacency_vector, n_nodes=100)
        
        for k in range(file_count):
            tep_raw = data[f"arr_{k}"]  # shape (1000, 100)
            
            # 1) Downsample from (1000, 100) to (10, 100)
            #    e.g. chunk_size=100 => we get shape (10, 100)
            tep = downsample_binary_series(tep_raw, chunk_size=chunk_size)
            
            # 2) Now 'tep' is shorter and continuous. 
            #    We can feed it into PairwiseSlidingDataset 
            #    with window_size= (whatever <= 10).
            ds = PairwiseSlidingDataset(
                tep,
                adjacency_matrix,
                window_size=tep_raw.shape[0]//chunk_size, 
                stride=tep_raw.shape[0]//chunk_size
            )
            datasets.append(ds)
    
    return ConcatDataset(datasets)

def create_full_dataset_with_scaling(
    file_paths,
    scaler,
    train=True,
    window_size=10,
    stride=10,
    chunk_size=100
):
    """
    Similar to your existing create_full_dataset, but we:
      1) load each TEP,
      2) (optionally) downsample,
      3) transform with the 'scaler' (already fitted on training data),
      4) build PairwiseSlidingDataset
    """
    from torch.utils.data import ConcatDataset
    
    datasets = []
    file_count = 20 if train else 5
    
    for path in file_paths:
        data = np.load(path)
        adjacency_matrix = vector_to_adjacency(data["graph"], n_nodes=100)
        
        for k in range(file_count):
            tep_raw = data[f"arr_{k}"]  # e.g., shape (1000, 100)
            
            # (Optional) downsample 
            tep_down = downsample_binary_series(tep_raw, chunk_size=chunk_size)  # => (10, 100)
            
            # Transform the shape to (samples=10, features=100) for scaling
            # We apply 'scaler.transform'
            tep_scaled = scaler.transform(tep_down)  # shape (10, 100)
            
            # Now pass the scaled TEP to your PairwiseSlidingDataset
            ds = PairwiseSlidingDataset(
                tep_scaled,
                adjacency_matrix,
                window_size=tep_raw.shape[0]//chunk_size,
                stride=tep_raw.shape[0]//chunk_size
            )
            datasets.append(ds)
    
    return ConcatDataset(datasets)

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

def gather_and_balance_dataset(dataset: Dataset):
    """
    1) Iterates through 'dataset' to collect all (X, y) in memory.
    2) Identifies which class is the majority (0 or 1).
    3) Randomly undersamples the majority class so that both classes have equal count.
    4) Returns a new balanced TensorDataset (X_balanced, y_balanced).
    """
    # Gather the entire dataset in memory
    X_list = []
    y_list = []
    for i in range(len(dataset)):
        X_i, y_i = dataset[i]
        # X_i shape = (seq_len, 2), y_i shape = scalar
        X_list.append(X_i.unsqueeze(0))   # add batch dimension => (1, seq_len, 2)
        y_list.append(y_i)
    
    # Stack them
    X_full = torch.cat(X_list, dim=0)  # shape: (N, seq_len, 2)
    y_full = torch.stack(y_list, dim=0)  # shape: (N,)

    # Identify minority & majority
    pos_indices = (y_full == 1).nonzero(as_tuple=True)[0]
    neg_indices = (y_full == 0).nonzero(as_tuple=True)[0]
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    if n_pos == 0 or n_neg == 0:
        # Edge case: if your data is all one class or the other, skip balancing
        print("Warning: dataset has only one class; cannot balance.")
        return TensorDataset(X_full, y_full)
    
    # Randomly shuffle whichever set is larger
    pos_indices = pos_indices[torch.randperm(n_pos)]
    neg_indices = neg_indices[torch.randperm(n_neg)]
    # Undersample to match the smaller count
    min_count = min(n_pos, n_neg)
    pos_indices = pos_indices[:min_count]
    neg_indices = neg_indices[:min_count]
    
    # Combine & shuffle
    balanced_indices = torch.cat([pos_indices, neg_indices], dim=0)
    balanced_indices = balanced_indices[torch.randperm(len(balanced_indices))]
    
    X_balanced = X_full[balanced_indices]
    y_balanced = y_full[balanced_indices]

    # Return as a TensorDataset
    return TensorDataset(X_balanced, y_balanced)

def downsample_binary_series(tep, chunk_size=100):
    """
    tep: np.ndarray of shape (T, N), e.g. (1000, N).
    chunk_size: how many timesteps per segment, e.g. 100 => 10 segments if T=1000.

    Returns a new array of shape (T_new, N) = (T/chunk_size, N).
    Each entry is the mean (or fraction of 1s) in that segment.
    """
    T, N = tep.shape
    assert T % chunk_size == 0, "For simplicity, T should be divisible by chunk_size."
    
    num_chunks = T // chunk_size
    # We'll build an output of shape (num_chunks, N)
    pooled = np.zeros((num_chunks, N), dtype=np.float32)
    
    for i in range(num_chunks):
        start = i * chunk_size
        end   = start + chunk_size
        # slice shape: (chunk_size, N)
        segment = tep[start:end, :]
        # e.g. average across time dimension => shape (N,)
        # fraction of 1s => segment.mean(axis=0)
        pooled[i, :] = segment.mean(axis=0)
    
    return pooled

def fit_scaler_on_train(train_files, file_count=20):
    """
    1) Loads all TEPs from 'train_files' (arr_0..arr_19 each).
    2) (Optional) Downsamples them to shape (10, 100) or leaves them as (1000, 100).
    3) Stacks them for the scaler.
    4) Fits and returns the scaler.
    """
    all_data = []
    
    for path in train_files:
        data = np.load(path)
        for k in range(file_count):
            tep_raw = data[f"arr_{k}"]  # shape (1000, 100)
            
            # (Optional) downsample from 1000 -> 10 (or do no downsampling if you want raw)
            # Here we assume a chunk_size=100 => (10, 100)
            tep_down = downsample_binary_series(tep_raw, chunk_size=100)
            
            # Now tep_down has shape (10, 100). We want to feed it to the scaler
            # But the StandardScaler expects 2D: (n_samples, n_features).
            # We'll interpret each row as a new "sample", each column as a "feature".
            # So shape (10, 100) is already (samples=10, features=100).
            # Alternatively, you might flatten time+node dimensions differently.
            
            all_data.append(tep_down)  # shape (10, 100)
    
    # Stack into one big (N_total, 100) array
    # Suppose we have M TEPs, each with shape (10, 100). Then shape => (M*10, 100).
    all_data_stacked = np.concatenate(all_data, axis=0)
    
    # Create and fit a StandardScaler
    scaler = StandardScaler()
    scaler.fit(all_data_stacked)  # learns mean & std for each of the 100 features
    
    return scaler