"""
# read_datasets.py

Helper file to read mutual_information datasets and corresponding graph labels.

`handle_data(prefix)` is the main function and given a path prefix, it reads the dataset and labels files.
For example
```python
data, labels = handle_dataset(BASE_DIR + "graphs/real/malawi/full")
```
loads the mutual information and underlying graph labels for the malawi dataset over the complete period.

`handle_multiple_datasets(path, graph_names)` is a helper function to read multiple datasets at once.
For example
```python
weeks = ["week_01", "week_02"]
data, labels = handle_multiple_datasets(BASE_DIR + "graphs/real/malawi", weeks)
```
or for the daily slicing
```python
days = [f"day_{i:02d}" for i in range(1, 15)]
data, labels = handle_multiple_datasets(BASE_DIR + "graphs/real/malawi", days)
```

With `handle_sis_graphs` we can read the mutual information and labels for TEPS for SIS on multiple networks.
For example
```python
handle_sis_graphs(100, ["er", "ba"], [1, 2, 3])
```
loads the mutual information and labels for the first three graphs with the erdos-renyi and barabasi-albert models with 100 vertices.
To load all data for all models, we can use
```python
models = ["er", "ba", "ws", "euc", "geo", "reg", "sf"]
models = models + [f"{model}md" for model in models]
models.append("gridmd")
handle_sis_graphs(100, models, range(1, 51))
```

"""

import math
import torch
import pandas as pd
from tqdm import tqdm

BASE_DIR = '/home/DATA/datasets/TEP/gillespie_SIS/results/'

def abrv_to_full(abrv):
    abrv_to_full = {
        "er": "erdos-renyi",
        "ba": "barabasi-albert",
        "ws": "watts-strogatz",
        "euc": "euclidean",
        "geo": "geometric",
        "grid": "grid",
        "reg": "regular",
        "sf": "scale-free",
    }

    if abrv[-2:] == "md":
        return abrv_to_full[abrv[:-2]] + "-multi-degree"
    return abrv_to_full[abrv]

def vector_to_symmetric_matrix(v, include_diagonal=True, dtype=torch.float32):
    """
    Reconstruct a symmetric NxN matrix from its upper triangular vector.

    Args:
        v (numpy.ndarray): Vector containing the upper triangular elements.
        include_diagonal (bool): Whether the vector includes the diagonal elements.

    Returns:
        torch.Tensor: Reconstructed symmetric matrix.
    """
    if include_diagonal:
        N = int((-1 + math.sqrt(1 + 8 * len(v))) / 2)
    else:
        # Solve N from v.numel() = N(N - 1)/2
        N = int((1 + math.sqrt(1 + 8 * len(v))) / 2)

    mat = torch.zeros(N, N, dtype=dtype)

    offset = 0 if include_diagonal else 1
    triu_indices = torch.triu_indices(N, N, offset=offset)

    mat[triu_indices[0], triu_indices[1]] = torch.tensor(v, dtype=dtype)
    mat = mat + mat.t() - torch.diag(mat.diag())

    return mat


def handle_dataset(prefix, dtype=torch.float32):
    """
    Load a dataset (mutual information matrix) and its corresponding labels (adjacency matrix),
    and process them into a format suitable for training.

    Args:
        prefix (str): Prefix for the dataset and labels files, i.e. the path such
            that the mutual information is stored with "_dataset.csv" and the graph
            labels with "_labels.csv" appended.

    Returns:
        torch.Tensor: Processed samples (shape: (N_teps X N_edges_to_predict X 2N_verices)).
        torch.Tensor: Processed labels (shape: (N_teps X N_edges_to_predict)).
    """
    samples = []
    targets = []

    graph_name = prefix.split("/")[-1]
    data = pd.read_csv(f'{prefix}_dataset.csv', header=None).values
    labels = pd.read_csv(f'{prefix}_labels.csv', header=None).values

    adjacency_matrix = vector_to_symmetric_matrix(labels[0], include_diagonal=False, dtype=dtype)
    N = adjacency_matrix.shape[0]
    assert N * (N + 1) == 2 * data.shape[1], "Invalid data and labels dimensions"

    pair_indices = torch.triu_indices(N, N, offset=1)

    for sample in tqdm(data, desc=f"Processing {graph_name}", leave=False):
        sample_tensor = vector_to_symmetric_matrix(sample, include_diagonal=True, dtype=dtype)
        sample_processed = torch.vstack([torch.cat((sample_tensor[i,:], sample_tensor[j,:]))
                                          for i, j in zip(pair_indices[0], pair_indices[1])])
        
        label_processed = torch.tensor([adjacency_matrix[i, j] for i, j in zip(pair_indices[0], pair_indices[1])],
                                        dtype=dtype)
        
        samples.append(sample_processed)
        targets.append(label_processed)

    return torch.stack(samples), torch.stack(targets)

def handle_multiple_datasets(path, graph_names, dtype=torch.float32):
    """
    Load the mutual information data and adjacency matrix labels for multiple graphs stored in
    the same directory
    """
    samples = []
    targets = []

    for graph_name in tqdm(graph_names, desc="Processing graphs", leave=False):
        g_samples, g_targets = handle_dataset(f'{path}/{graph_name}', dtype=dtype)

        samples.append(g_samples)
        targets.append(g_targets)

    return torch.cat(samples), torch.cat(targets)

def handle_sis_graphs(graph_size, graph_models, graph_idxs, nb_digits_g=2, base_dir=BASE_DIR+"sis/", dtype=torch.float32):
    """
    Load the mutual information data and adjacency matrix labels for multiple SIS networks

    Args:
        graph_size (int): Number of vertices in the graph.
        graph_models (list): List of graph models to load (abbreviated).
        graph_idxs (list): List of graph indices to load.
        nb_digits_g (int): Number of digits used to store the graph index.
        base_dir (str): Base directory for the SIS datasets.
    """
    samples = []
    targets = []

    for graph_model in tqdm(graph_models, desc="Processing graph models"):
        graphs = [f"{graph_model}-{graph_idx:0{nb_digits_g}d}" for graph_idx in graph_idxs]
        dir = f"{base_dir}/{abrv_to_full(graph_model)}/N{graph_size}"
        g_samples, g_targets = handle_multiple_datasets(dir, graphs, dtype=dtype)
        samples.append(g_samples)
        targets.append(g_targets)

    return torch.cat(samples), torch.cat(targets)

if __name__ == "__main__":
    N = 100
    models = ["er", "ba", "ws", "euc", "geo", "reg", "sf"]
    models = models + [f"{model}md" for model in models]
    models.append("gridmd")

    data, labels = handle_sis_graphs(N, models, range(1, 51))
    torch.save(data, f'sis_N{N}_data.pt')
    torch.save(labels, f'sis_N{N}_labels.pt')
