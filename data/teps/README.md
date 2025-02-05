This folder contains the teps on which LSTM's can be trained.
Each TEP is sampled from a continuous stochastic process, and is sampled with timestep 0.1 (see gillespie_SIS for details).
Per network size, there is a train and test folder.
Each folder contains files in the format `{graph_name}.npz`, where `graph_name` usually consits of the abbriviation of a model and a number.
Each file contains a dictionary with the following keys:
 - `graph`: The adjacency matrix of the graph, the upper triangular part vectorized
 - `arr_i`: The i'th tep on `graph` as a (n_timesteps, n_vertices) array

All data is stored as arrays of `uint8`.

Below is code that can load the data
```python
import numpy as np
data = np.load("ba-01.npz")
labels = data["graph"]
teps = [data[f"arr_{i}"] for i in range(len(data)-1)]
```

For each of 40 networks 20 teps are added as training data, and 5 as test.
25 teps of an additional 10 networks are added as test data. 
