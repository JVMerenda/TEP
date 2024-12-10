import numpy as np
from pathlib import Path

from mutual_information import mutual_information_matrix

class SIS_TEP:
    """
    A class for handling Temporal Event Process (TEP) data from SIS simulations.

    This class provides functionality to:
    - Load and process TEP data from various network types (e.g., erdos-renyi, barabasi-albert)
    - Sample system states at specified time points or intervals
    - Calculate and store mutual information matrices
    - Manage associated graph data

    Attributes:
        root (str): Root directory for specific network type and size
        network_type (str): Type of network (abbreviated)
        graph_id (str): Graph identifier
        tep_id (str): TEP identifier
        data (ndarray): Loaded TEP data
        time_points (ndarray): Transition time points
        vertex_indices (ndarray): Vertex indices for transitions
        N_vertices (int): Number of vertices
        max_T (float): Maximum time point

    Example:
        >>> tep = SIS_TEP("er", 100, 1, 1)
        >>> state = tep(0.5)  # Get system state at t=0.5
        >>> samples = tep.sample(0.1)  # Sample with dt=0.1
        >>> M = tep.generate_mutual_info(0.1)  # Generate mutual information matrix
        >>> M = tep.load_or_generate_mutual_info(0.1)  # Load mutual information matrix
    """
    
    __base_dir__ = "/home/DATA/datasets/TEP/gillespie_SIS/results/sis"
    __abrv_to_full__ = {
        "er": "erdos-renyi",
        "ba": "barabasi-albert",
        "ws": "watts-strogatz",
        "euc": "euclidean",
        "geo": "geometric",
        "grid": "grid",
        "reg": "regular",
        "sf": "scale-free",
    }

    def __init__(self, abrv, nb_vertices, i_graph, j_tep, nb_digits_g=2, nb_digits_tep=3):
        """
        Initialize the TEP reader.

        Args:
            abrv (str): Abbreviation for the network type.
            nb_vertices (int): Number of vertices.
            i_graph (int): Graph identifier.
            j_tep (int): TEP identifier.
            nb_digits_g (int, optional): Number of digits that the graph ID is stored in. Defaults to 2.
            nb_digits_tep (int, optional): Number of digits for TEP ID is stored in. Defaults to 3
        """
        self.root = f"{self.__base_dir__}/{self.__abrv_to_full__[abrv]}/N{nb_vertices}"
        self.network_type = abrv
        self.graph_id = f"{i_graph:0{nb_digits_g}d}"
        self.tep_id = f"{j_tep:0{nb_digits_tep}d}"
        self.load_tep()

    def get_tep_location(self):
        return f"{self.root}/tep-{self.network_type}-{self.graph_id}-{self.tep_id}.npz"

    def get_graph_location(self):
        return f"{self.root}/{self.network_type}-{self.graph_id}.npz"

    def get_mutual_info_location(self, dt):
        return f"{self.root}/mim-{self.network_type}-{self.graph_id}-{self.tep_id}-{dt:.2f}.npz"

    def get_sample_location(self, dt):
        return f"{self.root}/tep-{self.network_type}-{self.graph_id}-{self.tep_id}-{dt:.2f}.npy"

    def get_sample_location(self, dt, p):
        return f"{self.root}/tep-{self.network_type}-{self.graph_id}-{self.tep_id}-{dt:.2f}-p{p:.3f}.npy"

    def load_tep(self):
        """
        Load the TEP data, on the location specified by the constructor data.
        """
        filename = self.get_tep_location()
        self.data = np.load(filename)
        # time points at which a transition occurs
        self.time_points = self.data[:, 0]
        # vertex indices of the transitions
        self.vertex_indices = self.data[:, 1].astype(int) - 1
        self.N_vertices = max(self.vertex_indices) + 1
        self.max_T = self.time_points[-1]

    def __call__(self, t):
        """
        Returns the state of the system at time t.
        """
        if t < 0:
            raise ValueError("t must be non-negative.")
        idx = np.searchsorted(self.time_points, t, side='right')
        # The state of the system at time t is the parity of the number of times each vertex has transitioned up to time t.
        return np.array([np.count_nonzero(self.vertex_indices[:idx+1] == v) % 2 for v in range(self.N_vertices)])

    def __call__(self, t, p):
        """
        Returns the state of the system at time t with noise.
        The state of each vertex at time t is changed with probability p.
        """
        x = self(t)
        for i in range(self.N_vertices):
            if np.random.rand() < p:
                x[i] = 1 - x[i]

    def sample_at_ts(self, ts, p=0):
        """
        Sample with noise at time points ts.
        The state of each vertex at each time point is chosen randomly with probability p.
        """
        call_f = (lambda t: self(t, p)) if p > 0 else (lambda t: self(t))
        return np.array([call_f(t) for t in ts])

    def sample_with_dt(self, dt, p=0):
        """
        Samples the TEP with a time step dt and noise.
        """
        ts = np.arange(0, self.max_T + dt, dt)
        return self.sample_at_ts(ts, p) if p > 0 else self(ts)

    def sample(self, t, p=0):
        """
        Generate a TEP at time points ts (if ts is an array) or with a time step dt (if t is a number).
        If p is provided it acts as a probability of flipping the state of each vertex.
        """
        if isinstance(t, float):
            return self.sample_with_dt(t, p) if p > 0 else self.sample_with_dt(t)
        else:
            return self.sample_at_ts(t, p) if p > 0 else self.sample_at_ts(t)

    def store_sample(self, dt):
        """
        Store the TEP sample with time step dt at location
        "../tep-{network_type}-{graph_id}-{tep_id}-{dt:.2f}.npy".
        """
        np.save(self.get_sample_location(dt), self.sample_with_dt(dt))

    def store_sample(self, dt, p):
        """
        Store the TEP sample with time step dt and noise paramater p at location
        "../tep-{network_type}-{graph_id}-{tep_id}-{dt:.2f}-p{p:.3f}.npy".
        """
        np.save(self.get_sample_location(dt, p), self.sample_with_dt(dt, p))

    def load_graph(self):
        """
        Return the adjacency matrix of the graph associated with the TEP.
        """
        return np.load(self.get_graph_location())

    def generate_mutual_info(self, dt, store=True):
        """
        Generate the mutual information matrix for the TEP.
        """
        M = mutual_information_matrix(self.sample_with_dt(dt))
        if store:
            np.savez(self.get_mutual_info_location(dt), M=M)
        return M

    def load_mutual_info(self, dt):
        """
        Load the mutual information matrix for the TEP.
        """
        return np.load(self.get_mutual_info_location(dt))['M']

    def load_or_generate_mutual_info(self, dt):
        """
        Load the mutual information matrix if it exists, otherwise generate (and store) it.
        """
        path = Path(self.get_mutual_info_location(dt))
        if path.exists():
            try:
                return np.load(path)['M']
            except:
                pass
        return self.generate_mutual_info(dt)

