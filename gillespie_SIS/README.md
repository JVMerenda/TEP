## Step to run the code

### 1. install julia
`curl -fsSL https://install.julialang.org | s`

For more info see [here](https://github.com/JuliaLang/juliaup)

### 2. Clone and initialize the repository
```bash
git clone git@github.com:VMerenda/TEP.git
cd TEP/gillespie_SIS
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### 3. Run the code from the command line
`julia --project generate_tep.jl --N_vertices 900 --N 4 --dt 1.`

### 4. Experiments

See the [experimental setup](experimental_setup.md) for more information on the processes that generated the data in `results`.

## Information about the script

This script is meant to be used by command line, with optional arguments defined in the function below.

Read graphs as adjacency matrices from a directory and generate TEPs from them, using a stochastic SIS process.

Alternatively, generate multiple random graphs using the Erdős-Rényi model, simulate stochastic SIS model on it and store a tep of the results.
For each invocation one single new graph is generated, and the SIS model is simulated on it `N_teps` times.

The simulated process is exact in continuous time. By the optional argument --dt it is possible to sample the tep at discrete time steps.
If not given, the exact tep is stored as a list of time points and the vertex indices that change state at that point.

### Optional arguments
- `--input` Input file or directory with adjacency matrices, if not given ER graphs are created (default "")
- `--N_teps` Number of teps to generate per graph (default 1)
- `--lambda` Infection rate (default 0.03)
- `--mu` Healing rate (default 0.09)
- `--T` Time period (default 100.0)
- `--use-msis` Use the Metapopulation model with mobility MSIS (default false)
- `--delta` Mobility rate in MSIS (default 0.1)
- `--ppn` Initial number of people per vertex in MSIS (default 30)
- `--input` Input file or directory with adjacency matrices
- `--output` Output directory (default ".")
- `--dt` Sampling steps as an array (e.g. [.1,1.,10.]; if nothing is given, the exact tep is returned
- `--plot` Flag to plot the evolution of infectious density (recommend to use with only one thread)
- `--allow-dieout` Flag to also store the result if the infection has died out by time `T`
- `--store-tep` Flag to store the TEP
- `--store-mutual-info` Flag to store the mutual information
- `--mutual-info-word-length` Word length for the mutual information calculation (default 5)
- `--mutual-info-dt` Time step for the mutual information calculation (default .1)

### Stores in the output directory
- `er-\$i.npz` Adjacency matrix of the generated graph
- `tep-'graphname'-\$i-\$j.npz` Tep of the \$j-th simulation of the \$i-th graph (if --dt is not given)
- `tep-'graphname'-\$i-\$j-\$dt.npz` Tep of the \$j-th simulation of the \$i-th graph sampled at time step \$dt
- `rho-$i-$j-$dt.npz` If `--plot`. Evolution of the infectious density of the $j-th simulation of the $i-th graph sampled at time step $dt

### Example

#### Importing networks
For the example first generate the networks (but create zero teps)
```bash
julia --project generate_tep.jl --N_graphs 4 --N_vertices 0 --N_teps 0 --output graphs/
```
Evaluate a single network
```bash
julia --project generate_tep.jl --input graphs/graph-1.npz --N_teps 10 --output graphs/ --dt [1.,]
```
Evaluate all networks
```bash
julia --project -t 4 generate_tep.jl --input graphs/ --N_teps 10 --output graphs/ --dt [.1,]
```
