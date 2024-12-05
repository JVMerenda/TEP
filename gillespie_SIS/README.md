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
- `--N_graphs` Number of graphs to generate (default 1)
- `--N_vertices` Number of vertices per graph (default 1000)
- `--N_teps` Number of teps to generate per graph (default 1)
- `--p` Edge probability (default 0.01)
- `--lambda` Infection rate (default 0.03)
- `--mu` Healing rate (default 0.09)
- `--T` Time period (default 100.0)
- `--output` Output directory (default ".")
- `--dt` Sampling steps as an array (e.g. [.1,1.,10.]; if nothing is given, the exact tep is returned
- `--plot` Flag to plot the evolution of infectious density (recommend to use with only one thread)
- `--allow-dieout` Flag to also store the result if the infection has died out by time `T`

### Stores in the output directory
- `graph-$i.npz` Adjacency matrix of the generated graph
- `graph-$i-$j.npz` Tep of the $j-th simulation of the $i-th graph (if --dt is not given)
- `graph-$i-$j-$dt.npz` Tep of the $j-th simulation of the $i-th graph sampled at time step $dt
- `rho-$i-$j-$dt.npz` If `--plot`. Evolution of the infectious density of the $j-th simulation of the $i-th graph sampled at time step $dt

### Example
#### Generating networks
The `-t` flag is used to specify the number of threads that julia is allowed to use.
The `--project` flag ensures that Julia uses the correct environment.

```bash
julia --project -t 2 generate_tep.jl --N_graphs 4 --N_vertices 100 --N_teps 10 --p 0.04 --lambda 0.01 --mu 0.03 --T 300.0 --output N100/ --dt [1.,]
```
```bash
julia  --project -t 1 generate_tep.jl --N_vertices 100 --N_teps 5 --p 0.01 --lambda 0.08 --mu 0.06 --output N100/ --plot
```

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
