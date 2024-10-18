## Step to run the code

### 1. install julia
`curl -fsSL https://install.julialang.org | s`

For more info see [here](https://github.com/JuliaLang/juliaup)

### 2. Clone and initialize the repository
```bash
git clone git@github.com:TimVWese/contact-recovery.git
cd contact-recovery
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### 3. Run the code from the command line
`julia --project generate_tep.jl --n 900 --N 4 --dt 1.`

## Information about the script

This script is meant to be used by command line, with optional arguments defined in the function below.

Generate a random graph using the Erdős-Rényi model, simulate stochastic SIS model on it and store a tep of the results.
For each invocation one single new graph is generated, and the SIS model is simulated on it `N` times.

The simulated process is exact in continuous time. By the optional argument --dt it is possible to sample the tep at discrete time steps.
If not given, the exact tep is stored as a list of time points and the vertex indices that change state at that point.

### Optional arguments
- `--n` Number of vertices in the graph (default 1000)
- `--p` Probability of an edge between two vertices (default 0.01)
- `--N` Number of simulations to run (default 1)
- `--lambda` Infection rate (default 0.03)
- `--mu` Recovery rate (default 0.09)
- `--T` Maximum time to simulate (default 100)
- `--dt` Time step for the discrete time sampling of the tep, if not given the exact tep is stored

### Example
```bash
julia --project generate_tep.jl --n 900 --N 4 --dt 1.
```
