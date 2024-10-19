using Graphs
using NPZ

include(joinpath(@__DIR__, "src", "GenerateTep.jl"))
using .GenerateTep
"""
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
"""
function main()
    args = parse_command_line_args()
    graph = erdos_renyi(args["n"], args["p"])
    jset, vtj, jtv = generate_jump_sets(graph)
    npzwrite("graph.npz", adjacency_matrix(graph))
    Threads.@threads for i in 1:args["N"]
        sol = solve_problem(args["lambda"], args["mu"], args["n"], args["T"], jset, vtj, jtv)
        tep = isnothing(args["dt"]) ? to_tep(sol) : to_tep(sol, args["dt"])
        npzwrite("tep-$i.npz", tep)
    end
    return 0
end

main()
