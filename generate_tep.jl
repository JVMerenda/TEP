using Graphs
using NPZ
using ProgressMeter

include(joinpath(@__DIR__, "src", "GenerateTep.jl"))
using .GenerateTep
"""
This script is meant to be used by command line, with optional arguments defined in the function below.

Generate a random graph using the Erdős-Rényi model, simulate stochastic SIS model on it and store a tep of the results.
For each invocation one single new graph is generated, and the SIS model is simulated on it `N` times.

The simulated process is exact in continuous time. By the optional argument --dt it is possible to sample the tep at discrete time steps.
If not given, the exact tep is stored as a list of time points and the vertex indices that change state at that point.

### Optional arguments
- `--N_graphs` Number of graphs to generate (default 1)
- `--N_vertices` Number of vertices per graph (default 1000)
- `--N_teps` Number of teps to generate per graph (default 1)
- `--p` Edge probability (default 0.01)
- `--lambda` Infection rate (default 0.03)
- `--mu` Healing rate (default 0.09)
- `--T` Time period (default 100.0)
- `--output` Output directory (default ".")
- `--dt` Sampling steps as an array (e.g. [.1,1.,10.]; if nothing is given, the exact tep is returned

### Stores in the output directory
- `graph-$i.npz` Adjacency matrix of the generated graph
- `tep-$i-$j.npz` Tep of the $j-th simulation of the $i-th graph (if --dt is not given)
- `tep-$i-$j-$dt.npz` Tep of the $j-th simulation of the $i-th graph sampled at time step $dt

### Example
```bash
julia generate_tep.jl --N_graphs 4 --N_vertices 100 --N_teps 10 --p 0.04 --lambda 0.01 --mu 0.03 --T 300.0 --output g1/ --dt [1.,]
```
"""
function main()
    # Preparation
    args = parse_command_line_args()
    dts = args["dt"]
    isdir(args["output"]) || mkdir(args["output"])
    cd(args["output"])

    # Generate N_graphs graphs
    @showprogress for i in 1:args["N_graphs"]
        graph = erdos_renyi(args["N_vertices"], args["p"])
        jset, vtj, jtv = generate_jump_sets(graph)
        npzwrite("graph-$i.npz", adjacency_matrix(graph))
        
        # And simulate N_teps times
        inner_pb = Progress(args["N_teps"]; dt=1, desc="TEPs")
        Threads.@threads for j in 1:args["N_teps"]
            sol = solve_problem(args["lambda"], args["mu"], args["N_vertices"], args["T"], jset, vtj, jtv)
            
            # Result storage
            if isnothing(dts)
                npzwrite("tep-$i-$j.npz", to_tep(sol))
            else
                map(dt -> npzwrite("tep-$i-$j-$dt.npz", to_tep(sol, dt)), dts)
            end
            next!(inner_pb)
        end
    end
    return 0
end

main()
