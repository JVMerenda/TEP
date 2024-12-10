using Graphs
using NPZ
using ProgressMeter

include(joinpath(@__DIR__, "src", "GenerateTep.jl"))
using .GenerateTep
"""
This script is meant to be used by command line, with optional arguments defined in the function below.

Read graphs as adjacency matrices from a directory and generate TEPs from them, using a stochastic SIS process.

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

### Stores in the output directory
- `graph-\$i.npz` Adjacency matrix of the generated graph
- `tep-'graphname'-\$i-\$j.npz` Tep of the \$j-th simulation of the \$i-th graph (if --dt is not given)
- `tep-'graphname'-\$i-\$j-\$dt.npz` Tep of the \$j-th simulation of the \$i-th graph sampled at time step \$dt

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
julia --project -t 4 generate_tep.jl --input graphs/ --N_teps 10 --output graphs/ --dt [.1,.5]
```
"""
function main(
        N_teps::Int64, λ::Float64, μ::Float64, T::Float64,dts::Vector{Float64},
        input::AbstractString, output_dir::AbstractString, use_msis::Bool,
        δ::Float64, ppn::Int64, allow_dieout::Bool, create_plot::Bool
    )
    Dynamic = use_msis ? MSIS : SIS

    tep_pad = i -> lpad(i, length(string(N_teps)), '0')
    graphs = read_graph(input)

    isdir(output_dir) || mkpath(output_dir)
    cd(output_dir)

    # Generate N_graphs graphs
    for (g_name, graph) in graphs
        jset, vtj, jtv = Dynamic.generate_jump_sets(graph)
        npzwrite("$(g_name).npz", adjacency_matrix(graph))

        # And simulate N_teps times
        inner_pb = Progress(N_teps; dt=1, desc="TEPs for graph $(g_name)")
        Threads.@threads for j in 1:N_teps
            tepname = "tep-$g_name-$(tep_pad(j)).npz"
            if isfile(tepname)
                @info "Skipping $(tepname), since it exists"
                next!(inner_pb)
                continue
            end

            sol = Dynamic.solve_problem(λ, μ, nv(graph), T, jset, vtj, jtv; δ, ppn)
            while !Dynamic.is_success(sol, allow_dieout)
                sol = Dynamic.solve_problem(λ, μ, nv(graph), T, jset, vtj, jtv; δ, ppn)
            end

            # Result storage
            if isempty(dts)
                npzwrite(tepname, Dynamic.to_tep(sol))
            else
                map(dt -> npzwrite("tep-$(g_name)-$(tep_pad(j))-$dt.npz", Dynamic.to_tep(sol, dt)), dts)
            end
            if(create_plot)
                ts = 0:0.1:T
                p = Dynamic.plot_density(sol, ts, graph, g_name, j)
                savefig(p, "rho-$(g_name)-$(tep_pad(j)).png")
            end
            next!(inner_pb)
        end
    end
    return 0
end

# Preparation
args = parse_command_line_args()
args["plot"] && using Plots
main(args["N_teps"], args["lambda"], args["mu"], args["T"],
     args["dt"], args["input"], args["output"], args["use-msis"],
     args["delta"], args["ppn"], args["allow-dieout"], args["plot"])
